import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';
import { Point, Angle, toRad, toDeg } from '../math';
import { Truck } from '../model/truck';
import { Dock } from '../model/world';
import { PlantConventions, DockReference, AngleWrapping } from '../model/conventions';
import { NeuralNet, NetConfig, LayerConfig } from '../neuralnet/net';
import { ActivationFunction, Tanh, Sigmoid, SymmetricSigmoid } from '../neuralnet/activation';
import { AdalineUnit } from '../neuralnet/unit';
import { Optimizer, SGD, SGDNesterovMomentum } from '../neuralnet/optimizers';
import { WeightInitializer, TwoLayerInitializer, RandomWeightInitializer } from '../neuralnet/weightinitializer';
import { MSE, ControllerError, TruckControllerError, D2ControllerError, BestApproachD2ControllerError } from '../neuralnet/error';
import { Observation, observationForInputCount } from '../neuralnet/observation';
import { Args } from './args';

export type HiddenActivationName = "tanh" | "logistic";
export type OutputMapName = "tanh" | "2s-1";
export type ObjectiveName = "demo" | "d2-terminal" | "d2-best";
export type StartSchemeName = "curriculum" | "ensemble" | "point" | "yard";
export type OptimizerName = "nesterov" | "sgd";

export interface NetOptions {
    inputs: number;
    hidden: number;
    activation: HiddenActivationName;
    outputMap: OutputMapName;
}

export interface OptimizerOptions {
    name: OptimizerName;
    learningRate: number;
    momentum: number;
}

export interface StartOptions {
    scheme: StartSchemeName;
    seed: number;
    count: number;
}

/** One start state of a rollout or a training sample. */
export interface Start {
    x: number;
    y: number;
    trailerAngle: Angle;
    cabinAngle: Angle;
    scheme: string;
    idx: number;
}

/**
 * A seeded linear congruential generator, so that a run is reproducible from
 * its seed. Installed over Math.random because the weight initializers and the
 * plant's randomization both reach for it directly.
 */
export function installSeededRandom(seed: number): void {
    let state = (seed >>> 0) || 1;
    Math.random = () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

export function makeSeededRandom(seed: number): () => number {
    let state = (seed >>> 0) || 1;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

/** The repository this build came from, so that a result can be traced back. */
export function getRepoCommit(): string {
    try {
        return execFileSync("git", ["-C", getRepoRoot(), "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
    } catch (e) {
        return "unknown";
    }
}

export function isRepoDirty(): boolean {
    try {
        return execFileSync("git", ["-C", getRepoRoot(), "status", "--porcelain"], { encoding: "utf8" }).trim().length > 0;
    } catch (e) {
        return false;
    }
}

/**
 * The repository root, resolved from this file rather than the working
 * directory: the original trainers only ran from one specific cwd because they
 * read the emulator weights through a relative path.
 */
export function getRepoRoot(): string {
    return path.resolve(__dirname, "..", "..");
}

export function getDefaultEmulatorWeights(): string {
    return path.join(getRepoRoot(), "src", "weights", "truck_emulator_weights");
}

export function hiddenActivation(name: HiddenActivationName): ActivationFunction {
    return name == "logistic" ? new Sigmoid() : new Tanh();
}

export function outputActivation(name: OutputMapName): ActivationFunction {
    return name == "2s-1" ? new SymmetricSigmoid() : new Tanh();
}

export function makeOptimizer(options: OptimizerOptions): () => Optimizer {
    if (options.name == "sgd") {
        return () => new SGD(options.learningRate);
    }
    return () => new SGDNesterovMomentum(options.learningRate, options.momentum);
}

export function describeOptimizer(options: OptimizerOptions): string {
    return options.name == "sgd"
        ? "SGD(" + options.learningRate + ")"
        : "SGDNesterovMomentum(" + options.learningRate + ", " + options.momentum + ")";
}

/**
 * The controller net, shaped from the options rather than hardcoded: the demo's
 * 4-45-1 tanh, the paper's 4-9-1 logistic with a 2s-1 output, its 3-7-1 and its
 * 8-17-1 are all the same net with different arguments.
 */
export function buildControllerNet(net: NetOptions, optimizer: OptimizerOptions): NeuralNet {
    let hiddenLayer: LayerConfig = {
        neuronCount: net.hidden,
        weightInitializer: new TwoLayerInitializer(0.7, net.hidden),
        unitConstructor: (weights: number, activation: ActivationFunction, initialWeightRange: WeightInitializer, opt: Optimizer) =>
            new AdalineUnit(weights, activation, initialWeightRange, opt),
        activation: hiddenActivation(net.activation)
    };
    let outputLayer: LayerConfig = {
        neuronCount: 1,
        weightInitializer: new RandomWeightInitializer(0.01),
        unitConstructor: (weights: number, activation: ActivationFunction, initialWeightRange: WeightInitializer, opt: Optimizer) =>
            new AdalineUnit(weights, activation, initialWeightRange, opt),
        activation: outputActivation(net.outputMap)
    };
    let config: NetConfig = {
        inputs: net.inputs,
        optimizer: makeOptimizer(optimizer),
        errorFunction: new MSE(), // ignored for the controller
        layerConfigs: [hiddenLayer, outputLayer]
    };
    return new NeuralNet(config);
}

export function netShape(net: NetOptions): number[] {
    return [net.inputs, net.hidden, 1];
}

export function buildObservation(net: NetOptions): Observation {
    return observationForInputCount(net.inputs, 4);
}

export function buildObjective(name: ObjectiveName, dock: Dock): ControllerError {
    if (name == "d2-terminal") {
        return new D2ControllerError();
    }
    if (name == "d2-best") {
        return new BestApproachD2ControllerError();
    }
    let normalizedDock = new Point((dock.position.x - 50) / 50, dock.position.y / 50);
    return new TruckControllerError(normalizedDock);
}

/**
 * What the objective actually descends. TruckControllerError's scalar and its
 * derivative do not agree -- the derivative weights y by ten and zeroes x,
 * while the scalar does neither -- and that stays as it is, so the CLI says so.
 */
export function describeObjective(name: ObjectiveName): string[] {
    if (name == "d2-terminal") {
        return [
            "d2 of the state the episode stopped in:",
            "  scalar:     x^2 + y^2 + min(ts^2, (ts-2pi)^2, (ts+2pi)^2)   [physical units]",
            "  descends:   the same, chain ruled onto the normalized inputs"
        ];
    }
    if (name == "d2-best") {
        return [
            "d2 at the closest approach of the episode, the paper's grading:",
            "  scalar:     x^2 + y^2 + min(ts^2, (ts-2pi)^2, (ts+2pi)^2)   [physical units]",
            "  descends:   the same, at the step which minimised it, chain ruled",
            "              onto the normalized inputs; the episode is rolled out",
            "              once to find that step, then replayed up to it"
        ];
    }
    return [
        "TruckControllerError, what the bundled weights were trained toward:",
        "  scalar:     xDiff^2 + yDiff^2 + thetaDiff^2, with xDiff clamped at -1",
        "  descends:   [0, 20*yDiff, 0, 2*thetaDiff]",
        "  NOTE:       the derivative weights y by ten and zeroes x while the",
        "              scalar does neither, so the reported error is not the",
        "              quantity being minimised"
    ];
}

export function parseNetOptions(args: Args): NetOptions {
    let net: NetOptions = {
        inputs: args.integer("inputs", 4),
        hidden: args.integer("hidden", 45),
        activation: args.choice<HiddenActivationName>("activation", ["tanh", "logistic"], "tanh"),
        outputMap: args.choice<OutputMapName>("output-map", ["tanh", "2s-1"], "tanh")
    };
    // fails here rather than deep inside a backward pass
    observationForInputCount(net.inputs, 4);
    if (net.hidden < 1) {
        throw new Error("--hidden must be at least 1");
    }
    return net;
}

export function parseOptimizerOptions(args: Args, defaultLearningRate: number): OptimizerOptions {
    return {
        name: args.choice<OptimizerName>("optimizer", ["nesterov", "sgd"], "nesterov"),
        learningRate: args.number("lr", defaultLearningRate),
        momentum: args.number("momentum", 0.9)
    };
}

export function parseConventions(args: Args): PlantConventions {
    return new PlantConventions(
        args.choice<DockReference>("dock-ref", ["truckEnd", "trailerEnd"], "truckEnd"),
        args.choice<AngleWrapping>("wrap", ["none", "pi"], "none"),
        args.number("r", 1)
    );
}

export function parseStartOptions(args: Args, defaultScheme: StartSchemeName): StartOptions {
    return {
        scheme: args.choice<StartSchemeName>("starts", ["curriculum", "ensemble", "point", "yard"], defaultScheme),
        seed: args.integer("start-seed", 7),
        count: args.integer("start-count", 100)
    };
}

/**
 * The paper's fixed training set: five trailer orientations at each of three
 * positions. Cabin and trailer start aligned.
 */
export function ensembleStarts(): Start[] {
    let positions = [[100, 0], [80, 50], [80, -50]];
    let orientations = [-90, -30, 0, 30, 90];
    let starts: Start[] = [];
    for (let p = 0; p < positions.length; p++) {
        for (let o = 0; o < orientations.length; o++) {
            let angle = toRad(orientations[o]);
            starts.push({
                x: positions[p][0], y: positions[p][1],
                trailerAngle: angle, cabinAngle: angle,
                scheme: "ensemble", idx: p * orientations.length + o
            });
        }
    }
    return starts;
}

/** The paper's single training point, (20, 10, -2). */
export function pointStarts(): Start[] {
    return [{ x: 20, y: 10, trailerAngle: -2, cabinAngle: -2, scheme: "point", idx: 0 }];
}

/** Seeded uniform draws over the far yard, so that toy can draw the same ones. */
export function yardStarts(count: number, seed: number): Start[] {
    let random = makeSeededRandom(seed);
    let starts: Start[] = [];
    for (let i = 0; i < count; i++) {
        let angle = -Math.PI + random() * 2 * Math.PI;
        starts.push({
            x: 50 + random() * 50,
            y: -50 + random() * 100,
            trailerAngle: angle, cabinAngle: angle,
            scheme: "yard", idx: i
        });
    }
    return starts;
}

export function buildStarts(options: StartOptions): Start[] {
    if (options.scheme == "ensemble") {
        return ensembleStarts();
    }
    if (options.scheme == "point") {
        return pointStarts();
    }
    if (options.scheme == "yard") {
        return yardStarts(options.count, options.seed);
    }
    throw new Error("The curriculum draws its starts from lesson bounds and has no fixed start list");
}

export function applyStart(truck: Truck, start: Start): void {
    truck.setTruckPosition(new Point(start.x, start.y), start.trailerAngle, start.cabinAngle);
}

export function describeStart(start: Start): string {
    return "(" + start.x.toFixed(1) + ", " + start.y.toFixed(1) + ") trailer "
        + toDeg(start.trailerAngle).toFixed(0) + " deg, cabin " + toDeg(start.cabinAngle).toFixed(0) + " deg";
}

export function readJson(file: string): any {
    return JSON.parse(fs.readFileSync(file).toString());
}

export function writeJson(file: string, content: any): void {
    let directory = path.dirname(path.resolve(file));
    if (!fs.existsSync(directory)) {
        fs.mkdirSync(directory, { recursive: true } as any);
    }
    fs.writeFileSync(file, JSON.stringify(content));
}
