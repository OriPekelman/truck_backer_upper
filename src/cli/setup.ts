import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';
import { Point, Angle, toRad, toDeg } from '../math';
import { Truck } from '../model/truck';
import { Dock } from '../model/world';
import { PlantConventions, DockReference, AngleWrapping } from '../model/conventions';
import { NeuralNet } from '../neuralnet/net';
import { Optimizer, SGD, SGDNesterovMomentum } from '../neuralnet/optimizers';
import { ControllerError, TruckControllerError, D2ControllerError, BestApproachD2ControllerError } from '../neuralnet/error';
import { Observation, observationForInputCount } from '../neuralnet/observation';
import {
    NetOptions, HiddenActivationName, OutputMapName, Start, StartSchemeName,
    buildControllerNet as buildNet, buildObservation as buildObs, buildStarts as buildStartSet,
    applyStart as applyStartToTruck, describeStart as describeStartState, netShape as shapeOf,
    makeSeededRandom as seededRandom, ensembleStarts as ensemble, pointStarts as point, yardStarts as yard
} from '../model/replay';
import { Args } from './args';

export type ObjectiveName = "demo" | "d2-terminal" | "d2-best";
export type OptimizerName = "nesterov" | "sgd";

export interface OptimizerOptions {
    name: OptimizerName;
    learningRate: number;
    momentum: number;
}

export interface StartOptions {
    scheme: StartSchemeName | "curriculum";
    seed: number;
    count: number;
}

export { NetOptions, Start, HiddenActivationName, OutputMapName, StartSchemeName };

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

export function makeSeededRandom(seed: number): () => number {
    return seededRandom(seed);
}

export function buildControllerNet(net: NetOptions, optimizer: OptimizerOptions): NeuralNet {
    return buildNet(net, makeOptimizer(optimizer));
}

export function netShape(net: NetOptions): number[] {
    return shapeOf(net);
}

export function buildObservation(net: NetOptions): Observation {
    return buildObs(net);
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

/** One hidden width, or several separated by commas for a deeper net. */
export function parseHiddenLayers(value: string): number[] {
    let hidden = value.split(",").map((part) => Number.parseInt(part.trim()));
    for (let i = 0; i < hidden.length; i++) {
        if (!isFinite(hidden[i]) || hidden[i] < 1) {
            throw new Error("--hidden must be one or more whole numbers of at least 1, got \"" + value + "\"");
        }
    }
    return hidden;
}

export function parseNetOptions(args: Args): NetOptions {
    let net: NetOptions = {
        inputs: args.integer("inputs", 4),
        hidden: parseHiddenLayers(args.string("hidden", "45")),
        activation: args.choice<HiddenActivationName>("activation", ["tanh", "logistic"], "tanh"),
        outputMap: args.choice<OutputMapName>("output-map", ["tanh", "2s-1"], "tanh")
    };
    // fails here rather than deep inside a backward pass
    observationForInputCount(net.inputs, 4);
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

export function parseStartOptions(args: Args, defaultScheme: StartSchemeName | "curriculum"): StartOptions {
    return {
        scheme: args.choice<StartSchemeName | "curriculum">("starts", ["curriculum", "ensemble", "point", "yard"], defaultScheme),
        seed: args.integer("start-seed", 7),
        count: args.integer("start-count", 100)
    };
}

export function buildStarts(options: StartOptions): Start[] {
    if (options.scheme == "curriculum") {
        throw new Error("The curriculum draws its starts from lesson bounds and has no fixed start list");
    }
    return buildStartSet(options.scheme, options.count, options.seed);
}

export function applyStart(truck: Truck, start: Start): void {
    applyStartToTruck(truck, start);
}

export function describeStart(start: Start): string {
    return describeStartState(start);
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
