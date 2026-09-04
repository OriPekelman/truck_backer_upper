import { Point, Angle, toRad, toDeg } from '../math'
import { Truck, NormalizedTruck } from './truck'
import { Dock, TraceEventType } from './world'
import { PlantConventions } from './conventions'
import { NeuralNet, NetConfig, LayerConfig } from '../neuralnet/net'
import { ActivationFunction, Tanh, Sigmoid, SymmetricSigmoid } from '../neuralnet/activation'
import { AdalineUnit } from '../neuralnet/unit'
import { Optimizer, SGD } from '../neuralnet/optimizers'
import { WeightInitializer, TwoLayerInitializer, RandomWeightInitializer } from '../neuralnet/weightinitializer'
import { MSE } from '../neuralnet/error'
import { Observation, observationForInputCount } from '../neuralnet/observation'
import { traceBundleFormat, dockingD2, RunOutcome } from './traceBundle'

/**
 * Replaying a controller: building it from a weight file, driving it over a
 * start set, and writing the result out as a trace bundle. Shared by the
 * headless CLI and the browser, so that a rollout is the same rollout wherever
 * it was run -- which is the only reason comparing the two is worth anything.
 *
 * Nothing here touches the filesystem, so it can be bundled for the browser.
 */

export type HiddenActivationName = "tanh" | "logistic";
export type OutputMapName = "tanh" | "2s-1";

export interface NetOptions {
    inputs: number;
    // one entry per hidden layer; the output layer is always a single unit
    hidden: number[];
    activation: HiddenActivationName;
    outputMap: OutputMapName;
}

export function netShape(net: NetOptions): number[] {
    return [net.inputs].concat(net.hidden).concat([1]);
}

export function describeNet(net: NetOptions): string {
    return netShape(net).join("-") + " " + net.activation + ", output " + net.outputMap;
}

export function hiddenActivation(name: HiddenActivationName): ActivationFunction {
    return name == "logistic" ? new Sigmoid() : new Tanh();
}

export function outputActivation(name: OutputMapName): ActivationFunction {
    return name == "2s-1" ? new SymmetricSigmoid() : new Tanh();
}

const makeUnit = (weights: number, activation: ActivationFunction, initialWeightRange: WeightInitializer, optimizer: Optimizer) =>
    new AdalineUnit(weights, activation, initialWeightRange, optimizer);

/**
 * A controller net of the requested shape. The demo's 4-45-1 tanh, the paper's
 * 4-9-1 logistic with a 2s-1 output, its 3-7-1 and its 8-17-1 are all this with
 * different arguments; nothing here is hardcoded to one of them.
 */
export function buildControllerNet(net: NetOptions, optimizer: () => Optimizer): NeuralNet {
    return new NeuralNet(buildControllerNetConfig(net, optimizer));
}

/** The NetConfig behind buildControllerNet, for callers that also need to show or edit it (the Controller tab). */
export function buildControllerNetConfig(net: NetOptions, optimizer: () => Optimizer): NetConfig {
    let layerConfigs: LayerConfig[] = net.hidden.map((neurons) => ({
        neuronCount: neurons,
        weightInitializer: new TwoLayerInitializer(0.7, neurons),
        unitConstructor: makeUnit,
        activation: hiddenActivation(net.activation)
    }));
    layerConfigs.push({
        neuronCount: 1,
        weightInitializer: new RandomWeightInitializer(0.01),
        unitConstructor: makeUnit,
        activation: outputActivation(net.outputMap)
    });
    let config: NetConfig = {
        inputs: net.inputs,
        optimizer: optimizer,
        errorFunction: new MSE(), // ignored for the controller
        layerConfigs: layerConfigs
    };
    return config;
}

export function buildObservation(net: NetOptions): Observation {
    return observationForInputCount(net.inputs, 4);
}

export class WeightFileError extends Error {
    public constructor(message: string) {
        super(message);
    }
}

/**
 * The net's shape, read off a weight file in the format this repo has always
 * used: [layer][unit][weight..., bias], the bias last. A loaded controller's
 * architecture comes from its weights rather than from a constant.
 */
export function inferNetShape(weights: any): { inputs: number, hidden: number[] } {
    if (!(weights instanceof Array) || weights.length < 2) {
        throw new WeightFileError("expected an array of at least two layers of [unit][weight..., bias]");
    }
    let unitCounts: number[] = [];
    let inputDims: number[] = [];
    for (let layer = 0; layer < weights.length; layer++) {
        let units = weights[layer];
        if (!(units instanceof Array) || units.length == 0) {
            throw new WeightFileError("layer " + layer + " has no units");
        }
        for (let unit = 0; unit < units.length; unit++) {
            if (!(units[unit] instanceof Array) || units[unit].length < 2) {
                throw new WeightFileError("layer " + layer + " unit " + unit + " is not a weight vector");
            }
            if (units[unit].length != units[0].length) {
                throw new WeightFileError("layer " + layer + " mixes units of "
                    + units[0].length + " and " + units[unit].length + " weights");
            }
        }
        unitCounts.push(units.length);
        // the last weight of a unit is its bias, so the rest are its inputs
        inputDims.push(units[0].length - 1);
    }
    for (let layer = 1; layer < weights.length; layer++) {
        if (inputDims[layer] != unitCounts[layer - 1]) {
            throw new WeightFileError("layer " + layer + " takes " + inputDims[layer]
                + " inputs but layer " + (layer - 1) + " has " + unitCounts[layer - 1] + " units");
        }
    }
    if (unitCounts[unitCounts.length - 1] != 1) {
        throw new WeightFileError("a controller's last layer must be a single unit, but it has "
            + unitCounts[unitCounts.length - 1]);
    }
    return { inputs: inputDims[0], hidden: unitCounts.slice(0, unitCounts.length - 1) };
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

export type StartSchemeName = "ensemble" | "point" | "yard";

/** A seeded linear congruential generator, so a start set is reproducible. */
export function makeSeededRandom(seed: number): () => number {
    let state = (seed >>> 0) || 1;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
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

export function buildStarts(scheme: StartSchemeName, count: number, seed: number): Start[] {
    if (scheme == "ensemble") {
        return ensembleStarts();
    }
    if (scheme == "point") {
        return pointStarts();
    }
    return yardStarts(count, seed);
}

export function applyStart(truck: Truck, start: Start): void {
    truck.setTruckPosition(new Point(start.x, start.y), start.trailerAngle, start.cabinAngle);
}

export function describeStart(start: Start): string {
    return "(" + start.x.toFixed(1) + ", " + start.y.toFixed(1) + ") trailer "
        + toDeg(start.trailerAngle).toFixed(0) + " deg, cabin " + toDeg(start.cabinAngle).toFixed(0) + " deg";
}

export interface RolloutOptions {
    stepCap: number;
    yardXMax: number;
    yardYAbs: number;
}

export const defaultRolloutOptions: RolloutOptions = { stepCap: 300, yardXMax: 100, yardYAbs: 50 };

export interface Row {
    step: number;
    signal: number;
    u: number;
    x: number;
    y: number;
    tc: number;
    ts: number;
    clamped: number;
}

export interface RunResult {
    rows: Row[];
    outcome: RunOutcome;
    bestD2: number;
    bestStep: number;
    terminalD2: number;
    clampCount: number;
    pathLen: number;
}

export const rolloutColumns = ["signal", "u", "x", "y", "tc", "ts", "clamped", "step"];

function round(value: number): number {
    return Math.round(value * 1e9) / 1e9;
}

function outcomeFromEvents(truck: Truck): RunOutcome {
    let events = truck.consumeTraceEvents();
    // the last termination event is the one which ended the episode
    for (let i = events.length - 1; i >= 0; i--) {
        if (events[i].type == TraceEventType.DOCKED) {
            return "docked";
        }
        if (events[i].type == TraceEventType.HIT_DOCK_WALL) {
            return "wall";
        }
        if (events[i].type == TraceEventType.LEFT_AREA) {
            return "bound";
        }
    }
    // the plant refused to move without saying why, which means it started invalid
    return "wall";
}

/**
 * Drives one episode and records it. Termination comes from the plant itself --
 * the same dock and validity checks the interactive simulation uses, read off
 * its trace events -- except for the yard bound and the step cap, which are the
 * rollout's own. The bound only ends a run which has been inside the yard,
 * since the paper's ensemble starts sit exactly on it.
 */
export function rollOut(truck: Truck, plant: NormalizedTruck, net: NeuralNet, observation: Observation,
    start: Start, options: RolloutOptions): RunResult {
    applyStart(truck, start);
    truck.consumeTraceEvents(); // discard anything from positioning

    let rows: Row[] = [];
    let outcome: RunOutcome = "cap";
    let bestD2 = Infinity;
    let bestStep = 0;
    let clampCount = 0;
    let pathLen = 0;
    let previous = new Point(start.x, start.y);
    let maxSteeringAngle = truck.getMaxSteeringAngle();
    let enteredYard = false;

    for (let step = 1; step <= options.stepCap; step++) {
        let signal = net.forward(observation.observe(plant.getStateVector())).entries[0];
        let clampedSignal = Math.min(Math.max(-1, signal), 1);
        let canContinue = truck.nextState(signal, 1);

        let position = truck.getTrailerEndPosition();
        let row: Row = {
            step: step,
            signal: round(signal),
            u: round(maxSteeringAngle * clampedSignal),
            x: round(position.x),
            y: round(position.y),
            tc: round(truck.getTruckAngle()),
            ts: round(truck.getTrailerAngle()),
            clamped: truck.isJackKnifed() ? 1 : 0
        };
        rows.push(row);
        if (row.clamped) {
            clampCount++;
        }
        pathLen += Math.sqrt(Math.pow(position.x - previous.x, 2) + Math.pow(position.y - previous.y, 2));
        previous = new Point(position.x, position.y);

        let d2 = dockingD2(position.x, position.y, truck.getTrailerAngle());
        if (d2 < bestD2) {
            bestD2 = d2;
            bestStep = step;
        }

        if (!canContinue) {
            outcome = outcomeFromEvents(truck);
            break;
        }
        let outsideYard = position.x > options.yardXMax || Math.abs(position.y) > options.yardYAbs;
        if (!outsideYard) {
            enteredYard = true;
        } else if (enteredYard) {
            outcome = "bound";
            break;
        }
    }

    let last = rows[rows.length - 1];
    return {
        rows: rows,
        outcome: outcome,
        bestD2: round(bestD2),
        bestStep: bestStep,
        terminalD2: last ? round(dockingD2(last.x, last.y, last.ts)) : NaN,
        clampCount: clampCount,
        pathLen: round(pathLen)
    };
}

export function strideRows(rows: Row[], stride: number, bestStep: number): Row[] {
    if (stride <= 1 || rows.length == 0) {
        return rows;
    }
    let keep: Row[] = [];
    for (let i = 0; i < rows.length; i++) {
        // the last and best-approach rows are always kept, as the format requires
        if (i % stride == 0 || i == rows.length - 1 || rows[i].step == bestStep) {
            keep.push(rows[i]);
        }
    }
    return keep;
}

export interface BundleProvenanceOptions {
    arm: string;
    engine: string;
    engineGit: string;
    weights: string;
    trainSeed: number | undefined;
    objective: string;
    sidecar?: any;
}

/** A tbu-traces/1 bundle of the given runs, the format the overlay reads. */
export function buildBundle(provenance: BundleProvenanceOptions, truck: Truck, net: NetOptions,
    options: RolloutOptions, starts: Start[], results: RunResult[], stride: number): any {
    let runs = [];
    for (let i = 0; i < starts.length; i++) {
        let start = starts[i];
        let result = results[i];
        runs.push({
            id: i,
            start: {
                x: round(start.x), y: round(start.y),
                ts: round(start.trailerAngle), tc: round(start.cabinAngle),
                scheme: start.scheme, idx: start.idx
            },
            end: result.outcome,
            steps: result.rows.length,
            best_d2: result.bestD2,
            best_step: result.bestStep,
            terminal_d2: result.terminalD2,
            clamp_count: result.clampCount,
            path_len: result.pathLen,
            trace: strideRows(result.rows, stride, result.bestStep)
                .map((row) => [row.signal, row.u, row.x, row.y, row.tc, row.ts, row.clamped, row.step])
        });
    }
    return {
        format: traceBundleFormat,
        provenance: {
            arm: provenance.arm,
            engine: provenance.engine,
            engine_git: provenance.engineGit,
            weights: provenance.weights,
            train_seed: provenance.trainSeed,
            net: { shape: netShape(net), activation: net.activation, output_map: net.outputMap, obs: net.inputs },
            objective: provenance.objective,
            sidecar: provenance.sidecar
        },
        plant: describePlant(truck, options.stepCap),
        columns: rolloutColumns,
        runs: runs
    };
}

/** The vocabulary the trace bundles declare, rather than this code's own. */
export function bundleDockRef(conventions: PlantConventions): string {
    return conventions.dockReference == "trailerEnd" ? "trailer" : "truck-end";
}

export function describePlant(truck: Truck, stepCap: number): any {
    return {
        ls: truck.getTrailerLength(),
        lc: truck.getTruckLength(),
        u_max_deg: truck.getMaxSteeringAngle() * 180 / Math.PI,
        r: truck.conventions.stepLength,
        step_cap: stepCap,
        dock_ref: bundleDockRef(truck.conventions),
        wrap: truck.conventions.wrapping
    };
}

/**
 * Everything a replay needs: a plant under the given conventions, a net of the
 * given shape carrying the given weights, and the observation its input count
 * implies.
 */
export function prepareReplay(net: NetOptions, conventions: PlantConventions, weights: any):
    { truck: Truck, plant: NormalizedTruck, controllerNet: NeuralNet, observation: Observation } {
    let dock = new Dock(new Point(0, 0));
    let truck = new Truck(new Point(50, 0), 0, 0, dock, []);
    truck.conventions = conventions;
    let controllerNet = buildControllerNet(net, () => new SGD(0));
    controllerNet.loadWeights(weights);
    controllerNet.fixWeights(true); // a replay never learns, so store no inputs
    return {
        truck: truck,
        plant: new NormalizedTruck(truck),
        controllerNet: controllerNet,
        observation: buildObservation(net)
    };
}

/** Rolls one controller out over a whole start set. */
export function replayStarts(net: NetOptions, conventions: PlantConventions, weights: any,
    starts: Start[], options: RolloutOptions): { results: RunResult[], truck: Truck } {
    let prepared = prepareReplay(net, conventions, weights);
    let results: RunResult[] = [];
    for (let i = 0; i < starts.length; i++) {
        results.push(rollOut(prepared.truck, prepared.plant, prepared.controllerNet, prepared.observation, starts[i], options));
    }
    return { results: results, truck: prepared.truck };
}
