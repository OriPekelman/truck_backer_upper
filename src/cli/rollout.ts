import * as path from 'path';
import { Point, toDeg } from '../math';
import { Truck, NormalizedTruck } from '../model/truck';
import { Dock, TraceEventType } from '../model/world';
import { NeuralNet } from '../neuralnet/net';
import { Observation } from '../neuralnet/observation';
import { traceBundleFormat, dockingD2, RunOutcome } from '../model/traceBundle';
import { Args } from './args';
import { Sidecar, buildSidecarPlant, netOptionsFromSidecar, conventionsFromSidecar, bundleDockRef } from './sidecar';
import {
    NetOptions, Start, applyStart, buildControllerNet, buildObservation, buildStarts, describeStart,
    getRepoCommit, netShape, parseConventions, parseNetOptions, parseStartOptions, readJson, writeJson
} from './setup';

export const rolloutFlags = ["quiet"];

export const rolloutUsage = [
    "tbu rollout --weights <file> --out <bundle.json> [options]",
    "",
    "  Rolls a controller out over a start set and writes a " + traceBundleFormat + " bundle,",
    "  the format the rollout overlay reads and toy emits, so that this code's",
    "  controllers and toy's arms can be compared on identical starts.",
    "",
    "  --weights       weight file, [layer][unit][w..., bias]. Required.",
    "  --sidecar       sidecar describing the net and conventions. Defaults to",
    "                  <weights>.sidecar.json when that exists; without it the net",
    "                  and plant options below have to be given.",
    "  --out           bundle to write. Required.",
    "  --arm           name recorded in the bundle. Defaults to the sidecar's.",
    "  --starts        ensemble | point | yard. Default ensemble.",
    "  --start-seed    seed for the yard draws. Default 7.",
    "  --start-count   how many yard draws. Default 100.",
    "  --step-cap      steps before a run is cut off. Default from the sidecar, else 300.",
    "  --stride        keep every k-th step; the last and best-approach rows are",
    "                  always kept. Default 1.",
    "  --yard-x-max    x beyond which a run counts as out of bounds. Default 100.",
    "  --yard-y-abs    |y| beyond which a run counts as out of bounds. Default 50.",
    "                  The bound only ends a run which has been inside the yard, since",
    "                  the paper's ensemble starts sit exactly on it; a run which never",
    "                  comes back in reaches the step cap instead.",
    "  --quiet         only print the summary.",
    "",
    "  Net and plant options are read from the sidecar; --inputs, --hidden,",
    "  --activation, --output-map, --r, --dock-ref and --wrap override them."
].join("\n");

const columns = ["signal", "u", "x", "y", "tc", "ts", "clamped", "step"];

interface Row {
    step: number;
    signal: number;
    u: number;
    x: number;
    y: number;
    tc: number;
    ts: number;
    clamped: number;
}

interface RunResult {
    rows: Row[];
    outcome: RunOutcome;
    bestD2: number;
    bestStep: number;
    terminalD2: number;
    clampCount: number;
    pathLen: number;
}

function round(value: number): number {
    return Math.round(value * 1e9) / 1e9;
}

/**
 * Drives one episode and records it. Termination comes from the plant itself --
 * the same dock and validity checks the interactive simulation uses, read off
 * its trace events -- except for the yard bound and the step cap, which are the
 * rollout's own.
 */
function rollOut(truck: Truck, plant: NormalizedTruck, net: NeuralNet, observation: Observation,
    start: Start, stepCap: number, yardXMax: number, yardYAbs: number): RunResult {
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
    // the paper's own ensemble starts sit exactly on the yard's y bound, so the
    // bound only ends a run once that run has been inside the yard; one which
    // never comes back in reaches the step cap instead
    let enteredYard = false;

    for (let step = 1; step <= stepCap; step++) {
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
        let outsideYard = position.x > yardXMax || Math.abs(position.y) > yardYAbs;
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

function strideRows(rows: Row[], stride: number, bestStep: number): Row[] {
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

export function runRollout(args: Args): void {
    let weightsFile = args.string("weights", "");
    let outFile = args.string("out", "");
    if (weightsFile.length == 0 || outFile.length == 0) {
        throw new Error("--weights <file> and --out <bundle.json> are required");
    }
    let sidecarFile = args.string("sidecar", weightsFile + ".sidecar.json");
    let sidecar: Sidecar | undefined = undefined;
    try {
        sidecar = readJson(sidecarFile) as Sidecar;
    } catch (e) {
        if (args.has("sidecar")) {
            throw new Error("Could not read the sidecar " + sidecarFile);
        }
    }

    let net: NetOptions;
    let conventions;
    if (sidecar) {
        net = netOptionsFromSidecar(sidecar);
        conventions = conventionsFromSidecar(sidecar);
        // anything given explicitly still wins over the sidecar
        net.inputs = args.integer("inputs", net.inputs);
        net.hidden = args.integer("hidden", net.hidden);
        net.activation = args.choice("activation", ["tanh", "logistic"], net.activation);
        net.outputMap = args.choice("output-map", ["tanh", "2s-1"], net.outputMap);
        conventions.stepLength = args.number("r", conventions.stepLength);
        conventions.dockReference = args.choice("dock-ref", ["truckEnd", "trailerEnd"], conventions.dockReference);
        conventions.wrapping = args.choice("wrap", ["none", "pi"], conventions.wrapping);
    } else {
        net = parseNetOptions(args);
        conventions = parseConventions(args);
    }
    let stepCap = args.integer("step-cap", sidecar ? sidecar.plant.step_cap || 300 : 300);
    let stride = args.integer("stride", 1);
    let yardXMax = args.number("yard-x-max", 100);
    let yardYAbs = args.number("yard-y-abs", 50);
    let startOptions = parseStartOptions(args, "ensemble");
    let arm = args.string("arm", sidecar ? sidecar.arm : "unnamed");
    let quiet = args.flag("quiet");

    let unused = args.getUnused();
    if (unused.length > 0) {
        throw new Error("Unknown option(s): " + unused.map((u) => "--" + u).join(", "));
    }
    if (startOptions.scheme == "curriculum") {
        throw new Error("--starts curriculum has no fixed start list; use ensemble, point or yard");
    }

    let dock = new Dock(new Point(0, 0));
    let truck = new Truck(new Point(50, 0), 0, 0, dock, []);
    truck.conventions = conventions;
    let plant = new NormalizedTruck(truck);
    let controllerNet = buildControllerNet(net, { name: "sgd", learningRate: 0, momentum: 0 });
    controllerNet.loadWeights(readJson(weightsFile));
    controllerNet.fixWeights(true); // a rollout never learns, so store no inputs
    let observation = buildObservation(net);
    let starts = buildStarts(startOptions);

    console.log("arm            " + arm);
    console.log("net            " + netShape(net).join("-") + " " + net.activation
        + ", output " + net.outputMap + ", inputs via " + observation.getName());
    console.log("plant          r=" + conventions.stepLength + ", step cap=" + stepCap
        + ", dock=" + bundleDockRef(conventions) + ", wrap=" + conventions.wrapping);
    console.log("starts         " + startOptions.scheme + " (" + starts.length + ")"
        + (starts.length == 1 ? ": " + describeStart(starts[0]) : ""));
    console.log("yard           x <= " + yardXMax + ", |y| <= " + yardYAbs);
    console.log("sidecar        " + (sidecar ? sidecarFile : "none, options taken from the command line"));
    console.log("");

    let runs: any[] = [];
    let counts: { [outcome: string]: number } = { docked: 0, wall: 0, bound: 0, cap: 0 };
    let bestD2Sum = 0;
    let minBestD2 = Infinity;
    let stepSum = 0;

    for (let i = 0; i < starts.length; i++) {
        let start = starts[i];
        let result = rollOut(truck, plant, controllerNet, observation, start, stepCap, yardXMax, yardYAbs);
        counts[result.outcome] = (counts[result.outcome] || 0) + 1;
        bestD2Sum += result.bestD2;
        minBestD2 = Math.min(minBestD2, result.bestD2);
        stepSum += result.rows.length;

        let kept = strideRows(result.rows, stride, result.bestStep);
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
            trace: kept.map((row) => [row.signal, row.u, row.x, row.y, row.tc, row.ts, row.clamped, row.step])
        });
        if (!quiet) {
            console.log("run " + i + "  " + describeStart(start) + "  -> " + result.outcome
                + " after " + result.rows.length + " steps, best d2 " + result.bestD2.toFixed(3)
                + " at step " + result.bestStep + ", clamped " + result.clampCount);
        }
    }

    let bundle = {
        format: traceBundleFormat,
        provenance: {
            arm: arm,
            engine: "truck_backer_upper",
            engine_git: getRepoCommit(),
            weights: path.basename(weightsFile),
            train_seed: sidecar ? sidecar.seed : undefined,
            net: { shape: netShape(net), activation: net.activation, output_map: net.outputMap, obs: net.inputs },
            objective: sidecar ? sidecar.objective.name : "unknown",
            sidecar: sidecar ? sidecar : undefined
        },
        plant: buildSidecarPlant(truck, stepCap),
        columns: columns,
        runs: runs
    };
    writeJson(outFile, bundle);

    console.log("");
    console.log("bundle         " + outFile);
    console.log("runs           " + starts.length
        + "  docked " + counts["docked"] + ", wall " + counts["wall"]
        + ", bound " + counts["bound"] + ", cap " + counts["cap"]);
    console.log("best d2        mean " + (bestD2Sum / starts.length).toFixed(3) + ", min " + minBestD2.toFixed(3));
    console.log("steps          mean " + (stepSum / starts.length).toFixed(1));
}
