import * as path from 'path';
import { Point, toDeg } from '../math';
import { Truck, NormalizedTruck } from '../model/truck';
import { Dock, TraceEventType } from '../model/world';
import { NeuralNet } from '../neuralnet/net';
import { Observation } from '../neuralnet/observation';
import { traceBundleFormat, RunOutcome } from '../model/traceBundle';
import {
    RolloutOptions, RunResult, Start as ReplayStart, buildBundle, describePlant, prepareReplay, rollOut
} from '../model/replay';
import { Args } from './args';
import { Sidecar, buildSidecarPlant, netOptionsFromSidecar, conventionsFromSidecar, bundleDockRef } from './sidecar';
import {
    NetOptions, Start, applyStart, buildControllerNet, buildObservation, buildStarts, describeStart,
    getRepoCommit, netShape, parseConventions, parseHiddenLayers, parseNetOptions, parseStartOptions,
    readJson, writeJson
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
        net.hidden = args.has("hidden") ? parseHiddenLayers(args.string("hidden", "")) : net.hidden;
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

    let prepared = prepareReplay(net, conventions, readJson(weightsFile));
    let truck = prepared.truck;
    let observation = prepared.observation;
    let starts = buildStarts(startOptions);
    let options: RolloutOptions = { stepCap: stepCap, yardXMax: yardXMax, yardYAbs: yardYAbs };

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

    let counts: { [outcome: string]: number } = { docked: 0, wall: 0, bound: 0, cap: 0 };
    let bestD2Sum = 0;
    let minBestD2 = Infinity;
    let stepSum = 0;
    let results: RunResult[] = [];

    for (let i = 0; i < starts.length; i++) {
        let start = starts[i];
        let result = rollOut(truck, prepared.plant, prepared.controllerNet, observation, start, options);
        results.push(result);
        counts[result.outcome] = (counts[result.outcome] || 0) + 1;
        bestD2Sum += result.bestD2;
        minBestD2 = Math.min(minBestD2, result.bestD2);
        stepSum += result.rows.length;
        if (!quiet) {
            console.log("run " + i + "  " + describeStart(start) + "  -> " + result.outcome
                + " after " + result.rows.length + " steps, best d2 " + result.bestD2.toFixed(3)
                + " at step " + result.bestStep + ", clamped " + result.clampCount);
        }
    }

    let bundle = buildBundle({
        arm: arm,
        engine: "truck_backer_upper",
        engineGit: getRepoCommit(),
        weights: path.basename(weightsFile),
        trainSeed: sidecar ? sidecar.seed : undefined,
        objective: sidecar ? sidecar.objective.name : "unknown",
        sidecar: sidecar ? sidecar : undefined
    }, truck, net, options, starts, results, stride);
    writeJson(outFile, bundle);

    console.log("");
    console.log("bundle         " + outFile);
    console.log("runs           " + starts.length
        + "  docked " + counts["docked"] + ", wall " + counts["wall"]
        + ", bound " + counts["bound"] + ", cap " + counts["cap"]);
    console.log("best d2        mean " + (bestD2Sum / starts.length).toFixed(3) + ", min " + minBestD2.toFixed(3));
    console.log("steps          mean " + (stepSum / starts.length).toFixed(1));
}
