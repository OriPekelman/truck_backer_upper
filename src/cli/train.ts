import * as fs from 'fs';
import * as path from 'path';
import { Point } from '../math';
import { Truck, NormalizedTruck, TruckEmulator } from '../model/truck';
import { World, Dock } from '../model/world';
import { TrainController } from '../neuralnet/train';
import { NeuralNetEmulator, Emulator } from '../neuralnet/emulator';
import { emulatorNet } from '../neuralnet/implementations';
import { TruckLesson, Range, createTruckControllerLessons } from '../neuralnet/lesson';
import { Args } from './args';
import { buildSidecar } from './sidecar';
import {
    Start, applyStart, buildControllerNet, buildObjective, buildObservation, buildStarts,
    describeObjective, describeOptimizer, describeStart, getDefaultEmulatorWeights, getRepoCommit,
    installSeededRandom, isRepoDirty, makeOptimizer, netShape, parseConventions, parseNetOptions,
    parseOptimizerOptions, parseStartOptions, readJson, writeJson
} from './setup';

export type Backend = "emulator-bptt" | "jacobian-bptt";

export const trainFlags = ["quiet"];

export const trainUsage = [
    "tbu train --backend emulator-bptt|jacobian-bptt --out <dir> [options]",
    "",
    "  Trains a controller with one of the original code's two trainers and writes",
    "  the weights plus a sidecar describing every convention they were trained under.",
    "",
    "  --backend       emulator-bptt (through the learned 5-100-4 emulator, Nguyen &",
    "                  Widrow proper) or jacobian-bptt (through the analytic plant",
    "                  Jacobian). Default emulator-bptt.",
    "  --out           directory for the weight file and its sidecar. Required.",
    "  --arm           name recorded in the sidecar. Defaults to the backend.",
    "",
    "  net",
    "  --inputs        4 (all state), 3 (no cabin angle) or 8 (the four plus the same",
    "                  four at 10x). Default 4.",
    "  --hidden        hidden units. Default 45 (the demo's); the paper uses 9, 7, 17.",
    "  --activation    tanh | logistic. Default tanh (the demo's).",
    "  --output-map    tanh | 2s-1. Default tanh; the paper maps a logistic by 2s-1.",
    "",
    "  plant conventions (see PlantConventions)",
    "  --r             step length. Default 1 (the demo's); the paper uses 3.",
    "  --step-cap      steps before a sample is abandoned. Default 1000.",
    "  --dock-ref      truckEnd | trailerEnd. Default truckEnd (the demo's).",
    "  --wrap          none | pi. Default none (the demo's and the paper's).",
    "",
    "  training",
    "  --starts        curriculum (the 20-lesson default) | ensemble (the paper's 15",
    "                  starts) | point ((20, 10, -2)) | yard (seeded draws).",
    "  --start-seed    seed for the yard draws. Default 7.",
    "  --start-count   how many yard draws. Default 100.",
    "  --samples       samples per lesson. Default 1000.",
    "  --objective     demo | d2-terminal | d2-best. Default demo.",
    "  --optimizer     nesterov | sgd. Default nesterov.",
    "  --lr            learning rate. Default 0.1 for demo, 1e-8 for the d2 objectives,",
    "                  whose gradients are in physical units and far larger.",
    "  --momentum      nesterov momentum. Default 0.9.",
    "  --seed          seeds weight initialisation and start draws. Default 1.",
    "  --resume        weight file to start from.",
    "  --emulator-weights  emulator weights for emulator-bptt. Defaults to the",
    "                  bundled src/weights/truck_emulator_weights.",
    "  --first-lesson  lesson to start the curriculum at. Default 0.",
    "  --log-every     samples between progress lines. Default 100.",
    "  --quiet         only print the summary."
].join("\n");

/**
 * Positions the plant for each sample, and does so identically when asked twice
 * for the same sample, which is what lets the closest-approach objective replay
 * an episode. The curriculum draws from the lesson's bounds, so its draw is
 * remembered rather than repeated.
 */
class StartSequence {
    private lastIndex: number = -1;
    private lastState: { x: number, y: number, trailerAngle: number, cabinAngle: number } | undefined = undefined;
    private lesson: TruckLesson | undefined = undefined;

    public constructor(private truck: Truck, private starts: Start[] | undefined) {
    }

    public setLesson(lesson: TruckLesson) {
        this.lesson = lesson;
        this.lastIndex = -1;
        this.lastState = undefined;
    }

    public prepare(sampleIndex: number) {
        if (sampleIndex == this.lastIndex && this.lastState) {
            let state = this.lastState;
            this.truck.setTruckPosition(new Point(state.x, state.y), state.trailerAngle, state.cabinAngle);
            return;
        }
        if (this.starts) {
            applyStart(this.truck, this.starts[sampleIndex % this.starts.length]);
        } else {
            this.truck.randomizePosition(this.lesson as TruckLesson);
        }
        let position = this.truck.getTrailerEndPosition();
        this.lastState = {
            x: position.x, y: position.y,
            trailerAngle: this.truck.getTrailerAngle(), cabinAngle: this.truck.getTruckAngle()
        };
        this.lastIndex = sampleIndex;
    }
}

/**
 * A lesson pinning nothing: with a fixed start set the bounds are unused, and
 * the lesson only carries the sample count, the step cap and the optimizer.
 */
function fixedStartLesson(truck: Truck, samples: number, stepCap: number, optimizer: () => any): TruckLesson {
    let zero = new Range(0, 0);
    return new TruckLesson(truck, 0, samples, optimizer, zero, zero, zero, zero, stepCap);
}

export function runTrain(args: Args): void {
    let backend = args.choice<Backend>("backend", ["emulator-bptt", "jacobian-bptt"], "emulator-bptt");
    let outDir = args.string("out", "");
    if (outDir.length == 0) {
        throw new Error("--out <dir> is required");
    }
    let net = parseNetOptions(args);
    let conventions = parseConventions(args);
    let stepCap = args.integer("step-cap", 1000);
    let objective = args.choice<"demo" | "d2-terminal" | "d2-best">("objective", ["demo", "d2-terminal", "d2-best"], "demo");
    let startOptions = parseStartOptions(args, "curriculum");
    let samples = args.integer("samples", 1000);
    let seed = args.integer("seed", 1);
    let firstLesson = args.integer("first-lesson", 0);
    let logEvery = args.integer("log-every", 100);
    let quiet = args.flag("quiet");
    let arm = args.string("arm", backend);
    let resume = args.string("resume", "");
    let emulatorWeightsFile = args.string("emulator-weights", getDefaultEmulatorWeights());
    // the d2 objectives are in physical units, so their gradients are ~1e4 larger
    let optimizerOptions = parseOptimizerOptions(args, objective == "demo" ? 0.1 : 1e-8);

    let unused = args.getUnused();
    if (unused.length > 0) {
        throw new Error("Unknown option(s): " + unused.map((u) => "--" + u).join(", "));
    }

    installSeededRandom(seed);

    let dock = new Dock(new Point(0, 0));
    let truck = new Truck(new Point(15, 15), 0, 0, dock, []);
    truck.conventions = conventions;
    let world = new World(truck, dock);
    let normalizedTruck = new NormalizedTruck(truck);

    let controllerNet = buildControllerNet(net, optimizerOptions);
    let observation = buildObservation(net);

    let emulator: Emulator;
    if (backend == "emulator-bptt") {
        emulatorNet.loadWeights(readJson(emulatorWeightsFile));
        emulator = new NeuralNetEmulator(emulatorNet);
    } else {
        emulator = new TruckEmulator(truck);
    }

    let errorFunction = buildObjective(objective, dock);
    let trainer = new TrainController(world, normalizedTruck, controllerNet, emulator, errorFunction);
    trainer.setObservation(observation);

    let starts = startOptions.scheme == "curriculum" ? undefined : buildStarts(startOptions);
    let sequence = new StartSequence(truck, starts);
    trainer.setStartProvider((sampleIndex: number) => sequence.prepare(sampleIndex));

    let lessons: TruckLesson[];
    if (starts) {
        lessons = [fixedStartLesson(truck, samples, stepCap, makeOptimizer(optimizerOptions))];
    } else {
        lessons = createTruckControllerLessons(truck);
        for (let i = 0; i < lessons.length; i++) {
            lessons[i].samples = samples;
            lessons[i].maxSteps = stepCap;
            lessons[i].optimizer = makeOptimizer(optimizerOptions);
        }
    }
    if (firstLesson < 0 || firstLesson >= lessons.length) {
        throw new Error("--first-lesson must be between 0 and " + (lessons.length - 1));
    }
    if (resume.length > 0) {
        controllerNet.loadWeights(readJson(resume));
    }

    console.log("arm            " + arm + " (" + backend + ")");
    console.log("net            " + netShape(net).join("-") + " " + net.activation
        + ", output " + net.outputMap + ", inputs via " + observation.getName());
    console.log("plant          r=" + conventions.stepLength + ", step cap=" + stepCap
        + ", dock=" + conventions.dockReference + ", wrap=" + conventions.wrapping);
    console.log("starts         " + startOptions.scheme
        + (starts ? " (" + starts.length + " fixed" + (starts.length == 1 ? ": " + describeStart(starts[0]) : "") + ")"
            : " (" + lessons.length + " lessons from lesson " + firstLesson + ")"));
    console.log("optimizer      " + describeOptimizer(optimizerOptions));
    console.log("seed           " + seed);
    console.log("objective      " + describeObjective(objective).join("\n               "));
    if (resume.length > 0) {
        console.log("resumed from   " + resume);
    }
    console.log("");

    let weightsFile = path.join(outDir, arm + "_weights");
    let sidecarFile = path.join(outDir, arm + "_weights.sidecar.json");
    let commit = getRepoCommit();
    let dirty = isRepoDirty();
    let totalSamples = 0;
    let abandoned = 0;
    let skipped = 0;

    for (let l = firstLesson; l < lessons.length; l++) {
        let lesson = lessons[l];
        sequence.setLesson(lesson);
        trainer.setLesson(lesson);
        let lessonErrors: number[] = [];
        for (let i = 0; i < lesson.samples; i++) {
            // a sample abandoned at the step cap reports an error of 0 without
            // having updated anything, so it must not enter the mean
            let abandonedBefore = trainer.maxStepErrors;
            let error = trainer.trainSingleStep();
            totalSamples++;
            if (trainer.maxStepErrors > abandonedBefore) {
                // abandoned, counted below
            } else if (Number.isNaN(error)) {
                skipped++;
            } else {
                lessonErrors.push(error);
            }
            if (!quiet && logEvery > 0 && i > 0 && i % logEvery == 0) {
                console.log("lesson " + l + "  sample " + i + "/" + lesson.samples
                    + "  mean error " + mean(lessonErrors).toExponential(3)
                    + " over " + lessonErrors.length + " updates"
                    + "  abandoned " + trainer.maxStepErrors);
            }
        }
        abandoned += trainer.maxStepErrors;
        console.log("lesson " + l + " done  mean error " + mean(lessonErrors).toExponential(3)
            + " over " + lessonErrors.length + " updates"
            + "  abandoned " + trainer.maxStepErrors + "/" + lesson.samples);
        // written after every lesson, as the original scripts did
        writeJson(weightsFile, controllerNet.getWeights());
    }

    writeJson(weightsFile, controllerNet.getWeights());
    writeJson(sidecarFile, buildSidecar({
        arm: arm, commit: commit, dirty: dirty, seed: seed,
        weightsFile: path.basename(weightsFile), net: net, truck: truck, stepCap: stepCap,
        objective: objective,
        training: {
            backend: backend,
            starts: startOptions.scheme,
            start_seed: startOptions.seed,
            start_count: starts ? starts.length : undefined,
            lessons: lessons.length,
            first_lesson: firstLesson,
            samples_per_lesson: samples,
            samples_trained: totalSamples,
            samples_abandoned_at_step_cap: abandoned,
            samples_skipped: skipped,
            optimizer: describeOptimizer(optimizerOptions),
            emulator_weights: backend == "emulator-bptt" ? emulatorWeightsFile : undefined,
            resumed_from: resume.length > 0 ? resume : undefined
        }
    }));

    console.log("");
    console.log("weights        " + weightsFile);
    console.log("sidecar        " + sidecarFile);
    console.log("samples        " + totalSamples + " trained, " + abandoned + " abandoned at the step cap, " + skipped + " skipped");
}

function mean(values: number[]): number {
    if (values.length == 0) {
        return NaN;
    }
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
        sum += values[i];
    }
    return sum / values.length;
}
