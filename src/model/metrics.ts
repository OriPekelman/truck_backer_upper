import { Vector } from '../neuralnet/math'
import { Point } from '../math'
import { TruckControllerError } from '../neuralnet/error'
import { LoadedBundle, LoadedRun, RunOutcome, TraceRow, dockingD2 } from './traceBundle'

/**
 * Metrics for comparing arms, per the paper's readings rather than only the
 * one a controller was trained toward.
 */

// the normalization NormalizedTruck applies before the net sees a state
const xScale = 50;
const yScale = 50;
const angleScale = Math.PI;

/**
 * The demo's own error function, on the state an episode ended in. Bundled
 * controllers were trained toward this number, so they are worth judging on it
 * as well as on the paper's d2 -- and it is the demo's implementation being
 * called here, not a second copy of the formula.
 */
const demoError = new TruckControllerError(new Point((0 - xScale) / xScale, 0));
demoError.setSaveErrors(false);

export function demoTerminalError(row: TraceRow): number {
    return demoError.getError(new Vector([
        (row.x - xScale) / xScale,
        row.y / yScale,
        row.tc / angleScale,
        row.ts / angleScale
    ]));
}

export interface RunMetrics {
    bestD2: number;
    bestStep: number;
    terminalD2: number;
    demoTerminalError: number;
    steps: number;
    outcome: RunOutcome;
    clampCount: number;
    clampCountKnown: boolean;
    pathLen: number;
    pathLenKnown: boolean;
    isFar: boolean;
}

export function runMetrics(run: LoadedRun): RunMetrics {
    let last = run.rows[run.rows.length - 1];
    return {
        bestD2: run.recomputed.bestD2,
        bestStep: run.recomputed.bestStep,
        terminalD2: run.recomputed.terminalD2,
        demoTerminalError: last ? demoTerminalError(last) : NaN,
        steps: run.steps,
        outcome: run.outcome,
        // a strided bundle cannot support either from its rows, so the engine's
        // own count is used when it declared one
        clampCount: run.complete ? run.recomputed.clampCount : run.declared.clampCount,
        clampCountKnown: run.complete || isFinite(run.declared.clampCount),
        pathLen: run.complete ? run.recomputed.pathLen : run.declared.pathLen,
        pathLenKnown: run.complete || isFinite(run.declared.pathLen),
        isFar: run.isFar
    };
}

export interface Aggregate {
    label: string;
    n: number;
    docked: number;
    wall: number;
    bound: number;
    cap: number;
    bestD2Mean: number;
    bestD2Min: number;
    terminalD2Mean: number;
    terminalD2Min: number;
    demoErrorMean: number;
    stepsMean: number;
    pathLenMean: number;
    // runs which touched the jack-knife clamp at least once
    clampedRuns: number;
    clampedKnown: number;
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

function min(values: number[]): number {
    return values.length == 0 ? NaN : Math.min.apply(Math, values);
}

function finite(values: number[]): number[] {
    return values.filter((value) => isFinite(value));
}

/**
 * Mean and min over a start set, both of the aggregations the paper reports.
 */
export function aggregate(label: string, runs: LoadedRun[]): Aggregate {
    let metrics = runs.map(runMetrics);
    let counts: { [outcome: string]: number } = { docked: 0, wall: 0, bound: 0, cap: 0 };
    let clampedRuns = 0;
    let clampedKnown = 0;
    for (let i = 0; i < metrics.length; i++) {
        counts[metrics[i].outcome] = (counts[metrics[i].outcome] || 0) + 1;
        if (metrics[i].clampCountKnown) {
            clampedKnown++;
            if (metrics[i].clampCount > 0) {
                clampedRuns++;
            }
        }
    }
    let bestD2s = finite(metrics.map((m) => m.bestD2));
    let terminalD2s = finite(metrics.map((m) => m.terminalD2));
    return {
        label: label,
        n: runs.length,
        docked: counts["docked"],
        wall: counts["wall"],
        bound: counts["bound"],
        cap: counts["cap"],
        bestD2Mean: mean(bestD2s),
        bestD2Min: min(bestD2s),
        terminalD2Mean: mean(terminalD2s),
        terminalD2Min: min(terminalD2s),
        demoErrorMean: mean(finite(metrics.map((m) => m.demoTerminalError))),
        stepsMean: mean(finite(metrics.map((m) => m.steps))),
        pathLenMean: mean(finite(metrics.filter((m) => m.pathLenKnown).map((m) => m.pathLen))),
        clampedRuns: clampedRuns,
        clampedKnown: clampedKnown
    };
}

/**
 * The paper's stated result is that the gradient-free method won the far field
 * and lost close-up manoeuvring, so whether an arm reproduces that split is
 * one of the things a comparison is for.
 */
export function farNearAggregates(label: string, runs: LoadedRun[]): Aggregate[] {
    return [
        aggregate(label, runs),
        aggregate(label + " far", runs.filter((run) => run.isFar)),
        aggregate(label + " near", runs.filter((run) => !run.isFar))
    ];
}

export function bundleAggregates(bundle: LoadedBundle, runs: LoadedRun[]): Aggregate[] {
    return farNearAggregates(bundle.provenance.arm, runs);
}
