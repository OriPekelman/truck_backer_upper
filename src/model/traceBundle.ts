import { Point, Vector, Angle, plus, minus, rotate } from '../math'

/**
 * Reader for the tbu-traces/1 bundles produced by headless rollouts (defined in
 * OriPekelman/toy#190). A bundle holds every run of one arm: the per-step
 * trajectory plus the engine's own per-run summaries, which are recomputed here
 * so that a disagreement between the engine's arithmetic and ours is visible.
 */
export const traceBundleFormat = "tbu-traces/1";

export type RunOutcome = "docked" | "wall" | "bound" | "cap";
export const runOutcomes: RunOutcome[] = ["docked", "wall", "bound", "cap"];

export interface NetDescription {
    shape: number[];
    activation: string;
    output_map: string;
    obs: number;
}

export interface Provenance {
    arm: string;
    engine: string;
    engine_git: string;
    weights: string;
    train_seed: number;
    net: NetDescription | undefined;
    objective: string;
}

export interface PlantDescription {
    ls: number; // trailer length
    lc: number; // cabin length
    u_max_deg: number;
    r: number; // step length
    step_cap: number;
    dock_ref: string;
    wrap: string;
}

export interface RunStart {
    x: number;
    y: number;
    ts: Angle; // trailer angle
    tc: Angle; // cabin angle
    scheme: string | undefined;
    idx: number | undefined;
}

/** One sample of a run, in physical units. */
export interface TraceRow {
    step: number;
    signal: number;
    u: Angle;
    x: number;
    y: number;
    tc: Angle;
    ts: Angle;
    clamped: boolean;
}

export interface RunSummary {
    steps: number;
    bestD2: number;
    bestStep: number;
    terminalD2: number;
    clampCount: number;
    pathLen: number;
}

/**
 * The paper's score: squared distance of the trailer end from the dock,
 * tolerating one full turn of the trailer angle.
 */
export function dockingD2(x: number, y: number, ts: Angle): number {
    let wrapped = Math.min(ts * ts, Math.pow(ts - 2 * Math.PI, 2), Math.pow(ts + 2 * Math.PI, 2));
    return x * x + y * y + wrapped;
}

// relative tolerance when comparing the engine's summaries against ours
const summaryTolerance = 1e-6;

function disagrees(declared: number | undefined, recomputed: number): boolean {
    if (declared == undefined || !isFinite(declared)) {
        return false;
    }
    let scale = Math.max(1, Math.abs(declared), Math.abs(recomputed));
    return Math.abs(declared - recomputed) / scale > summaryTolerance;
}

export class LoadedRun {
    // fields of the summary which the engine and we do not agree on
    public summaryMismatch: string[] = [];

    public constructor(
        public id: number,
        public start: RunStart,
        public outcome: RunOutcome,
        public rows: TraceRow[],
        public declared: RunSummary,
        public recomputed: RunSummary,
        // whether the bundle carries every step, rather than a strided subset
        public complete: boolean
    ) {
        this.summaryMismatch = LoadedRun.compareSummaries(declared, recomputed, complete);
    }

    private static compareSummaries(declared: RunSummary, recomputed: RunSummary, complete: boolean): string[] {
        let mismatch: string[] = [];
        if (disagrees(declared.bestD2, recomputed.bestD2)) {
            mismatch.push("best_d2");
        }
        if (disagrees(declared.terminalD2, recomputed.terminalD2)) {
            mismatch.push("terminal_d2");
        }
        // a strided bundle cannot support these two, since rows are missing
        if (complete) {
            if (disagrees(declared.steps, recomputed.steps)) {
                mismatch.push("steps");
            }
            if (disagrees(declared.clampCount, recomputed.clampCount)) {
                mismatch.push("clamp_count");
            }
            if (disagrees(declared.pathLen, recomputed.pathLen)) {
                mismatch.push("path_len");
            }
        }
        return mismatch;
    }

    public get steps(): number {
        return this.declared.steps;
    }

    /** Whether this run started in the far field, the paper's x >= 50 split. */
    public get isFar(): boolean {
        return this.start.x >= 50;
    }

    /** The last sample at or before the given step, or undefined before the first. */
    public rowAt(step: number): TraceRow | undefined {
        let rows = this.rows;
        if (rows.length == 0 || step < rows[0].step) {
            return undefined;
        }
        let low = 0;
        let high = rows.length - 1;
        while (low < high) {
            let mid = Math.ceil((low + high) / 2);
            if (rows[mid].step <= step) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return rows[low];
    }

    public get bestRow(): TraceRow | undefined {
        return this.rowAt(this.recomputed.bestStep);
    }
}

export class LoadedBundle {
    public constructor(
        public label: string,
        public fileName: string,
        public provenance: Provenance,
        public plant: PlantDescription,
        public runs: LoadedRun[]
    ) {
    }

    public get outcomeCounts(): { [outcome: string]: number } {
        let counts: { [outcome: string]: number } = {};
        for (let i = 0; i < runOutcomes.length; i++) {
            counts[runOutcomes[i]] = 0;
        }
        for (let i = 0; i < this.runs.length; i++) {
            let outcome = this.runs[i].outcome;
            counts[outcome] = (counts[outcome] || 0) + 1;
        }
        return counts;
    }

    public get maxSteps(): number {
        let max = 0;
        for (let i = 0; i < this.runs.length; i++) {
            max = Math.max(max, this.runs[i].steps);
        }
        return max;
    }

    public get mismatchedRuns(): number {
        let count = 0;
        for (let i = 0; i < this.runs.length; i++) {
            if (this.runs[i].summaryMismatch.length > 0) {
                count++;
            }
        }
        return count;
    }
}

export class TraceBundleError extends Error {
    public constructor(message: string) {
        super(message);
    }
}

function requireObject(value: any, what: string): any {
    if (value == undefined || typeof value !== "object") {
        throw new TraceBundleError("missing or malformed \"" + what + "\"");
    }
    return value;
}

function number(value: any, fallback: number): number {
    return typeof value === "number" && isFinite(value) ? value : fallback;
}

function parseOutcome(value: any): RunOutcome {
    for (let i = 0; i < runOutcomes.length; i++) {
        if (value === runOutcomes[i]) {
            return runOutcomes[i];
        }
    }
    // an unknown termination is still a termination; treat it as the step cap
    return "cap";
}

class ColumnIndex {
    private index: { [name: string]: number } = {};

    public constructor(columns: any) {
        if (!(columns instanceof Array)) {
            throw new TraceBundleError("missing or malformed \"columns\"");
        }
        for (let i = 0; i < columns.length; i++) {
            this.index["" + columns[i]] = i;
        }
        if (this.index["x"] == undefined || this.index["y"] == undefined) {
            throw new TraceBundleError("\"columns\" must contain x and y");
        }
    }

    public get(row: any[], name: string, fallback: number): number {
        let i = this.index[name];
        if (i == undefined) {
            return fallback;
        }
        return number(row[i], fallback);
    }

    public has(name: string): boolean {
        return this.index[name] != undefined;
    }
}

function parseRun(raw: any, columns: ColumnIndex, index: number): LoadedRun {
    let rawStart = requireObject(raw.start, "runs[].start");
    let start: RunStart = {
        x: number(rawStart.x, 0),
        y: number(rawStart.y, 0),
        ts: number(rawStart.ts, 0),
        tc: number(rawStart.tc, 0),
        scheme: typeof rawStart.scheme === "string" ? rawStart.scheme : undefined,
        idx: typeof rawStart.idx === "number" ? rawStart.idx : undefined
    };
    let rawTrace = raw.trace;
    if (!(rawTrace instanceof Array)) {
        throw new TraceBundleError("run " + index + " has no \"trace\"");
    }

    let declaredSteps = number(raw.steps, rawTrace.length);
    let hasStepColumn = columns.has("step");
    let rows: TraceRow[] = [];
    for (let i = 0; i < rawTrace.length; i++) {
        let rawRow = rawTrace[i];
        if (!(rawRow instanceof Array)) {
            throw new TraceBundleError("run " + index + " row " + i + " is not an array");
        }
        rows.push({
            // without a step column, rows are assumed evenly spaced over the
            // episode, which is exact for an unstrided bundle
            step: hasStepColumn ? columns.get(rawRow, "step", i + 1) : stepOf(i, rawTrace.length, declaredSteps),
            signal: columns.get(rawRow, "signal", 0),
            u: columns.get(rawRow, "u", 0),
            x: columns.get(rawRow, "x", 0),
            y: columns.get(rawRow, "y", 0),
            tc: columns.get(rawRow, "tc", 0),
            ts: columns.get(rawRow, "ts", 0),
            clamped: columns.get(rawRow, "clamped", 0) != 0
        });
    }

    let complete = rows.length >= declaredSteps;
    let declared: RunSummary = {
        steps: declaredSteps,
        bestD2: number(raw.best_d2, NaN),
        bestStep: number(raw.best_step, NaN),
        terminalD2: number(raw.terminal_d2, NaN),
        clampCount: number(raw.clamp_count, NaN),
        pathLen: number(raw.path_len, NaN)
    };
    return new LoadedRun(number(raw.id, index), start, parseOutcome(raw.end), rows, declared, recompute(start, rows, declaredSteps), complete);
}

function stepOf(rowIndex: number, rowCount: number, steps: number): number {
    if (rowCount >= steps || rowCount <= 1) {
        return rowIndex + 1;
    }
    return Math.round(1 + rowIndex * (steps - 1) / (rowCount - 1));
}

function recompute(start: RunStart, rows: TraceRow[], declaredSteps: number): RunSummary {
    let bestD2 = Infinity;
    let bestStep = 0;
    let clampCount = 0;
    let pathLen = 0;
    let previousX = start.x;
    let previousY = start.y;
    for (let i = 0; i < rows.length; i++) {
        let row = rows[i];
        let d2 = dockingD2(row.x, row.y, row.ts);
        if (d2 < bestD2) {
            bestD2 = d2;
            bestStep = row.step;
        }
        if (row.clamped) {
            clampCount++;
        }
        pathLen += Math.sqrt(Math.pow(row.x - previousX, 2) + Math.pow(row.y - previousY, 2));
        previousX = row.x;
        previousY = row.y;
    }
    let last = rows[rows.length - 1];
    return {
        steps: rows.length > 0 ? Math.max(declaredSteps, rows[rows.length - 1].step) : 0,
        bestD2: rows.length > 0 ? bestD2 : NaN,
        bestStep: bestStep,
        terminalD2: last ? dockingD2(last.x, last.y, last.ts) : NaN,
        clampCount: clampCount,
        pathLen: pathLen
    };
}

export function parseTraceBundle(fileName: string, raw: any): LoadedBundle {
    requireObject(raw, "bundle");
    if (raw.format !== traceBundleFormat) {
        throw new TraceBundleError("unsupported format \"" + raw.format + "\", expected \"" + traceBundleFormat + "\"");
    }
    let rawProvenance = requireObject(raw.provenance, "provenance");
    let rawPlant = requireObject(raw.plant, "plant");
    if (!(raw.runs instanceof Array) || raw.runs.length == 0) {
        throw new TraceBundleError("bundle has no runs");
    }
    let columns = new ColumnIndex(raw.columns);

    let provenance: Provenance = {
        arm: "" + (rawProvenance.arm || "unnamed"),
        engine: "" + (rawProvenance.engine || "unknown"),
        engine_git: "" + (rawProvenance.engine_git || ""),
        weights: "" + (rawProvenance.weights || ""),
        train_seed: number(rawProvenance.train_seed, NaN),
        net: rawProvenance.net ? {
            shape: rawProvenance.net.shape instanceof Array ? rawProvenance.net.shape : [],
            activation: "" + (rawProvenance.net.activation || ""),
            output_map: "" + (rawProvenance.net.output_map || ""),
            obs: number(rawProvenance.net.obs, NaN)
        } : undefined,
        objective: "" + (rawProvenance.objective || "")
    };
    let plant: PlantDescription = {
        ls: number(rawPlant.ls, 14),
        lc: number(rawPlant.lc, 6),
        u_max_deg: number(rawPlant.u_max_deg, 70),
        r: number(rawPlant.r, 1),
        step_cap: number(rawPlant.step_cap, 0),
        dock_ref: "" + (rawPlant.dock_ref || "trailer"),
        wrap: "" + (rawPlant.wrap || "")
    };

    let runs: LoadedRun[] = [];
    for (let i = 0; i < raw.runs.length; i++) {
        runs.push(parseRun(raw.runs[i], columns, i));
    }
    return new LoadedBundle(provenance.arm, fileName, provenance, plant, runs);
}

/**
 * Cabin & trailer outlines of a rig in the given state, the same geometry the
 * live simulation draws, so a replayed run looks like a driven one.
 */
export function rigOutlines(x: number, y: number, tc: Angle, ts: Angle, plant: PlantDescription): Point[][] {
    let width = 0.5 * plant.lc;
    let tep = new Point(x, y);
    let trailerDirection = rotate(new Vector(1, 0), ts);
    let cdp = plus(tep, new Vector(trailerDirection.x * plant.ls, trailerDirection.y * plant.ls));
    let cabinDirection = rotate(new Vector(1, 0), tc);
    let cfp = plus(cdp, new Vector(cabinDirection.x * plant.lc, cabinDirection.y * plant.lc));
    let eot = plus(cdp, new Vector(cabinDirection.x * 2, cabinDirection.y * 2));

    let trailerOffset = trailerDirection.getOrthogonalVector().scale(width / 2);
    let cabinOffset = cabinDirection.getOrthogonalVector().scale(width / 2);
    return [
        [plus(tep, trailerOffset), minus(tep, trailerOffset), minus(cdp, trailerOffset), plus(cdp, trailerOffset)],
        [plus(eot, cabinOffset), minus(eot, cabinOffset), minus(cfp, cabinOffset), plus(cfp, cabinOffset)]
    ];
}
