import { toDeg } from '../math'
import { LoadedBundle, LoadedRun, traceBundleFormat } from '../model/traceBundle'
import { runMetrics } from '../model/metrics'

/**
 * Exports for the cross-engine parity check: toy emits the same per-step rows
 * for the same weights on the same start, and diffing the two is how a
 * disagreement in the controller forward pass gets found rather than dressed
 * up as a result about credit rules. Values are written at full precision, not
 * as displayed, since the comparison is to 1e-12.
 */

const traceColumns = ["step", "signal", "u", "x", "y", "tc", "ts", "clamped"];

function full(value: number): string {
    return isFinite(value) ? String(value) : "";
}

function header(bundle: LoadedBundle, run: LoadedRun): { [key: string]: any } {
    return {
        format: traceBundleFormat,
        arm: bundle.provenance.arm,
        engine: bundle.provenance.engine,
        engine_git: bundle.provenance.engine_git,
        weights: bundle.provenance.weights,
        train_seed: bundle.provenance.train_seed,
        objective: bundle.provenance.objective,
        net: bundle.provenance.net,
        plant: bundle.plant,
        run_id: run.id,
        start: {
            x: run.start.x, y: run.start.y, ts: run.start.ts, tc: run.start.tc,
            scheme: run.start.scheme, idx: run.start.idx
        },
        end: run.outcome,
        steps: run.steps,
        strided: !run.complete
    };
}

/** One run's per-step trace, with the header a diff needs to be meaningful. */
export function runToJson(bundle: LoadedBundle, run: LoadedRun): string {
    return JSON.stringify({
        header: header(bundle, run),
        columns: traceColumns,
        trace: run.rows.map((row) => [row.step, row.signal, row.u, row.x, row.y, row.tc, row.ts, row.clamped ? 1 : 0])
    }, undefined, 2);
}

export function runToCsv(bundle: LoadedBundle, run: LoadedRun): string {
    let lines: string[] = [];
    let head = header(bundle, run);
    for (let key in head) {
        let value = head[key];
        lines.push("# " + key + ": " + (typeof value === "object" ? JSON.stringify(value) : value));
    }
    lines.push(traceColumns.join(","));
    for (let i = 0; i < run.rows.length; i++) {
        let row = run.rows[i];
        lines.push([row.step, full(row.signal), full(row.u), full(row.x), full(row.y),
            full(row.tc), full(row.ts), row.clamped ? 1 : 0].join(","));
    }
    return lines.join("\n") + "\n";
}

const metricColumns = [
    "arm", "run", "start_x", "start_y", "start_ts_deg", "start_tc_deg", "field", "end",
    "steps", "best_d2", "best_step", "terminal_d2", "demo_terminal_error", "clamp_crossings", "path_len"
];

/** One row per run: the metrics table as it stands, for reading elsewhere. */
export function metricsToCsv(rows: { bundle: LoadedBundle, run: LoadedRun }[]): string {
    let lines = [metricColumns.join(",")];
    for (let i = 0; i < rows.length; i++) {
        let bundle = rows[i].bundle;
        let run = rows[i].run;
        let metrics = runMetrics(run);
        lines.push([
            bundle.provenance.arm,
            run.id,
            full(run.start.x),
            full(run.start.y),
            full(toDeg(run.start.ts)),
            full(toDeg(run.start.tc)),
            metrics.isFar ? "far" : "near",
            metrics.outcome,
            metrics.steps,
            full(metrics.bestD2),
            metrics.bestStep,
            full(metrics.terminalD2),
            full(metrics.demoTerminalError),
            metrics.clampCountKnown ? full(metrics.clampCount) : "",
            metrics.pathLenKnown ? full(metrics.pathLen) : ""
        ].join(","));
    }
    return lines.join("\n") + "\n";
}

export function download(fileName: string, mimeType: string, content: string): void {
    let blob = new Blob([content], { type: mimeType });
    let url = URL.createObjectURL(blob);
    let link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    // the object URL would otherwise hold the blob for the life of the document
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
