import * as React from 'react'
import { toDeg } from '../math'
import { LoadedBundle, LoadedRun, RunOutcome, runOutcomes, parseTraceBundle, traceBundleFormat } from '../model/traceBundle'
import { TraceOverlay, DisplayRun, RunRef, sameRun } from './TraceOverlay'

type ColorMode = "arm" | "outcome";
type FieldFilter = "all" | "far" | "near";

interface TraceBundleViewState {
    bundles: LoadedBundle[];
    errors: string[];
    colorMode: ColorMode;
    hiddenOutcomes: { [outcome: string]: boolean };
    fieldFilter: FieldFilter;
    clampedOnly: boolean;
    time: number;
    playing: boolean;
    stepsPerSecond: number;
    selected: RunRef | undefined;
    hovered: RunRef | undefined;
}

// one colour per loaded bundle, i.e. per arm
const armColors = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#e15759", "#edc948", "#9c755f"];

// the same colours the live simulation marks terminations with
const outcomeColors: { [outcome: string]: string } = {
    docked: "#5cb85c",
    wall: "#d9534f",
    bound: "#8e44ad",
    cap: "#f0ad4e"
};

// the yard's aspect ratio, so the drawing fills the canvas
const canvasWidth = 663;
const canvasHeight = 640;

const playbackSpeeds = [15, 30, 60, 120];

export class TraceBundleView extends React.Component<{}, TraceBundleViewState> {
    private lastFrameTime: number = 0;
    private animation: number | undefined = undefined;

    public constructor(props: {}) {
        super(props)
        this.state = {
            bundles: [],
            errors: [],
            colorMode: "arm",
            hiddenOutcomes: {},
            fieldFilter: "all",
            clampedOnly: false,
            time: 0,
            playing: false,
            stepsPerSecond: 30,
            selected: undefined,
            hovered: undefined
        };
    }

    public componentWillUnmount() {
        this.stopPlaying();
    }

    private get maxSteps(): number {
        let max = 0;
        for (let i = 0; i < this.state.bundles.length; i++) {
            max = Math.max(max, this.state.bundles[i].maxSteps);
        }
        return max;
    }

    private isVisible(run: LoadedRun): boolean {
        if (this.state.hiddenOutcomes[run.outcome]) {
            return false;
        }
        if (this.state.fieldFilter == "far" && !run.isFar) {
            return false;
        }
        if (this.state.fieldFilter == "near" && run.isFar) {
            return false;
        }
        if (this.state.clampedOnly && run.declared.clampCount <= 0 && run.recomputed.clampCount <= 0) {
            return false;
        }
        return true;
    }

    private get displayRuns(): DisplayRun[] {
        let selected = this.state.selected;
        let display: DisplayRun[] = [];
        for (let bundleIndex = 0; bundleIndex < this.state.bundles.length; bundleIndex++) {
            let bundle = this.state.bundles[bundleIndex];
            for (let runIndex = 0; runIndex < bundle.runs.length; runIndex++) {
                let run = bundle.runs[runIndex];
                let ref = { bundleIndex: bundleIndex, runIndex: runIndex };
                // a selected run is isolated, but stays visible whatever the filters
                if (selected) {
                    if (!sameRun(ref, selected)) {
                        continue;
                    }
                } else if (!this.isVisible(run)) {
                    continue;
                }
                display.push({
                    ref: ref,
                    run: run,
                    plant: bundle.plant,
                    color: this.state.colorMode == "arm" ? armColors[bundleIndex % armColors.length] : outcomeColors[run.outcome]
                });
            }
        }
        return display;
    }

    /** Changes whenever the drawn set of paths changes, not on every scrub step. */
    private get staticKey(): string {
        let selected = this.state.selected;
        return [
            this.state.bundles.map((b) => b.label + ":" + b.runs.length).join(","),
            this.state.colorMode,
            runOutcomes.filter((o) => this.state.hiddenOutcomes[o]).join("|"),
            this.state.fieldFilter,
            this.state.clampedOnly ? "clamped" : "",
            selected ? selected.bundleIndex + "/" + selected.runIndex : ""
        ].join(";");
    }

    private async readBundleFile(file: File): Promise<any> {
        let buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = () => resolve(reader.result as ArrayBuffer);
            reader.onerror = () => reject(new Error("could not read " + file.name));
            reader.readAsArrayBuffer(file);
        });
        let bytes = new Uint8Array(buffer);
        let text: string;
        if (bytes.length > 1 && bytes[0] == 0x1f && bytes[1] == 0x8b) {
            let decompressor = (window as any).DecompressionStream;
            if (!decompressor) {
                throw new Error(file.name + " is gzipped and this browser has no DecompressionStream; gunzip it first");
            }
            let stream = (new Blob([bytes]) as any).stream().pipeThrough(new decompressor("gzip"));
            text = await new Response(stream).text();
        } else {
            text = new TextDecoder("utf-8").decode(bytes);
        }
        return JSON.parse(text);
    }

    private async loadFiles(files: File[]) {
        let bundles: LoadedBundle[] = [];
        let errors: string[] = [];
        for (let i = 0; i < files.length; i++) {
            let file = files[i];
            try {
                bundles.push(parseTraceBundle(file.name, await this.readBundleFile(file)));
            } catch (e) {
                errors.push(file.name + ": " + (e instanceof Error ? e.message : e));
            }
        }
        this.setState((state) => ({
            bundles: state.bundles.concat(bundles),
            errors: errors,
            selected: undefined,
            hovered: undefined,
            time: 0
        }));
    }

    /**
     * A FileList is live: clearing the input, or letting the drop event go,
     * empties it while the files are still being read. Copy it first.
     */
    private static toArray(files: FileList | null): File[] {
        let copy: File[] = [];
        for (let i = 0; files && i < files.length; i++) {
            copy.push(files[i]);
        }
        return copy;
    }

    private handleFilesChosen(e: React.ChangeEvent<HTMLInputElement>) {
        let files = TraceBundleView.toArray(e.currentTarget.files);
        e.currentTarget.value = "";
        if (files.length > 0) {
            this.loadFiles(files);
        }
    }

    private handleDrop(e: React.DragEvent<HTMLDivElement>) {
        e.preventDefault();
        let files = e.dataTransfer ? TraceBundleView.toArray(e.dataTransfer.files) : [];
        if (files.length > 0) {
            this.loadFiles(files);
        }
    }

    private handleRemoveBundle(index: number) {
        this.stopPlaying();
        this.setState((state) => ({
            bundles: state.bundles.filter((b, i) => i != index),
            selected: undefined,
            hovered: undefined,
            playing: false
        }));
    }

    private stopPlaying() {
        if (this.animation != undefined) {
            window.cancelAnimationFrame(this.animation);
            this.animation = undefined;
        }
    }

    private handleTogglePlay() {
        if (this.state.playing) {
            this.stopPlaying();
            this.setState({ playing: false });
            return;
        }
        this.lastFrameTime = performance.now();
        this.setState({ playing: true, time: this.state.time >= this.maxSteps ? 0 : this.state.time }, () => {
            this.animation = window.requestAnimationFrame(this.playFrame.bind(this));
        });
    }

    private playFrame(timestamp: number) {
        let delta = (timestamp - this.lastFrameTime) / 1000;
        this.lastFrameTime = timestamp;
        let time = this.state.time + delta * this.state.stepsPerSecond;
        if (time >= this.maxSteps) {
            this.stopPlaying();
            this.setState({ time: this.maxSteps, playing: false });
            return;
        }
        this.setState({ time: time }, () => {
            this.animation = window.requestAnimationFrame(this.playFrame.bind(this));
        });
    }

    private handleTimeChanged(e: React.ChangeEvent<HTMLInputElement>) {
        this.stopPlaying();
        this.setState({ time: Number.parseFloat(e.currentTarget.value), playing: false });
    }

    private handleColorModeChanged(mode: ColorMode) {
        this.setState({ colorMode: mode });
    }

    private handleOutcomeToggled(outcome: RunOutcome) {
        this.setState((state) => {
            let hidden: { [outcome: string]: boolean } = {};
            for (let key in state.hiddenOutcomes) {
                hidden[key] = state.hiddenOutcomes[key];
            }
            hidden[outcome] = !hidden[outcome];
            return { hiddenOutcomes: hidden };
        });
    }

    private handleShowEverything() {
        this.setState({ hiddenOutcomes: {}, fieldFilter: "all", clampedOnly: false, selected: undefined });
    }

    private renderLoader() {
        return <div className="bundleDropZone" onDrop={this.handleDrop.bind(this)} onDragOver={(e) => e.preventDefault()}>
            <p className="mb">
                Drop <code>{traceBundleFormat}</code> bundles here (<code>.json</code> or gzipped), or
            </p>
            <input type="file" multiple accept=".json,.gz,.jsonc,application/json,application/gzip" onChange={this.handleFilesChosen.bind(this)} />
            <p className="bundleHint">One bundle per arm. Load several to compare them on the same start states.</p>
        </div>
    }

    private renderErrors() {
        if (this.state.errors.length == 0) {
            return null;
        }
        return <div className="alert alert-danger">
            {this.state.errors.map((error, i) => <div key={i}>{error}</div>)}
        </div>
    }

    private renderSummaries() {
        if (this.state.bundles.length == 0) {
            return null;
        }
        let rows = this.state.bundles.map((bundle, index) => {
            let visible = bundle.runs.filter((run) => this.isVisible(run));
            let docked = visible.filter((run) => run.outcome == "docked").length;
            let bestD2s = visible.map((run) => run.recomputed.bestD2).filter((d) => isFinite(d));
            let meanBestD2 = bestD2s.length > 0 ? bestD2s.reduce((a, b) => a + b, 0) / bestD2s.length : NaN;
            let minBestD2 = bestD2s.length > 0 ? Math.min.apply(Math, bestD2s) : NaN;
            let meanSteps = visible.length > 0 ? visible.reduce((sum, run) => sum + run.steps, 0) / visible.length : NaN;
            let color = armColors[index % armColors.length];
            return <tr key={index}>
                <td><span className="bundleSwatch" style={{ backgroundColor: color }}></span>{bundle.provenance.arm}</td>
                <td>{bundle.provenance.engine}{bundle.provenance.engine_git ? " @ " + bundle.provenance.engine_git.substring(0, 8) : ""}</td>
                <td>{bundle.provenance.net ? bundle.provenance.net.shape.join("-") + " " + bundle.provenance.net.activation : "-"}</td>
                <td>r={bundle.plant.r}, cap={bundle.plant.step_cap}, dock={bundle.plant.dock_ref}</td>
                <td>{visible.length} / {bundle.runs.length}</td>
                <td>{visible.length > 0 ? (100 * docked / visible.length).toFixed(1) + "%" : "-"}</td>
                <td>{isFinite(meanBestD2) ? meanBestD2.toFixed(2) : "-"}</td>
                <td>{isFinite(minBestD2) ? minBestD2.toFixed(2) : "-"}</td>
                <td>{isFinite(meanSteps) ? meanSteps.toFixed(1) : "-"}</td>
                <td>{bundle.mismatchedRuns > 0
                    ? <span className="text-danger" title="the engine's per-run summaries disagree with the same quantities recomputed from its own traces">{bundle.mismatchedRuns} run(s) disagree</span>
                    : <span className="text-muted">agrees</span>}</td>
                <td><button type="button" className="btn btn-sm btn-outline-danger" onClick={() => this.handleRemoveBundle(index)}>Remove</button></td>
            </tr>
        });
        return <table className="table table-sm bundleTable">
            <thead>
                <tr>
                    <th>Arm</th><th>Engine</th><th>Net</th><th>Plant</th><th>Runs shown</th>
                    <th>Docked</th><th>Mean best d²</th><th>Min best d²</th><th>Mean steps</th><th>Summaries</th><th></th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    }

    private renderControls() {
        let counts: { [outcome: string]: number } = {};
        for (let i = 0; i < this.state.bundles.length; i++) {
            let bundleCounts = this.state.bundles[i].outcomeCounts;
            for (let outcome in bundleCounts) {
                counts[outcome] = (counts[outcome] || 0) + bundleCounts[outcome];
            }
        }
        return <div className="bundleControls">
            <span className="bundleControlGroup">
                <strong>Colour by</strong>
                <label className="trace-legend-toggle"><input type="radio" checked={this.state.colorMode == "arm"} onChange={() => this.handleColorModeChanged("arm")} /><span>arm</span></label>
                <label className="trace-legend-toggle"><input type="radio" checked={this.state.colorMode == "outcome"} onChange={() => this.handleColorModeChanged("outcome")} /><span>outcome</span></label>
            </span>
            <span className="bundleControlGroup">
                <strong>Outcome</strong>
                {runOutcomes.map((outcome) => <label key={outcome} className="trace-legend-toggle">
                    <input type="checkbox" checked={!this.state.hiddenOutcomes[outcome]} onChange={() => this.handleOutcomeToggled(outcome)} />
                    <span><span className="bundleSwatch" style={{ backgroundColor: outcomeColors[outcome] }}></span>{outcome} ({counts[outcome] || 0})</span>
                </label>)}
            </span>
            <span className="bundleControlGroup">
                <strong>Start field</strong>
                {(["all", "far", "near"] as FieldFilter[]).map((field) => <label key={field} className="trace-legend-toggle">
                    <input type="radio" checked={this.state.fieldFilter == field} onChange={() => this.setState({ fieldFilter: field })} />
                    <span>{field == "far" ? "far (x ≥ 50)" : field == "near" ? "near (x < 50)" : "all"}</span>
                </label>)}
                <label className="trace-legend-toggle"><input type="checkbox" checked={this.state.clampedOnly} onChange={(e) => this.setState({ clampedOnly: e.currentTarget.checked })} /><span>jack-knifed only</span></label>
                <button type="button" className="btn btn-sm btn-outline-secondary" onClick={this.handleShowEverything.bind(this)}>Show all</button>
            </span>
        </div>
    }

    private renderScrub(runCount: number) {
        let maxSteps = this.maxSteps;
        return <div className="bundleScrub">
            <button type="button" className="btn btn-sm btn-primary" disabled={maxSteps == 0} onClick={this.handleTogglePlay.bind(this)}>
                {this.state.playing ? "Pause" : "Play"}
            </button>
            <input type="range" className="bundleTimeSlider" min={0} max={Math.max(maxSteps, 1)} step={1}
                value={this.state.time} onChange={this.handleTimeChanged.bind(this)} />
            <span className="bundleStepLabel">step {this.state.time.toFixed(0)} / {maxSteps}</span>
            <span className="btn-group btn-group-sm" role="group">
                {playbackSpeeds.map((speed) => <button key={speed} type="button"
                    className={"btn " + (this.state.stepsPerSecond == speed ? "btn-secondary" : "btn-outline-secondary")}
                    onClick={() => this.setState({ stepsPerSecond: speed })}>{speed}/s</button>)}
            </span>
            <span className="bundleStepLabel">{runCount} run(s) drawn</span>
        </div>
    }

    private renderRunDetails(display: DisplayRun | undefined) {
        if (!display) {
            return <p className="bundleHint">Hover a truck to see its rig, click it to isolate the run and its metrics.</p>
        }
        let run = display.run;
        let bundle = this.state.bundles[display.ref.bundleIndex];
        let declared = run.declared;
        let recomputed = run.recomputed;
        let row = (name: string, fromEngine: string, recomputedHere: string, flagged: boolean) => <tr key={name} className={flagged ? "table-danger" : ""}>
            <td>{name}</td><td>{fromEngine}</td><td>{recomputedHere}</td>
        </tr>;
        return <div className="bundleRunDetails">
            <h5>
                <span className="bundleSwatch" style={{ backgroundColor: display.color }}></span>
                {bundle.provenance.arm} run {run.id} — ended <strong>{run.outcome}</strong> after {run.steps} steps
                {run.complete ? "" : " (strided bundle: positions between samples are interpolated)"}
            </h5>
            <p className="bundleHint">
                start ({run.start.x.toFixed(1)}, {run.start.y.toFixed(1)}) trailer {toDeg(run.start.ts).toFixed(0)}° cabin {toDeg(run.start.tc).toFixed(0)}°
                {run.start.scheme ? ", " + run.start.scheme + (run.start.idx != undefined ? " #" + run.start.idx : "") : ""}
                {" — "}{run.isFar ? "far field" : "near field"}
            </p>
            <table className="table table-sm bundleTable">
                <thead><tr><th>Metric</th><th>Engine</th><th>Recomputed here</th></tr></thead>
                <tbody>
                    {row("best-approach d²", isFinite(declared.bestD2) ? declared.bestD2.toFixed(4) : "-", recomputed.bestD2.toFixed(4), run.summaryMismatch.indexOf("best_d2") >= 0)}
                    {row("best-approach step", isFinite(declared.bestStep) ? "" + declared.bestStep : "-", "" + recomputed.bestStep, false)}
                    {row("terminal d²", isFinite(declared.terminalD2) ? declared.terminalD2.toFixed(4) : "-", recomputed.terminalD2.toFixed(4), run.summaryMismatch.indexOf("terminal_d2") >= 0)}
                    {row("clamp crossings", isFinite(declared.clampCount) ? "" + declared.clampCount : "-", run.complete ? "" + recomputed.clampCount : "n/a (strided)", run.summaryMismatch.indexOf("clamp_count") >= 0)}
                    {row("path length", isFinite(declared.pathLen) ? declared.pathLen.toFixed(1) : "-", run.complete ? recomputed.pathLen.toFixed(1) : "n/a (strided)", run.summaryMismatch.indexOf("path_len") >= 0)}
                    {row("steps", "" + declared.steps, run.complete ? "" + recomputed.steps : "n/a (strided)", run.summaryMismatch.indexOf("steps") >= 0)}
                </tbody>
            </table>
        </div>
    }

    public render() {
        let displayRuns = this.displayRuns;
        let focus = this.state.selected || this.state.hovered;
        let focused: DisplayRun | undefined = undefined;
        for (let i = 0; i < displayRuns.length; i++) {
            if (sameRun(displayRuns[i].ref, focus)) {
                focused = displayRuns[i];
            }
        }
        return <div className="row">
            <div className="col-sm-12">
                <h2>Rollout overlay</h2>
                <p>
                    Headless rollouts exported as <code>{traceBundleFormat}</code> bundles, overlaid on one yard.
                    Nothing is run in the browser here — the picture is what the engine drew.
                </p>
                {this.renderLoader()}
                {this.renderErrors()}
                {this.renderSummaries()}
                {this.state.bundles.length > 0 ? this.renderControls() : null}
                <TraceOverlay runs={displayRuns} time={this.state.time} selected={this.state.selected} hovered={this.state.hovered}
                    staticKey={this.staticKey} width={canvasWidth} height={canvasHeight}
                    onHover={(ref) => this.setState({ hovered: ref })}
                    onSelect={(ref) => this.setState({ selected: ref })} />
                {this.state.bundles.length > 0 ? this.renderScrub(displayRuns.length) : null}
                {this.renderRunDetails(focused)}
            </div>
        </div>
    }
}
