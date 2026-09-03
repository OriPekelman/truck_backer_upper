import * as React from 'react'
import { toDeg } from '../math'
import { LoadedBundle, LoadedRun } from '../model/traceBundle'
import { Aggregate, RunMetrics, runMetrics, bundleAggregates } from '../model/metrics'
import { RunRef, sameRun } from './TraceOverlay'

export interface MetricsRow {
    ref: RunRef;
    bundle: LoadedBundle;
    run: LoadedRun;
    color: string;
}

interface MetricsTableProps {
    // the runs currently drawn, in the order they were loaded
    rows: MetricsRow[];
    // every loaded bundle with the runs of it which pass the filters
    bundles: { bundle: LoadedBundle, runs: LoadedRun[], color: string }[];
    selected: RunRef | undefined;
    onSelect: (ref: RunRef | undefined) => void;
}

type SortKey = "arm" | "run" | "start" | "field" | "end" | "steps" | "bestD2" | "terminalD2" | "demoError" | "clamps" | "pathLen";

interface MetricsTableState {
    sortKey: SortKey;
    ascending: boolean;
    showAll: boolean;
}

// a few hundred rows is readable, several thousand is not
const rowLimit = 100;

function fixed(value: number, digits: number): string {
    return isFinite(value) ? value.toFixed(digits) : "-";
}

function percent(part: number, whole: number): string {
    return whole > 0 ? (100 * part / whole).toFixed(1) + "%" : "-";
}

export class MetricsTable extends React.Component<MetricsTableProps, MetricsTableState> {

    public constructor(props: MetricsTableProps) {
        super(props)
        this.state = { sortKey: "bestD2", ascending: true, showAll: false };
    }

    private sortValue(row: MetricsRow, metrics: RunMetrics): number | string {
        switch (this.state.sortKey) {
            case "arm": return row.bundle.provenance.arm;
            case "run": return row.run.id;
            case "start": return row.run.start.x;
            case "field": return metrics.isFar ? "far" : "near";
            case "end": return metrics.outcome;
            case "steps": return metrics.steps;
            case "terminalD2": return metrics.terminalD2;
            case "demoError": return metrics.demoTerminalError;
            case "clamps": return metrics.clampCountKnown ? metrics.clampCount : -1;
            case "pathLen": return metrics.pathLenKnown ? metrics.pathLen : -1;
            default: return metrics.bestD2;
        }
    }

    private sortedRows(): { row: MetricsRow, metrics: RunMetrics }[] {
        let entries = this.props.rows.map((row) => ({ row: row, metrics: runMetrics(row.run) }));
        let direction = this.state.ascending ? 1 : -1;
        entries.sort((a, b) => {
            let left = this.sortValue(a.row, a.metrics);
            let right = this.sortValue(b.row, b.metrics);
            if (typeof left === "string" || typeof right === "string") {
                return direction * String(left).localeCompare(String(right));
            }
            // a run which cannot report a metric sorts last either way
            if (!isFinite(left)) {
                return 1;
            }
            if (!isFinite(right)) {
                return -1;
            }
            return direction * (left - right);
        });
        return entries;
    }

    private handleSort(key: SortKey) {
        this.setState((state) => ({
            sortKey: key,
            ascending: state.sortKey == key ? !state.ascending : true
        }));
    }

    private sortableHeader(key: SortKey, label: string, title?: string) {
        let marker = this.state.sortKey == key ? (this.state.ascending ? " ▲" : " ▼") : "";
        return <th key={key} className="metricsSortable" title={title} onClick={() => this.handleSort(key)}>{label}{marker}</th>
    }

    private renderAggregate(aggregate: Aggregate, color: string, emphasise: boolean) {
        return <tr key={aggregate.label} className={emphasise ? "" : "text-muted"}>
            <td>{emphasise ? <span className="bundleSwatch" style={{ backgroundColor: color }}></span> : null}{aggregate.label}</td>
            <td>{aggregate.n}</td>
            <td>{percent(aggregate.docked, aggregate.n)}</td>
            <td>{aggregate.wall} / {aggregate.bound} / {aggregate.cap}</td>
            <td>{fixed(aggregate.bestD2Mean, 3)}</td>
            <td>{fixed(aggregate.bestD2Min, 3)}</td>
            <td>{fixed(aggregate.terminalD2Mean, 3)}</td>
            <td>{fixed(aggregate.terminalD2Min, 3)}</td>
            <td>{fixed(aggregate.demoErrorMean, 5)}</td>
            <td>{fixed(aggregate.stepsMean, 1)}</td>
            <td>{fixed(aggregate.pathLenMean, 1)}</td>
            <td>{aggregate.clampedKnown > 0 ? percent(aggregate.clampedRuns, aggregate.clampedKnown) : "-"}</td>
        </tr>
    }

    private renderAggregates() {
        let rows: JSX.Element[] = [];
        for (let i = 0; i < this.props.bundles.length; i++) {
            let entry = this.props.bundles[i];
            let aggregates = bundleAggregates(entry.bundle, entry.runs);
            for (let j = 0; j < aggregates.length; j++) {
                rows.push(this.renderAggregate(aggregates[j], entry.color, j == 0));
            }
        }
        return <table className="table table-sm bundleTable">
            <thead>
                <tr>
                    <th>Arm</th><th>n</th><th>Docked</th>
                    <th title="wall / yard bound / step cap">wall / bound / cap</th>
                    <th title="the paper's score at the closest approach of the episode">Mean best d²</th>
                    <th>Min best d²</th>
                    <th title="the paper's score at the state the episode stopped in">Mean terminal d²</th>
                    <th>Min terminal d²</th>
                    <th title="TruckControllerError on the terminal state: the number the bundled controllers were trained toward">Mean demo error</th>
                    <th>Mean steps</th><th>Mean path</th>
                    <th title="runs which touched the jack-knife clamp at least once">Clamped</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    }

    private renderRuns() {
        let entries = this.sortedRows();
        let shown = this.state.showAll ? entries : entries.slice(0, rowLimit);
        return <div>
            <table className="table table-sm table-hover bundleTable">
                <thead>
                    <tr>
                        {this.sortableHeader("arm", "Arm")}
                        {this.sortableHeader("run", "Run")}
                        {this.sortableHeader("start", "Start")}
                        {this.sortableHeader("field", "Field")}
                        {this.sortableHeader("end", "Ended")}
                        {this.sortableHeader("steps", "Steps")}
                        {this.sortableHeader("bestD2", "Best d²", "the paper's score at the closest approach")}
                        {this.sortableHeader("terminalD2", "Terminal d²")}
                        {this.sortableHeader("demoError", "Demo error", "TruckControllerError on the terminal state")}
                        {this.sortableHeader("clamps", "Clamps", "steps where the jack-knife clamp had to project the cabin angle back")}
                        {this.sortableHeader("pathLen", "Path")}
                    </tr>
                </thead>
                <tbody>
                    {shown.map((entry) => {
                        let metrics = entry.metrics;
                        let run = entry.row.run;
                        let isSelected = sameRun(entry.row.ref, this.props.selected);
                        return <tr key={entry.row.ref.bundleIndex + "/" + entry.row.ref.runIndex}
                            className={"metricsRow" + (isSelected ? " table-active" : "")}
                            onClick={() => this.props.onSelect(isSelected ? undefined : entry.row.ref)}>
                            <td><span className="bundleSwatch" style={{ backgroundColor: entry.row.color }}></span>{entry.row.bundle.provenance.arm}</td>
                            <td>{run.id}</td>
                            <td>({fixed(run.start.x, 0)}, {fixed(run.start.y, 0)}) {fixed(toDeg(run.start.ts), 0)}°</td>
                            <td>{metrics.isFar ? "far" : "near"}</td>
                            <td>{metrics.outcome}</td>
                            <td>{metrics.steps}</td>
                            <td>{fixed(metrics.bestD2, 3)}<span className="text-muted"> @{metrics.bestStep}</span></td>
                            <td>{fixed(metrics.terminalD2, 3)}</td>
                            <td>{fixed(metrics.demoTerminalError, 5)}</td>
                            <td>{metrics.clampCountKnown ? metrics.clampCount : <span className="text-muted" title="a strided bundle cannot support this and the engine declared none">n/a</span>}</td>
                            <td>{metrics.pathLenKnown ? fixed(metrics.pathLen, 1) : <span className="text-muted">n/a</span>}</td>
                        </tr>
                    })}
                </tbody>
            </table>
            {entries.length > shown.length || this.state.showAll
                ? <p className="bundleHint">
                    showing {shown.length} of {entries.length} runs{" "}
                    <button type="button" className="btn btn-sm btn-link" onClick={() => this.setState({ showAll: !this.state.showAll })}>
                        {this.state.showAll ? "show fewer" : "show all"}
                    </button>
                </p>
                : null}
        </div>
    }

    public render() {
        if (this.props.rows.length == 0) {
            return null;
        }
        return <div className="metricsSection">
            <h4>Per arm</h4>
            <p className="bundleHint">
                Mean and min, both of the aggregations the paper reports, over the runs currently shown,
                split by the paper's far/near boundary at x = 50.
            </p>
            {this.renderAggregates()}
            <h4>Per run</h4>
            <p className="bundleHint">Click a column to sort, a row to isolate that run on the yard.</p>
            {this.renderRuns()}
        </div>
    }
}
