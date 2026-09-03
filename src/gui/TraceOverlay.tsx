import * as React from 'react'
import { LoadedRun, PlantDescription, TraceRow, rigOutlines } from '../model/traceBundle'

export interface RunRef {
    bundleIndex: number;
    runIndex: number;
}

export function sameRun(a: RunRef | undefined, b: RunRef | undefined): boolean {
    return a != undefined && b != undefined && a.bundleIndex == b.bundleIndex && a.runIndex == b.runIndex;
}

export interface DisplayRun {
    ref: RunRef;
    run: LoadedRun;
    plant: PlantDescription;
    color: string;
}

interface TraceOverlayProps {
    runs: DisplayRun[];
    // step every run is drawn at, in lockstep
    time: number;
    selected: RunRef | undefined;
    hovered: RunRef | undefined;
    // changes whenever the static layer (paths, starts) has to be redrawn
    staticKey: string;
    width: number;
    height: number;
    onHover: (ref: RunRef | undefined) => void;
    onSelect: (ref: RunRef | undefined) => void;
}

// the yard is [0,100] x [-50,50]; a margin keeps starts and glyphs off the edge
const viewMinX = -8;
const viewMaxX = 108;
const viewMinY = -56;
const viewMaxY = 56;

const pathAlpha = 0.35;
const yardColor = "#e3e3e3";
const wallColor = "#333333";
const dockColor = "#d9534f";
const startGlyphColor = "#9a9a9a";
const rigColor = "#111111";

/**
 * Draws every loaded run on one canvas. Konva nodes are not used here: a
 * thousand runs of a few hundred steps is far past what a node per run can
 * repaint at scrub speed, so the paths are rendered once into an offscreen
 * canvas and only the per-step markers are redrawn each frame.
 */
export class TraceOverlay extends React.Component<TraceOverlayProps, {}> {
    private canvas: HTMLCanvasElement | null = null;
    private staticLayer: HTMLCanvasElement | null = null;
    private staticLayerKey: string = "";
    private scale: number = 1;
    private offsetX: number = 0;
    private offsetY: number = 0;

    public constructor(props: TraceOverlayProps) {
        super(props)
    }

    public componentDidMount() {
        this.draw();
    }

    public componentDidUpdate() {
        this.draw();
    }

    private mapX(x: number): number {
        return this.offsetX + (x - viewMinX) * this.scale;
    }

    private mapY(y: number): number {
        // canvas y grows downwards, the yard's does not
        return this.offsetY + (viewMaxY - y) * this.scale;
    }

    private updateScale() {
        this.scale = Math.min(this.props.width / (viewMaxX - viewMinX), this.props.height / (viewMaxY - viewMinY));
        // centre the yard, so a canvas wider than the yard does not look empty
        this.offsetX = (this.props.width - (viewMaxX - viewMinX) * this.scale) / 2;
        this.offsetY = (this.props.height - (viewMaxY - viewMinY) * this.scale) / 2;
    }

    private prepare(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
        let ratio = window.devicePixelRatio || 1;
        canvas.width = this.props.width * ratio;
        canvas.height = this.props.height * ratio;
        canvas.style.width = this.props.width + "px";
        canvas.style.height = this.props.height + "px";
        let context = canvas.getContext("2d");
        if (context) {
            context.setTransform(ratio, 0, 0, ratio, 0, 0);
        }
        return context;
    }

    private drawYard(context: CanvasRenderingContext2D) {
        context.clearRect(0, 0, this.props.width, this.props.height);
        context.strokeStyle = yardColor;
        context.lineWidth = 1;
        context.strokeRect(this.mapX(0), this.mapY(50), 100 * this.scale, 100 * this.scale);

        // the dock wall, which a run can end against
        context.strokeStyle = wallColor;
        context.lineWidth = 1.5;
        context.beginPath();
        context.moveTo(this.mapX(0), this.mapY(viewMaxY));
        context.lineTo(this.mapX(0), this.mapY(viewMinY));
        context.stroke();

        context.fillStyle = dockColor;
        context.beginPath();
        context.arc(this.mapX(0), this.mapY(0), 3, 0, 2 * Math.PI);
        context.fill();
    }

    private drawStartGlyph(context: CanvasRenderingContext2D, display: DisplayRun) {
        let start = display.run.start;
        let length = 4;
        context.beginPath();
        context.moveTo(this.mapX(start.x), this.mapY(start.y));
        context.lineTo(this.mapX(start.x + Math.cos(start.ts) * length), this.mapY(start.y + Math.sin(start.ts) * length));
        context.stroke();
    }

    private drawPath(context: CanvasRenderingContext2D, display: DisplayRun) {
        let rows = display.run.rows;
        if (rows.length == 0) {
            return;
        }
        context.beginPath();
        context.moveTo(this.mapX(display.run.start.x), this.mapY(display.run.start.y));
        for (let i = 0; i < rows.length; i++) {
            context.lineTo(this.mapX(rows[i].x), this.mapY(rows[i].y));
        }
        context.stroke();
    }

    private drawStaticLayer() {
        if (!this.staticLayer) {
            this.staticLayer = document.createElement("canvas");
        }
        let context = this.prepare(this.staticLayer);
        if (!context) {
            return;
        }
        this.drawYard(context);

        context.strokeStyle = startGlyphColor;
        context.lineWidth = 1;
        for (let i = 0; i < this.props.runs.length; i++) {
            this.drawStartGlyph(context, this.props.runs[i]);
        }

        context.globalAlpha = pathAlpha;
        context.lineWidth = 1;
        for (let i = 0; i < this.props.runs.length; i++) {
            context.strokeStyle = this.props.runs[i].color;
            this.drawPath(context, this.props.runs[i]);
        }

        // where each run came closest to the dock, which is what the paper scores
        context.globalAlpha = 0.7;
        for (let i = 0; i < this.props.runs.length; i++) {
            let display = this.props.runs[i];
            let best = display.run.bestRow;
            if (!best) {
                continue;
            }
            context.fillStyle = display.color;
            context.beginPath();
            context.arc(this.mapX(best.x), this.mapY(best.y), 1.8, 0, 2 * Math.PI);
            context.fill();
        }
        context.globalAlpha = 1;
        this.staticLayerKey = this.props.staticKey;
    }

    private drawRig(context: CanvasRenderingContext2D, row: TraceRow, display: DisplayRun) {
        let outlines = rigOutlines(row.x, row.y, row.tc, row.ts, display.plant);
        context.strokeStyle = rigColor;
        context.lineWidth = 1.5;
        for (let i = 0; i < outlines.length; i++) {
            let corners = outlines[i];
            context.beginPath();
            context.moveTo(this.mapX(corners[0].x), this.mapY(corners[0].y));
            for (let j = 1; j < corners.length; j++) {
                context.lineTo(this.mapX(corners[j].x), this.mapY(corners[j].y));
            }
            context.closePath();
            context.stroke();
        }
    }

    private drawMarker(context: CanvasRenderingContext2D, display: DisplayRun, row: TraceRow, finished: boolean) {
        let x = this.mapX(row.x);
        let y = this.mapY(row.y);
        context.beginPath();
        context.arc(x, y, finished ? 1.6 : 2.6, 0, 2 * Math.PI);
        if (finished) {
            // a run which already terminated is left as an outline
            context.strokeStyle = display.color;
            context.lineWidth = 1;
            context.stroke();
        } else {
            context.fillStyle = display.color;
            context.fill();
            // a short tick showing where the trailer points
            context.strokeStyle = display.color;
            context.lineWidth = 1;
            context.beginPath();
            context.moveTo(x, y);
            context.lineTo(this.mapX(row.x + Math.cos(row.ts) * 4), this.mapY(row.y + Math.sin(row.ts) * 4));
            context.stroke();
        }
    }

    private draw() {
        if (!this.canvas) {
            return;
        }
        this.updateScale();
        if (this.staticLayerKey !== this.props.staticKey || !this.staticLayer) {
            this.drawStaticLayer();
        }
        let context = this.prepare(this.canvas);
        if (!context) {
            return;
        }
        context.clearRect(0, 0, this.props.width, this.props.height);
        if (this.staticLayer) {
            context.save();
            context.setTransform(1, 0, 0, 1, 0, 0);
            context.drawImage(this.staticLayer, 0, 0);
            context.restore();
        }

        let highlighted: DisplayRun[] = [];
        for (let i = 0; i < this.props.runs.length; i++) {
            let display = this.props.runs[i];
            let row = display.run.rowAt(this.props.time);
            if (!row) {
                continue;
            }
            let isHighlighted = sameRun(display.ref, this.props.selected) || sameRun(display.ref, this.props.hovered);
            if (isHighlighted) {
                highlighted.push(display);
            }
            this.drawMarker(context, display, row, this.props.time > display.run.steps);
        }

        // the selected & hovered runs are drawn as full rigs, on top
        for (let i = 0; i < highlighted.length; i++) {
            let display = highlighted[i];
            context.strokeStyle = display.color;
            context.lineWidth = 2;
            context.globalAlpha = 0.9;
            this.drawPath(context, display);
            context.globalAlpha = 1;
            let row = display.run.rowAt(this.props.time);
            if (row) {
                this.drawRig(context, row, display);
            }
        }
    }

    /**
     * The run whose current position is closest to the given canvas position,
     * within a few pixels.
     */
    private pick(offsetX: number, offsetY: number): RunRef | undefined {
        let closest: RunRef | undefined = undefined;
        let closestDistance = 10;
        for (let i = 0; i < this.props.runs.length; i++) {
            let display = this.props.runs[i];
            let row = display.run.rowAt(this.props.time);
            if (!row) {
                continue;
            }
            let distance = Math.sqrt(Math.pow(this.mapX(row.x) - offsetX, 2) + Math.pow(this.mapY(row.y) - offsetY, 2));
            if (distance < closestDistance) {
                closestDistance = distance;
                closest = display.ref;
            }
        }
        return closest;
    }

    private handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
        if (!this.canvas) {
            return;
        }
        let rect = this.canvas.getBoundingClientRect();
        let picked = this.pick(e.clientX - rect.left, e.clientY - rect.top);
        if (!sameRun(picked, this.props.hovered) || (picked == undefined) !== (this.props.hovered == undefined)) {
            this.props.onHover(picked);
        }
    }

    private handleMouseLeave() {
        if (this.props.hovered) {
            this.props.onHover(undefined);
        }
    }

    private handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
        if (!this.canvas) {
            return;
        }
        let rect = this.canvas.getBoundingClientRect();
        let picked = this.pick(e.clientX - rect.left, e.clientY - rect.top);
        this.props.onSelect(sameRun(picked, this.props.selected) ? undefined : picked);
    }

    public render() {
        return <canvas className="traceOverlay"
            ref={(canvas) => { this.canvas = canvas; }}
            onMouseMove={this.handleMouseMove.bind(this)}
            onMouseLeave={this.handleMouseLeave.bind(this)}
            onClick={this.handleClick.bind(this)} />
    }
}
