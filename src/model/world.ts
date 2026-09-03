import { Point, Vector, isLeftOf, plus, StraightLine, Angle, rotate } from '../math'
import { Truck } from './truck'
import * as nnMath from '../neuralnet/math';
import { TruckLesson } from '../neuralnet/lesson'
export class Dock {
    public dockDirection: Vector;

    constructor(public position: Point) {
        this.dockDirection = new Vector(0, 1);
    }

}

export interface Normalized {
    getNormalizedDock(dock: Dock): Point;
}
export interface HasLength {
    getLength(): number;
}

export interface HasState {
    getOriginalState(): nnMath.Vector; // unnormalized
    getStateVector(): nnMath.Vector // normalized
    getStateDescription(): string[];
    nextState(input: number, time: number): boolean;
    getMaxSteeringAngle(): number;
    randomizePosition(lesson: TruckLesson): void;
    randomizePosition(): void;
}

/**
 * Objects which can leave a visible trace of the path they took.
 * getTracePoints returns one point per traced reference point (e.g. end of
 * trailer & front of cabin); the returned points must not be mutated later on.
 */
export interface Traceable {
    getTracePoints(): Point[];

    /**
     * The object's outline as a list of closed polygons (e.g. cabin & trailer),
     * used to draw the box trace. Copies, as for getTracePoints.
     */
    getTraceOutlines(): Point[][];
}

/**
 * One continuous trajectory. A new trace is started whenever the object is
 * teleported, so that jumps are not drawn as if they had been driven.
 */
export class Trace {
    // one polyline per traced reference point of getTracePoints()
    public paths: Point[][] = [];
    // outlines of the object sampled along the way; one entry per sample,
    // each holding the polygons of getTraceOutlines()
    public outlines: Point[][][] = [];
    // where the last outline was sampled, to space the samples out
    public lastOutlinePosition: Point | undefined = undefined;
}

export function isTraceable(obj: any): obj is Traceable {
    return obj != undefined
        && typeof obj.getTracePoints === "function"
        && typeof obj.getTraceOutlines === "function";
}

export interface Limitable {
    setLimits(limits: Array<StraightLine>): void;
    setLimited(limited: boolean): void;
}

export enum AngleType {
    CAB,
    TRAILER,
    BOTH
}

export class World {
    private limits = [
        new StraightLine(new Point(0, 0), new Vector(0, 1)), // left
        new StraightLine(new Point(0, 100), new Vector(1, 0)), // top
        new StraightLine(new Point(200, 100), new Vector(0, -1)), // left
        new StraightLine(new Point(200, -100), new Vector(-1, 0)), // left
    ];

    // minimum distance (in m) between two recorded points of a trace
    private static minTraceDistance = 0.25;
    // minimum distance (in m) between two sampled outlines of the object
    private static minOutlineDistance = 6;
    // every trajectory ever driven, oldest first
    private traces: Trace[] = [];
    // the trace currently being extended
    private currentTrace: Trace | undefined = undefined;

    constructor(public movableObject: HasState & Limitable, public dock: Dock) {
        this.recordTrace();
    }

    /**
     * All trajectories driven so far, oldest first. Traces are only ever
     * removed by clearTrace().
     */
    public getTraces(): Trace[] {
        return this.traces;
    }

    public clearTrace(): void {
        this.traces = [];
        this.currentTrace = undefined;
        this.recordTrace();
    }

    /**
     * Begins a new trajectory instead of extending the current one. Call this
     * whenever the object is moved without driving there.
     */
    public startNewTrace(): void {
        this.currentTrace = undefined;
        this.recordTrace();
    }

    public recordTrace(): void {
        let traceable = this.movableObject;
        if (!isTraceable(traceable)) {
            return;
        }
        let points = traceable.getTracePoints();
        if (points.length == 0) {
            return;
        }
        if (!this.currentTrace) {
            this.currentTrace = new Trace();
            this.traces.push(this.currentTrace);
        }
        let trace = this.currentTrace;
        for (let i = 0; i < points.length; i++) {
            if (trace.paths.length <= i) {
                trace.paths.push([]);
            }
            let path = trace.paths[i];
            let last = path[path.length - 1];
            if (last && last.getVectorTo(points[i]).getLength() < World.minTraceDistance) {
                continue;
            }
            path.push(points[i]);
        }
        this.recordOutline(traceable, trace, points[0]);
    }

    private recordOutline(traceable: Traceable, trace: Trace, reference: Point): void {
        let last = trace.lastOutlinePosition;
        if (last && last.getVectorTo(reference).getLength() < World.minOutlineDistance) {
            return;
        }
        trace.outlines.push(traceable.getTraceOutlines());
        trace.lastOutlinePosition = reference;
    }


    public setWorldLimited(limited: boolean) {
        this.movableObject.setLimited(limited);
    }

    public getLimits(): Array<StraightLine> {
        return this.limits;
    }

    public nextTimeStep(steeringSignal: number, time: number = 1): boolean {
        let result = this.movableObject.nextState(steeringSignal, time);
        this.recordTrace();
        return result;
    }
}