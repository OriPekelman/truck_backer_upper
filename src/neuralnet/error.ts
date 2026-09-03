import { Vector, Scalar, plus } from './math';
import { Point } from '../math'
import { Dock } from '../model/world';
import { getCiphers } from 'crypto';

export abstract class ErrorFunction {
    abstract getName(): string;
    abstract getError(is: Vector, should: Vector): Scalar;
    abstract getErrorDerivative(is: Vector, sholud: Vector): Vector;
}

export class MSE extends ErrorFunction {

    public getName() {
        return "MSE";
    }

    public getError(is: Vector, should: Vector): Scalar {
        let diff = 0;
        for (let i = 0; i < is.length; i++) {
            let d = is.entries[i] - should.entries[i];
            diff += d * d;
        }
        return diff / is.length;
    }

    public getErrorDerivative(is: Vector, should: Vector): Vector {
        // 2 * (is - should) or -2 * (should - is)
        return plus(is, should.getScaled(-1)).scale(2 / is.length);
    }
}

export class WeightedMSE extends ErrorFunction {
    private weightSum: number;

    public getName() {
        return "WeightedMSE";
    }
    public constructor(private weights: Vector) {
        super();
        this.weightSum = weights.entries.reduce((prev, next) => prev + next, 0);
    }

    public getError(is: Vector, should: Vector): Scalar {
        let diff = 0;
        for (let i = 0; i < is.length; i++) {
            let d = is.entries[i] - should.entries[i];
            diff += this.weights.entries[i] * d * d;
        }
        return diff / this.weightSum;
    }

    public getErrorDerivative(is: Vector, should: Vector): Vector {
        return plus(is, should.getScaled(-1)).multiplyElementWise(this.weights).scale(2 / this.weightSum);
    }
}

export abstract class ControllerError extends ErrorFunction {
    abstract getError(finalState: Vector): Scalar;
    abstract getErrorDerivative(finalState: Vector): Vector;
    abstract setSaveErrors(saveErrors: boolean): void;
}

export class SimpleControllerError extends ControllerError {

    public getName() {
        return "SimpleControllerError"
    }

    public getError(finalState: Vector): Scalar {
        return (0 - finalState.entries[0]) * (0 - finalState.entries[0]);
    }

    public getErrorDerivative(finalState: Vector): Vector {
        return new Vector([- 2 * (0 - finalState.entries[0])]);
    }

    public setSaveErrors(saveErrors: boolean) {
    }
}

/**
 * A controller error which grades the closest approach of an episode rather
 * than the state it stopped in. The paper's GA was graded on the closest point
 * of the trajectory, and a controller which arrives and then drifts off scores
 * very differently under the two readings.
 */
export interface BestApproachError {
    /**
     * Marks this error as grading the closest approach. Every scorable error
     * can score a single state, so the flag rather than the method is what
     * says which reading is in force.
     */
    readonly gradeBestApproach: boolean;
    /** The score of one state along the way, lower being closer. */
    scoreState(state: Vector): Scalar;
}

export function usesBestApproachGrading(errorFunction: any): errorFunction is BestApproachError {
    return errorFunction != undefined
        && errorFunction.gradeBestApproach === true
        && typeof errorFunction.scoreState === "function";
}

/**
 * The paper's score, d2 = x^2 + y^2 + min(ts^2, (ts - 2pi)^2, (ts + 2pi)^2),
 * of the trailer against a dock at the origin, in physical units. The wrapped
 * angle term is what lets the paper leave the angles unwrapped: it tolerates
 * one full turn of the trailer.
 *
 * The state handed in is normalized -- x as (x - 50) / 50, y as y / 50, angles
 * over pi -- so the score is computed in physical units while its derivative
 * is taken with respect to those normalized inputs, which is what the emulator
 * behind it consumes. That makes the gradients numerically much larger than
 * TruckControllerError's, so the learning rate has to be set accordingly.
 */
export class D2ControllerError extends ControllerError {
    public errors: Array<number> = [];
    private saveErrors: boolean = true;

    // the normalization NormalizedTruck applies to the plant's state
    private static xScale = 50;
    private static yScale = 50;
    private static angleScale = Math.PI;

    public getName() {
        return "D2ControllerError";
    }

    public setSaveErrors(saveErrors: boolean) {
        this.saveErrors = saveErrors;
    }

    /** Physical (x, y, trailer angle) of a normalized state. */
    private toPhysical(state: Vector): number[] {
        return [
            state.entries[0] * D2ControllerError.xScale + D2ControllerError.xScale,
            state.entries[1] * D2ControllerError.yScale,
            state.entries[3] * D2ControllerError.angleScale
        ];
    }

    /** The turn of the trailer angle which the wrapped term selects. */
    private static closestTurn(angle: Scalar): Scalar {
        let candidates = [0, 2 * Math.PI, -2 * Math.PI];
        let best = candidates[0];
        for (let i = 1; i < candidates.length; i++) {
            if (Math.abs(angle - candidates[i]) < Math.abs(angle - best)) {
                best = candidates[i];
            }
        }
        return best;
    }

    public scoreState(state: Vector): Scalar {
        let physical = this.toPhysical(state);
        let x = physical[0];
        let y = physical[1];
        let angle = physical[2] - D2ControllerError.closestTurn(physical[2]);
        return x * x + y * y + angle * angle;
    }

    public getError(finalState: Vector): Scalar {
        let error = this.scoreState(finalState);
        if (this.saveErrors) {
            this.errors.push(error);
        }
        return error;
    }

    public getErrorDerivative(finalState: Vector): Vector {
        let physical = this.toPhysical(finalState);
        let x = physical[0];
        let y = physical[1];
        let angle = physical[2] - D2ControllerError.closestTurn(physical[2]);
        // chain rule back onto the normalized inputs
        return new Vector([
            2 * x * D2ControllerError.xScale,
            2 * y * D2ControllerError.yScale,
            0, // the cabin angle is not scored
            2 * angle * D2ControllerError.angleScale
        ]);
    }
}

/** D2ControllerError graded at the closest approach instead of the last state. */
export class BestApproachD2ControllerError extends D2ControllerError implements BestApproachError {
    public readonly gradeBestApproach = true;

    public getName() {
        return "BestApproachD2ControllerError";
    }
}

export class TruckControllerError extends ControllerError {
    public angleError: Array<number>;
    public yError: Array<number>;
    public errors: Array<number>;

    private saveErrors: boolean = true;

    public getName() {
        return "TruckControllerError";
    }
    public constructor(private dock: Point) {
        super();
        this.angleError = [];
        this.yError = [];
        this.errors = [];
    }

    public setSaveErrors(saveErrors: boolean) {
        this.saveErrors = saveErrors;
    }
    public getErrorDerivative(finalState: Vector): Vector {
        let xTrailer = finalState.entries[0];
        let yTrailer = finalState.entries[1];
        let thetaTrailer = finalState.entries[3];

        // Derivative of SSE
        let xDiff = Math.max(xTrailer, -1) - this.dock.x;
        let yDiff = yTrailer - this.dock.y;
        let thetaDiff = thetaTrailer - 0;

        return new Vector([0, 10 * 2 * yDiff, 0, 2 * thetaDiff]);
    }

    // 3 elements: x trailer y trailer theta trailer at position 3 4 and 5
    public getError(finalState: Vector) {
        let xTrailer = finalState.entries[0];
        let yTrailer = finalState.entries[1];
        // 2 is cabin angle
        let thetaTrailer = finalState.entries[3];
        // IMPORTANT: x = 0 is at -1 because of the x transformation!
        // we just ignore x < 0 This also explains why it tries to drive a circle with max(xTrailer, 0)
        let xDiff = Math.max(xTrailer, -1) - this.dock.x
        let yDiff = yTrailer - this.dock.y
        let thetaDiff = thetaTrailer - 0

        // We input the final state in emulator output space => angle / Math.PI and y divided by 50

        if (this.saveErrors) {
            this.angleError.push(Math.abs(thetaDiff * Math.PI))
            this.yError.push(Math.abs(yDiff * 50))
        }

        let error = xDiff * xDiff + yDiff * yDiff + thetaDiff * thetaDiff;
        if (this.saveErrors) {
            this.errors.push(error);
        }
        return error;
    }
}
