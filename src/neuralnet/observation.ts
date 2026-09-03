import { Vector } from './math'

/**
 * Maps the plant's normalized state onto a controller's inputs, and a
 * controller's input gradient back onto the plant's state, so that nets with
 * different input sets can be trained by the same loop.
 *
 * The plant's state is [x, y, cabin angle, trailer angle], all normalized.
 * The paper uses three input sets: those four, the same without the cabin
 * angle, and the four plus the same four rescaled by ten, which it reports cut
 * its GA from about 1500 generations to about 800.
 */
export abstract class Observation {
    public abstract getName(): string;
    /** Number of controller inputs for a plant state of the given size. */
    public abstract getInputDim(stateDim: number): number;
    public abstract observe(state: Vector): Vector;
    /** The state-space gradient of an observation-space gradient. */
    public abstract mapGradient(gradient: Vector, stateDim: number): Vector;
}

/** All four state variables, as the demo's 4-45-1 controller takes them. */
export class FullObservation extends Observation {
    public getName(): string {
        return "full";
    }

    public getInputDim(stateDim: number): number {
        return stateDim;
    }

    public observe(state: Vector): Vector {
        return state;
    }

    public mapGradient(gradient: Vector, stateDim: number): Vector {
        return gradient;
    }
}

/** The four without the cabin angle, the paper's 3-7-1 input set. */
export class WithoutCabinAngleObservation extends Observation {
    // the plant's state is [x, y, cabin angle, trailer angle]
    private static cabinAngleIndex = 2;

    public getName(): string {
        return "withoutCabinAngle";
    }

    public getInputDim(stateDim: number): number {
        return stateDim - 1;
    }

    public observe(state: Vector): Vector {
        let entries: number[] = [];
        for (let i = 0; i < state.entries.length; i++) {
            if (i != WithoutCabinAngleObservation.cabinAngleIndex) {
                entries.push(state.entries[i]);
            }
        }
        return new Vector(entries);
    }

    public mapGradient(gradient: Vector, stateDim: number): Vector {
        let entries: number[] = [];
        let read = 0;
        for (let i = 0; i < stateDim; i++) {
            // the dropped input cannot carry any gradient
            entries.push(i == WithoutCabinAngleObservation.cabinAngleIndex ? 0 : gradient.entries[read++]);
        }
        return new Vector(entries);
    }
}

/** The four plus the same four scaled, the paper's 8-17-1 input set. */
export class ScaledDuplicateObservation extends Observation {
    public constructor(private factor: number = 10) {
        super();
    }

    public getName(): string {
        return "scaledDuplicate(" + this.factor + ")";
    }

    public getInputDim(stateDim: number): number {
        return 2 * stateDim;
    }

    public observe(state: Vector): Vector {
        let entries = state.entries.slice();
        for (let i = 0; i < state.entries.length; i++) {
            entries.push(state.entries[i] * this.factor);
        }
        return new Vector(entries);
    }

    public mapGradient(gradient: Vector, stateDim: number): Vector {
        let entries: number[] = [];
        for (let i = 0; i < stateDim; i++) {
            // both copies of a state variable feed back into it
            entries.push(gradient.entries[i] + this.factor * gradient.entries[i + stateDim]);
        }
        return new Vector(entries);
    }
}

export function observationForInputCount(inputs: number, stateDim: number = 4): Observation {
    if (inputs == stateDim) {
        return new FullObservation();
    }
    if (inputs == stateDim - 1) {
        return new WithoutCabinAngleObservation();
    }
    if (inputs == 2 * stateDim) {
        return new ScaledDuplicateObservation(10);
    }
    throw new Error("No observation for " + inputs + " inputs; expected "
        + stateDim + ", " + (stateDim - 1) + " or " + (2 * stateDim));
}
