/**
 * Conventions where this demo and Schoenauer & Ronald, "Neuro-Genetic Truck
 * Backer-Upper Controller" (ICEC'94) disagree. Paper and demo implement the
 * same arcsin reformulation of the plant; they differ on the three settings
 * below, and in two of them the demo is internally inconsistent.
 *
 * These are switches, not fixes: the bundled weights were trained under the
 * demo's conventions, so those have to stay reachable.
 */

/**
 * Which point has to reach the dock for an episode to be over.
 *
 * "truckEnd" is what Truck.continue() has always measured — the coupling
 * device plus two units towards the cab — even though TruckControllerError
 * grades the trailer. "trailerEnd" is the paper's: the centre rear of the
 * trailer at the origin, which is also the point the error term scores.
 */
export type DockReference = "truckEnd" | "trailerEnd";

/**
 * Whether the angles are wrapped back into (-pi, pi] after every step.
 *
 * The demo wraps in the constructor and in setTruckPosition but not after
 * driving, so a long rollout can leave the interval while
 * NormalizedTruck.getStateVector divides by pi and hands the net something
 * outside [-1, 1]. The paper does not wrap either, absorbing the wrap in its
 * distance term, min(ts^2, (ts-2pi)^2, (ts+2pi)^2), which tolerates one full
 * turn. Both are defensible; not knowing which is in force is not.
 */
export type AngleWrapping = "none" | "pi";

export class PlantConventions {
    /**
     * @param dockReference the point which has to reach the dock
     * @param wrapping whether angles are wrapped after each step
     * @param stepLength r, the distance covered per step; the paper reports
     *                   convergence was "only achieved by setting a fairly
     *                   large value r = 3", the demo uses 1
     */
    public constructor(
        public dockReference: DockReference = "truckEnd",
        public wrapping: AngleWrapping = "none",
        public stepLength: number = 1
    ) {
    }

    /** What the bundled weights_0..19 were trained under. */
    public static demo(): PlantConventions {
        return new PlantConventions("truckEnd", "none", 1);
    }

    /** Schoenauer & Ronald's. */
    public static paper(): PlantConventions {
        return new PlantConventions("trailerEnd", "none", 3);
    }

    public copy(): PlantConventions {
        return new PlantConventions(this.dockReference, this.wrapping, this.stepLength);
    }

    public equals(other: PlantConventions): boolean {
        return this.dockReference == other.dockReference
            && this.wrapping == other.wrapping
            && this.stepLength == other.stepLength;
    }
}
