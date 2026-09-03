import { PlantConventions } from '../model/conventions';
import { Truck } from '../model/truck';
import { NetOptions, ObjectiveName, netShape, describeObjective, buildObservation } from './setup';

export const sidecarFormat = "tbu-controller/1";

/** The vocabulary the trace bundles declare, rather than this code's own. */
export function bundleDockRef(conventions: PlantConventions): string {
    return conventions.dockReference == "trailerEnd" ? "trailer" : "truck-end";
}

export interface SidecarNet {
    shape: number[];
    activation: string;
    output_map: string;
    obs: number;
}

export interface SidecarPlant {
    ls: number;
    lc: number;
    u_max_deg: number;
    r: number;
    step_cap: number;
    dock_ref: string;
    wrap: string;
}

/**
 * What a weight file needs alongside it to be worth anything a week later:
 * which arm it is, which commit produced it, from which seed, and every
 * convention it was trained under. #2's loader reads this, and rollout copies
 * it into a trace bundle's provenance.
 */
export interface Sidecar {
    format: string;
    arm: string;
    engine: string;
    engine_git: string;
    engine_dirty: boolean;
    trained_at: string;
    seed: number;
    weights: string;
    net: SidecarNet;
    plant: SidecarPlant;
    objective: { name: string; descends: string[] };
    training: { [key: string]: any };
}

export function buildSidecarNet(net: NetOptions): SidecarNet {
    return {
        shape: netShape(net),
        activation: net.activation,
        output_map: net.outputMap,
        obs: net.inputs
    };
}

export function buildSidecarPlant(truck: Truck, stepCap: number): SidecarPlant {
    return {
        ls: truck.getTrailerLength(),
        lc: truck.getTruckLength(),
        u_max_deg: truck.getMaxSteeringAngle() * 180 / Math.PI,
        r: truck.conventions.stepLength,
        step_cap: stepCap,
        dock_ref: bundleDockRef(truck.conventions),
        wrap: truck.conventions.wrapping
    };
}

export function buildSidecar(options: {
    arm: string,
    commit: string,
    dirty: boolean,
    seed: number,
    weightsFile: string,
    net: NetOptions,
    truck: Truck,
    stepCap: number,
    objective: ObjectiveName,
    training: { [key: string]: any }
}): Sidecar {
    return {
        format: sidecarFormat,
        arm: options.arm,
        engine: "truck_backer_upper",
        engine_git: options.commit,
        engine_dirty: options.dirty,
        trained_at: new Date().toISOString(),
        seed: options.seed,
        weights: options.weightsFile,
        net: buildSidecarNet(options.net),
        plant: buildSidecarPlant(options.truck, options.stepCap),
        objective: { name: options.objective, descends: describeObjective(options.objective) },
        training: options.training
    };
}

/** Net options as recorded in a sidecar, for a rollout of those weights. */
export function netOptionsFromSidecar(sidecar: Sidecar): NetOptions {
    let shape = sidecar.net.shape;
    if (!(shape instanceof Array) || shape.length < 3 || shape[shape.length - 1] != 1) {
        throw new Error("The sidecar's net.shape must be [inputs, hidden..., 1]");
    }
    let net: NetOptions = {
        inputs: shape[0],
        hidden: shape.slice(1, shape.length - 1),
        activation: sidecar.net.activation == "logistic" ? "logistic" : "tanh",
        outputMap: sidecar.net.output_map == "2s-1" ? "2s-1" : "tanh"
    };
    if (sidecar.net.obs != undefined && sidecar.net.obs != net.inputs) {
        throw new Error("The sidecar's net.obs (" + sidecar.net.obs + ") disagrees with its net.shape inputs (" + net.inputs + ")");
    }
    buildObservation(net); // fails now rather than mid-rollout
    return net;
}

export function conventionsFromSidecar(sidecar: Sidecar): PlantConventions {
    return new PlantConventions(
        sidecar.plant.dock_ref == "trailer" ? "trailerEnd" : "truckEnd",
        sidecar.plant.wrap == "pi" ? "pi" : "none",
        sidecar.plant.r
    );
}
