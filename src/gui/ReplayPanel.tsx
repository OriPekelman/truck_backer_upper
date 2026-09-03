declare var __REPO_COMMIT__: string;

import * as React from 'react'
import { toDeg } from '../math'
import { PlantConventions, DockReference, AngleWrapping } from '../model/conventions'
import {
    NetOptions, HiddenActivationName, OutputMapName, RolloutOptions, Start, StartSchemeName,
    buildBundle, buildStarts, describeNet, describeStart, defaultRolloutOptions, inferNetShape,
    netShape, replayStarts
} from '../model/replay'

interface LoadedController {
    fileName: string;
    weights: any;
    net: NetOptions;
    // the sidecar written next to the weights, when there is one
    sidecar: any | undefined;
    arm: string;
    trainSeed: number | undefined;
    objective: string;
}

interface ReplayPanelProps {
    onReplayed: (bundles: any[]) => void;
}

interface ReplayPanelState {
    controllers: LoadedController[];
    errors: string[];
    scheme: StartSchemeName;
    startSeed: number;
    startCount: number;
    stepCap: number;
    stepLength: number;
    dockReference: DockReference;
    wrapping: AngleWrapping;
    replaying: boolean;
}

const engineName = "truck_backer_upper";
const sidecarFormat = "tbu-controller/1";

/**
 * Loads externally trained controllers and replays them here, over one start
 * set shared by all of them: the comparison of interest is several arms from
 * identical starts, and reading their runs one after another does not show
 * what an overlay shows. Nothing is trained in the browser.
 */
export class ReplayPanel extends React.Component<ReplayPanelProps, ReplayPanelState> {

    public constructor(props: ReplayPanelProps) {
        super(props)
        let conventions = PlantConventions.demo();
        this.state = {
            controllers: [],
            errors: [],
            scheme: "ensemble",
            startSeed: 7,
            startCount: 100,
            stepCap: defaultRolloutOptions.stepCap,
            stepLength: conventions.stepLength,
            dockReference: conventions.dockReference,
            wrapping: conventions.wrapping,
            replaying: false
        };
    }

    private get conventions(): PlantConventions {
        return new PlantConventions(this.state.dockReference, this.state.wrapping, this.state.stepLength);
    }

    private static readText(file: File): Promise<string> {
        return new Promise<string>((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(new Error("could not read " + file.name));
            reader.readAsText(file);
        });
    }

    /** A FileList is live, so it has to be copied before any await. */
    private static toArray(files: FileList | null): File[] {
        let copy: File[] = [];
        for (let i = 0; files && i < files.length; i++) {
            copy.push(files[i]);
        }
        return copy;
    }

    private async loadFiles(files: File[]) {
        let errors: string[] = [];
        let weightFiles: { name: string, content: any }[] = [];
        let sidecars: { name: string, content: any }[] = [];

        for (let i = 0; i < files.length; i++) {
            try {
                let content = JSON.parse(await ReplayPanel.readText(files[i]));
                if (content instanceof Array) {
                    weightFiles.push({ name: files[i].name, content: content });
                } else if (content && content.format === sidecarFormat) {
                    sidecars.push({ name: files[i].name, content: content });
                } else {
                    errors.push(files[i].name + ": neither a weight file nor a " + sidecarFormat + " sidecar");
                }
            } catch (e) {
                errors.push(files[i].name + ": " + (e instanceof Error ? e.message : e));
            }
        }

        let loaded: LoadedController[] = [];
        for (let i = 0; i < weightFiles.length; i++) {
            let weights = weightFiles[i];
            try {
                loaded.push(this.buildController(weights, ReplayPanel.matchSidecar(weights, sidecars, weightFiles.length)));
            } catch (e) {
                errors.push(weights.name + ": " + (e instanceof Error ? e.message : e));
            }
        }
        this.setState((state) => ({ controllers: state.controllers.concat(loaded), errors: errors }));
    }

    /**
     * The sidecar belonging to a weight file: written as <weights>.sidecar.json
     * and naming the weight file it describes. With a single pair loaded, they
     * belong together whatever they are called.
     */
    private static matchSidecar(weights: { name: string, content: any }, sidecars: { name: string, content: any }[], weightFileCount: number): any | undefined {
        for (let i = 0; i < sidecars.length; i++) {
            if (sidecars[i].content.weights === weights.name || sidecars[i].name.indexOf(weights.name) == 0) {
                return sidecars[i].content;
            }
        }
        return weightFileCount == 1 && sidecars.length == 1 ? sidecars[0].content : undefined;
    }

    private buildController(weights: { name: string, content: any }, sidecar: any | undefined): LoadedController {
        // the architecture comes from the weight file, never from a constant
        let shape = inferNetShape(weights.content);
        let net: NetOptions = {
            inputs: shape.inputs,
            hidden: shape.hidden,
            activation: sidecar && sidecar.net && sidecar.net.activation == "logistic" ? "logistic" : "tanh",
            outputMap: sidecar && sidecar.net && sidecar.net.output_map == "2s-1" ? "2s-1" : "tanh"
        };
        if (sidecar && sidecar.net && sidecar.net.shape instanceof Array) {
            let declared = sidecar.net.shape.join("-");
            let actual = netShape(net).join("-");
            if (declared !== actual) {
                throw new Error("the sidecar declares a " + declared + " net but the weights are " + actual);
            }
        }
        return {
            fileName: weights.name,
            weights: weights.content,
            net: net,
            sidecar: sidecar,
            arm: sidecar && sidecar.arm ? sidecar.arm : weights.name,
            trainSeed: sidecar ? sidecar.seed : undefined,
            objective: sidecar && sidecar.objective ? sidecar.objective.name : "unknown"
        };
    }

    private handleFilesChosen(e: React.ChangeEvent<HTMLInputElement>) {
        let files = ReplayPanel.toArray(e.currentTarget.files);
        e.currentTarget.value = "";
        if (files.length > 0) {
            this.loadFiles(files);
        }
    }

    private handleDrop(e: React.DragEvent<HTMLDivElement>) {
        e.preventDefault();
        let files = e.dataTransfer ? ReplayPanel.toArray(e.dataTransfer.files) : [];
        if (files.length > 0) {
            this.loadFiles(files);
        }
    }

    private updateController(index: number, change: (controller: LoadedController) => void) {
        this.setState((state) => {
            let controllers = state.controllers.slice();
            let copy: LoadedController = {
                fileName: controllers[index].fileName,
                weights: controllers[index].weights,
                net: {
                    inputs: controllers[index].net.inputs,
                    hidden: controllers[index].net.hidden,
                    activation: controllers[index].net.activation,
                    outputMap: controllers[index].net.outputMap
                },
                sidecar: controllers[index].sidecar,
                arm: controllers[index].arm,
                trainSeed: controllers[index].trainSeed,
                objective: controllers[index].objective
            };
            change(copy);
            controllers[index] = copy;
            return { controllers: controllers };
        });
    }

    private handleRemove(index: number) {
        this.setState((state) => ({ controllers: state.controllers.filter((c, i) => i != index) }));
    }

    /**
     * The conventions a controller was trained under, where its sidecar says,
     * so that replaying it under different ones is visible rather than silent.
     */
    private conventionMismatch(controller: LoadedController): string[] {
        if (!controller.sidecar || !controller.sidecar.plant) {
            return [];
        }
        let plant = controller.sidecar.plant;
        let conventions = this.conventions;
        let differences: string[] = [];
        if (plant.r != undefined && plant.r != conventions.stepLength) {
            differences.push("trained at r=" + plant.r);
        }
        let dockRef = conventions.dockReference == "trailerEnd" ? "trailer" : "truck-end";
        if (plant.dock_ref != undefined && plant.dock_ref != dockRef) {
            differences.push("trained with dock=" + plant.dock_ref);
        }
        if (plant.wrap != undefined && plant.wrap != conventions.wrapping) {
            differences.push("trained with wrap=" + plant.wrap);
        }
        return differences;
    }

    private handleReplay() {
        if (this.state.controllers.length == 0) {
            return;
        }
        // let the button repaint before the rollouts block the thread
        this.setState({ replaying: true, errors: [] }, () => window.setTimeout(() => this.replay(), 0));
    }

    private replay() {
        let conventions = this.conventions;
        let options: RolloutOptions = {
            stepCap: this.state.stepCap,
            yardXMax: defaultRolloutOptions.yardXMax,
            yardYAbs: defaultRolloutOptions.yardYAbs
        };
        // one start list, shared by every arm: that is the whole point
        let starts = buildStarts(this.state.scheme, this.state.startCount, this.state.startSeed);
        let bundles: any[] = [];
        let errors: string[] = [];
        for (let i = 0; i < this.state.controllers.length; i++) {
            let controller = this.state.controllers[i];
            try {
                let replayed = replayStarts(controller.net, conventions, controller.weights, starts, options);
                bundles.push(buildBundle({
                    arm: controller.arm,
                    engine: engineName + " (browser)",
                    engineGit: typeof __REPO_COMMIT__ === "string" ? __REPO_COMMIT__ : "unknown",
                    weights: controller.fileName,
                    trainSeed: controller.trainSeed,
                    objective: controller.objective,
                    sidecar: controller.sidecar
                }, replayed.truck, controller.net, options, starts, replayed.results, 1));
            } catch (e) {
                errors.push(controller.arm + ": " + (e instanceof Error ? e.message : e));
            }
        }
        this.setState({ replaying: false, errors: errors }, () => this.props.onReplayed(bundles));
    }

    /** A small toggle group; a two-option select is a click too many anyway. */
    private renderChoice<T extends string>(options: T[], selected: T, onChange: (value: T) => void) {
        return <span className="btn-group btn-group-sm" role="group">
            {options.map((option) => <button key={option} type="button"
                className={"btn " + (selected == option ? "btn-secondary" : "btn-outline-secondary")}
                onClick={() => onChange(option)}>{option}</button>)}
        </span>
    }

    private renderController(controller: LoadedController, index: number) {
        let mismatch = this.conventionMismatch(controller);
        return <tr key={index}>
            <td>
                <input type="text" className="form-control form-control-sm replayArmInput" value={controller.arm}
                    onChange={(e) => { let value = e.currentTarget.value; this.updateController(index, (c) => c.arm = value); }} />
            </td>
            <td><code>{controller.fileName}</code></td>
            <td>{netShape(controller.net).join("-")}</td>
            <td>{this.renderChoice(["tanh", "logistic"] as HiddenActivationName[], controller.net.activation,
                (value) => this.updateController(index, (c) => c.net.activation = value))}</td>
            <td>{this.renderChoice(["tanh", "2s-1"] as OutputMapName[], controller.net.outputMap,
                (value) => this.updateController(index, (c) => c.net.outputMap = value))}</td>
            <td>{controller.net.inputs}</td>
            <td>{controller.sidecar
                ? <span title={"objective " + controller.objective}>
                    {controller.sidecar.engine || "?"}{controller.sidecar.engine_git ? " @ " + String(controller.sidecar.engine_git).substring(0, 8) : ""}
                    {controller.trainSeed != undefined ? ", seed " + controller.trainSeed : ""}
                </span>
                : <span className="text-muted" title="without a sidecar the activation and output mapping are assumed to be the demo's; set them here if they are not">no sidecar</span>}</td>
            <td>{mismatch.length > 0 ? <span className="text-danger" title="replaying under conventions it was not trained under">{mismatch.join(", ")}</span> : ""}</td>
            <td><button type="button" className="btn btn-sm btn-outline-danger" onClick={() => this.handleRemove(index)}>Remove</button></td>
        </tr>
    }

    private renderControllers() {
        if (this.state.controllers.length == 0) {
            return null;
        }
        return <table className="table table-sm bundleTable">
            <thead>
                <tr>
                    <th>Arm</th><th>Weights</th><th>Shape</th><th>Hidden</th><th>Output</th>
                    <th title="input set: 4 state variables, 3 without the cabin angle, or 8 with the four duplicated at 10x">Inputs</th>
                    <th>Trained by</th><th>Conventions</th><th></th>
                </tr>
            </thead>
            <tbody>{this.state.controllers.map((controller, index) => this.renderController(controller, index))}</tbody>
        </table>
    }

    private renderSettings() {
        return <div className="bundleControls">
            <span className="bundleControlGroup">
                <strong>Starts</strong>
                {(["ensemble", "point", "yard"] as StartSchemeName[]).map((scheme) => <label key={scheme} className="trace-legend-toggle">
                    <input type="radio" checked={this.state.scheme == scheme} onChange={() => this.setState({ scheme: scheme })} />
                    <span>{scheme == "ensemble" ? "the paper's 15" : scheme == "point" ? "(20, 10, -2)" : "yard"}</span>
                </label>)}
                {this.state.scheme == "yard"
                    ? <span>
                        <input type="number" className="form-control form-control-sm replayNumber" value={this.state.startCount}
                            onChange={(e) => this.setState({ startCount: Math.max(1, Number.parseInt(e.currentTarget.value) || 1) })} />
                        <span className="bundleStepLabel"> draws, seed </span>
                        <input type="number" className="form-control form-control-sm replayNumber" value={this.state.startSeed}
                            onChange={(e) => this.setState({ startSeed: Number.parseInt(e.currentTarget.value) || 0 })} />
                    </span>
                    : null}
            </span>
            <span className="bundleControlGroup">
                <strong>Plant</strong>
                <span className="bundleStepLabel">r </span>
                <input type="number" step="0.5" className="form-control form-control-sm replayNumber" value={this.state.stepLength}
                    onChange={(e) => this.setState({ stepLength: Number.parseFloat(e.currentTarget.value) || 1 })} />
                <span className="bundleStepLabel"> cap </span>
                <input type="number" className="form-control form-control-sm replayNumber" value={this.state.stepCap}
                    onChange={(e) => this.setState({ stepCap: Math.max(1, Number.parseInt(e.currentTarget.value) || 1) })} />
                <label className="trace-legend-toggle">
                    <input type="checkbox" checked={this.state.dockReference == "trailerEnd"}
                        onChange={(e) => this.setState({ dockReference: e.currentTarget.checked ? "trailerEnd" : "truckEnd" })} />
                    <span>dock at the trailer rear</span>
                </label>
                <label className="trace-legend-toggle">
                    <input type="checkbox" checked={this.state.wrapping == "pi"}
                        onChange={(e) => this.setState({ wrapping: e.currentTarget.checked ? "pi" : "none" })} />
                    <span>wrap angles</span>
                </label>
            </span>
            <span className="bundleControlGroup">
                <button type="button" className="btn btn-sm btn-primary"
                    disabled={this.state.controllers.length == 0 || this.state.replaying}
                    onClick={this.handleReplay.bind(this)}>
                    {this.state.replaying ? "Replaying…" : "Replay over the same starts"}
                </button>
            </span>
        </div>
    }

    public render() {
        return <div className="replayPanel">
            <h4>Replay weights here</h4>
            <p className="bundleHint">
                Load controllers trained elsewhere and drive them from one shared start set.
                The net's shape is read from the weight file, so nothing is fixed to the demo's 4-45-1;
                a <code>{sidecarFormat}</code> sidecar alongside it supplies the activation, output
                mapping, arm and seed, and without one the demo's are assumed.
            </p>
            <div className="bundleDropZone" onDrop={this.handleDrop.bind(this)} onDragOver={(e) => e.preventDefault()}>
                <p className="mb">Drop weight files (and their sidecars) here, or</p>
                <input type="file" multiple onChange={this.handleFilesChosen.bind(this)} />
            </div>
            {this.state.errors.length > 0
                ? <div className="alert alert-danger">{this.state.errors.map((error, i) => <div key={i}>{error}</div>)}</div>
                : null}
            {this.renderControllers()}
            {this.state.controllers.length > 0 ? this.renderSettings() : null}
        </div>
    }
}
