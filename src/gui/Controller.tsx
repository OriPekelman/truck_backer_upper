import * as React from 'react'
import { Point } from "../math";
import { World } from "../model/world";
import { Truck, NormalizedTruck } from '../model/truck';
import { NetworkCreator } from './NetworkCreator';

import { NetConfig } from '../neuralnet/net';
import { MSE, ErrorFunction } from '../neuralnet/error';
import { SGD, Optimizer, SGDNesterovMomentum } from '../neuralnet/optimizers';
import { RandomWeightInitializer, TwoLayerInitializer, WeightInitializer } from '../neuralnet/weightinitializer';
import { Tanh, Sigmoid, ActivationFunction, ReLu, Linear } from '../neuralnet/activation';
import { AdalineUnit } from '../neuralnet/unit';
import { NeuralNet } from '../neuralnet/net';
import { TrainController } from '../neuralnet/train';
import { TruckControllerError } from '../neuralnet/error';
import { NeuralNetEmulator } from '../neuralnet/emulator';
import { TruckLesson } from '../neuralnet/lesson';
import { createTruckControllerLessons } from '../neuralnet/lesson';
import { LayerConfig } from '../neuralnet/net';
import { LessonsComponent } from './LessonsComponent'
import { NetOptions, buildControllerNetConfig, inferNetShape, HiddenActivationName, OutputMapName } from '../model/replay';
const ReactHighcharts = require('react-highcharts');

interface ControllerProps {
    object: Truck;
    world: World;
    emulatorNet: NeuralNet | undefined;
    onControllerTrained: (net: TrainController | null) => void;
}

interface ControllerState {
    network: NetConfig;
    nn: NeuralNet | undefined;
    loadingWeights: boolean;
    loadWeightsSuccessful: boolean | null;
    loadWeightsFailureMsg: string | null;
    updatedController: boolean;
    train: boolean;
    isTrainedNetwork: boolean;
    errors: number[];
    lessons: TruckLesson[];
    currentLessonIndex: number;
    weightLessonIndex: number;
    trainedControllers: TrainedControllerEntry[];
    selectedTrained: number;
    loadedTrainedLabel: string | null;
    trainedHint: string | null;
    loadedLessonWeights: number;
    maxStepErrors: number;
}

/** One entry of weights/index.json: a controller trained elsewhere, shipped with the page. */
interface TrainedControllerEntry {
    weights: string;
    sidecar: string;
    label: string;
}

export class Controller extends React.Component<ControllerProps, ControllerState> {
    public static MAX_LESSON = 19;
    private emulatorController: TrainController | null = null;
    public STEPS_PER_FRAME = 1;
    public readonly STEPS_PER_ERROR = 100;
    private lastIteration: number = 0;

    private errorCache: number[] = [];
    private errorCount: number = 0;
    private errorSum: number = 0;
    private currentLessonSteps: number = 0;

    public constructor(props: ControllerProps) {
        super(props);
        this.state = {
            updatedController: false,
            network: this.getDefaultNetConfig(),
            nn: undefined,
            loadingWeights: false,
            loadWeightsSuccessful: null,
            loadWeightsFailureMsg: null,
            isTrainedNetwork: false,
            train: false,
            errors: [],
            lessons: createTruckControllerLessons(this.props.object),
            currentLessonIndex: 0,
            weightLessonIndex: Controller.MAX_LESSON,
            trainedControllers: [],
            selectedTrained: 0,
            loadedTrainedLabel: null,
            trainedHint: null,
            loadedLessonWeights: -1,
            maxStepErrors: 0
        };
        this.handleLoadPretrainedWeights();
        this.fetchTrainedControllers();
    }

    private handleResetLessons() {
        this.setState({ currentLessonIndex: 0, lessons: createTruckControllerLessons(this.props.object) })
    }

    private handleStopTrain(errorMsg: string | undefined = undefined) {
        this.errorCache = [];
        this.setState({ train: false, isTrainedNetwork: errorMsg === undefined, nn: this.state.nn, errors: this.state.errors }, () => {
            if (errorMsg === undefined) {
                let ctrl = this.makeTrainController();
                if (!ctrl) {
                    alert("Failed to create train controller after training!");
                } else {
                    this.onControllerTrained(ctrl);
                }
            } else {
                alert("Error: " + errorMsg.toString());
            }
        });
    }

    private onControllerTrained(ctrl: TrainController) {
        this.props.onControllerTrained(ctrl);
        this.setState({updatedController: true});
    }

    public handleMaxStepErrors() {
        this.setState({ maxStepErrors: this.state.maxStepErrors + 1 }, () => {
            if (this.state.maxStepErrors >= 90) {
                this.handleStopTrain("The truck diverged and did not find the dock!");
            }
        });
    }

    public handleTrain() {
        // TODO: disable train button, show hover hint!
        if (this.props.emulatorNet === undefined) {
            alert("You need to load emulator weights or train an emulator net before using the controller!");
            return;
        }

        if (this.state.lessons.length == 0) {
            alert("You need to create at least one lesson!");
            return;
        }

        let nn = this.state.nn;
        if (!this.state.nn) {
            nn = new NeuralNet(this.state.network);
        }

        this.setState({ updatedController: false, nn: nn, train: true, maxStepErrors: 0 }, () => {
            // we updated the gui
            // start animation
            let ctrl = this.makeTrainController();
            if (ctrl) {
                this.emulatorController = ctrl;
                this.emulatorController.setLesson(this.state.lessons[this.state.currentLessonIndex]);
                this.lastIteration = performance.now();
                this.errorCache = [];
                this.errorSum = 0;
                this.errorCount = 0;
                requestAnimationFrame(this.trainNeuralNetAniFrame);
            } else {
                alert("Failed to start training because emulator network was not loaded or controller net is missing!");
            }
        });
    }

    private makeTrainController(): TrainController | null {
        if (!this.state.nn) {
            return null;
        }

        let normalizedObject = new NormalizedTruck(this.props.object);
        let dock = normalizedObject.getNormalizedDock(this.props.world.dock);
        let error = new TruckControllerError(dock)
        error.setSaveErrors(false);
        let emulator = null;
        if (this.props.emulatorNet) {
            emulator = new NeuralNetEmulator(this.props.emulatorNet);
        }
        let ctrl = new TrainController(this.props.world, normalizedObject, this.state.nn, emulator, error);
        ctrl.addMaxStepListener(this.handleMaxStepErrors.bind(this));
        return ctrl;
    }

    private trainNeuralNetAniFrame = this.trainNeuralNetCallback.bind(this);
    private trainNeuralNetCallback() {
        if (!this.emulatorController) {
            throw new Error("You must initialize the controller before starting training!");
        }
        for (let i = 0; i < this.STEPS_PER_FRAME; i++) {
            this.props.object.randomizeNoLimits();
            let error = 0;
            try {
                error = this.emulatorController.trainSingleStep();
            } catch (e) {
                this.setState({ train: false }, () => {
                    alert("Error during Training: " + e);
                })
                return;
            }
            if (error) {
                this.errorCount++;
                this.errorSum += error;
            }

            if (this.errorCount > 0 && this.errorCount % this.STEPS_PER_ERROR === 0) {
                this.errorCache.push(this.errorSum / this.errorCount);
                (this.refs.chart as any).getChart().series[0].addPoint(this.errorSum / this.errorCount, true);

                this.errorCount = 0;
                this.errorSum = 0;
            }
        }
        // TODO: chart disable decimals
        if (this.currentLessonSteps + this.STEPS_PER_FRAME >= this.state.lessons[this.state.currentLessonIndex].samples) {
            // end training
            if (this.state.currentLessonIndex + 1 >= this.state.lessons.length) {
                this.handleStopTrain();
                return;
            } else {
                this.emulatorController.setLesson(this.state.lessons[this.state.currentLessonIndex + 1]);
                this.currentLessonSteps = 0;
                this.setState({ currentLessonIndex: this.state.currentLessonIndex + 1, maxStepErrors: 0 }, () => {
                    this.trainNextStep();
                });
            }
        } else {
            this.currentLessonSteps += this.STEPS_PER_FRAME;
            if (this.currentLessonSteps % 100 === 0) {
                this.setState({ maxStepErrors: 0 }, () => {
                    this.trainNextStep()
                });
            } else {
                this.trainNextStep()
            }
        }
    }

    private trainNextStep(): void {
        // dynamically adjust steps per frame;
        let duration = performance.now() - this.lastIteration;
        this.lastIteration = performance.now();
        if (duration > 1.05 * 1000 / 60) {
            this.STEPS_PER_FRAME = Math.min(1, this.STEPS_PER_FRAME);
        } else if (duration < 0.95 * 1000 / 60) {
            this.STEPS_PER_FRAME += 1;
        }

        if (this.state.train) {
            requestAnimationFrame(this.trainNeuralNetAniFrame);
        }
    }

    /**
     * Controllers trained outside this page (toy's dfa-vs-bp arc), listed in
     * weights/index.json with a tbu-controller/1 sidecar each. The sidecar
     * fixes the net's activation and output mapping, so a logistic 4-9-1 with
     * a 2s-1 output is driven as it was trained, not as the demo's tanh net.
     */
    private fetchTrainedControllers() {
        fetch("weights/index.json")
            .then((r) => r.ok ? r.json() : Promise.reject(new Error("" + r.status)))
            .then((index) => this.setState({ trainedControllers: (index && index.controllers) || [] }))
            .catch(() => { /* none shipped: the file loader still works */ });
    }

    private installTrainedController(weights: any, sidecar: any, label: string) {
        let shape = inferNetShape(weights);
        let net: NetOptions = {
            inputs: shape.inputs,
            hidden: shape.hidden,
            activation: (sidecar && sidecar.net && sidecar.net.activation == "logistic" ? "logistic" : "tanh") as HiddenActivationName,
            outputMap: (sidecar && sidecar.net && sidecar.net.output_map == "2s-1" ? "2s-1" : "tanh") as OutputMapName
        };
        if (net.inputs != 4) {
            throw new Error("this controller takes " + net.inputs + " inputs; the simulator observes 4 (x, y, cab angle, trailer angle)");
        }
        let config = buildControllerNetConfig(net, () => new SGD(0.8));
        let nn = new NeuralNet(config);
        nn.loadWeights(weights);
        let plant = sidecar && sidecar.plant ? sidecar.plant : null;
        let hint = plant
            ? "trained at r = " + plant.r + ", " + plant.step_cap + "-step episodes, dock reference \"" + plant.dock_ref + "\" — set the Plant Conventions panel to match"
            : "no sidecar: assumed " + net.activation + " hidden units and a " + net.outputMap + " output";
        this.setState({
            loadingWeights: false,
            updatedController: false,
            nn: nn,
            network: config,
            loadWeightsSuccessful: true,
            loadedTrainedLabel: label + " (" + [net.inputs].concat(net.hidden).concat([1]).join("-") + " " + net.activation + ", " + net.outputMap + ")",
            trainedHint: hint,
            errors: [],
            isTrainedNetwork: false
        }, () => {
            let ctrl = this.makeTrainController();
            if (!ctrl) {
                alert("Failed to create controller!");
            } else {
                this.onControllerTrained(ctrl);
            }
        });
    }

    public handleLoadTrainedController() {
        let entry = this.state.trainedControllers[this.state.selectedTrained];
        if (!entry) {
            return;
        }
        Promise.all([fetch(entry.weights).then((r) => r.json()), fetch(entry.sidecar).then((r) => r.ok ? r.json() : null)])
            .then(([weights, sidecar]) => this.installTrainedController(weights, sidecar, entry.label))
            .catch((e) => this.setState({ nn: undefined, loadWeightsSuccessful: false, loadedTrainedLabel: null, loadWeightsFailureMsg: "" + e }));
    }

    /** A weights file, optionally with its sidecar, from disk — e.g. toy's ctrl.json + ctrl.json.meta.json. */
    public handleTrainedControllerFiles(e: React.ChangeEvent<HTMLInputElement>) {
        let files: File[] = [];
        for (let i = 0; e.currentTarget.files && i < e.currentTarget.files.length; i++) {
            files.push(e.currentTarget.files[i]);
        }
        e.currentTarget.value = "";
        if (files.length == 0) {
            return;
        }
        let read = (file: File) => new Promise<any>((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = () => { try { resolve(JSON.parse(reader.result as string)); } catch (err) { reject(new Error(file.name + ": " + err)); } };
            reader.onerror = () => reject(new Error("could not read " + file.name));
            reader.readAsText(file);
        });
        Promise.all(files.map(read)).then((contents) => {
            let sidecar = contents.find((c) => c && (c.format === "tbu-controller/1" || ("" + c.schema).indexOf("toy-truck-controller") == 0)) || null;
            let weights = contents.find((c) => c instanceof Array);
            if (!weights) {
                throw new Error("no weights file among " + files.map((f) => f.name).join(", ") + " (expected a JSON array of layers)");
            }
            if (sidecar && !sidecar.net && sidecar.schema) {
                // toy's own sidecar shape: flat fields, sigmoid/logistic naming
                sidecar = { net: { activation: sidecar.activation == "sigmoid" || sidecar.activation == "logistic" ? "logistic" : "tanh",
                                   output_map: sidecar.output_map == "2sigma-1" || sidecar.output_map == "2s-1" ? "2s-1" : "tanh" },
                            plant: { r: sidecar.r, step_cap: sidecar.step_cap, dock_ref: "trailer" } };
            }
            let label = files.map((f) => f.name).filter((n) => n.indexOf("meta") < 0)[0] || files[0].name;
            this.installTrainedController(weights, sidecar, label);
        }).catch((err) => this.setState({ nn: undefined, loadWeightsSuccessful: false, loadedTrainedLabel: null, loadWeightsFailureMsg: "" + err }));
    }

    public handleLoadPretrainedWeights() {
        let weightName = "truck_emulator_controller_weights_" + this.state.weightLessonIndex;;
        let lessonIndex = this.state.weightLessonIndex;

        $.ajax({
            url: "weights/" + weightName,
            dataType: "text",
            mimeType: "application/json",
            success: (data) => {
                let network = this.getDefaultNetConfig();
                let neuralNet = new NeuralNet(network);

                try {
                    neuralNet.loadWeights(JSON.parse(data));
                    this.setState({
                        loadingWeights: false,
                        updatedController: false,
                        nn: neuralNet,
                        network: network,
                        loadWeightsSuccessful: true,
                        loadedTrainedLabel: null,
                        trainedHint: null,
                        loadedLessonWeights: lessonIndex,
                        errors: [],
                        isTrainedNetwork: false
                    }, () => {
                        let ctrl = this.makeTrainController()
                        if (!ctrl) {
                            alert("Failed to create controller!");
                        } else {
                            this.onControllerTrained(ctrl);
                        }
                    });
                } catch (e) {
                    this.setState({
                        network: network,
                        loadingWeights: false,
                        nn: undefined,
                        loadWeightsSuccessful: false,
                        loadWeightsFailureMsg: "" + e
                    })
                }
            }
        })
    }

    private getDefaultNetConfig() {
        return this.getTruckNet();
    }

    private getTruckNet(): NetConfig {
        const hiddenControllerLayer: LayerConfig = {
            neuronCount: 45,
            weightInitializer: new TwoLayerInitializer(0.7, 45),
            unitConstructor: (weights: number, activation: ActivationFunction, initialWeightRange: WeightInitializer, optimizer: Optimizer) => new AdalineUnit(weights, activation, initialWeightRange, optimizer),
            activation: new Tanh()
        }

        const outputControllerLayer: LayerConfig = {
            neuronCount: 1,
            weightInitializer: new TwoLayerInitializer(0.7, 1),
            unitConstructor: (weights: number, activation: ActivationFunction, initialWeightRange: WeightInitializer, optimizer: Optimizer) => new AdalineUnit(weights, activation, initialWeightRange, optimizer),
            activation: new Tanh() // [-1, 1]
        }

        return {
            inputs: 4,
            optimizer: () => new SGD(0.8),
            errorFunction: new MSE(), // ignored
            layerConfigs: [
                hiddenControllerLayer,
                outputControllerLayer
            ]
        }
    }
    // TODO: add visualization for training area from TrainController.lastTrainedLesson
    // i.e. add red square in simulation

    public handleResetNetwork() {
        this.setState({
            network: this.getDefaultNetConfig(),
            nn: undefined,
            loadWeightsSuccessful: null,
            updatedController: false,
            errors: []
        }, () => {
            this.props.onControllerTrained(null);
        })
    }

    public onNetworkChange(net: NetConfig, keepWeights: boolean) {
        let nn = this.state.nn;
        let errors = this.state.errors;
        let isTrained = this.state.isTrainedNetwork;
        let currentLesson = this.state.currentLessonIndex;
        if (keepWeights && this.state.nn) {
            let weights = this.state.nn.getWeights();
            nn = new NeuralNet(net);
            nn.loadWeights(weights);
        } else {
            nn = undefined;
            errors = [];
            isTrained = false;
            currentLesson = 0;
        }
        this.setState({ currentLessonIndex: currentLesson, network: net, nn: nn, errors: errors, isTrainedNetwork: isTrained });
    }

    private updateLessons(lessons: TruckLesson[]) {
        let newIndex = this.state.currentLessonIndex < lessons.length ? this.state.currentLessonIndex : lessons.length - 1;
        newIndex = newIndex < 0 ? 0 : newIndex;
        if (this.emulatorController) {
            let lesson = undefined;
            if (newIndex < lessons.length)
                lesson = lessons[newIndex];
            if (lesson !== undefined) {
                this.emulatorController.setLesson(lesson);
            }
        }

        this.setState({ lessons: lessons, currentLessonIndex: newIndex });
    }

    private setCurrentLesson(index: number) {
        if (this.emulatorController)
            this.emulatorController.setLesson(this.state.lessons[index]);
        this.setState({ currentLessonIndex: index })
    }
    private getErrorDiagram() {
        let config = {
            title: {
                text: "Controller Error"
            },
            plotOptions: {
                line: {
                    animation: false
                }
            },
            yAxis: {
                min: 0,
                title: {
                    text: "Error"
                }
            },
            xAxis: {
                labels: {
                    formatter: function(): string {
                        return ((this as any).value * 100 + 100).toFixed(0);
                    }
                }
            },
            series: [
                {
                    name: "Error",
                    data: this.state.errors
                }
            ]
        }
        return <div>
            <button type="button" disabled={this.state.train} onClick={() => this.handleResetErrors()} className="btn btn-danger pb">Clear diagram</button>
            <ReactHighcharts
                config={config}
                ref="chart"
            />
        </div>;
    }

    private handleResetErrors() {
        this.setState({
            errors: []
        }, () => {
            (this.refs.chart as any).getChart().series[0].setData([], true)
        })
    }

    private handleLessonWeightIndexChanged(e: React.ChangeEvent<HTMLSelectElement>) {
        this.setState({
            weightLessonIndex: Number.parseInt(e.currentTarget.value)
        })
    }

    private renderController() {
        let normalizedDockPosition = new Point((this.props.world.dock.position.x - 50) / 50, this.props.world.dock.position.y / 50);
        let mse = new TruckControllerError(normalizedDockPosition);

        let errorFunctions: { [key: string]: ErrorFunction } = {};
        if (mse != undefined) {
            errorFunctions = {};
            errorFunctions[mse.getName()] = mse;
        }

        let optimizers: { [key: string]: () => Optimizer } = {};
        let sgd = new SGD(0.5);
        let nesterov = new SGDNesterovMomentum(0.5, 0.9);
        optimizers[sgd.getName()] = () => new SGD(0.5);
        optimizers[nesterov.getName()] = () => new SGDNesterovMomentum(0.5, 0.9);

        let weightInitializers: { [key: string]: WeightInitializer } = {};
        let random = new RandomWeightInitializer(0.5);
        let twoLayer = new TwoLayerInitializer(0.7, 25);
        weightInitializers[random.getName()] = random;
        weightInitializers[twoLayer.getName()] = twoLayer;

        let activations: { [key: string]: ActivationFunction } = {}
        activations[new Tanh().getName()] = new Tanh();
        activations[new Sigmoid().getName()] = new Sigmoid();
        activations[new ReLu(0.01).getName()] = new ReLu(0.01);
        activations[new Linear().getName()] = new Linear();

        let alert = undefined;

        if (this.state.loadWeightsSuccessful !== null) {
            if (this.state.loadWeightsSuccessful) {
                alert = this.state.loadedTrainedLabel
                    ? <div className="row alert alert-success" role="alert">
                        <strong>Controller loaded: {this.state.loadedTrainedLabel}</strong>
                        {this.state.trainedHint ? <span> — {this.state.trainedHint}</span> : null}
                    </div>
                    : <div className="row alert alert-success" role="alert">
                        <strong>Network for lesson {this.state.loadedLessonWeights} loaded!</strong>
                    </div>
            } else {
                alert = <div className="row alert alert-danger" role="alert">
                    <strong>Failed to load weights!</strong> The demo's nets are 4-45-1 tanh; a trained controller needs a
                 weights file of nested layers with the bias last, and its sidecar for the activation.<br />{this.state.loadWeightsFailureMsg}
                </div>
            }
        }

        let updatedController = undefined;
        if (this.state.updatedController) {
            updatedController = <div className="row alert alert-success" role="alert">
            <strong>Updated controller!</strong>
        </div>
        }

        let alertInstability = <div className="row alert alert-warning" role="alert">
            The training is not very stable - the truck might diverge during the earlier lessons and learn to drive
            hard left or hard right only. This depends on the random weight initialization and the chosen random starting
            positions during training.
        </div>
        let trainButton = <button type="button" onClick={this.handleTrain.bind(this)} className="btn btn-primary">Train</button>;
        if (this.state.train) {
            trainButton = <button type="button" disabled={!this.state.train} onClick={() => this.handleStopTrain()} className="btn btn-danger">Stop</button>;
        }
        let diagram = undefined;
        if (this.state.train || this.state.isTrainedNetwork) {
            let lesson = this.state.lessons[this.state.currentLessonIndex];
            diagram = <div className="row">
                <div className="col-sm-12">
                    Training lesson {lesson.no} for {lesson.samples} samples. {this.state.maxStepErrors} max step violations occurred in the last 100 training samples.
                {this.getErrorDiagram()}
                </div>
            </div>;
        }

        let lessonOptions = [];
        for (let i = 0; i <= Controller.MAX_LESSON; i++) {
            lessonOptions.push(
                <option key={i} value={i}>{i}</option>
            )
        }

        return <div className="container">
            <div className="row mt-large">
                <div className="btn-toolbar form-inline">
                    {trainButton}
                    <button type="button" onClick={this.handleResetNetwork.bind(this)} disabled={this.state.train} className="btn btn-danger">Reset Network</button>
                    <button type="button" onClick={this.handleResetLessons.bind(this)} disabled={this.state.train} className="btn btn-danger mr">Reset Lessons</button>
                </div>
            </div>
            <div className="row mt mb">
                <div className="form-inline">
                    <b>Original Lesson:</b>
                    <select className="ml mr select form-control" defaultValue={this.state.weightLessonIndex.toString()} onChange={this.handleLessonWeightIndexChanged.bind(this)}>
                        {lessonOptions}
                    </select>
                    <button type="button" onClick={this.handleLoadPretrainedWeights.bind(this)} disabled={this.state.train} className="btn btn-warning">Load pretrained network</button>
                </div>
            </div>
            <div className="row mb">
                <div className="form-inline">
                    <b>Trained controller:</b>
                    <select className="ml mr select form-control" value={this.state.selectedTrained.toString()}
                        onChange={(e) => this.setState({ selectedTrained: Number.parseInt(e.currentTarget.value) })}
                        disabled={this.state.trainedControllers.length == 0}>
                        {this.state.trainedControllers.length == 0
                            ? <option value="0">none shipped</option>
                            : this.state.trainedControllers.map((c, i) => <option key={i} value={i}>{c.label}</option>)}
                    </select>
                    <button type="button" onClick={this.handleLoadTrainedController.bind(this)}
                        disabled={this.state.train || this.state.trainedControllers.length == 0} className="btn btn-warning mr">Drive it in the simulator</button>
                    <span className="ml">or your own: </span>
                    <input type="file" multiple accept=".json,application/json" className="ml" onChange={this.handleTrainedControllerFiles.bind(this)} disabled={this.state.train} />
                </div>
            </div>
            {alert}
            {updatedController}
            {alertInstability}
            {diagram}
            <div className="row">
                <div className="col-12">
                    <ul className="nav nav-tabs">
                        <li className="nav-item">
                            <a className="nav-link active" data-toggle="tab" href="#lessons">Lessons</a>
                        </li>
                        <li className="nav-item">
                            <a className="nav-link" data-toggle="tab" href="#network">Network Architecture</a>
                        </li>
                    </ul>

                    <div className="tab-content">
                        <div className="tab-pane container active" id="lessons">
                            <LessonsComponent disabled={this.state.train} onSelectRow={this.setCurrentLesson.bind(this)} activeLessonIndex={this.state.currentLessonIndex} object={this.props.object} lessons={this.state.lessons} onChange={this.updateLessons.bind(this)} />
                        </div>
                        <div className="tab-pane container" id="network">
                            <NetworkCreator disabled={this.state.train} showOptimizer={false} showInfo={false} activations={activations} weightInitializers={weightInitializers} optimizers={optimizers} network={this.state.network} onChange={this.onNetworkChange.bind(this)} errorFunctions={errorFunctions} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }

    public render() {
        return this.renderController();
    }
}