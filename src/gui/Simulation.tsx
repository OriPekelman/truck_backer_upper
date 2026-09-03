declare var require: any; // trust me require exists
import * as React from 'react'
import WorldVisualization from "./WorldVisualization"
import { World } from '../model/world'
import { toDeg, toRad } from '../math';
import { TrainController } from '../neuralnet/train'
import { Point } from '../math'
import { Truck } from '../model/truck';
import { Dock } from '../model/world';
import { PlantConventions, DockReference, AngleWrapping } from '../model/conventions';
import Slider from 'rc-slider';


interface SimulationProps {
    object: Truck;
    dock: Dock;
    controller: TrainController | undefined;
}
interface SimulationState {
    world: World;
    steeringSignal: number;
    simulationSpeed: number;
    isDriving: boolean;
    cabAngle: number;
    trailerAngle: number;
    showTrace: boolean;
    showTraceOutlines: boolean;
    conventions: PlantConventions;
}

export class Simulation extends React.Component<SimulationProps, SimulationState> {
    static instance: Simulation;
    private lastTimestamp: number = -1;
    private stepLengthInMs = 1000;

    public constructor(props: SimulationProps) {
        super(props)
        if (Simulation.instance) throw Error("Already instantiated")
        else Simulation.instance = this;

        let cabAngle = toDeg(this.props.object.getTruckAngle());
        let trailerAngle = toDeg(this.props.object.getTrailerAngle());

        this.state = {
            world: new World(this.props.object, this.props.dock),
            steeringSignal: 0,
            simulationSpeed: 4,
            isDriving: false,
            cabAngle: cabAngle,
            trailerAngle: trailerAngle,
            showTrace: true,
            showTraceOutlines: false,
            conventions: this.props.object.conventions.copy()
        };
    }

    public drive(steeringSignal: number, done: (cont: boolean) => void) {
        this.setState({ isDriving: true }, () => {
            this.lastTimestamp = performance.now();
            const callback = (cont: boolean) => {
                if (cont) {
                    done(cont);
                } else {
                    this.setState({ isDriving: false }, () => {
                        done(cont);
                    });
                }
            }
            window.requestAnimationFrame(this.driveFrameCallback(steeringSignal, 0, callback));    
        });
    }

    private driveFrameCallback = (steeringSignal: number, totalTime: number, done: (cont: boolean) => void) => {
        return (timestamp: number) => this.driveStep(timestamp, steeringSignal, totalTime, done);
    };

    private driveStep(timestamp: number, steeringSignal: number, totalTime: number, done: (cont: boolean) => void) {
        const stepLength = this.stepLengthInMs / this.state.simulationSpeed;
        const delta = (timestamp - this.lastTimestamp);
        let cont = true;

        if (delta >= 0) {
            const realDelta = this.state.simulationSpeed * Math.min(stepLength - totalTime, delta);

            cont = this.state.world.nextTimeStep(steeringSignal, realDelta / this.stepLengthInMs);
            totalTime += delta;
            this.onFrame(true);
        }

        if (totalTime < stepLength && cont && this.state.isDriving) {
            this.lastTimestamp = performance.now();
            window.requestAnimationFrame(this.driveFrameCallback(steeringSignal, totalTime, done));
        } else {
            done(cont && this.state.isDriving);
        }
    }

    public onFrame(forceRedraw: boolean) {
        if (forceRedraw)
            this.forceUpdate();

    }

    private handleDriveButton() {
        this.drive(this.state.steeringSignal, (cont: boolean) => { 
            if (cont) {
                if (cont) {
                    this.handleDriveButton();
                }
            }
        });
    }

    private handleSteeringSignalChanged(value: number) {
        this.setState({ steeringSignal: value })
    }
    private handleSimulationSpeedChanged(value: number) {
        this.setState({ simulationSpeed: value });
    }

    private handleSetRandomPosition() {
        this.state.world.movableObject.randomizePosition();
        this.state.world.startNewTrace();
        //        this.forceUpdate();
        console.log("New Random position: ");
        console.log("Cab Angle: ", toDeg(this.props.object.getTruckAngle()));
        console.log("Trailer Angle: ", toDeg(this.props.object.getTrailerAngle()))
        this.setState({ 
            isDriving: false, 
            cabAngle: toDeg(this.props.object.getTruckAngle()), 
            trailerAngle: toDeg(this.props.object.getTrailerAngle()) 
        })
    }

    private handleDriveController() {
        if (!this.props.controller) {
            throw new Error("The controller must be initialized for driving to work!");
        }
        let steeringSignal = this.props.controller.predict();
        this.drive(steeringSignal, (cont: boolean) => {
            if (cont) {
                this.handleDriveController();
            }
        })
    }

    private getTruckAngleSettings() {
        return <div className="form-group">
            <div className="form-inline">
                <div className="row mb w-100">
                    <div className="col-6">
                        <label htmlFor="formGroupExampleInput" className="float-left">Trailer Angle</label>
                    </div>
                    <div className="col-6">
                        <input key={this.state.trailerAngle} defaultValue={this.state.trailerAngle.toFixed(0)} id="trailerAngle" type="text" onBlur={(e) => this.handleTrailerAngleChanged(e)} className="form-control ml float-right" />
                    </div>
                </div>
            </div>
            <div className="form-inline">
                <div className="row mb w-100">
                    <div className="col-6">
                        <label htmlFor="formGroupExampleInput" className="float-left">Cabin Angle (rel. to Trailer)</label>
                    </div>
                    <div className="col-6">
                        <input key={this.state.cabAngle} defaultValue={this.state.cabAngle.toFixed(0)} id="cabAngle" type="text" onBlur={(e) => this.handleCabinAngleChanged(e)} className="form-control ml float-right" />
                    </div>
                </div>
            </div>
            <div className="form-inline">
                <div className="row w-100">
                    <div className="col-12">
                        <button type="button" className="btn btn-primary float-right" disabled={this.state.isDriving} onClick={this.handleChangeTruckAngle.bind(this)} >Change Angles</button>
                    </div>
                </div>
            </div>
        </div>
    }

    private handleChangeTruckAngle() {
        if (this.props.object instanceof Truck) {
            this.props.object.setTruckPosition(
                this.props.object.getTrailerEndPosition(),
                toRad(this.state.trailerAngle),
                toRad(this.state.trailerAngle + this.state.cabAngle)
            )
            this.state.world.startNewTrace();
            this.forceUpdate();
        }
    }

    private handleTrailerAngleChanged(e: React.ChangeEvent<HTMLInputElement>) {
        let trailerAngle = Number.parseFloat(e.currentTarget.value);
        this.setState({
            trailerAngle: trailerAngle
        })
    }

    private handleCabinAngleChanged(e: React.ChangeEvent<HTMLInputElement>) {
        let current = Number.parseFloat(e.currentTarget.value);

        let cabAngle = Math.max(-90, Math.min(90, current));

        this.setState({
            cabAngle: cabAngle
        })
    }

    private handlePositionChange(translation: Point) {
        let truck = this.props.object;
        let tep = truck.getTrailerEndPosition();
        let oldTep = new Point(tep.x, tep.y);
        tep.x += translation.x;
        tep.y += translation.y;
        this.props.object.setTruckPosition(tep, truck.getTrailerAngle(), truck.getTruckAngle());
        this.state.world.startNewTrace();
        this.forceUpdate();
    }

    /**
     * The conventions live on the plant; the state only mirrors them so that
     * the controls re-render.
     */
    private applyConventions(conventions: PlantConventions) {
        this.props.object.conventions = conventions;
        this.setState({ conventions: conventions.copy() });
    }

    private handleDockReferenceChanged(dockReference: DockReference) {
        let conventions = this.state.conventions.copy();
        conventions.dockReference = dockReference;
        this.applyConventions(conventions);
    }

    private handleWrappingChanged(wrapping: AngleWrapping) {
        let conventions = this.state.conventions.copy();
        conventions.wrapping = wrapping;
        this.applyConventions(conventions);
    }

    private handleStepLengthChanged(e: React.FocusEvent<HTMLInputElement>) {
        let stepLength = Number.parseFloat(e.currentTarget.value);
        if (!isFinite(stepLength) || stepLength <= 0) {
            return;
        }
        let conventions = this.state.conventions.copy();
        conventions.stepLength = stepLength;
        this.applyConventions(conventions);
    }

    private getConventionSettings() {
        let conventions = this.state.conventions;
        let disabled = this.state.isDriving;
        return <div className="form-group">
            <div className="alert alert-info">
                Where this demo and Schoenauer &amp; Ronald's paper disagree. The bundled
                controller weights were trained under the demo's settings.
            </div>
            <div className="row mb w-100">
                <div className="col-6"><label className="float-left">Dock reference point</label></div>
                <div className="col-6">
                    <label className="trace-legend-toggle"><input type="radio" disabled={disabled} checked={conventions.dockReference == "truckEnd"} onChange={() => this.handleDockReferenceChanged("truckEnd")} /><span>End of truck (demo)</span></label>
                    <label className="trace-legend-toggle"><input type="radio" disabled={disabled} checked={conventions.dockReference == "trailerEnd"} onChange={() => this.handleDockReferenceChanged("trailerEnd")} /><span>Rear of trailer (paper, and what the error term grades)</span></label>
                </div>
            </div>
            <div className="row mb w-100">
                <div className="col-6"><label className="float-left">Angle wrapping</label></div>
                <div className="col-6">
                    <label className="trace-legend-toggle"><input type="radio" disabled={disabled} checked={conventions.wrapping == "none"} onChange={() => this.handleWrappingChanged("none")} /><span>None (demo &amp; paper)</span></label>
                    <label className="trace-legend-toggle"><input type="radio" disabled={disabled} checked={conventions.wrapping == "pi"} onChange={() => this.handleWrappingChanged("pi")} /><span>Wrap into (-180°, 180°] after each step</span></label>
                </div>
            </div>
            <div className="form-inline">
                <div className="row mb w-100">
                    <div className="col-6">
                        <label className="float-left">Step length r (paper: 3)</label>
                    </div>
                    <div className="col-6">
                        <input key={conventions.stepLength} defaultValue={"" + conventions.stepLength} type="text" disabled={disabled} onBlur={(e) => this.handleStepLengthChanged(e)} className="form-control ml float-right" />
                    </div>
                </div>
            </div>
            <div className="row w-100">
                <div className="col-12 btn-toolbar float-right">
                    <button type="button" className="btn btn-secondary" disabled={disabled || conventions.equals(PlantConventions.demo())} onClick={() => this.applyConventions(PlantConventions.demo())}>Demo defaults</button>
                    <button type="button" className="btn btn-secondary" disabled={disabled || conventions.equals(PlantConventions.paper())} onClick={() => this.applyConventions(PlantConventions.paper())}>Paper defaults</button>
                </div>
            </div>
        </div>
    }

    private handleClearTrace() {
        this.state.world.clearTrace();
        this.forceUpdate();
    }

    private handleShowTraceChanged(e: React.ChangeEvent<HTMLInputElement>) {
        this.setState({ showTrace: e.currentTarget.checked });
    }

    private handleShowTraceOutlinesChanged(e: React.ChangeEvent<HTMLInputElement>) {
        this.setState({ showTraceOutlines: e.currentTarget.checked });
    }

    public handleStopDriving() {
        this.setState({ isDriving: false })
    }
    public render() {
        let marksSteering: any = {};
        for (let i = -1; i <= 1; i += 0.2) {
            marksSteering[i] = "" + (this.state.world.movableObject.getMaxSteeringAngle() * i * 180 / Math.PI).toFixed(2);
        }
        let marksSimulationSpeed: any = { 1: "1", 2: "2" };
        let maxSimSpeed = 64;
        for (let i = 4; i <= maxSimSpeed; i += 4) {
            marksSimulationSpeed[i] = "" + i.toFixed(0);
        }
        return <div>
            <div className="container">
                <div className="row">
                    <div className="col-sm-6 pad">
                        <div className="col-sm-12 panel panel-default">
                            <WorldVisualization draggable={!this.state.isDriving} world={this.state.world} showTrace={this.state.showTrace} showTraceOutlines={this.state.showTraceOutlines} onObjectMoved={this.handlePositionChange.bind(this)} />
                            <div className="trace-legend">
                                <label className="trace-legend-toggle">
                                    <input type="checkbox" checked={this.state.showTrace} onChange={this.handleShowTraceChanged.bind(this)} />
                                    <span>Show trace</span>
                                </label>
                                <span className="trace-legend-entry"><span className="trace-legend-line trace-legend-trailer"></span>End of trailer</span>
                                <span className="trace-legend-entry"><span className="trace-legend-line trace-legend-cabin"></span>Front of cabin</span>
                                <span className="trace-legend-entry"><span className="trace-legend-marker trace-legend-dockref"></span>Dock reference</span>
                            </div>
                            <div className="trace-legend">
                                <label className="trace-legend-toggle">
                                    <input type="checkbox" checked={this.state.showTraceOutlines} onChange={this.handleShowTraceOutlinesChanged.bind(this)} />
                                    <span>Show boxes (as in the paper)</span>
                                </label>
                            </div>
                            <div className="trace-legend">
                                <span className="trace-legend-entry"><span className="trace-legend-marker trace-legend-jackknife"></span>Jack-knifed</span>
                                <span className="trace-legend-entry"><span className="trace-legend-marker trace-legend-wall"></span>Hit dock wall</span>
                                <span className="trace-legend-entry"><span className="trace-legend-marker trace-legend-area"></span>Left area</span>
                                <span className="trace-legend-entry"><span className="trace-legend-marker trace-legend-docked"></span>Docked</span>
                            </div>
                        </div>
                    </div>
                    <div className="col-sm-6 pad">
                        <div className="row">
                            <div className="col-sm-12 panel panel-default h-100">
                                <h3>Simulation Settings</h3>
                                <div className="form-group pad-slider">
                                    <label htmlFor="formGroupExampleInput">Steering Angle (in Degree)</label>
                                    <Slider min={-1} max={1} marks={marksSteering} onChange={this.handleSteeringSignalChanged.bind(this)} value={this.state.steeringSignal} step={0.05} />
                                </div>
                                <div className="form-group pad-slider">
                                    <label htmlFor="formGroupExampleInput">Simulation Speed</label>
                                    <Slider min={1} max={maxSimSpeed} marks={marksSimulationSpeed} onChange={this.handleSimulationSpeedChanged.bind(this)} value={this.state.simulationSpeed} step={1} />
                                </div>
                                <div className="h3 btn-toolbar">
                                    <button type="button" className="btn btn-primary" disabled={this.state.isDriving} onClick={this.handleDriveButton.bind(this)} >Manual Drive</button>
                                    <button type="button" className="btn btn-warning" onClick={this.handleSetRandomPosition.bind(this)}>Random Position</button>
                                    <button type="button" className="btn btn-primary" disabled={!this.props.controller || this.state.isDriving} onClick={this.handleDriveController.bind(this)}>Drive using Controller</button>
                                    <button type="button" className="btn btn-danger" disabled={!this.state.isDriving} onClick={this.handleStopDriving.bind(this)}>Stop</button>
                                    <button type="button" className="btn btn-default" onClick={this.handleClearTrace.bind(this)}>Clear Trace</button>
                                </div>
                                <h3>Truck Orientation</h3>
                                <div className="alert alert-info">
                                    Drag & Drop the truck to change its position, then set the angles here. Cabin Angle must be less than +/- 90 degrees
                                </div>
                                {this.getTruckAngleSettings()}
                                <h3>Plant Conventions</h3>
                                {this.getConventionSettings()}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }
}
