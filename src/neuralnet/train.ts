import { World, Dock, HasState } from '../model/world';
import { NeuralNet } from './net'
import { Vector } from './math'
import { Angle, Point } from '../math'
import { TruckLesson } from './lesson'
import { ControllerError, usesBestApproachGrading, BestApproachError } from './error';
import { Observation, FullObservation } from './observation';
import { emulatorNet } from './implementations';
import { Emulator } from './emulator';

export type MaxStepListener = (steps: number) => void;

/**
 * Positions the plant for one training sample. Must be deterministic in the
 * sample index: grading the closest approach replays the same episode, so
 * asking twice for the same sample has to give the same start.
 */
export type StartProvider = (sampleIndex: number) => void;

export class TrainTruckEmulator {
    private lastError: number = 0;
    public cabAngleError: number[] = [];
    public xCabError: number[] = []
    public yCabError: number[] = []
    public trailerAngleError: number[] = []
    public xTrailerError: number[] = []
    public yTrailerError: number[] = []

    private trainedSteps = 0;
    // TODO: disable data collection?
    public constructor(private plant: HasState, private neuralNet: NeuralNet, private batchSize: number = 1) {
    }

    public getPerformedSteps() {
        return this.trainedSteps;
    }

    public getEmulatorNet(): NeuralNet {
        return this.neuralNet;
    }

    public getErrorCurve(): Array<number> {
        return []//this.neuralNet.errors;
    }

    public trainStep(nextSteeringAngle: number): boolean {
        let stateVector = this.plant.getStateVector();

        stateVector = stateVector.getWithNewElement(nextSteeringAngle);

        let result = this.neuralNet.forward(stateVector);
        let retVal = this.plant.nextState(nextSteeringAngle, 1);

        let expectedVector = this.plant.getStateVector();

        //[cdp.x, cdp.y, this.cabinAngle, this.tep.x, this.tep.y, this.trailerAngle]
        // Record errors
        this.xTrailerError.push(Math.abs(expectedVector.entries[0] - result.entries[0]) * 50);
        this.yTrailerError.push(Math.abs(expectedVector.entries[1] - result.entries[1]) * 50);
        this.cabAngleError.push(Math.abs(expectedVector.entries[2] - result.entries[2]) * 180);// * Math.PI * 180 / Math.PI
        if (result.entries.length >= 4) {
            this.trailerAngleError.push(Math.abs(expectedVector.entries[3] - result.entries[3]) * 180);
        }
        this.lastError = this.neuralNet.getError(result, expectedVector);

        let error = this.neuralNet.backward(result, expectedVector, true); // batch update

        this.trainedSteps++;
        if (this.trainedSteps % this.batchSize == 0) {
            this.neuralNet.updateWithAccumulatedWeights();
        }

        return retVal && !result.isEntryNaN();
    }

    public train(epochs: number) {
        let nextSteeringAngle = Math.random() * 2 - 1;
        let err = 0;
        let count = 0;
        for (let i = 0; i < epochs; i++) {
            let cont = this.trainStep(nextSteeringAngle);
            err += this.lastError;
            count++;
            if (!cont) {
                return [i, err / count];
            }
        }
        return [epochs, err / count];
    }
}

// let's not use this for now
export class TrainController {
    private lastTrainedLesson: TruckLesson | null = null
    public errors: Array<number> = [];
    public steeringSignals: Array<number> = [];
    public angleError: Array<number> = [];
    public yError: Array<number> = [];

    public fixedEmulator = false;
    private performedTrainSteps = 0;
    public maxStepErrors = 0;

    public emulatorInputs: any = [];
    private currentLesson: TruckLesson | null = null;
    private maxStepListeners: Set<MaxStepListener> = new Set<MaxStepListener>();
    // how the plant's state reaches the controller's inputs
    private observation: Observation = new FullObservation();
    // where each sample starts; the lesson's own randomization when unset
    private startProvider: StartProvider | null = null;

    public constructor(private world: World, private realPlant: HasState, private controllerNet: NeuralNet, private emulatorNet: Emulator | null, private errorFunction: ControllerError) {
    }

    /**
     * Sets the input set the controller sees. The plant's state is unchanged;
     * only what the net is shown, and where its input gradient goes back to.
     */
    public setObservation(observation: Observation) {
        this.observation = observation;
    }

    public getObservation(): Observation {
        return this.observation;
    }

    public setStartProvider(startProvider: StartProvider | null) {
        this.startProvider = startProvider;
    }

    public setEmulatorNet(emulator: Emulator) {
        this.emulatorNet = emulator;
    }

    public addMaxStepListener(listener: MaxStepListener) {
        this.maxStepListeners.add(listener);
    }

    private informListeners(steps: number) {
        for (let listener of this.maxStepListeners) {
            listener(steps);
        }
    }

    public setPlant(realPlant: HasState) {
        this.realPlant = realPlant;
    }

    public setLastTrainedLesson(lesson: TruckLesson) {
        this.lastTrainedLesson = lesson;
    }

    public getEmulatorNet() {
        return this.emulatorNet;
    }

    public getControllerNet() {
        return this.controllerNet;
    }

    public predict(): number {
        let currentState = this.realPlant.getStateVector();
        this.controllerNet.fixWeights(true); // do not safe input in units
        let controllerSignal = this.controllerNet.forward(this.observation.observe(currentState));
        return controllerSignal.entries[0];
    }

    public setLesson(lesson: TruckLesson): void {
        this.currentLesson = lesson;
        if (lesson !== undefined)
            this.controllerNet.changeOptimizer(lesson.optimizer);

        this.performedTrainSteps = 0;
        this.maxStepErrors = 0;
    }

    public getPerformedTrainSteps(): number {
        return this.performedTrainSteps;
    }

    public hasNextStep(): boolean {
        if (!this.currentLesson) {
            return false;
        }
        return this.performedTrainSteps < this.currentLesson.samples;
    }

    public getCurrentLesson(): TruckLesson | null {
        return this.currentLesson;
    }

    public trainSingleStep(): number {
        if (!this.currentLesson) {
            throw new Error("You have to set the current lesson before calling this function!");
        }
        this.prepareTruckPosition();
        let error = this.trainStep();
        this.lastTrainedLesson = this.currentLesson;
        this.performedTrainSteps++;
        return error;
    }

    public getErrorCurve(): Array<number> {
        return this.errors;
    }

    private prepareTruckPosition() {
        if (this.startProvider) {
            this.startProvider(this.performedTrainSteps);
            return;
        }
        this.realPlant.randomizePosition(this.currentLesson as TruckLesson);
    }

    /**
     * Rolls the episode out once without training, to find the step at which
     * the controller came closest, then puts the plant back at the same start
     * so that the training pass can stop there. Returns 0 when the start was
     * already the closest point, i.e. when there is nothing to learn from.
     */
    private measureBestApproachStep(maxSteps: number): number {
        if (!this.startProvider) {
            throw new Error("Grading the closest approach needs a start provider, \
                so that the episode can be replayed up to it");
        }
        let errorFunction = this.errorFunction as any as BestApproachError;
        this.controllerNet.fixWeights(true); // do not record inputs for this pass
        let bestScore = errorFunction.scoreState(this.realPlant.getStateVector());
        let bestStep = 0;
        let steps = 0;
        let canContinue = true;
        while (canContinue && steps < maxSteps) {
            let observed = this.observation.observe(this.realPlant.getStateVector());
            let steeringSignal = this.controllerNet.forward(observed).entries[0];
            canContinue = this.realPlant.nextState(steeringSignal, 1);
            steps++;
            let score = errorFunction.scoreState(this.realPlant.getStateVector());
            if (score < bestScore) {
                bestScore = score;
                bestStep = steps;
            }
        }
        this.controllerNet.fixWeights(false);
        this.startProvider(this.performedTrainSteps);
        return bestStep;
    }

    private fixEmulator(fix: boolean) {
        if (this.emulatorNet && this.fixedEmulator != fix) {
            this.emulatorNet.setNotTrainable(fix); // do not train emulator
            this.fixedEmulator = fix;
        }
    }

    // TODO: duplicate code
    private normalizeDock(d: Dock) {
        let normX = (d.position.x - 50) / 50
        let normY = (d.position.y) / 50;
        return new Point(normX, normY);
    }

    private trainStep(): number {
        if (!this.emulatorNet) {
            throw new Error("The emulator net has to be initialized before trainin gcan begin!");
        }
        this.fixEmulator(true);
        let canContinue = true;
        let controllerSignals = [];
        let statesFromEmulator = [];
        this.emulatorInputs = [];
        let i = 0;

        let outputState = this.realPlant.getOriginalState();

        let maxSteps = (this.currentLesson as TruckLesson).maxSteps;
        // -1 grades the state the episode stopped in, as it always has
        let gradeAtStep = -1;
        if (usesBestApproachGrading(this.errorFunction)) {
            gradeAtStep = this.measureBestApproachStep(maxSteps);
            if (gradeAtStep == 0) {
                return NaN; // the start was the closest approach
            }
        }

        // start at current state
        let positions = [];
        while (canContinue) {
            let currentState = this.realPlant.getStateVector();
            positions.push(this.realPlant.getOriginalState());

            let controllerSignal = this.controllerNet.forward(this.observation.observe(currentState));

            let steeringSignal = controllerSignal.entries[0];

            let stateWithSteering = currentState.getWithNewElement(steeringSignal);

            this.emulatorNet.forward(stateWithSteering);

            canContinue = this.realPlant.nextState(steeringSignal, 1);

            // set the next state
            currentState = this.realPlant.getStateVector();
            outputState = this.realPlant.getOriginalState();

            if (gradeAtStep < 0 && canContinue && i + 1 >= maxSteps) {
                this.informListeners(i);
                this.controllerNet.clearInputs();
                this.emulatorNet.clearInputs();
                this.maxStepErrors++;

                return 0;
            }
            i++;
            if (gradeAtStep > 0 && i >= gradeAtStep) {
                break; // grade the closest approach rather than the last state
            }
        }
        let realState = this.realPlant.getStateVector();

        if (i == 0) { // we didn't do anything => no update!
            return NaN;
        }

        // we hit the end => calculate performance error (real position - real target), backpropagate
        let finalState = this.realPlant.getStateVector();
        let dock = this.world.dock;
        let normalizedDock: Point = this.normalizeDock(dock);

        // performance error i.e. real position - real target
        let controllerDerivative = this.calculateErrorDerivative(finalState, normalizedDock);
        let controllerError = this.calculateError(finalState, normalizedDock);

        let error = this.calculateError(finalState, normalizedDock);

        for (let j = i - 1; j >= 0; j--) {

            let emulatorDerivative = this.emulatorNet.backward(controllerDerivative); //.backwardWithGradient(controllerDerivative, false);

            let steeringSignalDerivative = emulatorDerivative.entries[emulatorDerivative.entries.length - 1]; // last entry

            let observationDerivative = this.controllerNet.backwardWithGradient(new Vector([steeringSignalDerivative]), true);
            // the controller's input gradient is in observation space, the
            // emulator's in state space; both are gradients wrt the same state
            controllerDerivative = this.observation.mapGradient(observationDerivative, emulatorDerivative.entries.length - 1);

            // get the error from the emulator and add it to the input error for the controller
            // remove the last element
            let errorFromEmulator = new Vector(emulatorDerivative.entries.slice(0, emulatorDerivative.entries.length - 1));

            controllerDerivative.add(errorFromEmulator);
        }

        this.controllerNet.updateWithAccumulatedWeights();
        this.fixEmulator(false);
        let endState = this.realPlant.getOriginalState();
        let endError = this.errorFunction.getError(this.realPlant.getStateVector());

        return error;
    }

    private calculateError(finalState: Vector, dock: Point): number {
        return this.errorFunction.getError(finalState);
    }

    private calculateErrorDerivative(finalState: Vector, dock: Point): Vector {
        return this.errorFunction.getErrorDerivative(finalState);
    }
}
