/**
 * Model checks, run against the CLI build: node scripts/check.js
 *
 * These exist because this repo's rollouts are compared against another
 * engine's, so the properties they rely on have to be pinned rather than
 * assumed. The seam checks come from the parity investigation in issue #10.
 */
const path = require('path');

const build = path.resolve(__dirname, '..', 'cli-build');
const { Truck, NormalizedTruck } = require(build + '/model/truck');
const { Dock, World } = require(build + '/model/world');
const { PlantConventions } = require(build + '/model/conventions');
const { Point } = require(build + '/math');
const { Vector } = require(build + '/neuralnet/math');
const { observationForInputCount } = require(build + '/neuralnet/observation');
const { SymmetricSigmoid } = require(build + '/neuralnet/activation');

let failures = 0;
let checks = 0;
function check(name, ok, detail) {
    checks++;
    if (!ok) {
        failures++;
        console.log('FAIL ' + name + (detail ? '  [' + detail + ']' : ''));
    } else if (process.env.VERBOSE) {
        console.log('pass ' + name + (detail ? '  [' + detail + ']' : ''));
    }
}
function section(name) {
    if (process.env.VERBOSE) console.log('\n== ' + name + ' ==');
}

const dock = () => new Dock(new Point(0, 0));
function truckAt(conventions, x, y, ts, tc) {
    const t = new Truck(new Point(x, y), ts, tc, dock(), []);
    t.conventions = conventions;
    t.setLimited(false);
    return t;
}
/** The shortest signed angle, which is the only meaningful cab/trailer angle. */
function shortest(angle) {
    let a = angle % (2 * Math.PI);
    if (a > Math.PI) a -= 2 * Math.PI;
    if (a < -Math.PI) a += 2 * Math.PI;
    return a;
}

section('jack-knife clamp across the +/-pi seam');
{
    // the state both engines reach at step 34 of ensemble start (80, -50, +90deg)
    // with the shipped DFA controller: the rig pinned at exactly the +90 degree
    // limit, with the trailer just wrapped and the cab not. The difference of
    // the two stored angles is -1.5 pi; the angle between them is +0.5 pi.
    const seamTs = -3.096561063;
    const seamTc = 1.615827918;
    check('the seam state really is at the clamp limit',
        Math.abs(shortest(seamTs - seamTc) - Math.PI / 2) < 1e-9,
        'shortest difference ' + (shortest(seamTs - seamTc) / Math.PI).toFixed(4) + ' pi');
    check('and its raw difference is not',
        Math.abs((seamTs - seamTc) + 1.5 * Math.PI) < 1e-9,
        'raw difference ' + ((seamTs - seamTc) / Math.PI).toFixed(4) + ' pi');

    // whatever the steering, the cab must stay on the side of the trailer it was
    // on. Reading the raw difference here flips it to the other side instead.
    for (const wrapping of ['none', 'pi']) {
        let worstDrift = 0;
        for (let signal = -1; signal <= 1.0001; signal += 0.1) {
            const t = truckAt(new PlantConventions('trailerEnd', wrapping, 3), 60, -70, seamTs, seamTc);
            t.nextState(signal, 1);
            const after = shortest(t.getTrailerAngle() - t.getTruckAngle());
            check('wrap=' + wrapping + ': the cab stays on its side of the trailer at the seam (signal ' + signal.toFixed(1) + ')',
                after > 0, 'cab/trailer angle went to ' + (after / Math.PI).toFixed(4) + ' pi');
            worstDrift = Math.max(worstDrift, Math.abs(after) - Math.PI / 2);
        }
        check('wrap=' + wrapping + ': the clamp still holds at the seam', worstDrift < 1e-9,
            'exceeded 90 degrees by ' + worstDrift.toExponential(2));
    }
}

section('clamp and wrapping over a long spin');
for (const wrapping of ['none', 'pi']) {
    const t = truckAt(new PlantConventions('trailerEnd', wrapping, 3), 60, 0, 0, 0);
    let worstCabTrailer = 0;
    let worstStored = 0;
    for (let i = 0; i < 400; i++) {
        t.nextState(1, 1);
        worstCabTrailer = Math.max(worstCabTrailer, Math.abs(shortest(t.getTrailerAngle() - t.getTruckAngle())));
        worstStored = Math.max(worstStored, Math.abs(t.getTrailerAngle()), Math.abs(t.getTruckAngle()));
    }
    check('wrap=' + wrapping + ': the clamp holds at 90 degrees over 400 steps at full lock',
        worstCabTrailer <= Math.PI / 2 + 1e-9, 'reached ' + (worstCabTrailer * 180 / Math.PI).toFixed(4) + ' deg');
    if (wrapping === 'pi') {
        check('wrap=pi: both stored angles stay within [-pi, pi]', worstStored <= Math.PI + 1e-9,
            'reached ' + worstStored.toFixed(6) + ' rad');
    } else {
        check('wrap=none: the stored angles are free to leave [-pi, pi]', worstStored > Math.PI,
            'reached ' + worstStored.toFixed(6) + ' rad');
    }
}

section('plant conventions');
{
    for (const r of [1, 3, 0.5]) {
        const t = truckAt(new PlantConventions('truckEnd', 'none', r), 45, 15, 0, 0);
        const from = new Point(t.getTrailerEndPosition().x, t.getTrailerEndPosition().y);
        t.nextState(0, 1);
        const moved = Math.hypot(t.getTrailerEndPosition().x - from.x, t.getTrailerEndPosition().y - from.y);
        check('r=' + r + ' moves r per step', Math.abs(moved - r) < 1e-9, 'moved ' + moved.toFixed(9));
        check('r=' + r + ' is what the emulator Jacobian reads', t.velocity === r);
    }
    const demo = truckAt(PlantConventions.demo(), 0, 0, 0, 0);
    check('truckEnd does not stop with the trailer rear at the dock', demo.nextState(0, 0.0001) === true);
    const paper = truckAt(PlantConventions.paper(), 0, 0, 0, 0);
    check('trailerEnd stops with the trailer rear at the dock', paper.nextState(0, 0.0001) === false);
    const eot = truckAt(PlantConventions.demo(), -16, 0, 0, 0);
    check('truckEnd stops with the end of truck at the dock', eot.nextState(0, 0.0001) === false);
    check('a fresh truck defaults to the demo conventions',
        new Truck(new Point(45, 15), 0, 0, dock(), []).conventions.equals(PlantConventions.demo()));
    const copy = paper.getDockReferencePoint();
    copy.x = 999;
    check('the dock reference point is a copy', paper.getTrailerEndPosition().x !== 999);
}

section('a normalized plant honours its time step');
{
    const t = new Truck(new Point(45, 15), 0, 0, dock(), []);
    const plant = new NormalizedTruck(t);
    const world = new World(plant, dock());
    const from = new Point(t.getTrailerEndPosition().x, t.getTrailerEndPosition().y);
    for (let i = 0; i < 4; i++) world.nextTimeStep(0, 0.25);
    const moved = Math.hypot(t.getTrailerEndPosition().x - from.x, t.getTrailerEndPosition().y - from.y);
    check('four quarter steps cover one whole step', Math.abs(moved - 1) < 1e-9, 'moved ' + moved.toFixed(9));
}

section('observations map gradients back correctly');
for (const inputs of [4, 3, 8]) {
    const observation = observationForInputCount(inputs, 4);
    const state = [0.3, -0.7, 0.15, -0.45];
    const outDim = observation.getInputDim(4);
    const eps = 1e-7;
    const jacobian = [];
    for (let o = 0; o < outDim; o++) jacobian.push(new Array(4).fill(0));
    for (let s = 0; s < 4; s++) {
        const up = state.slice(); up[s] += eps;
        const down = state.slice(); down[s] -= eps;
        const a = observation.observe(new Vector(up)).entries;
        const b = observation.observe(new Vector(down)).entries;
        for (let o = 0; o < outDim; o++) jacobian[o][s] = (a[o] - b[o]) / (2 * eps);
    }
    const g = [];
    for (let o = 0; o < outDim; o++) g.push(Math.sin(o + 1) * 1.7);
    const expected = new Array(4).fill(0);
    for (let s = 0; s < 4; s++) for (let o = 0; o < outDim; o++) expected[s] += jacobian[o][s] * g[o];
    const actual = observation.mapGradient(new Vector(g), 4).entries;
    let worst = 0;
    for (let s = 0; s < 4; s++) worst = Math.max(worst, Math.abs(actual[s] - expected[s]));
    check(inputs + ' inputs: mapGradient is the transpose of the observation Jacobian', worst < 1e-5,
        'max difference ' + worst.toExponential(2));
}

section('the output mapping');
{
    const s = new SymmetricSigmoid();
    check('2s-1 maps 0 to 0', Math.abs(s.apply(0)) < 1e-12);
    check('2s-1 stays within the steering range', Math.abs(s.apply(1e3)) <= 1 && Math.abs(s.apply(-1e3)) <= 1);
    let worst = 0;
    for (const x of [-3, -1, -0.2, 0, 0.4, 1.5, 4]) {
        worst = Math.max(worst, Math.abs((s.apply(x + 1e-6) - s.apply(x - 1e-6)) / 2e-6 - s.applyDerivative(x)));
    }
    check('2s-1 derivative matches finite differences', worst < 1e-6, 'max difference ' + worst.toExponential(2));
}

console.log(failures === 0
    ? '\n' + checks + ' checks passed'
    : '\n' + failures + ' of ' + checks + ' checks FAILED');
process.exit(failures === 0 ? 0 : 1);
