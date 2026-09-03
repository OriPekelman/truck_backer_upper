# Truck Backer Upper

This repository contains the [Truck Backer Upper demonstration project](http://www-isl.stanford.edu/~widrow/papers/j1990neuralnetworks.pdf) for the lecture “Neural Networks” at the [Interactive Systems Lab (ISL)](http://isl.anthropomatik.kit.edu/english/) at the [Karlsruhe Institute of Technology](https://kit.edu).

This demo runs completely client-side in the browser. [A hosted version of upstream is available here](https://tifu.github.io/truck_backer_upper/).

This fork is the frontend for the `dfa-for-dynamic-control` arc: the plant and the
arms live in `OriPekelman/toy`, and what this repository contributes is seeing how
a controller drives. Alongside the original demo it has a rollout overlay for
comparing arms over the same start states, selectable plant conventions, and a
headless CLI.

## Development

This project is written in [TypeScript](http://www.typescriptlang.org/), a statically typed superset of JavaScript.

I recommend [Visual Studio Code](https://code.visualstudio.com/) (crossplatform and opensource) for development.
An alternative is [Atom](https://atom.io/) + [atom-typescript](https://atom.io/packages/atom-typescript).


[React](https://facebook.github.io/react/) is used with [JSX](https://facebook.github.io/jsx/) for GUI state handling.

## Setup

```bash
npm install
npm run build          # or npm run watch to rebuild on every change
```

The build writes into `bin/`, which is untracked. Serve it on a local webserver
and open `index.html`:

```bash
cd bin && python3 -m http.server 8000
```

`npm run build-prod` produces the minified build.

Upstream builds into a `gh-pages` worktree (`git worktree add bin gh-pages`), so
that `bin/` is a checked-in publishable branch. This fork has no `gh-pages`
branch and does not use that scheme: `bin/` is plain build output, and publishing
is a separate concern. To publish the current build somewhere, copy the contents
of `bin/` there; nothing in the build depends on it being a worktree.

## Headless CLI

`npm run tbu -- <command>` trains and rolls out controllers without a browser.
It runs from any working directory.

```bash
# train with either of the original trainers: through the learned emulator
# (Nguyen & Widrow proper) or through the analytic plant Jacobian
npm run tbu -- train --backend jacobian-bptt --out weights/mine \
    --hidden 9 --activation logistic --output-map 2s-1 \
    --r 3 --dock-ref trailerEnd --starts ensemble --objective d2-best

# roll a controller out over a start set and write a tbu-traces/1 bundle,
# which the "Rollout overlay" tab reads
npm run tbu -- rollout --weights src/weights/truck_emulator_controller_weights_19 \
    --out bundles/demo19.json --starts yard --start-count 200 --dock-ref trailerEnd
```

The net shape, input set, plant conventions, start set and objective are all
arguments rather than hardcoded; `npm run tbu -- help` lists them with their
defaults, which reproduce the demo's own settings. Training writes a sidecar
next to the weights recording the commit, the seed and every convention in
force, and `rollout` reads that sidecar so a rollout matches its training.

## Plant conventions

This code and Schoenauer & Ronald's paper implement the same reformulation of
the plant but disagree on three conventions around it: which point has to reach
the dock, whether the angles are wrapped after each step, and the step length.
All three are selectable, in the simulation's "Plant Conventions" panel and as
CLI arguments, and default to what the bundled weights were trained under. See
`src/model/conventions.ts`.

## Layout

- `src/model` -- the plant: the truck's kinematics, the world, the selectable
  conventions, and the reader for `tbu-traces/1` rollout bundles.
- `src/neuralnet` -- nets, layers, activations, optimizers, the two controller
  trainers and the error functions.
- `src/gui` -- React components: the interactive simulation, the emulator and
  controller tabs, and the rollout overlay with its comparison tables.
- `src/cli` -- the headless `tbu` CLI.
- `src/train` -- the original standalone training scripts.
- `src/weights` -- the bundled emulator and controller weights.
