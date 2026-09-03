# Truck Backer Upper

This repository contains the [Truck Backer Upper demonstration project](http://www-isl.stanford.edu/~widrow/papers/j1990neuralnetworks.pdf) for the lecture “Neural Networks” at the [Interactive Systems Lab (ISL)](http://isl.anthropomatik.kit.edu/english/) at the [Karlsruhe Institute of Technology](https://kit.edu).

This demo runs completely client-side in the browser. The `index.html` file in the `gh-pages` branch can be opened on a local webserver (`python3 -m http.server`). [A hosted version is available here](https://tifu.github.io/truck_backer_upper/).

## Development

This project is written in [TypeScript](http://www.typescriptlang.org/), a statically typed superset of JavaScript.

I recommend [Visual Studio Code](https://code.visualstudio.com/) (crossplatform and opensource) for development.
An alternative is [Atom](https://atom.io/) + [atom-typescript](https://atom.io/packages/atom-typescript).


[React](https://facebook.github.io/react/) is used with [JSX](https://facebook.github.io/jsx/) for GUI state handling.

## Setup

```bash
git checkout gh-pages # to set upstream
git checkout master
git worktree add bin gh-pages
npm install
```

Then build via ```npm run build``` or ```npm run prod-build``` and open the `index.html` in the `bin` directory.

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

## Updating binaries

Use `cd bin; git add -A; git commit -m'update binaries'; git push` to update the binaries.
