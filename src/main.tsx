import * as React from "react"
import * as $ from "jquery"
//import * as bootstrap from "bootstrap";

// import needed styles. This project's own stylesheet comes last, so that its
// rules win over the framework's on order rather than needing to out-specify
// them; bootstrap was previously imported again below it, which put a second
// copy of every rule after it.
import 'bootstrap/dist/css/bootstrap.css'
import 'rc-slider/assets/index.css';
import './style.css'

import { render } from "react-dom"

import { NeuralNet, NetConfig, LayerConfig } from './neuralnet/net'
import { MSE } from './neuralnet/error'
import { AdalineUnit } from './neuralnet/unit'
import { ActivationFunction, Tanh } from './neuralnet/activation'
import { Vector } from './neuralnet/math'
import { MainComponent } from "./gui/MainComponent";

$(document).ready(() => {
    render(<MainComponent />, document.getElementById("mainContainer"));
})