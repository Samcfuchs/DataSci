console.log("main")

import '../style/style.scss';

const HEIGHT = 400;
const WIDTH = 400;

var d3 = require('d3');

var main = d3.select('#game')
    .attr('height', HEIGHT)
    .attr('width', WIDTH);
