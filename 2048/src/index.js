console.log("main")

import '../style/style.scss';

const HEIGHT = 420;
const WIDTH = 420;

const PADDING = 20;
const TILE_HEIGHT = 80;
const TILE_WIDTH = 80;

var d3 = require('d3');

var main = d3.select('#game')
    .attr('height', HEIGHT)
    .attr('width', WIDTH);

// add tiles to main
for (var i = 0; i<16; i++) {
    let row = Math.floor(i / 4);
    let col = i % 4;
    let value = 0
    
    main.append('rect')
        .attr('class', 'tile')
        .attr('x', PADDING + col * (PADDING + TILE_WIDTH))
        .attr('y', PADDING + row * (PADDING + TILE_HEIGHT))
        .attr('rx', 10);
}

var values = [
    0, 2, 4, 8,
    16, 32, 64, 128,
    256, 512, 1024, 2048,
    4096, 4096*2, 0, 0
]

function color(n) {
    let norm = Math.log2(n) / 12;
    return d3.interpolateGreens(norm);
}

function draw_tiles(t) {
    let tiles = main.selectAll(".tile")
    tiles.data(t)
        .attr('fill', function (d) { return color(d) });
    
    console.log(tiles);
    console.log("Colored the tiles");
    for (var i=0; i<16; i++) {

    }
}

draw_tiles(values)

window.draw_tiles = draw_tiles