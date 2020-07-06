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
    console.log(t)
    let tiles = main.selectAll(".tile")
    tiles.data(t).attr('fill', function (d) { return color(d) });
}

function init() {
    console.log('initting')
    fetch('/init', { method: 'POST' })
        .then(response => response.json())
        .then(json => draw_tiles(json.board));
}
d3.select('button#init').on('click', init);

function move(d) {
    console.log('moving', d);
    let b = JSON.stringify({direction: d});
    console.log(b);
    fetch('/move?direction=' + d, { method: 'POST' })
        .then(response => response.json())
        .then(json => draw_tiles(json.board));
    /*
    fetch('/move', {
        method: 'POST', 
        headers: { 'Content-type': 'application/json' },
        body: b
    }).then(response => response.json())
        .then(console.log);
        //.then(json => draw_tiles(json.board));
    */
}


window.draw_tiles = draw_tiles;
window.move = move;
