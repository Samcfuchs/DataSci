from flask import Flask
from flask import render_template, url_for, jsonify, request
from game import Game

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

game=0
@app.route('/init', methods=['POST'])
def init():
    global game
    game = Game()
    game.start()

    # I have no idea why this is necessary but without the list comprehension it
    # breaks.
    data = [int(x) for x in game.grid.flatten()]
    response = jsonify({'board': data})
    return response


directions = { 'U': (1,-1), 'D': (1, 1), 'L': (0,-1), 'R': (0, 1) }

@app.route('/move', methods=['POST'])
def move():
    d = request.args['direction']
    game.move(*directions[d.upper()])
    
    response = jsonify({ 'board': [int(x) for x in game.grid.flatten()] })
    return response
