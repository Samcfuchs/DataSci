include("game.jl")
using .TwentyFortyEight
using Blink

function init()
    w = Window()
    load!(w, "style.css")
    load!(w, "index.html")
    load!(w, "index.js")

    game = TwentyFortyEight.init()
    display(game.board)

    return game, w
end

function show(game)
end

function loop!(game, move)
    res = TwentyFortyEight.turn!(game, move)
    show(game)
    if res == 1
        game_over!(w)
    end
end

function game_over!(w)
    # Display message
end

# Javascript should call loop! with the move
