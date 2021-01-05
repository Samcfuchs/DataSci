module TwentyFortyEight

export Game, init, turn!

moves = Dict("up"=>(-1,0), "down"=>(1,0), "left"=>(0,-1), "right"=>(0,1))

struct Game
    board::Array{Int64, 2}
    score::Int64
end

function init()
    game = Game(zeros(Int, 4,4), 0)
    populate!(game)
    populate!(game)

    return game
end

function merge_left(list)
    original_length = length(list)
    list = list[list .!= 0] # remove zeros
    points = 0

    skip = false
    merged = []
    for i in eachindex(list)
        #print("Item 1 ($(i)): ")
        if skip || list[i] == 0
            skip = false
            #print("Skipped\n")
            continue
        end
        
        if i == length(list)
            push!(merged, list[i])
            #print("Pushed at end\n")
            continue
        end

        if list[i] == list[i+1]
            points += list[i] + list[i+1]
            push!(merged, list[i] + list[i+1])
            skip = true
            #print("Merged\n")
            continue
        end

        push!(merged, list[i])
    end
    #print(merged)
    merged = vcat(merged, repeat([0], original_length - length(merged)))
    return (merged, points)
end

# Axis: which axis to move on
# L/R: axis 0
# U/D: axis 1
#
# Direction: which way to go
# 1: D/R
# -1: U/L
function move!(game::Game, axis, direction)
    board = game.board
    g = axis == 0 ? board : permutedims(board)

    for (i,row) in enumerate(eachrow(g))
        (g[i,:], points) = direction == -1 ? merge_left(row) : reverse(merge_left(reverse(row)))
        game.score += points
    end

    game.board = axis == 0 ? g : permutedims(g)
    return game
end

function populate!(game::Game, fours_rate=0.10)
    empty_indices = findall(game.board .== 0)
    if length(empty_indices) == 0
        # TODO End the game
        return 1
    end

    selection = empty_indices[rand(1:end)]
    game.board[selection] = rand() > fours_rate ? 2 : 4
    return 0
end

function turn!(game::Game, move)
    move!(game, move[1], move[2])
    res = populate!(board)
    return res
end

# Player is a callback to the interface with the player.
function play!(player)
    game = init()
    while true
        move = player(game)
        res = turn!(game, move)
        if res == 1
            println("Game Over")
            return game.score
        end
    end
end

end
