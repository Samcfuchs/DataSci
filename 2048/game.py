import numpy as np
# Convention: axis 0 is rows, axis 1 is columns

class Game(object):

    def __init__(self, size=4, four_rate=0.10):
        self.size = size
        self.four_rate = four_rate
        self.grid = np.zeros((size, size), dtype=int)

    """ Determine the random value of a new tile """
    def new(self):
        r = np.random.random()
        if r < self.four_rate:
            return 4
        else:
            return 2

    """ Set up the game and populate the board """
    def start(self):
        squares = np.random.randint(0, self.size, size=(2,2))
        self.grid[tuple(squares[0])] = self.new()
        self.grid[tuple(squares[1])] = self.new()

        self.score = 0

    # Axis: what axis to move on    
    # L/R : Rows : axis 0
    # U/D : Cols : axis 1
    # 
    # Direction: which way to go
    # 1: D/R
    # -1: U/L
    """ Execute the specified move """
    def move(self, axis, direction):
        #print("Update Matrix")
        g = self.grid.T if axis else self.grid
        for i,row in enumerate(g):
            row = row[::direction]
            #print(row)

            # Merge everything right
            merged = self.merge_left(row[::-1])[::-1]

            # Now shift everything right
            r = [0] * (self.size - len(merged)) + merged
            
            
            g[i] = r[::direction]
        
        self.grid = g.T if axis else g
    
    """ Add a tile in an empty space """
    def populate(self):
        empty_indices = np.transpose((self.grid==0).nonzero())
        if len(empty_indices) == 0:
            print("Game Over")
            return 1
        
        selection = np.random.choice(empty_indices)
        self.grid[selection] = self.new()
        return 0

    #@staticmethod
    """ Merge the elements of a list toward the left. Removes zeros """
    def merge_left(self, row):
        skip = False
        merged = []
        row = [v for v in row if v]
        for i in range(len(row)):
            if skip or row[i] == 0:
                #print("Skipped", i)
                skip = False
                continue

            if i == len(row)-1:
                #print("Left", i)
                merged.append(row[i])
                continue

            if row[i] == row[i+1]:
                merged.append(row[i]*2)
                skip = True
                self.score += row[i] * 2
                #print(f"Merged {i} and {i+1}")
                continue
            
            #print("Left", i)
            merged.append(row[i])

        return merged
    
    """ Display the board in the terminal """
    def display(self):
        print("="*10)
        print(self.grid)
        print("="*10)
    
    """ Execute a full turn """
    def turn(self, move):
        self.move(*move)
        self.populate()
        self.display()


if __name__ == "__main__":
    arrow = { "l": (0, -1), "r": (0, 1), "u": (1, -1), "d": (1, 1) }

    game = Game()
    game.start()
    game.turn(arrow['l'])

    


    
