# This project solves sudoku puzzles using a depth-first, brute-force algorithm

from time import sleep
import tkinter as tk
from tkinter import filedialog, StringVar, Entry
from tensorflow import keras

import copy
import numpy as np
import pandas as pd



""" functions """

""" callback for text objects (event listener) """
def callback(sv, x,y):
    string = sv.get()
    valid = True
    
    # if it was manually inserted
    if not '\n' in sv.get():  
        if len(string) > 1:
            string = string[-1:]
            sv.set(string)
        if not string[-1:].isnumeric():
            valid = False
        elif int(string) == 0:
            string = ''
            sv.set('')
            
    # else it was inputed through a file
    else:
        sv.set(sv.get().replace('\n', ''))
        valid = False
    
    # change font color
    if valid:
        entries[x][y].config(fg = "black")
    else:
        entries[x][y].config(fg = "red")

    return True

""" method for opening sudoku puzzle """
def openPuzzle():
    # lock other buttons
    solvePuzzle["state"] = "disabled"
    resetButton["state"] = "disabled"
    
    fileName = filedialog.askopenfilename(initialdir="/", title = "Select Puzzle",
                                          filetypes = (('Text Files', '.txt'), ('All Files', '*.*')))

    counter = 0
    file = open(fileName, "r")
    while(True):
        # get next line
        line = file.readline().split()
        
        # if its null, end loop
        if not line:
            break
        
        # save line
        for n in range(9):
            numbers[n][counter].set(line[n])
            
        counter += 1
        
    for x in range(9):
        for y in range(9):
            if len(numbers[y][x].get()) > 1:
                numbers[y][x].set(numbers[y][x].get() + '\n')
            if not numbers[y][x].get().isnumeric():
                numbers[y][x].set(numbers[y][x].get() + '\n')
            elif numbers[y][x].get() == '0':
                numbers[y][x].set(' ')
                
    # unlock other buttons
    solvePuzzle["state"] = "normal"
    resetButton["state"] = "normal"

""" a reset button that empties all numbers """
def reset():
    for rows in numbers:
        for num in rows:
            num.set('')

""" Normalize data """
def norm(data):
    return (data / 9) - .5 # normalize data to range 0-1 and -.5 to make the mean close to 0 (-.5 - .5)

""" Denormalize data """
def denorm(data):
    return (data + .5) * 9 # undo the normalization from the method above

""" method to validate puzzle before solving """
def validate():
    # lock all buttons
    openFile["state"] = "disabled"
    solvePuzzle["state"] = "disabled"
    resetButton["state"] = "disabled"
    
    # rows(x,y) x=row, y=1-9
    rows = [[None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None]]
    
    #each row represents a section
    sections = [[None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None],
              [None,None,None,None,None,None,None,None,None]]
    
    solvable = True
    
    for x in range(9):
        columns = [None,None,None,None,None,None,None,None,None]
        for y in range(9):
            if numbers[x][y].get() != '':
                valid = True
                number = int(numbers[x][y].get())
                
                # check if it is a number or length is > 1
                if not numbers[x][y].get().isnumeric or len(numbers[x][y].get()) > 1:
                    solvable = False
                elif number != 0:
                    # check rows for duplicates
                    if rows[y][number-1] is not None: # we want to lock the y height so we can iterate through the row
                        entries[x][y].config(fg = "red")
                        rows[y][number-1].config(fg = "red")
                        valid = False
                        solvable = False
                    else:
                        rows[y][number-1] = entries[x][y]
                        
                    # check columns for duplicates
                    if columns[number-1] is not None:
                        entries[x][y].config(fg = "green")
                        columns[number-1].config(fg = "green")
                        valid = False
                        solvable = False
                    else:
                        columns[number-1] = entries[x][y]
                        
                    # check sections for duplicates
                    sectionNumber = (x // 3) + (y // 3)*3
                    if sections[sectionNumber][number-1] is not None:
                        entries[x][y].config(fg = "blue")
                        sections[sectionNumber][number-1].config(fg = "blue")
                        valid = False
                        solvable = False
                    else:
                        sections[sectionNumber][number-1] = entries[x][y]
                    
                    # turn font to black if its valid
                    if valid:
                        entries[x][y].config(fg = "black")
    
    # if solvable, solve it
    if solvable:
        grid = []
        for i in numbers:
            for num in i:
                if num.get() == '':
                    grid.append(0)
                else:
                    grid.append(int(num.get()))
                
        solve(grid)
        
    # unlock all buttons
    openFile["state"] = "normal"
    solvePuzzle["state"] = "normal"
    resetButton["state"] = "normal"

""" Solve Sudoku Puzzle 1 step at a time """
def solve(game):
    game = np.array(game).reshape((9,9,1))
    game = norm(game)
    
    # loop until the puzzle is solved
    while(1):
        # make prediction. CNN requires 4D array
        output = model.predict(game.reshape(1,9,9,1))
        output = output.squeeze() # remove blanks
        
        # format the prediction into something readable
        pred = np.argmax(output, axis = 1).reshape((9,9))+1
        
        # get probability values for every space
        prob = np.around(np.max(output, axis = 1).reshape((9,9)), 2)
        
        # denormalize game
        game = denorm(game)
        
        # get a mask of only the empty spaces
        mask = copy.copy(game)
        for i,x in enumerate(mask):
            for j,num in enumerate(x):
                if num != 0:
                    mask[i][j] = 1
        
        # if there's no 0's solution is found
        if np.sum(mask) == 81:
            print('hello!')
            break
    
        # apply mask to probability values
        prob = np.ma.masked_array(prob, mask = mask)
        
        print(np.matrix(prob.reshape((9,9))))
        
        # get location of spot with the highest probability value
        index = np.argmax(prob)
        x, y = (index//9), (index%9)
        
        # get predicted value at that location
        val = pred[x][y]
        
        # set predicted value
        game[x][y] = val
        numbers[x][y].set(val)
        entries[x][y].config(fg = "blue")
        root.update()
        sleep(.01)
        
        # normalize
        game = norm(game)
    
    #return solution
    return game
   
""" Check if the number is a valid choice """
def valid(x, y, grid):
    valid = True
    val = grid[x][y]
    
    # check section offset
    sectionX = (x // 3) * 3
    sectionY = (y // 3) * 3
    
    # check for possible numbers
    for i in range(9):
        # check columns
        current = grid[i][y]
        if current != 0:
            if i != x and current == val:
                valid = False
        # check rows
        current = grid[x][i]
        if current != 0:
            if i != y and current == val:
                valid = False
        # check section
        current = grid[sectionX + (i//3)][sectionY + (i%3)]
        if current != 0:
            if sectionX + (i//3) != x and sectionY + (i%3) != y and current == val:
                valid = False
    return valid

""" Main """
# Create gui
root = tk.Tk()
root.title("Sudoku Solver")
root.resizable(False, False)

canvas = tk.Canvas(root, height = 500, width = 500, bg="#A5A5A5")
canvas.create_rectangle(20,20, 480,480, fill = "white", width = 5)
canvas.pack()

#puzzle outline
#columns
canvas.create_line(71,20, 71,480, width = 1)
canvas.create_line(122,20, 122,480, width = 1)
canvas.create_line(173,20, 173,480, width = 4)
canvas.create_line(224,20, 224,480, width = 1)
canvas.create_line(275,20, 275,480, width = 1)
canvas.create_line(328,20, 328,480, width = 4)
canvas.create_line(379,20, 379,480, width = 1)
canvas.create_line(430,20, 430,480, width = 1)

#rows
canvas.create_line(20,71, 480,71, width = 1)
canvas.create_line(20,122, 480,122, width = 1)
canvas.create_line(20,173, 480,173, width = 4)
canvas.create_line(20,224, 480,224, width = 1)
canvas.create_line(20,275, 480,275, width = 1)
canvas.create_line(20,328, 480,328, width = 4)
canvas.create_line(20,379, 480,379, width = 1)
canvas.create_line(20,430, 480,430, width = 1)

#number texts
font = {'font': (None, 20)}
entries = []
numbers = []
for x in range(9):
    # make numbers and entries 2d lists
    row1 = []
    row2 = [] # strange things were happening when trying to use only 1 'row[]'
    numbers.append(row1)
    entries.append(row2)
    for y in range(9):
        # get string var
        sv = StringVar()
        sv.set('')
        sv.trace("w", lambda *_, var = sv, x=x, y=y: callback(var,x,y))
        numbers[x].append(sv)
        
        # get entry and attach string var
        num = Entry( canvas, textvariable = sv, width = 2, **font)
        num.grid(column = x, row = y, padx = (8), pady = (7))
        entries[x].append(num)
        
        #offset grid
        if x == 0:
            num.grid(padx = (34,8))
        if x == 8:
            num.grid(padx = (8, 34))
        if y == 0:
            num.grid(pady = (33,7))
        if y == 8:
            num.grid(pady = (7, 33))

# load model
model = keras.models.load_model('model2.h5')

# button for opening a puzzle
openFile = tk.Button(root, text = "Open File", padx = 10, pady = 10, fg="Black", command = openPuzzle)
openFile.pack()

# button for solving puzzle
solvePuzzle = tk.Button(root, text = "Solve", padx = 10, pady = 10, fg="Black", command = validate)
solvePuzzle.pack()

# button to reset
resetButton = tk.Button(root, text = "Reset", padx = 10, pady = 10, fg="Black", command = reset)
resetButton.pack()

root.mainloop()