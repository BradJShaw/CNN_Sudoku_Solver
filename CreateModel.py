# creates and trains a sudoku solver with CNN

import copy
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape, Activation

from sklearn.model_selection import train_test_split

""" Normalize data """
def norm(data):
    return (data / 9) - .5 # normalize data to range 0-1 and -.5 to make the mean close to 0 (-.5 - .5)

""" Denormalize data """
def denorm(data):
    return (data + .5) * 9 # undo the normalization from the method above

""" Setup Model """
def getModel():
    model = keras.models.Sequential()
    
    # convolutional layer
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    
    # convolutional layer
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    
    # dense layer
    model.add(Flatten())
    model.add(Dense(81*9)) # 81 spots with 9 possible choices
    model.add(Reshape((-1,9)))
    model.add(Activation('softmax'))
    
    # use kera's adam optimizer
    model.compile(optimizer = keras.optimizers.Adam(), loss='sparse_categorical_crossentropy')
    
    return model

""" Prepare Dataset """
def getData(file):
    data = pd.read_csv(filename)
    data['puzzle'] = data['puzzle']

    # format puzzles from string to 9x9x1 matrix
    puzzles = []
    for i in data['puzzle']:
        puzzle = np.array(list(i), dtype=int).reshape((9,9,1))
        puzzle = norm(puzzle)
        puzzles.append(puzzle)
    puzzles = np.array(puzzles)
    print(puzzles[0])
    
    # format solutions from string to a matrix
    solutions = []
    for i in data['solution']:
        solution = np.array(list(i), dtype=int).reshape((81,1))-1
        solutions.append(solution)
    solutions = np.array(solutions)
    print(solutions[0])
    
    #split 3:1
    return train_test_split(puzzles, solutions, test_size = 0.1, random_state = 42)

""" Solve Sudoku Puzzle 1 step at a time """
def prediction(game):
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
            break
        
        # apply mask to probability values
        prob = np.ma.masked_array(prob, mask = mask)
    
        # get location of spot with the highest probability value
        index = np.argmax(prob)
        x, y = (index//9), (index%9)
        
        # get predicted value at that location
        val = pred[x][y]
        
        # set predicted value
        game[x][y] = val
        game = norm(game)
    
    #return solution
    return game
        

""" Calc Accuracy """
def getAcc(Puzzle_test, solutions_test):
    correct = 0
    
    # test all test dataset
    for i,puzzle in enumerate(Puzzle_test):
        # make a prediction
        predict = prediction(copy.copy(puzzle))
        
        # get solution 
        solution = solutions_test[i].reshape((9,9)) + 1
        
        if (solution - predict).sum() == 0:
            correct += 1
        else:
            model.fit(puzzle, solution, epoch = 3)
        
        print('accuracy: ' + str(correct/(i+1)))
        print( str(i+1) + ' out of ' + str(Puzzle_test.shape[0]))
        
    print(correct/Puzzle_test.shape[0])
    
""" Main """
# prep dataset filename
filename = 'sudoku.csv'

# get model
model = getModel()

# get train/test datasets by splitting dataset and normalizes data
Puzzle_train, Puzzle_test, solutions_train, solutions_test = getData(filename)

# train model
model.fit(Puzzle_train, solutions_train, epochs = 3)

# save model
#model.save('MODEL_NAME.h5')

# load model
#model = keras.models.load_model('MODEL_NAME.h5')

# Calc Accuracy
getAcc(Puzzle_test, solutions_test)
