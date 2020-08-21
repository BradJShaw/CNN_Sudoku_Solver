# CNN_Sudoku_Solver
Attempt to train a Convolutional Neural Net to play sudoku

# Files
CreateModel.py :  This script creates a cnn model using keras and tests its accuracy with 10% of the given data set.
                  This model uses 2 convolutional layers, a flatten layer, and a dense layer. It also uses adam as an optimizer.
CNN_Sudoku.py :   This script launches a small gui that is used to create the sudoku puzzle. when the solve button is pressed, the program will check if its a valid puzzle and                       will begin to solve it.

# How to Use
- obtain a dataset of sudoku puzzles from a source such as kaggle and place it in this project's directory.

- In line 138 in CreateModel.py, rename 'filename' to the name of the dataset

- uncomment line 150 to save model

- run CreateModel.py so it can create a model, let it continue running to calculate the accuracy

- run CNN_Sudoku.py to bring up a gui with the working puzzle solver.

# Last Mentions

Puzzle4 is an example of a puzzle that was not solved correctly with the model I trained.

The Success Rate I achieved was around 98.93% out of 16,460 puzzles
