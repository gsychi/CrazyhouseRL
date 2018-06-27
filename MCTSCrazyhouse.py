"""

Since strings can be turned into arrays, and these arrays can be turned back into the string for the move,
it is now time to create self-learning training data for the computer!

This will be done through MCTS.

At each position, there will be n number of legal moves.

Using the legalMovesFromSquare and MoveablePieces framework, we can create a list of moves for each position.
These nodes will then be updated each time a new playout is finished.

Things to consider:

a) Do lists allow for different sized entries? i.e. can list[0] be an array of 15 numbers but list[1] be an array of 2?

"""
import numpy as np
import chess.variant
from ChessEnvironment import ChessEnvironment
from MyDataset import MyDataset
import ActionToArray
import ChessConvNet
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# Creates a list of
def zeroList(n):
    temp = [0] * n
    return temp

# w stands for # of wins, n stands for number of times node has been visited.
# N stands for number of times parent node is visited, and c is just an exploration constant that can be tuned.
# Q is the evaluation from -1 to 1 of the neural network
# UCT Algorithm used by Alpha Zero.
def PUCT_Algorithm(w, n, c, N, q):
    # Provides a win rate score from 0 to 1
    selfPlayEvaluation = 0.5
    selfPlayEvaluation = np.divide(w, n, out=np.zeros_like(w), where=n != 0)
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * np.sqrt(N)/(1+n)

    PUCT = winRate + exploration
    return PUCT

class MCTS():

    # We will use three lists:
    # seenStates stores the gameStates that have been seen. This is a library.
    # parentSeen stores the times that the current gameState has been seen
    # Each game state corresponds to arrays of the possible moves
    # There are 3 points information stored for each of the children
    # - win count, number of times visited, and neural network evaluation
    # This is helpful because we get to use numpy stuffs.

    def __init__(self, directory):

        self.dictionary = {
            # 'string' = n position. Let this string be the FEN of the position.
        }
        self.childrenMoveNames = []     # a 2D list, each directory may be of different size, stores name of moves
        self.childrenStateSeen = []     # a 2D list, each directory contains numpy array
        self.childrenStateWin = []      # a 2D list, each directory contains numpy array
        self.childrenNNEvaluation = []  # a 2D list, each directory contains numpy array
        try:
            self.neuralNet = torch.load(directory)
        except:
            print("Network not found!")

    # This adds information into the MCTS database

    def printInformation(self):
        print(self.dictionary)
        print(self.childrenMoveNames)
        print(self.childrenStateSeen)
        print(self.childrenStateWin)
        print(self.childrenNNEvaluation)
        print("Parent states in tree: ", len(self.childrenMoveNames))

    def addPositionToMCTS(self, string, legalMoves, arrayBoard, prediction):
        self.dictionary[string] = len(self.dictionary)
        self.childrenMoveNames.append(legalMoves)
        self.childrenStateSeen.append(np.zeros(len(legalMoves)))
        self.childrenStateWin.append(np.zeros(len(legalMoves)))
        evaluations = ActionToArray.moveEvaluations(legalMoves, arrayBoard, prediction)
        self.childrenNNEvaluation.append(evaluations)

    def playout(self):
        tempBoard = ChessEnvironment()

        whiteParentStateDictionary = []
        whiteStateSeen = []
        whiteStateWin = []

        blackParentStateDictionary = []
        blackStateSeen = []
        blackStateWin = []

        while tempBoard.result == 2:

            position = tempBoard.boardToString()
            if position not in self.dictionary:
                # Create a new entry in the tree, if the state is not seen before.
                state = torch.from_numpy(tempBoard.boardToState())
                nullAction = torch.from_numpy(np.zeros((1, 4504)))  # this will not be used, is only a filler
                testSet = MyDataset(state, nullAction)
                generatePredic = torch.utils.data.DataLoader(dataset=testSet, batch_size=len(state), shuffle=False)
                with torch.no_grad():
                    for images, labels in generatePredic:
                        outputs = self.neuralNet(images)
                        self.addPositionToMCTS(tempBoard.boardToString(),
                                                     ActionToArray.legalMovesForState(tempBoard.arrayBoard,
                                                                                      tempBoard.board),
                                                     tempBoard.arrayBoard, outputs)
                        # find and make the preferred move
                        move = self.childrenMoveNames[len(self.childrenStateSeen) - 1][
                            np.argmax(self.childrenNNEvaluation[len(self.childrenStateSeen) - 1])]
                        print(move)
                        tempBoard.makeMove(move)
                        # currently, this is reliant on only the neural network. But it will use PUCT
                        # in the future.

                        actionVector = np.zeros(len(self.childrenMoveNames[len(self.childrenStateSeen) - 1]))
                        actionVector[np.argmax(self.childrenNNEvaluation[len(self.childrenStateSeen) - 1])] = 1
            else:
                # find the directory of the move
                directory = self.dictionary[position]
                move = self.childrenMoveNames[directory][np.argmax(self.childrenNNEvaluation[directory])]
                print(move)
                tempBoard.makeMove(move)

                #the move will have to be indexed correctly based on where thee position is.
                actionVector = np.zeros(len(self.childrenMoveNames[directory]))
                actionVector[np.argmax(self.childrenNNEvaluation[directory])] = 1

            # add this into the actions chosen.
            if tempBoard.plies % 2 == 1:  # white has moved.
                whiteParentStateDictionary.append(position)
                whiteStateSeen.append(actionVector)
                print(self.childrenMoveNames[self.dictionary[position]])
            else:  # black has moved
                blackParentStateDictionary.append(position)
                blackStateSeen.append(actionVector)
                print(self.childrenMoveNames[self.dictionary[position]])

            print(tempBoard.board)
            tempBoard.gameResult()

        print(tempBoard.gameStatus)
        print("PLIES:", tempBoard.plies)

        if tempBoard.result == 1:  # white victory
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i])
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j] * 0)
        if tempBoard.result == -1:  # black victory
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i]*0)
                blackStateWin.append(blackStateSeen[i])
                # this is okay, because if the game is played til checkmate then
                # this ensures that the move count for both sides is equal.
        if tempBoard.result == 0:  # 'tis a tie
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i]*0.5)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j]*0.5)

        # now, add the information into the MCTS database.
        for i in range(len(whiteStateSeen)):
            directory = self.dictionary[whiteParentStateDictionary[i]]
            self.childrenStateSeen[directory] = self.childrenStateSeen[directory] + whiteStateSeen[i]
            self.childrenStateWin[directory] = self.childrenStateWin[directory] + whiteStateWin[i]
        for i in range(len(blackStateSeen)):
            directory = self.dictionary[blackParentStateDictionary[i]]
            self.childrenStateSeen[directory] = self.childrenStateSeen[directory] + blackStateSeen[i]
            self.childrenStateWin[directory] = self.childrenStateWin[directory] + blackStateWin[i]

# Initialize board and the MCTS.

treeSearch = MCTS('NewWeights.pt')
treeSearch.playout()

#treeSearch.printInformation()

# Important: make sure that after the entire data matrix for training is complete,
# multiply the pick up plane (entries from square 320 to 383) by 2.
# This allows training to increase in speed.









