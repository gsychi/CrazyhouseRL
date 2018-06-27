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
import ActionToArray


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

    def __init__(self):

        self.dictionary = {
            # 'string' = n position. Let this string be the FEN of the position.
        }
        self.childrenMoveNames = []     # a 2D list, each directory may be of different size, stores name of moves
        self.childrenStateSeen = []     # a 2D list, each directory may be of different size, will use np.asarray
        self.childrenStateWin = []      # a 2D list, each directory may be of different size, will use np.asarray
        self.childrenNNEvaluation = []  # a 2D list, each directory may be of different size, will use np.asarray

    # This adds information into the MCTS database

    def printInformation(self):
        print(self.childrenMoveNames)
        print(self.childrenStateSeen)
        print(self.childrenStateWin)
        print(self.childrenNNEvaluation)

    def addPositionToMCTS(self, string, legalMoves):
        self.dictionary[string] = len(self.dictionary)
        self.childrenMoveNames.append(legalMoves)
        self.childrenStateSeen.append(zeroList(len(legalMoves)))
        self.childrenStateWin.append(zeroList(len(legalMoves)))
        self.childrenNNEvaluation.append(zeroList(len(legalMoves)))


tempBoard = ChessEnvironment()
treeSearch = MCTS()

treeSearch.printInformation()
treeSearch.addPositionToMCTS(tempBoard.boardToString(), ActionToArray.legalMovesForState(tempBoard.arrayBoard, tempBoard.board))
treeSearch.printInformation()









