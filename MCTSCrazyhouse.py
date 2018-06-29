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
import datetime

import numpy as np
import copy
import chess.variant
import chess.pgn
import chess
import time
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
    selfPlayEvaluation = np.divide(w, n, out=np.zeros_like(w), where=n != 0)
    for i in range(len(selfPlayEvaluation)):
        if n[i] == 0:
            selfPlayEvaluation[i] = 0.5
    nnEvaluation = q
    winRate = (nnEvaluation + selfPlayEvaluation) / 2

    # Exploration
    exploration = c * np.sqrt(N) / (1 + n)

    PUCT = winRate + exploration
    return PUCT

def noiseEvals(nnEvals, bounds):
    # this diversifies training data during its self-play games, in order to ensure that the computer looks at a lot of
    # different positions.
    noise = (np.random.rand(len(nnEvals))*2*bounds)-(bounds)
    return noise + nnEvals


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
        self.childrenMoveNames = []  # a 2D list, each directory may be of different size, stores name of moves
        self.childrenStateSeen = []  # a 2D list, each directory contains numpy array
        self.childrenStateWin = []  # a 2D list, each directory contains numpy array
        self.childrenNNEvaluation = []  # a 2D list, each directory contains numpy array
        try:
            self.neuralNet = torch.load(directory)
        except:
            print("Network not found!")
        self.nameOfNetwork = directory[0:-3]

    # This adds information into the MCTS database

    def printInformation(self):
        print(self.dictionary)
        print(self.childrenMoveNames)
        print(self.childrenStateSeen)
        print(self.childrenStateWin)
        print(self.childrenNNEvaluation)
        print("Parent states in tree: ", len(self.childrenMoveNames))

    def printSize(self):
        print("Size: ", len(self.childrenMoveNames))

    def addPositionToMCTS(self, string, legalMoves, arrayBoard, prediction):
        self.dictionary[string] = len(self.dictionary)
        self.childrenMoveNames.append(legalMoves)
        self.childrenStateSeen.append(np.zeros(len(legalMoves)))
        self.childrenStateWin.append(np.zeros(len(legalMoves)))

        # should scale the evaluations from 0 to 1.
        evaluations = ActionToArray.moveEvaluations(legalMoves, arrayBoard, prediction)
        if len(evaluations) > 0:
            minVal = np.amin(evaluations)  # pretend this is -2
            maxVal = np.amax(evaluations)  # pretend this is 10
            if minVal != maxVal:
                multiplier = maxVal - minVal  # then multiplier is 12. max = 10/12, min = -2/12
                try:
                    center = minVal / multiplier
                except ZeroDivisionError:
                    center = 0
                evaluations = (evaluations / multiplier)
                evaluations = evaluations - center
            else:
                evaluations = evaluations / maxVal

        self.childrenNNEvaluation.append(evaluations)


        # NEED TO ADD ACTUALLY A PAWN

    def playout(self, round,
                explorationConstant=0.15,   # lower? will test more.
                notFromBeginning=False, arrayBoard=0, pythonBoard=0, plies=0, wCap=0, bCap=0,
                actuallyAPawn=0,
                noise=True,
                printPGN=True): # Here is the information just for starting at a different position

        if printPGN:
            PGN = chess.pgn.Game()
            PGN.headers["Event"] = "Playout"
            PGN.headers["Site"] = "Nayoung's Home"
            PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d')
            PGN.headers["Round"] = round
            PGN.headers["White"] = "Network: " + self.nameOfNetwork
            PGN.headers["Black"] = "Network: " + self.nameOfNetwork
            PGN.headers["Variant"] = "Crazyhouse"

        whiteParentStateDictionary = []
        whiteStateSeen = []
        whiteStateWin = []

        blackParentStateDictionary = []
        blackStateSeen = []
        blackStateWin = []

        tempBoard = ChessEnvironment()
        if notFromBeginning:
            tempBoard.arrayBoard = arrayBoard
            tempBoard.board = pythonBoard
            tempBoard.plies = plies
            tempBoard.whiteCaptivePieces = wCap
            tempBoard.blackCaptivePieces = bCap
            tempBoard.actuallyAPawn = actuallyAPawn
            tempBoard.updateNumpyBoards()

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
                        if noise:
                            noiseConstant = 0.6/(2.5*(1+tempBoard.plies))
                        else:
                            noiseConstant = 0

                        if len(self.childrenStateWin) > 0:
                            index = np.argmax(PUCT_Algorithm(self.childrenStateWin[len(self.childrenStateSeen) - 1],
                                           self.childrenStateSeen[len(self.childrenStateSeen) - 1], explorationConstant,
                                           np.sum(self.childrenStateSeen[len(self.childrenStateSeen) - 1]),
                                           noiseEvals(self.childrenNNEvaluation[len(self.childrenStateSeen) - 1],
                                                      noiseConstant)))
                        else:
                            index = 0

                        move = self.childrenMoveNames[len(self.childrenStateSeen) - 1][index]

                        if chess.Move.from_uci(move) not in tempBoard.board.legal_moves:
                            print("Not legal move.")
                            # play a random move
                            move = self.childrenMoveNames[len(self.childrenStateSeen) - 1][0]

                        #print(move)
                        tempBoard.makeMove(move)

                        actionVector = np.zeros(len(self.childrenMoveNames[len(self.childrenStateSeen) - 1]))
                        actionVector[index] = 1
            else:
                # find the directory of the move
                directory = self.dictionary[position]

                if noise:
                    noiseConstant = 0.6 / (2.5 * (1 + tempBoard.plies))
                else:
                    noiseConstant = 0
                index = np.argmax(PUCT_Algorithm(self.childrenStateWin[directory],
                                                 self.childrenStateSeen[directory], explorationConstant,
                                                 np.sum(self.childrenStateSeen[directory]),
                                                 noiseEvals(self.childrenNNEvaluation[directory], noiseConstant)
                                                 ))
                move = self.childrenMoveNames[directory][index]

                #print(move)
                tempBoard.makeMove(move)


                # the move will have to be indexed correctly based on where the position is.
                actionVector = np.zeros(len(self.childrenMoveNames[directory]))
                actionVector[index] = 1

            if printPGN:
                if tempBoard.plies == 1:
                    node = PGN.add_variation(chess.Move.from_uci(move))
                else:
                    node = node.add_variation(chess.Move.from_uci(move))

            # add this into the actions chosen.
            if tempBoard.plies % 2 == 1:  # white has moved.
                whiteParentStateDictionary.append(position)
                whiteStateSeen.append(actionVector)
            else:  # black has moved
                blackParentStateDictionary.append(position)
                blackStateSeen.append(actionVector)

            #print(tempBoard.board)
            tempBoard.gameResult()

        if tempBoard.result == 1:  # white victory
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i])
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j] * 0)
            if printPGN:
                PGN.headers["Result"] = "1-0"
        if tempBoard.result == -1:  # black victory
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i] * 0)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j])
                # this is okay, because if the game is played til checkmate then
                # this ensures that the move count for both sides is equal.
            if printPGN:
                PGN.headers["Result"] = "0-1"
        if tempBoard.result == 0:  # 'tis a tie
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i] * 0.5)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j] * 0.5)
            if printPGN:
                PGN.headers["Result"] = "1/2-1/2"

        # Print PGN and final state
        if printPGN:
            print("PGN: ")
            print(PGN)

        # now, add the information into the MCTS database.
        for i in range(len(whiteStateSeen)):
            directory = self.dictionary[whiteParentStateDictionary[i]]
            self.childrenStateSeen[directory] = self.childrenStateSeen[directory] + whiteStateSeen[i]
            self.childrenStateWin[directory] = self.childrenStateWin[directory] + whiteStateWin[i]
        for i in range(len(blackStateSeen)):
            directory = self.dictionary[blackParentStateDictionary[i]]
            self.childrenStateSeen[directory] = self.childrenStateSeen[directory] + blackStateSeen[i]
            self.childrenStateWin[directory] = self.childrenStateWin[directory] + blackStateWin[i]

        if printPGN:
            print(tempBoard.board)
            self.printSize()

    def trainingPlayoutFromBeginning(self, runs, printPGN):
        for i in range(1, runs+1):
            print("GAME", str(i))
            self.playout(str(i), printPGN=printPGN)

    def competitivePlayoutFromBeginning(self, runs, printPGN):
        for i in range(1, runs+1):
            print("GAME", str(i))
            self.playout(str(i), noise=False, printPGN=printPGN)

    def trainingPlayoutsFromPosition(self, runs, sim):
        for i in range(runs):
            tempBoard = copy.deepcopy(sim)
            # playout from a certain position.
            self.playout(str(int(i+1)), notFromBeginning=True, arrayBoard=tempBoard.arrayBoard, pythonBoard=tempBoard.board,
                               plies=tempBoard.plies, wCap=tempBoard.whiteCaptivePieces, noise=True,
                               bCap=tempBoard.blackCaptivePieces, actuallyAPawn=tempBoard.actuallyAPawn, explorationConstant=0.3, printPGN=False)

            #self.printSize()
            #print(self.childrenMoveNames[self.dictionary[sim.boardToString()]])
            #print(self.childrenStateSeen[self.dictionary[sim.boardToString()]])

    def competitivePlayoutsFromPosition(self, runs, sim):
        for i in range(runs):
            tempBoard = copy.deepcopy(sim)
            # playout from a certain position.
            self.playout(str(int(i+1)), notFromBeginning=True, arrayBoard=tempBoard.arrayBoard, pythonBoard=tempBoard.board,
                               plies=tempBoard.plies, wCap=tempBoard.whiteCaptivePieces, explorationConstant=0.3,
                               bCap=tempBoard.blackCaptivePieces, noise=False, actuallyAPawn=tempBoard.actuallyAPawn, printPGN=False)

            #self.printSize()
            print(self.childrenMoveNames[self.dictionary[sim.boardToString()]])
            print(self.childrenStateSeen[self.dictionary[sim.boardToString()]])

    def simulateTrainingGame(self, playouts, round="1"):

        PGN = chess.pgn.Game()
        PGN.headers["Event"] = "Simulated Training Game"
        PGN.headers["Site"] = "Nayoung's Home"
        PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
        PGN.headers["Round"] = round
        PGN.headers["White"] = "Network: " + self.nameOfNetwork
        PGN.headers["Black"] = "Network: " + self.nameOfNetwork
        PGN.headers["Variant"] = "Crazyhouse"

        whiteParentState = np.zeros(1)
        whiteStateSeen = []
        whiteStateWin = []
        whiteStateNames = []

        blackParentState = np.zeros(1)
        blackStateSeen = []
        blackStateWin = []
        blackStateNames = []

        sim = ChessEnvironment()
        while sim.result == 2:
            self.trainingPlayoutsFromPosition(playouts, sim)
            directory = self.dictionary[sim.boardToString()]
            index = np.argmax(
                PUCT_Algorithm(self.childrenStateWin[directory], self.childrenStateSeen[directory], 0.1,
                               np.sum(self.childrenStateSeen[directory]),
                               noiseEvals(self.childrenNNEvaluation[directory], 1.8/(5*(sim.plies+1))))
                # if there are too many weird openings, decrease 0.6 to 0.48, decrease 5 to 4.

                )
            move = self.childrenMoveNames[directory][index]
            moveNames = self.childrenMoveNames[directory]

            actionVector = np.zeros(len(self.childrenMoveNames[directory]))
            actionVector[index] = 1

            if sim.plies == 0:
                whiteParentState = sim.boardToState()
                whiteStateSeen.append(actionVector)
                whiteStateNames.append(moveNames)
            if sim.plies == 1:
                blackParentState = sim.boardToState()
                blackStateSeen.append(actionVector)
                blackStateNames.append(moveNames)
            if sim.plies % 2 == 0 and sim.plies != 0:
                whiteParentState = np.concatenate((whiteParentState, sim.boardToState()))
                whiteStateSeen.append(actionVector)
                whiteStateNames.append(moveNames)
            if sim.plies % 2 == 1 and sim.plies != 1:
                blackParentState = np.concatenate((blackParentState, sim.boardToState()))
                blackStateSeen.append(actionVector)
                blackStateNames.append(moveNames)

            sim.makeMove(move)
            sim.gameResult()
            print(sim.board)

            if sim.plies == 1:
                node = PGN.add_variation(chess.Move.from_uci(move))
            else:
                node = node.add_variation(chess.Move.from_uci(move))

        if sim.result == 1:
            PGN.headers["Result"] = "1-0"
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i])
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j]*0)
        if sim.result == 0:
            PGN.headers["Result"] = "1/2-1/2"
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i]*0.5)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j]*0.5)
        if sim.result == -1:
            PGN.headers["Result"] = "0-1"
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i]*0)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j])
        if sim.result == 2:
            for i in range(len(whiteStateSeen)):
                whiteStateWin.append(whiteStateSeen[i]*0)
            for j in range(len(blackStateSeen)):
                blackStateWin.append(blackStateSeen[j]*0)

        parentStates = np.concatenate((whiteParentState, blackParentState))
        statesSeen = whiteStateSeen + blackStateSeen
        statesWin = whiteStateWin + blackStateWin
        statesNames = whiteStateNames + blackStateNames

        print(PGN)
        
        return parentStates, statesSeen, statesWin, statesNames

    def simulateCompetitiveGame(self, playouts):

        PGN = chess.pgn.Game()
        PGN.headers["Event"] = "Simulated Competitive Game"
        PGN.headers["Site"] = "Nayoung's Home"
        PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
        PGN.headers["Round"] = "1"
        PGN.headers["White"] = "Network: " + self.nameOfNetwork
        PGN.headers["Black"] = "Network: " + self.nameOfNetwork
        PGN.headers["Variant"] = "Crazyhouse"

        # refresh the MCTS tree from scratch initially.

        self.dictionary = {
            # 'string' = n position. Let this string be the FEN of the position.
        }
        self.childrenMoveNames = []  # a 2D list, each directory may be of different size, stores name of moves
        self.childrenStateSeen = []  # a 2D list, each directory contains numpy array
        self.childrenStateWin = []  # a 2D list, each directory contains numpy array
        self.childrenNNEvaluation = []  # a 2D list, each directory contains numpy array

        sim = ChessEnvironment()
        while sim.result == 2:

            # now start looking at variations
            self.competitivePlayoutsFromPosition(playouts, sim)
            directory = self.dictionary[sim.boardToString()]
            index = np.argmax(
                PUCT_Algorithm(self.childrenStateWin[directory], self.childrenStateSeen[directory], 0,
                               np.sum(self.childrenStateSeen[directory]),
                               self.childrenNNEvaluation[directory])
                )
            move = self.childrenMoveNames[directory][index]
            print(move)
            sim.makeMove(move)
            sim.gameResult()

            if sim.plies == 1:
                node = PGN.add_variation(chess.Move.from_uci(move))
            else:
                node = node.add_variation(chess.Move.from_uci(move))

            print(sim.board)

        if sim.result == 1:
            PGN.headers["Result"] = "1-0"
        if sim.result == 0:
            PGN.headers["Result"] = "1/2-1/2"
        if sim.result == -1:
            PGN.headers["Result"] = "0-1"

        print(PGN)

        PGN = chess.pgn.Game()
        PGN.headers["Event"] = "Simulated Competitive Game"
        PGN.headers["Site"] = "Nayoung's Home"
        PGN.headers["Date"] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
        PGN.headers["Round"] = "1"
        PGN.headers["White"] = "Network: " + self.nameOfNetwork
        PGN.headers["Black"] = "Network: " + self.nameOfNetwork
        PGN.headers["Variant"] = "Crazyhouse"

        sim = ChessEnvironment()
        while sim.result == 2:
            self.competitivePlayoutsFromPosition(playouts, sim)
            directory = self.dictionary[sim.boardToString()]
            index = np.argmax(
                PUCT_Algorithm(self.childrenStateWin[directory], self.childrenStateSeen[directory], 0,
                               np.sum(self.childrenStateSeen[directory]),
                               self.childrenNNEvaluation[directory])
                )
            move = self.childrenMoveNames[directory][index]
            print(move)
            sim.makeMove(move)
            sim.gameResult()

            if sim.plies == 1:
                node = PGN.add_variation(chess.Move.from_uci(move))
            else:
                node = node.add_variation(chess.Move.from_uci(move))

            print(sim.board)

    def createTrainingGames(self, numberOfGames, playouts):
        trainingParentStates = np.zeros(1)
        trainingStatesSeen = []
        trainingStatesWin = []
        trainingStatesName = []

        for i in range(numberOfGames):
            newParentStates, \
            newStatesSeen,\
            newStatesWin, \
            newStatesName = self.simulateTrainingGame(playouts, round=str(int(i+1)))

            if i == 0:  # if nothing has been added yet
                trainingParentStates = newParentStates
                trainingStatesSeen = newStatesSeen
                trainingStatesWin = newStatesWin
                trainingStatesName = newStatesName

            if i != 0:
                removeDirectories = []
                for k in range(len(trainingParentStates)):
                    for j in range(len(newParentStates)):
                        if np.sum((abs(trainingParentStates[k].flatten() - newParentStates[j].flatten()))) == 0:
                            # If information is already in dataset, edit existing data
                            trainingStatesWin[k] = trainingStatesWin[k] + newStatesWin[j]
                            trainingStatesSeen[k] = trainingStatesSeen[k] + newStatesSeen[j]
                            removeDirectories.append(j)

                removeDirectories.sort()
                while len(removeDirectories) > 0:
                    index = removeDirectories.pop()
                    newParentStates = np.delete(newParentStates, index, axis=0)
                    del newStatesSeen[index]
                    del newStatesWin[index]
                    del newStatesName[index]

                trainingParentStates = np.concatenate((trainingParentStates, newParentStates), axis=0)
                trainingStatesSeen = trainingStatesSeen + newStatesSeen
                trainingStatesWin = trainingStatesWin + newStatesWin
                trainingStatesName = trainingStatesName + newStatesName

                trainingWinPercentages = []
                # Create win percentage for all moves:
                for i in range(len(trainingStatesWin)): #length of tSW and tSS should be the same
                    newEntry = np.divide(trainingStatesWin[i], trainingStatesSeen[i], out=np.zeros_like(trainingStatesWin[i]),
                                                                                                       where=trainingStatesSeen[i]!=0)
                    trainingWinPercentages.append(newEntry)

        # return the information. trainingWinPercentage shas to be converted to a numpy array of correct shape!
        print("Size of Training Material: ", len(trainingParentStates))
        return trainingParentStates, trainingWinPercentages

            # create trainingData.
# Initialize board and the MCTS.

treeSearch = MCTS('newWeights.pt')

# check to see how fast a playout is.

"""
PLAYOUTS = 10
start = time.time()
treeSearch.competitivePlayoutFromBeginning(PLAYOUTS, True)
end = time.time()

print("\nStatistics:")
print(end-start, "seconds elapsed.")
print("Average time for a game", (end-start)/PLAYOUTS, "seconds.")
"""

newBoard = ChessEnvironment()

# FOR playout, update the legal moves, and should check if position that is currently seen and saved has the
# same length for the legal moves. if not, refresh the stateSeen and stateWin

treeSearch.createTrainingGames(100, 1)

# don't bother with competitive games yet.
# for boardtostring, check if there is en passant square

treeSearch.printSize()


# in the future, the number of playouts at a position can be dependent on how many possible moves there are
# this way, resources are allocated and used when most necessary.

# Important: make sure that after the entire data matrix for training is complete,
# multiply the pick up plane (entries from square 320 to 383) by 2.
# This allows training to increase in speed.
