"""
This is quite an important function, as it maps each action that the computer chooses
"""
import numpy as np


# move is a string
def moveArray(move, board):
    placedPlane = np.zeros((5, 8, 8))  # pawn, knight, bishop, rook, queen.
    pickUpPlane = np.zeros((8, 8))
    movePlane = np.zeros((8, 7, 8, 8))  # direction (N, NE, E, SE, S, SW, W, NW), squares.
    knightMovePlane = np.zeros((8, 8, 8))  # direction ([1, 2],[2, 1],[2, -1],[1, -2],[-1, -2],[-2, -1],[-2, 1],[-1, 2])
    underPromotion = np.zeros((3, 8))  # this can be a 8x8 plane, but for now we will not. Knight, Bishop, Rook

    if "PRNBQ".find(move[0]) == -1:
        # print("NORMAL MOVE")
        rowNames = "abcdefgh"
        pickUpPlane[8 - int(move[1])][int(rowNames.find(move[0]))] = 1

        # identify how far the piece has moved, and how far it will be moving.
        if "PBRQK".find(board[8 - int(move[1])][int(rowNames.find(move[0]))]) != -1:
            # print("its a", board[8 - int(move[1])][int(rowNames.find(move[0]))])
            if len(move) == 5:
                directory = "nbr".find(move[4].lower())  # .lower() just in case
                underPromotion[directory][int(rowNames.find(move[2]))] = 1
            else:
                rowMovement = int(move[3]) - int(move[1])  # positive = north, negative = south [NORTH = 0, SOUTH = 4]
                columnMovement = int(rowNames.find(move[2])) - int(
                    rowNames.find(move[0]))  # positive = east, negative = west [EAST = +1, WEST = -1]

                directory = 999
                if rowMovement > 0:  # NORTH
                    directory = 0
                    if columnMovement > 0:  # NORTH-EAST
                        directory = 1
                    if columnMovement < 0:  # NORTH-WEST
                        directory = 7
                elif rowMovement < 0:
                    directory = 4
                    if columnMovement > 0:  # SOUTH-EAST
                        directory = 3
                    if columnMovement < 0:  # SOUTH-WEST
                        directory = 5
                elif rowMovement == 0:
                    if columnMovement > 0:  # EAST:
                        directory = 2
                    if columnMovement < 0:  # WEST
                        directory = 6

                magnitude = max(rowMovement, columnMovement) - 1
                movePlane[directory][magnitude][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
        elif board[8 - int(move[1])][int(rowNames.find(move[0]))] == " ":
            print("not legal move")  # don't do anything
        else:
            # print("its a knight")
            rowMovement = int(move[3]) - int(move[1])  # positive = north, negative = south [NORTH = 0, SOUTH = 4]
            columnMovement = int(rowNames.find(move[2])) - int(
                rowNames.find(move[0]))  # positive = east, negative = west [EAST = +1, WEST = -1]
            directory = 999
            if rowMovement == 1:
                if columnMovement == 2:
                    directory = 0
                elif columnMovement == -2:
                    directory = 3
            elif rowMovement == 2:
                if columnMovement == 1:
                    directory = 1
                elif columnMovement == -1:
                    directory = 2
            elif rowMovement == -2:
                if columnMovement == 1:
                    directory = 6
                elif columnMovement == -1:
                    directory = 5
            elif rowMovement == -1:
                if columnMovement == 2:
                    directory = 7
                elif columnMovement == -2:
                    directory = 4
            knightMovePlane[directory][8 - int(move[3])][int(rowNames.find(move[2]))] = 1
    else:
        # print("PLACED PIECE")
        rowNames = "abcdefgh"
        placedPlane["PNBRQ".find(move[0])][8 - int(move[3])][int(rowNames.find(move[2]))] = 1

    # return the result.
    placedPlane = placedPlane.flatten()
    pickUpPlane = pickUpPlane.flatten()
    movePlane = movePlane.flatten()
    knightMovePlane = knightMovePlane.flatten()
    underPromotion = underPromotion.flatten()
    moveToArray = np.concatenate((placedPlane, pickUpPlane, movePlane, knightMovePlane, underPromotion))
    # ARRAY IS 4504 ENTRIES LONG
    return moveToArray.reshape((1, 4504))


# arraytomove = array to string, use reshape...
def MoveArrayToString(array, board):
    placedPlane = array[0, 0:64 * 5]
    pickUpPlane = array[0, 64 * 5:(64 * 5) + (64 * 1)]
    movePlane = array[0, 64 * 6:(64 * 6) + (8 * 7 * 8 * 8)]
    knightMovePlane = array[0, (64 * 6) + (8 * 7 * 8 * 8):(64 * 6) + (8 * 7 * 8 * 8) + (8 * 8 * 8)]
    underPromotion = array[0, -24:]
    print(placedPlane.shape)
    print(pickUpPlane.shape)
    print(movePlane.shape)
    print(knightMovePlane.shape)
    print(underPromotion.shape)
    moveToArray = np.concatenate((placedPlane, pickUpPlane, movePlane, knightMovePlane, underPromotion))
    return (moveToArray).shape

    return 0


testingBoard = [["r", "n", "b", "q", "k", "b", "n", "r"],
                ["p", "p", "p", "p", "p", "p", "p", "p"],
                [" ", " ", " ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " ", " ", " "],
                [" ", " ", " ", " ", " ", " ", " ", " "],
                ["P", "P", "P", "P", "P", "P", "P", "P"],
                ["R", "N", "B", "Q", "K", "B", "N", "R"]]

print(MoveArrayToString(moveArray("e6e5", testingBoard), testingBoard))
