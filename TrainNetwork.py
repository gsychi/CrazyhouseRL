import chess.variant
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data_utils
from ChessEnvironment import ChessEnvironment
from ChessConvNet import ChessConvNet
from MyDataset import MyDataset
import ActionToArray

def predictions(outputs):
    listOfMoves = []
    newBoard = ChessEnvironment()
    for i in range(len(outputs)):
        if newBoard.result != 2:
            listOfMoves.append('0000')
        else:
            legalMoves = ActionToArray.legalMovesForState(newBoard.arrayBoard, newBoard.board)
            evaluationScores = ActionToArray.moveEvaluations(
                ActionToArray.legalMovesForState(newBoard.arrayBoard, newBoard.board), newBoard.arrayBoard,
                outputs[i])
            move = legalMoves[np.argmax(evaluationScores)]
            newBoard.makeMove(move)
            newBoard.gameResult()
            listOfMoves.append(move)
    return listOfMoves

# inputs and outputs are numpy arrays. This method of checking accuracy only works with imported games.
# if it's not imported, accuracy will never be 100%, so it will just output the trained network after 10,000 epochs.
def trainNetwork(states, outputMoves, EPOCHS=10000, BATCH_SIZE=1000, LR=0.001, loadDirectory = 'TwelveTeenWeights.pt', saveDirectory='newWeights.pt', OUTPUT_ARRAY_LEN=4504, THRESHOLD_FOR_SAVE=100):

    states = torch.from_numpy(states)
    outputMoves = torch.from_numpy(outputMoves)

    boards, actions = states, outputMoves

    data = MyDataset(boards, actions)

    trainLoader = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=data, batch_size=len(boards), shuffle=False)
    # to create a prediction, create a new dataset with input of the states, and output should just be np.zeros()

    # TRAINING!
    model = ChessConvNet(OUTPUT_ARRAY_LEN).double()
    try:
        model = torch.load(loadDirectory)
    except:
        print("File not found!")

    criterion = nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_step = len(trainLoader)

    trainNotFinished = True
    for epoch in range(EPOCHS):
        if trainNotFinished:
            for i, (images, labels) in enumerate(trainLoader):
                images = images.to('cpu')
                labels = labels.to('cpu')

                # print(images.shape)

                # Forward pass
                outputMoves = model(images)
                loss = criterion(outputMoves, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 1 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))

                # Test the model
                model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
                answers = np.argmax(actions.numpy(), axis=1)
                with torch.no_grad():
                    for images, labels in testLoader:
                        images = images.to('cpu')
                        labels = labels.to('cpu')
                        outputMoves = model(images)
                        _, predicted = torch.max(outputMoves.data, 1)

                        # print expectations vs reality
                        print("MAX", np.amax(outputMoves.numpy()))
                        print("MIN", np.amin(outputMoves.numpy()))

                        correct = (predicted.numpy() == answers).sum()
                        acc = 100 * (correct / len(answers))
                        print("argmax prediction: ", acc, "% correct.")

                        if epoch % 200 == 100:
                            newBoard = ChessEnvironment()
                            for i in range(len(outputMoves.numpy())):
                                if newBoard.result == 2:
                                    move = ActionToArray.moveArrayToString(outputMoves.numpy()[i].reshape((1, 4504)),
                                                                           newBoard.arrayBoard, newBoard.board,
                                                                           newBoard.whiteCaptivePieces, newBoard.blackCaptivePieces,
                                                                           newBoard.plies)
                                    print("NN PREDICTED MOVE: ", move)

                                    # See if the evaluation score matches up with the argmax function!
                                    legalMoves = ActionToArray.legalMovesForState(newBoard.arrayBoard, newBoard.board)
                                    evaluationScores = ActionToArray.moveEvaluations(
                                        ActionToArray.legalMovesForState(newBoard.arrayBoard, newBoard.board), newBoard.arrayBoard,
                                        outputMoves[i])
                                    #print("Evaluation Rankings: ")
                                    print(" = " + legalMoves[np.argmax(evaluationScores)])
                                    #print(ActionToArray.sortEvals(legalMoves, evaluationScores))

                                    newBoard.makeMove(move)
                                    newBoard.gameResult()
                                else:
                                    print(newBoard.gameStatus)

                            print(newBoard.board)
                            newBoard.gameResult()
                            print(newBoard.boardToString())
                            print(newBoard.gameStatus)

                        print(listOfMoves)
                        predicts = predictions(outputMoves)
                        print(predicts)

                        correct = 0
                        for i in range(len(listOfMoves)):
                            if listOfMoves[i] == predicts[i]:
                                correct += 1
                        accuracy = 100 * (correct / len(listOfMoves))
                        print(accuracy, "% correct.")

                        if accuracy >= THRESHOLD_FOR_SAVE:
                            torch.save(model, saveDirectory)
                            print("Updated!")
                            trainNotFinished = False

    # make sure it saves the model regardless.
    torch.save(model, saveDirectory)
    print("Updated!")


board = ChessEnvironment()

# here is a fascinating crazyhouse game...
listOfMoves = ["e2e4", "e7e6", "d2d4", "b8c6", "g1f3", "d7d5", "e4e5", "g8e7", "b1c3", "e7f5", "f1d3", "c6d4",
               "f3d4", "f5d4", "e1g1", "d4f5", "d3f5", "e6f5", "c3d5", "B@e4", "d5e3", "P@h3", "d1d8", "e8d8",
               "f1d1", "c8d7", "d1d7", "d8d7", "Q@d4", "R@d6", "N@c5", "d7e8", "B@b5", "c7c6", "e5d6", "h3g2",
               "N@c7", "e8d8", "R@e8"]

listOfMoves = ["e2e4", "e7e5", "d2d4", "b8c6", "d4e5", "c6e5", "g1f3", "e5f3", "g2f3", "g8f6", "P@e5", "d7d5",
               "e5f6", "d8f6", "b1c3", "N@h4", "f1e2", "h4g2", "e1f1", "c8h3", "c3d5", "g2e3", "f1e1", "e3g2",
               "e1d2", "f6d4", "N@d3", "d4d5", "e4d5", "P@d4", "c2c3", "P@e3", "d2c2", "e3f2", "Q@e5", "f8e7",
               "e5g7", "h8f8", "P@e6", "N@f6", "e6f7", "f8f7", "g7f7", "e8f7", "d3e5", "f7f8", "R@h8", "f8g7",
               "h8a8", "Q@f5", "R@e4", "f6e4", "P@h6", "g7f6", "N@e8", "f6e5", "d1d4"]

inputs = np.zeros(1)
outputs = np.zeros(1)

for i in range(len(listOfMoves)):
    state = board.boardToState()
    action = ActionToArray.moveArray(listOfMoves[i], board.arrayBoard)
    #print(ActionToArray.moveArrayToString(action, board.arrayBoard, board.board,
                                          #board.whiteCaptivePieces, board.blackCaptivePieces,
                                          #board.plies))

    #print(board.boardToString())
    #print(board.boardToFEN())
    #print(board.board)
    if board.board.legal_moves.count() != len(ActionToArray.legalMovesForState(board.arrayBoard, board.board)):
        print("ERROR!")

    board.makeMove(listOfMoves[i])
    if i == 0:
        inputs = state
        outputs = action
    else:
        inputs = np.concatenate((inputs, state))
        outputs = np.concatenate((outputs, action))

print(inputs.shape)

print(board.board)

# Now, with this database, we start training the neural network.
trainNetwork(inputs, outputs, saveDirectory="TwelveTeenWeights.pt")
