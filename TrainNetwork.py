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
def trainNetwork(states, outputMoves, EPOCHS=10000, BATCH_SIZE=1000, LR=0.001, loadDirectory = 'none.pt', saveDirectory='network1.pt', OUTPUT_ARRAY_LEN=4504, THRESHOLD_FOR_SAVE=100):

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
        print("Pretrained NN model not found!")

    criterion = nn.PoissonNLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_step = len(trainLoader)

    trainNotFinished = True
    for epoch in range(EPOCHS):
        if trainNotFinished:
            for i, (images, labels) in enumerate(trainLoader):
                images = images.to('cpu')
                labels = labels.to('cpu')

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

                        print(predicted.numpy())
                        print(answers)

                        correct = (predicted.numpy() == answers).sum()
                        acc = 100 * (correct / len(answers))
                        print("argmax prediction: ", acc, "% correct.")

                        if epoch % 2000 == 100:
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

                        if acc >= THRESHOLD_FOR_SAVE:
                            torch.save(model, saveDirectory)
                            print("Updated!")
                            trainNotFinished = False

    # make sure it saves the model regardless.
    torch.save(model, saveDirectory)
    print("Updated!")


listOfMoves = [
                ["e2e4", "e7e6", "d2d4", "b8c6", "g1f3", "d7d5", "e4e5", "g8e7", "b1c3", "e7f5", "f1d3", "c6d4",
                 "f3d4", "f5d4", "e1g1", "d4f5", "d3f5", "e6f5", "c3d5", "B@e4", "d5e3", "P@h3", "d1d8", "e8d8",
                 "f1d1", "c8d7", "d1d7", "d8d7", "Q@d4", "R@d6", "N@c5", "d7e8", "B@b5", "c7c6", "e5d6", "h3g2",
                 "N@c7", "e8d8", "R@e8"],

                ["e2e4", "e7e5", "d2d4", "b8c6", "d4e5", "c6e5", "g1f3", "e5f3", "g2f3", "g8f6", "P@e5", "d7d5",
                 "e5f6", "d8f6", "b1c3", "N@h4", "f1e2", "h4g2", "e1f1", "c8h3", "c3d5", "g2e3", "f1e1", "e3g2",
                 "e1d2", "f6d4", "N@d3", "d4d5", "e4d5", "P@d4", "c2c3", "P@e3", "d2c2", "e3f2", "Q@e5", "f8e7",
                 "e5g7", "h8f8", "P@e6", "N@f6", "e6f7", "f8f7", "g7f7", "e8f7", "d3e5", "f7f8", "R@h8", "f8g7",
                 "h8a8", "Q@f5", "R@e4", "f6e4", "P@h6", "g7f6", "N@e8", "f6e5", "d1d4"],

                ["d2d4", "d7d6", "c1f4", "g8f6", "b1d2", "b8d7", "g1f3", "e7e5", "d4e5", "d6e5", "f3e5", "d7e5",
                 "f4e5", "f6g4", "e5g3", "P@e3", "d2e4", "f8b4", "P@c3", "b4c3", "b2c3", "P@d2", "e4d2", "e3d2",
                 "d1d2", "d8d2", "e1d2", "N@c4", "d2d3", "N@e5", "d3d4", "c7c5", "d4d5", "Q@c6"],

                ["e2e4", "e7e6", "b1c3", "b7b6", "d2d4", "c8b7", "f1d3", "g8f6", "g1e2", "f8b4", "f2f3", "d7d6",
                 "e1g1", "b8d7", "d1e1", "e8g8", "e1g3", "g8h8", "c1g5", "f8g8", "e4e5", "d6e5", "d4e5", "d7e5",
                 "g3e5", "b4d6", "e5e3", "P@h6", "g5h4", "P@g5", "P@e5", "d6e5", "e3e5", "g5h4", "B@d4", "B@d6",
                 "e5e3", "P@g3", "h2g3", "h4g3", "P@e5", "c7c5", "e5f6", "c5d4", "f6g7", "g8g7", "e3d4", "B@c5",
                 "e2g3", "c5d4", "P@f2", "d6g3", "P@h2", "N@h3", "g1h1", "Q@g1", "f1g1", "h3f2"],

                ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "f8c5", "f1c4", "g8f6", "d2d3", "e8g8", "e1g1", "d7d6",
                 "c1g5", "h7h6", "g5f6", "d8f6", "c3d5", "f6d8", "c2c3", "B@e6", "b2b4", "c5b6", "a2a4", "a7a5",
                 "b4b5", "c6e7", "N@f6", "g8h8", "d5e7", "d8e7", "N@d5", "e7f6", "d5f6", "g7f6", "Q@h4", "N@g8",
                 "c4e6", "c8e6", "B@e7", "B@g7", "e7f8", "a8f8", "R@g3", "N@g6", "g3g6", "f7g6", "N@g5", "f6g5",
                 "f3g5", "B@f7", "P@e7", "f8e8", "g5f7", "e6f7", "B@d5", "f7d5", "e4d5", "e8e7", "B@g5", "P@f6",
                 "g5e3", "N@f5", "h4h3", "b6e3", "f2e3", "B@h7", "f1f5", "g6f5", "B@h5", "N@g5", "h3g3", "N@f8",
                 "N@h4", "B@d7", "e3e4", "f5f4", "g3g5", "h6g5", "N@g6", "h7g6", "h4g6", "f8g6", "h5g6", "R@h1",
                 "g1h1", "N@f2", "h1g1", "R@h1", "g1f2", "Q@e3"],

                ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "d7d6", "d2d4", "c6d4", "f3d4", "e5d4", "d1d4", "P@h3",
                 "h1g1", "c8d7", "P@f5", "h3g2", "f1g2", "c7c5", "c3d5", "c5d4", "N@c7", "d8c7", "d5c7", "e8d8",
                 "c7a8", "Q@a5", "c1d2", "a5d2", "e1d2", "N@c4", "d2d1", "c4b2", "d1e1", "B@a5", "P@d2", "d8c8",
                 "a8b6", "a5b6", "Q@a8", "N@b8", "e4e5", "d6d5", "e5e6", "f7e6", "f5e6", "N@d3", "c2d3", "b2d3",
                 "e1f1", "P@e2", "f1e2", "c8c7", "N@a6", "c7d6", "a8b8", "P@c7", "b8f8", "g8e7", "N@f7", "d6e6",
                 "g2d5", "e6d5", "B@e4", "d5e4", "Q@f3"],

                ["e2e4", "b7b6", "d2d4", "c8b7", "f1d3", "g8f6", "d1e2", "b8c6", "c2c3", "e7e5", "d4d5", "c6e7",
                 "g1f3", "e7g6", "c1g5", "f8e7", "g5f6", "e7f6", "e1g1", "g6f4", "e2d2", "B@g4", "N@e1", "e8g8",
                 "b1a3", "d7d6", "a1c1", "c7c6", "d5c6", "b7c6", "P@e3", "f4g2", "g1g2", "d6d5", "e4d5", "c6d5",
                 "P@e4", "g4f3", "e1f3", "P@g4", "f3g1", "d5e4", "f2f3", "e4d3", "f3g4", "d3f1", "g2f1", "d8d2",
                 "N@e2", "R@f2", "f1f2", "N@d3", "f2f3", "Q@f2", "f3e4", "d3c5"],

                ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "g8e7", "f1d3", "b8c6", "c2c3", "e7f5", "g1e2", "f5h4",
                 "e1g1", "h4g2", "g1g2", "P@e4", "d3b5", "c8d7", "e2f4", "c6e5", "b5d7", "d8d7", "f2f3", "e4f3",
                 "g2g1", "P@e2", "f4e2", "f3e2", "d1e2", "B@d3", "e2e5", "N@h3", "g1g2", "d3f1", "g2f1", "R@g1",
                 "f1e2", "g1c1", "N@e3", "B@f4", "e5h5", "h3g1", "e2d2", "f4e3", "d2e3", "N@c4", "e3f2", "P@c2",
                 "B@d3", "c2b1q", "a1b1", "N@h3", "f2g3", "c1b1", "d3b1", "f8d6", "P@e5", "d6e5", "d4e5", "R@g2",
                 "g3g2", "c4e3", "g2g3", "P@f4", "g3h4", "e3g2", "h4g4", "h3f2", "g4g5", "h7h6", "h5h6", "g1f3",
                 "g5h5", "h8h6"]
                ]

inputs = np.zeros(1)
outputs = np.zeros(1)

for j in range(len(listOfMoves)):
    board = ChessEnvironment()
    for i in range(len(listOfMoves[j])):
        state = board.boardToState()
        action = ActionToArray.moveArray(listOfMoves[j][i], board.arrayBoard)
        if board.board.legal_moves.count() != len(ActionToArray.legalMovesForState(board.arrayBoard, board.board)):
            print("ERROR!")

        board.makeMove(listOfMoves[j][i])
        if j == 0:
            if i == 0:
                inputs = state
                outputs = action
            else:
                inputs = np.concatenate((inputs, state))
                outputs = np.concatenate((outputs, action))
        else:
            seenBefore = False
            for k in range(len(inputs)):
                if np.sum(abs(inputs[k].flatten()-state.flatten())) == 0:
                    seenBefore = True

            if not seenBefore:
                inputs = np.concatenate((inputs, state))
                outputs = np.concatenate((outputs, action))

            # otherwise, do something. but im too lazy.

    print(inputs.shape)

    print(board.board)
    board.gameResult()
    print(board.gameStatus)

# Now, with this database, we start training the neural network.
trainNetwork(inputs, outputs, loadDirectory="supervised.pt", saveDirectory="supervised.pt", EPOCHS=500)
