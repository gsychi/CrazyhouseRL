import chess.variant
import torch
import numpy as np

#explore the crazyhouse variant

#Declare the board
board = chess.variant.CrazyhouseBoard()

#Print the board
print(board)

#Print the legal moves
print(board.legal_moves)

#Check if a move is legal
print(chess.Move.from_uci("e2e4") in board.legal_moves)

#make a move (this creates any move, so you need to check if the move is legal)
board.push(chess.Move.from_uci("g2g4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g4g5"))
board.push(chess.Move.from_uci("e5e4"))
board.push(chess.Move.from_uci("g5g6"))
board.push(chess.Move.from_uci("e4e3"))
board.push(chess.Move.from_uci("g6h7"))
board.push(chess.Move.from_uci("d7d5"))
board.push(chess.Move.from_uci("h7g8q"))

#Print the legal moves
print(board.legal_moves)

haha = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
print(haha)
#work with numpy and then convert to tensor at very end.

#Print the board
print(board)