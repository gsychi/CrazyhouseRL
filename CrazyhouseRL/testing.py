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

#make a move (this creates any move, so you need to check if the move is legal)
board.push(chess.Move.from_uci("f2f4"))
board.push(chess.Move.from_uci("e7e5"))
board.push(chess.Move.from_uci("g2g4"))
board.push(chess.Move.from_uci("d8h4"))
print(board)
print(board.is_checkmate())


