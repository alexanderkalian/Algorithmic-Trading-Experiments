import chess.pgn
import numpy as np

# Define a function to convert a board into an 8x8 matrix
def board_to_matrix(board):
    piece_map = board.piece_map()  # Get a mapping of pieces on the board
    matrix = np.zeros((8, 8), dtype=int)  # Initialize an 8x8 matrix

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        # Map pieces to positive/negative integers
        piece_value = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6,
        }.get(piece.piece_type, 0)
        # Use positive for white and negative for black
        matrix[7 - row, col] = piece_value if piece.color == chess.WHITE else -piece_value

    return matrix

# Load the PGN file
pgn_file = "lichess_pgn_2024.12.26_canc3111_vs_LordJedizor.uvITU1qX.pgn"  # Replace with your PGN file path
with open(pgn_file) as pgn:
    game = chess.pgn.read_game(pgn)

# Go to move 20
move_number = 20
board = game.board()  # Start with an empty board
for i, move in enumerate(game.mainline_moves()):
    board.push(move)
    if i + 1 == move_number:
        break

# Convert to matrix
matrix = board_to_matrix(board)
print("Board at move 20:")
print(matrix)

# Go to move 20
move_number = 20
board = game.board()
for i, move in enumerate(game.mainline_moves()):
    board.push(move)
    if i + 1 == move_number:
        break

# Display the board in the terminal
print("Board at move 20:")
print(board)
