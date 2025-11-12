import chess
import matplotlib.pyplot as plt
import numpy as np

# Unicode chess symbols
UNICODE_PIECES = {
    'P': '\u2659', 'p': '\u265F',  # White pawn, Black pawn
    'R': '\u2656', 'r': '\u265C',  # White rook, Black rook
    'N': '\u2658', 'n': '\u265E',  # White knight, Black knight
    'B': '\u2657', 'b': '\u265D',  # White bishop, Black bishop
    'Q': '\u2655', 'q': '\u265B',  # White queen, Black queen
    'K': '\u2654', 'k': '\u265A',  # White king, Black king
}

def display_chess_board(fen):
    """
    Display a chessboard using python-chess and matplotlib with Unicode pieces.
    """
    board = chess.Board(fen)
    piece_map = board.piece_map()
    
    # Create an 8x8 board matrix
    board_matrix = [["" for _ in range(8)] for _ in range(8)]
    for square, piece in piece_map.items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        board_matrix[7 - rank][file] = UNICODE_PIECES[piece.symbol()]  # Flip rank for board orientation

    # Create the chessboard visual
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#f0d9b5", "#b58863"]  # Light and dark square colors
    
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            ax.add_patch(plt.Rectangle((file, rank), 1, 1, color=color))
            piece = board_matrix[rank][file]
            if piece:
                ax.text(file + 0.5, rank + 0.5, piece, ha='center', va='center', fontsize=32, fontname="DejaVu Sans", color='black')

    # Set axis labels and remove ticks
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks(np.arange(8) + 0.5)
    ax.set_yticks(np.arange(8) + 0.5)
    ax.set_xticklabels(["a", "b", "c", "d", "e", "f", "g", "h"])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7", "8"])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.show()

def get_piece_coordinates(fen):
    """
    Extract the piece positions from the FEN string.
    """
    board = chess.Board(fen)
    piece_map = board.piece_map()
    
    piece_coordinates = {}
    for square, piece in piece_map.items():
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        coordinate = (file + 1, rank + 1)  # Convert to 1-based indexing
        piece_coordinates[coordinate] = piece.symbol()
    
    return piece_coordinates

# Example FEN string
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Display the chessboard
display_chess_board(fen)

# Get the piece coordinates
piece_positions = get_piece_coordinates(fen)
print("Piece coordinates:")
for coordinate, piece in piece_positions.items():
    print(f"Piece: {piece} at {coordinate}")
