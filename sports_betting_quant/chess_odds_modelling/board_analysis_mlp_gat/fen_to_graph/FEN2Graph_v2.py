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

def list_attacked_and_defended_squares(fen):
    """
    List all squares being attacked or defended by each piece.
    """
    board = chess.Board(fen)
    attacked_squares = {}
    defended_squares = {}

    for square in chess.SQUARES:
        attackers = board.attackers(chess.WHITE, square)
        defenders = board.attackers(chess.BLACK, square)

        if attackers:
            attacked_squares[chess.square_name(square)] = [chess.square_name(sq) for sq in attackers]
        if defenders:
            defended_squares[chess.square_name(square)] = [chess.square_name(sq) for sq in defenders]

    return attacked_squares, defended_squares

def list_attacked_and_defended_pieces(fen):
    """
    List all pieces being attacked or defended by each piece.
    """
    board = chess.Board(fen)
    piece_map = board.piece_map()
    attacked_pieces = {}
    defended_pieces = {}

    for square, piece in piece_map.items():
        attackers = board.attackers(chess.WHITE, square)
        defenders = board.attackers(chess.BLACK, square)

        if attackers:
            attacked_pieces[chess.square_name(square)] = [
                (chess.square_name(sq), piece_map[sq].symbol()) for sq in attackers if sq in piece_map
            ]
        if defenders:
            defended_pieces[chess.square_name(square)] = [
                (chess.square_name(sq), piece_map[sq].symbol()) for sq in defenders if sq in piece_map
            ]

    return attacked_pieces, defended_pieces

# Example FEN string
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Display the chessboard
display_chess_board(fen)

# Get the piece coordinates
piece_positions = get_piece_coordinates(fen)
print("Piece coordinates:")
for coordinate, piece in piece_positions.items():
    print(f"Piece: {piece} at {coordinate}")

# Get squares attacked/defended
attacked_squares, defended_squares = list_attacked_and_defended_squares(fen)
print("\nSquares being attacked:")
for square, attackers in attacked_squares.items():
    print(f"{square}: {attackers}")

print("\nSquares being defended:")
for square, defenders in defended_squares.items():
    print(f"{square}: {defenders}")

# Get pieces attacked/defended
attacked_pieces, defended_pieces = list_attacked_and_defended_pieces(fen)
print("\nPieces being attacked:")
for square, attackers in attacked_pieces.items():
    print(f"{square}: {attackers}")

print("\nPieces being defended:")
for square, defenders in defended_pieces.items():
    print(f"{square}: {defenders}")
