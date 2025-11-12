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

total_positions = 0

# Open the PGN file
pgn_file = 'lichess_LordJedizor_2024-12-26.pgn'  # Replace with your PGN file path
with open(pgn_file) as pgn:
    # Iterate through all games in the PGN file
    game_count = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break  # No more games in the file

        # Print basic information about the game
        game_count += 1
        print(f'Game {game_count}:')
        print('Event:', game.headers.get('Event', 'Unknown'))
        print('White:', game.headers.get('White', 'Unknown'))
        print('Black:', game.headers.get('Black', 'Unknown'))
        print('Result:', game.headers.get('Result', 'Unknown'))
        print('-' * 40)

        # Process moves in the game (optional)
        board = game.board()
        # Go to move 20
        move_number = 20
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            total_positions += 1
            '''
            if i + 1 == move_number:
                break
            '''
        
        # Convert to matrix
        matrix = board_to_matrix(board)
        print('Board at move 20:')
        print(matrix)
        
        # Display the board in the terminal
        print('Board at move 20:')
        print(board)

print(f'Total games processed: {game_count}')

print(f'Total positions: {total_positions}')



