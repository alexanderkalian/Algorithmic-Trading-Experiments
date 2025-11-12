import pandas as pd
import numpy as np
import chess
from tqdm import tqdm


filename = 'stockfish_scores_depth_20_interim.csv'
df = pd.read_csv(filename)


def fen_to_matrix(fen):
    '''
    Converts a FEN string to an 8x8 matrix using python-chess.

    Args:
        fen (str): The FEN string of the chess position.

    Returns:
        np.ndarray: An 8x8 numpy array representing the board.
    '''
    # Mapping pieces to numbers
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6  # Black pieces
    }

    # Initialize a chess board from the FEN string
    board = chess.Board(fen)

    # Initialize an empty 8x8 matrix
    board_matrix = np.zeros((8, 8), dtype=int)

    # Loop through squares and populate the matrix
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            board_matrix[row, col] = piece_map[piece.symbol()]

    return board_matrix

# Example usage
if __name__ == '__main__':
    
    output_file = 'interim_dataset.csv'
    with open(output_file, 'w') as f:
        for n in range(64):
            f.write(f'x{n},')
        f.write('y')
    
    for fen, score in tqdm(zip(list(df['FEN']), list(df['stockfish_score'])), total=len(list(df['FEN']))):
        
        board_matrix = fen_to_matrix(fen)
        board_vector = board_matrix.flatten()
        
        with open(output_file, 'a') as f:
            f.write('\n')
            for i in board_vector:
                f.write(f'{i},')
            f.write(str(score))
        


