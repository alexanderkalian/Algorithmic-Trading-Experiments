import chess
import chess.pgn
from io import StringIO
import torch
from chess_transformers.play import load_model
from chess_transformers.configs import import_config

# Define the PGN text directly in the script
PGN_TEXT = """
[Event "Sample Game"]
[Site "?"]
[Date "2025.01.06"]
[Round "?"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3
"""

def parse_pgn_to_fens(pgn_text):
    """
    Parse PGN text and extract FENs for each move in the game.
    :param pgn_text: PGN content as a string.
    :return: List of FENs representing game states.
    """
    fens = []
    pgn_file = StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_file)

    if game is None:
        raise ValueError("No valid games found in the PGN text.")

    board = game.board()
    for move in game.mainline_moves():
        fens.append(board.fen())  # Record FEN before the move
        board.push(move)

    return fens

def extract_model_inputs_from_fen(fen):
    """
    Extract the required inputs for the model from a FEN string.
    :param fen: FEN string representing the board state.
    :return: A tuple of (turn, castling rights, board positions).
    """
    board = chess.Board(fen)

    # Extract castling rights
    white_kingside = int(board.has_kingside_castling_rights(chess.WHITE))
    white_queenside = int(board.has_queenside_castling_rights(chess.WHITE))
    black_kingside = int(board.has_kingside_castling_rights(chess.BLACK))
    black_queenside = int(board.has_queenside_castling_rights(chess.BLACK))

    # Extract board positions as a flat array
    board_positions = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        board_positions.append(piece.piece_type if piece else 0)

    # Extract turn (1 for white, 0 for black)
    turn = int(board.turn)

    return turn, white_kingside, white_queenside, black_kingside, black_queenside, board_positions

def generate_embeddings(fens, config_name="CT-EFT-85"):
    """
    Generate contextual embeddings for a sequence of FENs using the chess-transformers model.
    :param fens: List of FENs representing game states.
    :param config_name: Name of the configuration for the chess-transformers model.
    :return: Contextual embeddings for the FEN sequence.
    """
    # Load model and configuration
    CONFIG = import_config(config_name)
    model = load_model(CONFIG)

    # Ensure the model is in evaluation mode
    model.eval()

    # Prepare inputs
    turns = []
    white_kingside = []
    white_queenside = []
    black_kingside = []
    black_queenside = []
    board_positions = []

    for fen in fens:
        turn, wk, wq, bk, bq, bp = extract_model_inputs_from_fen(fen)
        turns.append(turn)
        white_kingside.append(wk)
        white_queenside.append(wq)
        black_kingside.append(bk)
        black_queenside.append(bq)
        board_positions.append(bp)

    # Convert to tensors
    turns = torch.tensor(turns).unsqueeze(1)  # Shape: (num_moves, 1)
    white_kingside = torch.tensor(white_kingside).unsqueeze(1)
    white_queenside = torch.tensor(white_queenside).unsqueeze(1)
    black_kingside = torch.tensor(black_kingside).unsqueeze(1)
    black_queenside = torch.tensor(black_queenside).unsqueeze(1)
    board_positions = torch.tensor(board_positions)  # Shape: (num_moves, 64)

    # Generate embeddings
    with torch.no_grad():
        initial_embeddings = model._orig_mod.board_encoder(
            turns,
            white_kingside,
            white_queenside,
            black_kingside,
            black_queenside,
            board_positions
        )
    
    # Pass through encoder layers for contextualization
    contextual_embeddings = initial_embeddings
    for layer in model._orig_mod.board_encoder.encoder_layers:
        contextual_embeddings = layer(contextual_embeddings)

    return contextual_embeddings

# Main flow
if __name__ == "__main__":
    # Parse PGN text into FENs
    print("Parsing PGN text...")
    fens = parse_pgn_to_fens(PGN_TEXT)
    print("FENs extracted:", fens)

    # Generate contextual embeddings
    print("\nGenerating contextual embeddings...")
    embeddings = generate_embeddings(fens)

    # Print the embeddings
    print("\nEmbeddings shape:", embeddings.shape)  # Expected: (num_moves, embedding_dim)
    print("Embeddings (first move):", embeddings[0])
    print("Embeddings (last move):", embeddings[-1])
