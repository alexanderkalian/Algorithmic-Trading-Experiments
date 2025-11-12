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

PGN_TEXT = """
[Event "Wijk aan Zee"]
[Site "Wijk aan Zee NED"]
[Date "1999.01.20"]
[Round "4"]
[White "Kasparov, Garry"]
[Black "Topalov, Veselin"]
[Result "1-0"]

1. e4 d6 2. d4 Nf6 3. Nc3 g6 4. Be3 Bg7 5. Qd2 O-O 6. Bh6 Bxh6 7. Qxh6 c5
8. dxc5 Qa5 9. O-O-O Qxc5 10. Nd5 Nxd5 11. Rxd5 Qb4 12. Nf3 Qxe4 13. Ng5 Qf4+
14. Kb1 Qf6 15. Qxh7# 1-0
"""

PGN_TEXT = """
[Event "World Chess Championship"]
[Site "Reykjavik"]
[Date "1972.07.23"]
[Round "6"]
[White "Fischer, Robert James"]
[Black "Spassky, Boris V"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 
8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. Nc3 Bb7 13. a3 Re8 
14. Ba2 Bf8 15. Bg5 h6 16. Bh4 g6 17. Rc1 Rc8 18. cxb5 axb5 19. dxe5 dxe5 
20. Qb3 Qe7 21. Red1 Nc5 22. Qc2 Ne6 23. b4 g5 24. Bg3 Nf4 25. Bxf4 gxf4 
26. Nh4 c5 27. Nf5 Qc7 28. bxc5 Qxc5 29. Qb3 Qc7 30. Nxb5 Qb6 31. Nbd6 Qxb3 
32. Bxb3 Bxd6 33. Nxd6 Rxc1 34. Rxc1 Re7 35. f3 Rd7 36. Nf5 Kh7 37. Rc5 Kg6 
38. Rxe5 Rd3 39. Rb5 Bc8 40. Nh4+ Kh7 41. Bxf7 Rd1+ 42. Kh2 Nd7 43. Be6 Kg7 
44. Bxd7 Bxd7 45. Rb7 Kf6 46. Rb6+ Kg5 47. Ng6 Bb5 48. h4+ Kh5 49. Nxf4+ Kxh4 
50. Rxb5 Rd2 51. Rh5# 1-0
"""

PGN_TEXT2 = """
[Event "World Chess Championship"]
[Site "Reykjavik"]
[Date "1972.07.23"]
[Round "6"]
[White "Fischer, Robert James"]
[Black "Spassky, Boris V"]
[Result "1-0"]

1. e4 e5
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

    # Generate initial embeddings
    with torch.no_grad():
        initial_embeddings = model._orig_mod.board_encoder(
            turns,
            white_kingside,
            white_queenside,
            black_kingside,
            black_queenside,
            board_positions
        )

    print("Initial embeddings shape:", initial_embeddings.shape)

    # Pass through nested encoder layers
    contextual_embeddings = initial_embeddings
    for i, encoder_layer in enumerate(model._orig_mod.board_encoder.encoder_layers):
        print(f"Processing Encoder Layer {i}")
        # Each encoder_layer contains two components: MultiHeadAttention and PositionWiseFCNetwork
        attention_layer = encoder_layer[0]  # MultiHeadAttention
        positionwise_layer = encoder_layer[1]  # PositionWiseFCNetwork

        # Prepare additional arguments for attention
        key_value_sequences = contextual_embeddings  # In self-attention, keys and values are the same as queries
        key_value_sequence_lengths = torch.full(
            (contextual_embeddings.shape[0],), contextual_embeddings.shape[1], dtype=torch.int32
        )

        # Apply MultiHeadAttention
        contextual_embeddings = attention_layer(contextual_embeddings, key_value_sequences, key_value_sequence_lengths)
        print(f"After MultiHeadAttention: {contextual_embeddings.shape}")

        # Apply PositionWiseFCNetwork
        contextual_embeddings = positionwise_layer(contextual_embeddings)
        print(f"After PositionWiseFCNetwork: {contextual_embeddings.shape}")

    return contextual_embeddings



# Main flow
if __name__ == "__main__":
    # Parse PGN text into FENs
    print("Parsing PGN text...")
    fens = parse_pgn_to_fens(PGN_TEXT2)
    print("FENs extracted:", fens)

    # Generate contextual embeddings
    print("\nGenerating contextual embeddings...")
    embeddings = generate_embeddings(fens)

    # Print the embeddings
    print("\nEmbeddings shape:", embeddings.shape)  # Expected: (num_moves, embedding_dim)
    #print("Embeddings (first move):", embeddings[0])
    #print("Embeddings (last move):", embeddings[-1])
    first_state = embeddings[-1].clone()
    
    
    # Parse PGN text into FENs
    fens = parse_pgn_to_fens(PGN_TEXT)

    # Generate contextual embeddings
    embeddings = generate_embeddings(fens)

    # Print the embeddings
    print("Embeddings (first move):", first_state)
    print("Embeddings (last move):", embeddings[-1])
    
