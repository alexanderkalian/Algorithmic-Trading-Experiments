import chess
import chess.pgn
import chess.engine

# Path to your Stockfish executable
STOCKFISH_PATH = "stockfish-windows-x86-64-sse41-popcnt/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe"

# PGN file to analyze
PGN_FILE_PATH = "../PGN to Matrix/lichess_LordJedizor_2024-12-26.pgn" 

# Function to get the board at the 20th move
def get_board_at_move(pgn_path, move_number):
    with open(pgn_path) as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        board = game.board()

        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            if i + 1 == move_number:
                return board
    return None

def evaluate_position(stockfish_path, board):
    import subprocess

    process = subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    commands = (
        "uci\n"
        f"position fen {board.fen()}\n"
        "go depth 2\n"
        "quit\n"
    )
    stdout, stderr = process.communicate(commands)

    # Debug raw output (optional)
    #print("Raw Stockfish Output:\n", stdout)
    #print("Error Output:\n", stderr)

    print(stdout)





if __name__ == "__main__":
    move_number = 20

    # Get the board at the specified move
    board = get_board_at_move(PGN_FILE_PATH, move_number)
    if board is None:
        print(f"Could not find move number {move_number} in the PGN file.")
    else:
        print(f"Board position at move {move_number}:\n{board}")

        # Evaluate the position using Stockfish
        score = evaluate_position(STOCKFISH_PATH, board)
        print(f"Evaluation by Stockfish: {score}")
