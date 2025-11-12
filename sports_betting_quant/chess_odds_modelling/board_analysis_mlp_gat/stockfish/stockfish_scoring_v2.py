import subprocess
import chess
import chess.pgn

# Path to Stockfish executable
stockfish_path = "stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"  # Update this with the path to your Stockfish binary

def score_position_from_pgn(pgn_path, game_number=1, depth=10):
    with open(pgn_path, "r") as pgn_file:
        game = None
        for i in range(game_number):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                raise ValueError(f"Game {game_number} not found in the PGN file.")

    board = game.end().board()

    # Start Stockfish process
    process = subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Send UCI commands and ensure readiness
        process.stdin.write("uci\n")
        process.stdin.write("isready\n")
        process.stdin.flush()
        while True:
            line = process.stdout.readline().strip()
            if line == "readyok":
                break

        # Send position and analyze
        commands = f"position fen {board.fen()}\ngo depth {depth}\n"
        print(f"Commands sent to Stockfish:\n{commands}")
        process.stdin.write(commands)
        process.stdin.flush()

        # Capture output
        score = None
        while True:
            line = process.stdout.readline().strip()
            if not line:
                break
            print(f"Stockfish Output: {line}")
            if "score cp" in line:
                score = line.split("score cp ")[-1].split()[0]
            elif "score mate" in line:
                score = "Mate in " + line.split("score mate")[-1].split()[0]

        if score is None:
            if board.is_checkmate():
                score = "Mate position"
            elif board.is_stalemate():
                score = "Stalemate"
            else:
                score = "Unable to evaluate position"

        # Send quit command
        process.stdin.write("quit\n")
        process.stdin.flush()

    finally:
        # Ensure process termination
        process.terminate()
        process.wait(timeout=5)  # Wait for the process to terminate

    return score



# Example usage
if __name__ == "__main__":
    # Path to the PGN file
    pgn_path = "../PGN to Matrix/lichess_LordJedizor_2024-12-26.pgn"  # Replace with your PGN file path

    # Score the first game's final position
    evaluation = score_position_from_pgn(pgn_path, game_number=1)
    print(f"Evaluation of the final position: {evaluation}")
    
    


