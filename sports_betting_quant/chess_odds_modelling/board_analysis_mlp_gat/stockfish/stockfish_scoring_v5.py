import chess
import chess.pgn
import subprocess

def get_position_from_pgn(pgn_file, move_number):
    """
    Reads a PGN file and extracts the position after the given move number.

    :param pgn_file: Path to the PGN file.
    :param move_number: Move number for which the position is extracted.
    :return: FEN string of the position after the given move number.
    """
    with open(pgn_file, 'r') as f:
        game = chess.pgn.read_game(f)
        
    if game is None:
        raise ValueError("No valid game found in the PGN file.")

    board = game.board()

    # Play moves to reach the desired position
    for i, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        if i == move_number:
            return board.fen()

    raise ValueError(f"Move number {move_number} exceeds the total moves in the game.")

def analyze_position_with_stockfish(fen, stockfish_path="stockfish"):
    """
    Analyzes a chess position using Stockfish.

    :param fen: FEN string of the chess position.
    :param stockfish_path: Path to the Stockfish binary.
    :return: Analysis output from Stockfish.
    """
    # Start the Stockfish process
    process = subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    # Send commands to Stockfish
    process.stdin.write(f"position fen {fen}\n")
    process.stdin.write("go depth 30\n")
    process.stdin.flush()

    # Capture the output
    output = []
    while True:
        line = process.stdout.readline().strip()
        if line == "" and process.poll() is not None:
            break
        if "bestmove" in line:
            output.append(line)
            break
        output.append(line)

    # Terminate the Stockfish process
    process.terminate()

    return "\n".join(output)

def main():
    pgn_file = "../PGN to Matrix/lichess_LordJedizor_2024-12-26.pgn"   # Replace with your PGN file path
    stockfish_path = "stockfish-windows-x86-64-sse41-popcnt/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe"  # Replace with the path to your Stockfish binary
    
    for n in range(1,21):
        
        try:
            # Get the FEN of the position after the 20th move
            fen = get_position_from_pgn(pgn_file, n)
            print(f"FEN for move {n}: {fen}")
    
            # Analyze the position using Stockfish
            analysis = analyze_position_with_stockfish(fen, stockfish_path)
            #print("\nStockfish Analysis:")
            #print(analysis)
            score = float(analysis.split("score cp ")[-1].split(" ")[0])
            print(f"Score: {score}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
