import subprocess
import chess
import chess.engine

# Path to Stockfish executable
stockfish_path = "stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"  # Update this with the path to your Stockfish binary

# Set up the board
board = chess.Board()
board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")
board.push_san("Nc6")

# Use subprocess to launch Stockfish
process = subprocess.Popen(
    [stockfish_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    universal_newlines=True,
)

# Send UCI commands to Stockfish
process.stdin.write("uci\n")
process.stdin.write("isready\n")
process.stdin.write(f"position fen {board.fen()}\n")
process.stdin.write("go depth 10\n")
process.stdin.flush()

# Capture the output
output = []
while True:
    line = process.stdout.readline().strip()
    if line == "":
        break
    output.append(line)

# Print the output from Stockfish
print("\n".join(output))

# Close the process
process.terminate()

