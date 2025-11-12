import chess
import chess.pgn
import subprocess


depth = 20

output_file = f'stockfish_scores_depth_{depth}.csv'

with open(output_file, 'w') as out:
    out.write('FEN,stockfish_score')


def get_position_from_game(game, move_number):
    '''
    Extracts the position after the given move number from a single game.

    :param game: A chess.pgn.Game object.
    :param move_number: Move number for which the position is extracted.
    :return: FEN string of the position after the given move number.
    '''
    board = game.board()

    # Play moves to reach the desired position
    for i, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        if i == move_number:
            return board.fen(), board.turn  # Return FEN and side to move (True = White, False = Black)

    raise ValueError(f'Move number {move_number} exceeds the total moves in the game.')

def analyze_position_with_stockfish(fen, is_white_to_move, stockfish_path='stockfish'):
    '''
    Analyzes a chess position using Stockfish.

    :param fen: FEN string of the chess position.
    :param is_white_to_move: Boolean indicating if it's White's turn to move.
    :param stockfish_path: Path to the Stockfish binary.
    :return: Normalized score from White's perspective.
    '''
    # Start the Stockfish process
    process = subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )

    # Send commands to Stockfish
    process.stdin.write(f'position fen {fen}\n')
    process.stdin.write(f'go depth {depth}\n')
    process.stdin.flush()

    # Capture the output
    output = []
    while True:
        line = process.stdout.readline().strip()
        if line == '' and process.poll() is not None:
            break
        if 'bestmove' in line:
            output.append(line)
            break
        output.append(line)

    # Terminate the Stockfish process
    process.terminate()

    # Extract and normalize the score
    for line in output:
        if 'score cp' in line:
            score = int(line.split('score cp ')[-1].split()[0])
            return score / 100 if is_white_to_move else -score / 100

    raise ValueError('Stockfish did not return a valid score.')

def main():
    pgn_file = '../PGN to Matrix/lichess_LordJedizor_2024-12-26.pgn'   # Replace with your PGN file path
    stockfish_path = 'stockfish-windows-x86-64-sse41-popcnt/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe'  # Replace with the path to your Stockfish binary

    try:
        with open(pgn_file, 'r') as f:
            game_number = 1
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break  # No more games in the PGN file

                print(f'\nAnalyzing Game {game_number}')
                board = game.board()

                total_moves = sum(1 for _ in game.mainline_moves())

                for n in range(1, total_moves + 1):
                    # Get the FEN and side to move
                    fen, is_white_to_move = get_position_from_game(game, n)

                    # Analyze the position using Stockfish
                    try:
                        score = analyze_position_with_stockfish(fen, is_white_to_move, stockfish_path)
                        move_type = 'W' if is_white_to_move else 'B'
                        move_number = (n + 1) // 2
                        print(f'Move {move_number} ({move_type}): Score = {score:.2f}')
                        csv_row = f'\n{fen},{score}'
                        with open(output_file, 'a') as out:
                            out.write(csv_row)
                    except ValueError as ve:
                        print(f'Error analyzing move {n}: {ve}')

                game_number += 1

    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    main()
