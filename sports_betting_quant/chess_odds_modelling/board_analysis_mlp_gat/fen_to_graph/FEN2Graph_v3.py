import chess
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# Unicode chess symbols
UNICODE_PIECES = {
    'P': '\u2659', 'p': '\u265F',  # White pawn, Black pawn
    'R': '\u2656', 'r': '\u265C',  # White rook, Black rook
    'N': '\u2658', 'n': '\u265E',  # White knight, Black knight
    'B': '\u2657', 'b': '\u265D',  # White bishop, Black bishop
    'Q': '\u2655', 'q': '\u265B',  # White queen, Black queen
    'K': '\u2654', 'k': '\u265A',  # White king, Black king
}

# Dictionary of some piece-specific node features.
pieces_dict = {
    'P': [0, 1, 1], 'p': [1, 1, 1],  # White pawn, Black pawn
    'R': [0, 2, 5], 'r': [1, 2, 5],  # White rook, Black rook
    'N': [0, 3, 3], 'n': [1, 3, 3],  # White knight, Black knight
    'B': [0, 4, 3], 'b': [1, 4, 3],  # White bishop, Black bishop
    'Q': [0, 5, 9], 'q': [1, 5, 9],  # White queen, Black queen
    'K': [0, 6, 20], 'k': [1, 6, 20],  # White king, Black king
}

# Dictionary of positional file conversions, form letters to coordinates.
files_dict = {
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 
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
    plt.savefig('figures/chess_board.png', dpi=500)
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


def list_attacked_and_defended_pieces(fen):
    """
    List all pieces being looked at (attacked or defended) by other pieces.
    """
    board = chess.Board(fen)
    piece_map = board.piece_map()

    attacked_defended_pieces = {}

    for square, piece in piece_map.items():
        attackers = list(board.attackers(chess.WHITE, square)) + list(board.attackers(chess.BLACK, square))
        #attackers_info = [(chess.square_name(attacker), piece_map[attacker].symbol()) for attacker in attackers if attacker in piece_map]
        attackers_info = [chess.square_name(attacker) for attacker in attackers if attacker in piece_map]
        
        if attackers_info:
            attacked_defended_pieces[chess.square_name(square)] = attackers_info

    return attacked_defended_pieces


def find_edge_feature(board, coord_pair, attacked_defended_pieces):
    """
    Find edge feature to represent relationship between two pieces as nodes.
    0 means no attacking/defending between either.
    1 means attacking/defending only by piece A to piece B.
    2 means attacking/defending only by piece B to piece A.
    3 means attacking/defending by both pieces to each other.
    """
    
    # Unpacks coordinate pair into two squares.
    square_1_pos, square_2_pos = coord_pair
    square_1 = chess.square_name(square_1_pos)
    square_2 = chess.square_name(square_2_pos)
    
    # Finds square_A and square_B, according to correct ordering required.
    
    # Starts by finding piece_1 and piece_2.
    piece_1 = board.piece_at(square_1_pos).symbol()
    piece_2 = board.piece_at(square_2_pos).symbol()
    
    # Rules out both pieces looking at each other, while flagging up other interactions.
    
    # Defaultly sets edge feature to 0 (no interaction).
    edge_feature = 0
    
    # Checks if both squares are being attacked/defended somehow.
    if square_1 in attacked_defended_pieces.keys() and square_2 in attacked_defended_pieces.keys():
        # Checks for both pieces attacking/defending each other.
        if square_2 in attacked_defended_pieces[square_1] and square_1 in attacked_defended_pieces[square_2]:
            # Sets edge feature to 3.
            edge_feature = 3
        # Checks for neither pieces attacking/defending each other.
        elif square_2 not in attacked_defended_pieces[square_1] and square_1 not in attacked_defended_pieces[square_2]:
            # Sets edge feature to 3.
            edge_feature = 0
        # Flags one-way interaction, if needed.
        else:
            edge_feature = None
    # Catches cases where one piece is attacking/defending the other (but not mutually).
    elif square_1 in attacked_defended_pieces.keys():
        if square_2 in attacked_defended_pieces[square_1]:
            edge_feature = None
    elif square_2 in attacked_defended_pieces.keys():
        if square_1 in attacked_defended_pieces[square_2]:
            edge_feature = None
    
    # If some other case (i.e. only one piece interacting with the other), then find out the order.
    if edge_feature is None:
        
        # Finds colours and point values.
        colour_1 = pieces_dict[piece_1][0]
        colour_2 = pieces_dict[piece_2][0]
        points_1 = pieces_dict[piece_1][2]
        points_2 = pieces_dict[piece_2][2]
        # Assigns a slightly higher points value to bishops, just as a convention here.
        if pieces_dict[piece_1][1] == 4:
            points_1 = 3.5
        if pieces_dict[piece_2][1] == 4:
            points_2 = 3.5
        
        # Checks if there is a colour difference.
        if colour_1 != colour_2:
            # Assigns white as A and black as B.
            if colour_1 == 0:
                square_A = square_1
                square_B = square_2
            else:
                square_A = square_2
                square_B = square_1
        # Otherwise checks if there is a points value difference.
        elif points_1 != points_2:
            # Assigns the higher value piece as A and the lower value as B.
            if points_1 > points_2:
                square_A = square_1
                square_B = square_2
            else:
                square_A = square_2
                square_B = square_1
        # Else, it must be two pieces of the same colour and type.
        else:
            # Assigns the closest piece to the a-file as A, and the other as B.
            ## Note: both cannot be equally distant! As that would then mean non-pawns, which would mutually interact and hence have already been covered.
            if files_dict[square_1[0]] < files_dict[square_2[0]]:
                square_A = square_1
                square_B = square_2
            else:
                square_A = square_2
                square_B = square_1
        
        # Checks if square A is attacking/defending square B.
        if square_B in attacked_defended_pieces.keys():
            if square_A in attacked_defended_pieces[square_B]:
                edge_feature = 1 
        # Checks if square B is attacking/defending square A.
        if square_A in attacked_defended_pieces.keys():
            if square_B in attacked_defended_pieces[square_A]:
                if edge_feature == 1:
                    print('Edge feature found to be both 1 and 2!')
                    raise(SystemError)
                edge_feature = 2
        if edge_feature is None:
            print('Edge feature not found!')
            print(square_1, square_2)
            raise(SystemError)
    
    return edge_feature
        

def build_graph(fen):
    
    board = chess.Board(fen)
    
    # Find and print all piece positions.
    for square, piece in board.piece_map().items():
        print(f"Piece: {piece} at {square}")
    
    # Get pieces being looked at (attacked or defended)
    attacked_defended = list_attacked_and_defended_pieces(fen)
    print("\nPieces being looked at (attacked or defended):")
    for square, attackers in attacked_defended.items():
        print(f"{square}: {attackers}")
    
    # Builds a NetworkX graph, to represent the board (in terms of interactions).
    G = nx.Graph()
    
    # Iterates through all possible pairing.
    for square1, piece1 in board.piece_map().items():
        for square2, piece2 in board.piece_map().items():
            # Ensures unique pairs.
            if square1 < square2:
                
                # Finds edge feature.
                square_name1 = chess.square_name(square1)
                square_name2 = chess.square_name(square2)
                edge_feature = find_edge_feature(board, [square1, square2], attacked_defended)
                
                # Finds node features for each (white/black, piece type, points value, x coordinate, y coordinate).
                
                # For piece 1.
                node_feature1 = pieces_dict[piece1.symbol()]
                node_feature1.append(files_dict[square_name1[0]])
                node_feature1.append(square_name1[1])
                
                # For piece 2.
                node_feature2 = pieces_dict[piece2.symbol()]
                node_feature2.append(files_dict[square_name2[0]])
                node_feature2.append(square_name2[1])
                
                # Adds the whole thing to the graph.
                
                # Nodes with features.
                G.add_node(square_name1, features=node_feature1)
                G.add_node(square_name2, features=node_feature2)
                
                # Add an edge between the nodes.
                G.add_edge(square_name1, square_name2, interaction=edge_feature)
    
    return G
        
    

# Example FEN string
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fen = "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1b1PPB2/2N2N2/PPP2PPP/R2Q1RK1 w - - 0 10"

board = chess.Board(fen)

# Display the chessboard
display_chess_board(fen)

# Finds positional graph.
G = build_graph(fen)


# Define edge colors
edge_colors = {0: ["grey", 0.3], 1: ["blue", 1.0], 2: ["green", 1.0], 3: ["red", 1.0]}
interactions = {0: 'A B', 1: r'A$\rightarrow$B', 
                2: r'B$\rightarrow$A', 3: r'A$\leftrightarrow$B'}

# Layout with better spacing
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Increase figure size
plt.figure(figsize=(8, 8))

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_color="lightblue", edgecolors="black", node_size=1000)
nx.draw_networkx_edges(
    G,
    pos,
    edge_color=[edge_colors[G[u][v]['interaction']][0] for u, v in G.edges()],
    alpha=[edge_colors[G[u][v]['interaction']][1] for u, v in G.edges()],
    width=2
)

# Draw node labels
nx.draw_networkx_labels(G, pos, labels={n: UNICODE_PIECES[board.piece_at(
    chess.parse_square(n)).symbol()] for n in G.nodes()}, font_size=10)

# Add node features below each node with offset
'''
for node, (x, y) in pos.items():
    features = G.nodes[node]['features']
    plt.text(x, y - 0.1, f"{features}", fontsize=8, ha="center", color="purple")
'''
# Add a legend for edge colors
for feature, params in edge_colors.items():
    color, alpha = params
    plt.plot([], [], color=color, label=f"{interactions[feature]}", alpha=alpha)
plt.legend(title="Interaction", loc="upper right", fontsize=8)

# Improve axis appearance
plt.axis("off")
plt.title('Knowledge Graph Representation of Chess FEN Position.')
plt.savefig('figures/knowledge_graph.png', dpi=500)
plt.show()

