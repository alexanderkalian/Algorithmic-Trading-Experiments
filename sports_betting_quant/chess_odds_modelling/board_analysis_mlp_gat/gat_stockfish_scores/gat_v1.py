import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np
import pickle
from tqdm import tqdm

# Function to process a NetworkX graph into PyG Data format
def process_nx_to_pyg(graph):
    # Extract node features
    node_features = torch.tensor(
        [[float(v) for v in graph.nodes[node]['features']] for node in graph.nodes], dtype=torch.float
    )

    # Extract edge indices and edge features
    edge_index = torch.tensor(
        list(graph.edges), dtype=torch.long
    ).t().contiguous()
    edge_features = torch.tensor(
        [graph[edge[0]][edge[1]]['interaction'] for edge in graph.edges], dtype=torch.float
    ).unsqueeze(1)  # Make edge features 2D

    # Add graph-level target (regression label)
    y = torch.tensor([graph.graph['target']], dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=y)

# Dummy function to generate synthetic NetworkX graphs for illustration
def generate_dummy_graphs(num_graphs):
    graphs = []
    for _ in range(num_graphs):
        num_nodes = np.random.randint(10, 33)
        graph = nx.erdos_renyi_graph(num_nodes, 0.2)

        # Add node features (4-5 features per node)
        for node in graph.nodes:
            graph.nodes[node]['features'] = np.random.rand(5)

        # Add edge features (one feature per edge, range 0-4)
        for edge in graph.edges:
            graph[edge[0]][edge[1]]['feature'] = np.random.randint(0, 5)

        # Add a graph-level target
        graph.graph['target'] = np.random.rand()
        graphs.append(graph)

    return graphs

# Define the GAT model
class GATRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, output_dim, heads=4):
        super(GATRegressor, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_dim, concat=False)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = self.gat2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)  # Pooling over nodes in each graph
        x = self.fc(x)
        return x

# Main function
def main():
    # Generate synthetic data
    '''
    num_graphs = 100
    nx_graphs = generate_dummy_graphs(num_graphs)
    '''
    
    # Load graphs back
    with open('../fen_to_graph/knowledge_graphs.pkl', 'rb') as f:
        nx_graphs = pickle.load(f)
    
    nx_graphs = nx_graphs[:int(0.3*len(nx_graphs))]
    num_graphs = len(nx_graphs)
    print('Number of knowledge graphs:',num_graphs)  # List of graphs

    # Process NetworkX graphs into PyG Data objects
    pyg_data_list = [process_nx_to_pyg(g) for g in tqdm(nx_graphs)]

    # Split data into train and test sets
    train_data = pyg_data_list[:80]
    test_data = pyg_data_list[80:]

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Model, optimizer, and loss function
    model = GATRegressor(input_dim=5, hidden_dim=16, edge_dim=1, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data):.4f}")

    # Evaluation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            loss = loss_fn(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
    print(f"Test Loss: {total_loss / len(test_data):.4f}")

if __name__ == "__main__":
    main()
