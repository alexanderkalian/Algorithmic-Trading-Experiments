import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Load data from CSV file
data = pd.read_csv('interim_dataset.csv')  # Replace 'data.csv' with your file name

# Extract X and y from the data
X = data[[col for col in data.columns if col.startswith('x')]].values
y = data['y'].values

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(128, 256, 512, 1024, 1024, 512, 256, 128, 64, 32), 
                   max_iter=1, warm_start=True, random_state=42, learning_rate_init=0.0001)

# Train the model for multiple epochs and record metrics
losses = []
r2_train_scores = []
r2_test_scores = []
epochs = 50  # Number of epochs

R2_text = r'$R^2$'

print('Training Epoch Results:')
for epoch in range(epochs):
    mlp.fit(X_train, y_train)
    loss = mlp.loss_
    losses.append(loss)
    
    # Calculate R^2 scores for training and testing data
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_train_scores.append(r2_train)
    r2_test_scores.append(r2_test)

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, R^2 Train: {r2_train:.4f}, R^2 Test: {r2_test:.4f}')

    if epoch%10 == 0:
        plt.scatter(y_test, y_test_pred, marker='x', alpha=0.4, label='Data')
        plt.plot([-100,100], [-100,100], color='red', label='Optimal y=x Line')
        plt.title(f'Real vs Predicted Stockfish Scores\n(Epoch: {epoch+1}, {R2_text} Score: {round(r2_test,3)})')
        plt.xlabel('Real Stockfish Score')
        plt.ylabel('Predicted Stockfish Score')
        plt.grid()
        plt.xlim([min(y_test)*1.2, max(y_test)*1.2])
        plt.ylim([min(y_test)*1.2, max(y_test)*1.2])
        plt.legend()
        plt.savefig(f'figures/mlp_real_vs_pred_epoch_{epoch+1}.png', dpi=500)
        plt.show()

# Final R^2 Scores
print(f'\nFinal R^2 Score on Training Data: {r2_train_scores[-1]:.4f}')
print(f'Final R^2 Score on Test Data: {r2_test_scores[-1]:.4f}')

# Plot training loss
plt.figure()
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.savefig('figures/mlp_training_loss.png', dpi=500)
plt.show()

# Plot R^2 scores
plt.figure()
plt.plot(range(1, epochs + 1), r2_train_scores, label=f'Train {R2_text} Score', marker='o')
plt.plot(range(1, epochs + 1), r2_test_scores, label=f'Test {R2_text} Score', marker='s')
plt.title(f'{R2_text} Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel(f'{R2_text} Score')
plt.legend()
plt.grid()
plt.savefig('figures/MLP_R2_score.png', dpi=500)
plt.show()

