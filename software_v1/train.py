import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Fixer la graine pour la reproductibilité
np.random.seed(40)  # Pour numpy
torch.manual_seed(40)  # Pour PyTorch
torch.cuda.manual_seed_all(40)  # Pour PyTorch sur GPU (si utilisé)
torch.backends.cudnn.deterministic = True  # S'assurer que les opérations sur GPU sont déterministes
torch.backends.cudnn.benchmark = False # Désactiver le benchmarking pour garantir des résultats reproductibles

# Définition du modèle LSTM pour la prédiction des ventes
class SalesPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SalesPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Prendre la dernière sortie du LSTM
        out = self.fc(lstm_out)
        return out

# Hyperparamètres
input_size = 20  # 20 caractéristiques par mois
hidden_size = 32  # Taille de l'état caché LSTM
output_size = 14  # Prédiction sur 14 mois
learning_rate = 0.0001
batch_size = 4
epochs = 5

# Initialisation du modèle
model = SalesPredictionModel(input_size, hidden_size, output_size)

# Fonction de perte et optimiseur
criterion = nn.MSELoss()  # Erreur quadratique moyenne
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_samples = 48
x = torch.load('dataset2\x_train_t2.pth', weights_only = True)
col0 = x[:, :, 0]
col1 = x[:, :, 1]
sum = col0 + col1
sum = sum.unsqueeze(-1)
x = torch.cat((x, sum), dim=-1)
mean = x.mean()
print("mean:", mean)
std = x.std()
print("std", std)
x = (x - mean) / std
y = torch.load('dataset2\y_true_t2.pth', weights_only = True)
mean = y.mean()
print("mean", mean)
std = y.std()
print("std", std)
y = (y - mean) / std

x_train = x[:42]
x_test = x[42:]
y_train = y[:42]
y_test = y[42:]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Charger les données dans un DataLoader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

train_all_losses = []
test_all_losses = []
# Entraînement du modèle
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        # Zéro les gradients
        optimizer.zero_grad()

        # Passer les entrées dans le modèle
        outputs = model(inputs)

        # Calcul de la perte
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    train_all_losses.append(avg_loss)
    #print(f'Époque [{epoch+1}/{epochs}], Train Set, Perte moyenne : {avg_loss:.4f}')

    # Exemple de prédiction avec un échantillon de test
    model.eval()
    with torch.no_grad():
        # Générer un échantillon de test
        predicted_sales = model(x_test)
        #print(f'Prédiction des ventes pour 12 mois : {predicted_sales * std + mean}')
        loss = criterion(predicted_sales, y_test)
        test_all_losses.append(loss)
        #print(f'Époque [{epoch+1}/{epochs}], Test Set, Perte moyenne : {loss.item()}')

# Générer un échantillon de test
torch.save(model.state_dict(), 'model_weights.pth')
factor_increase = 0
factor_decrease = 0
x_predict = torch.load('dataset2\predict_t2.pth', weights_only=True)
col0 = x_predict[:, :, 0]
col1 = x_predict[:, :, 1]
sum = col0 + col1
sum = sum.unsqueeze(-1)
x_predict = torch.cat((x_predict, sum), dim=-1)
#x_predict[:, :, 19] += 0.1 * x_predict[:, :, 19]
for i in range(x_predict.shape[1]):
    x_predict[0, i, 0] -= factor_increase * i
    x_predict[0, i, 1] -= factor_decrease * i
    x_predict[0, i, 19] -= 2 * factor_decrease * 1
x_predict[0, :, 1] = torch.clamp(x_predict[0, :, 1], min=0)
predicted_sales = model(x_predict)
print(x_predict.shape)
#x_predict[:, :, 1:] = 0 No more disease
#x_predict[:, :, 1] = 0
print("x_predict tensor shape:", x_predict.shape)
print("Aperçu du tenseur predict:\n", x_predict)
predicted_sales = model(x_predict)
print(f'Prédiction des ventes pour 12 mois : {predicted_sales * std + mean}')



# Affichage de la train loss
plt.figure(figsize=(10, 5))
plt.plot(range(len(train_all_losses)), train_all_losses, label='Train Loss', color='#007FFF')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Train Loss over Iterations')
plt.show()
         
# Affichage de la test loss
plt.figure(figsize=(10, 5))
plt.plot(range(len(test_all_losses)), test_all_losses, label='Test Loss', color='#5db804')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss over Iterations')
plt.show()