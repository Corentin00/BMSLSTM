import pandas as pd
import torch

context_size = 20

file_path = "dataset2\IN_E_FULL.xlsx"
df = pd.read_excel(file_path)

columns_to_use = ['Somme de Value Innovix', 'Somme de Value Yrex', 'Indication 1', 'Indication 10', 'Indication 11', 'Indication 12', 'Indication 13', 'Indication 14', 'Indication 15', 'Indication 16', 'Indication 18', 'Indication 19', 'Indication 2', 'Indication 21', 'Indication 22', 'Indication 23', 'Indication 7', 'Indication 8', 'Indication 9']

data = df[columns_to_use].values

tensor_data = torch.tensor(data, dtype=torch.float32)

tensor_data = tensor_data[-context_size:, :]
tensor_data = tensor_data.unsqueeze(0)
print("predict tensor shape:", tensor_data.shape)
print("Aperçu du tenseur predict\n", tensor_data)

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Fixer la graine pour la reproductibilité
np.random.seed(10)  # Pour numpy
torch.manual_seed(10)  # Pour PyTorch
torch.cuda.manual_seed_all(10)  # Pour PyTorch sur GPU (si utilisé)
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
epochs = 2000

# Initialisation du modèle
model = SalesPredictionModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("project\model_weights.pth", weights_only=True))

col0 = tensor_data[:, :, 0]
col1 = tensor_data[:, :, 1]
sum1 = col0 + col1
sum1 = sum1.unsqueeze(-1)
x_predict = torch.cat((tensor_data, sum1), dim=-1)
mean = torch.tensor(1550583.2500) # Computed on the x_train
std = torch.tensor(4149713.2500) # Computed on the x_train
x_predict = (x_predict - mean) / std

mean = torch.tensor(7372857.0000) # Computed on the y_train
std = torch.tensor(1640941.7500) # Computed on the y_train
predicted_sales = model(x_predict)
predicted_sales = predicted_sales * std + mean
print(predicted_sales)

total_predicted_sales = [2415700, 2546680, 2692040, 3064760, 2862080, 2877080, 2927400, 3153880, 3030560, 3810540, 5079380, 4899640, 4357480, 4568100, 4649660, 5515160, 4288000, 4600800, 4688740, 5303720, 5859060, 4057640, 4452560, 4994260, 4165660, 4406000, 5040540, 5860780, 4539720, 5276600, 5815360, 5150060, 5283020, 5659780, 5255620, 5899480, 5066920, 5357640, 6233060, 6691640, 5001040, 6072900, 5746840, 6913120, 6356520, 6470860, 6643160, 7302280, 6640320, 7044660, 8116600, 8428760, 7425980, 8568520, 8482500, 9179600, 8812520, 8775560, 8945980, 9272860, 8165240, 8475020, 9370660, 9527680, 8649920, 9342280, 8750300, 9662840, 8770500, 9398940, 9075300, 9474340, 8228220, 8726340, 7966980, 10391580, 8973780, 8540900, 9485580, 9147500, 8280260]
total_predicted_sales.extend(predicted_sales.squeeze(0).tolist())
print(total_predicted_sales)

plt.figure(figsize=(10, 5))
plt.plot(range(len(total_predicted_sales)), total_predicted_sales, label='Predicted Sales', color='#90EE90')
plt.xlabel('Months')
plt.ylabel('Sales Volume (×10⁷)')
plt.legend()
plt.title('Monthly Sales Volume Prediction for INNOVIX in Elbonie')
plt.show()