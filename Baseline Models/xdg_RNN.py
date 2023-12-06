import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import csv
from mystyle import *
class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)



class XdG_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_tasks):
        super(XdG_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks

        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(num_tasks, hidden_dim)

    def forward(self, x, task_id):
        task_vector = torch.zeros(self.num_tasks).to(x.device)
        task_vector[task_id] = 1
        gating_values = torch.sigmoid(self.gate(task_vector))

        rnn_out, _ = self.rnn(x.view(len(x), 1, -1))
        gated_rnn_out = rnn_out * gating_values
        predictions = self.linear(gated_rnn_out.view(len(x), -1))
        return predictions.squeeze()

def train_test_split_tasks(data, num_tasks):
    task_size = len(data) // num_tasks
    task_data = []
    for task in range(num_tasks):
        task_data.append(data[task * task_size: (task + 1) * task_size])
    return task_data

def load_data(task_data, test_size=0.2):
    # load your data using pandas, and split X and y
    X = task_data.drop(columns=['value', 'Date']).values
    y = task_data['value'].values

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    return DataLoader(MyDataset(X_train, y_train), batch_size=32, shuffle=True), DataLoader(
        MyDataset(X_test, y_test), batch_size=32, shuffle=False)
# Initialize parameters and model
input_dim = 7
hidden_dim = 32
output_dim = 1
num_tasks = 10

stored_performances = []
r2_scores = []
rmse_scores = []
forgetting_r2 = []
forgetting_rmse = []

df = pd.read_csv('data_file')
all_tasks = train_test_split_tasks(df, num_tasks)
all_data = [(load_data(task)) for task in all_tasks]

model = XdG_RNN(input_dim, hidden_dim, output_dim, num_tasks)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training and evaluation
for task, (train_data, test_data) in enumerate(all_data):
    for epoch in range(200):  # Number of epochs
        for i, (inputs, labels) in enumerate(train_data):
            optimizer.zero_grad()
            outputs = model(inputs, task).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            outputs = model(inputs, task).view(-1)
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2_scores.append(r2_score(actuals, predictions))
    rmse_scores.append(math.sqrt(mean_squared_error(actuals, predictions)))
    print(f'R-squared for task {task + 1}: {r2_scores[-1]}')
    #print(f'RMSE for task {task + 1}: {rmse_scores[-1]}')
    stored_performances.append((r2_scores[-1], rmse_scores[-1]))


# After all tasks, re-evaluate performance on all tasks
final_performances = []
r2_final_list=[]
final_rmse=[]
for task, (train_data, test_data) in enumerate(all_data):  # Unpack the tuple here
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_data:  # Use test_data here
            inputs = inputs.float()  # Ensure inputs are float type
            labels = labels.float()  # Ensure labels are float type
            outputs = model(inputs, task).view(-1)  # Added task_id here
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2_final = r2_score(actuals, predictions)
    r2_final_list.append(r2_final)
    rmse_final = math.sqrt(mean_squared_error(actuals, predictions))
    final_rmse.append(rmse_final)
    print(f'Final R-squared for task {task + 1}: {r2_final}')
    #print(f'Final RMSE for task {task + 1}: {rmse_final}')
    final_performances.append((r2_final, rmse_final))



# Compute forgetting
for task in range(num_tasks):
    forgetting_r2.append(stored_performances[task][0] - final_performances[task][0])
    forgetting_rmse.append(stored_performances[task][1] - final_performances[task][1])
    print(f'Forgetting for R-squared of task {task + 1}: {forgetting_r2[-1]}')
    #print(f'Forgetting for RMSE of task {task + 1}: {forgetting_rmse[-1]}')

# Compute memory stability
memory_stability_r2 = 1 - np.mean(np.abs(forgetting_r2))
memory_stability_rmse = 1 - np.mean(np.abs(forgetting_rmse))
print(f'Memory stability for R-squared: {memory_stability_r2}')
#print(f'Memory stability for RMSE: {memory_stability_rmse}')



