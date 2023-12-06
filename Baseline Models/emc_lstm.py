import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # new import
import warnings
import random

warnings.filterwarnings("ignore")
import seaborn as sns
sns.set_style("whitegrid")


# Define your dataset class
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
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        predictions = self.linear(lstm_out.view(len(x), -1))

input_dim = 7  # assuming  12 features
hidden_dim = 32
output_dim = 1
batch_size = 32
epochs = 200
consolidation_threshold = 0.2

def train_test_split_tasks(data, num_tasks):
    task_size = len(data) // num_tasks
    task_data = []
    for task in range(num_tasks):
        task_data.append(data[task * task_size: (task + 1) * task_size])
    return task_data

def load_data(task_data, test_size=0.2):
    X = task_data.drop(columns=['value', 'Date']).values
    y = task_data['value'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    return DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True), DataLoader(
        MyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

df = pd.read_csv('us_infleunza_output.csv')
num_tasks = 10
all_tasks = train_test_split_tasks(df, num_tasks)
all_data = [(load_data(task)) for task in all_tasks]

model = LSTM(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define the Memory class
class Memory:
    def __init__(self, memory_size, memory_update_strategy):
        self.input_variables = []
        self.output_variables = []
        self.learning_progress = []
        self.memory_size = memory_size
        self.memory_update_strategy = memory_update_strategy

    def update(self, input_window, output_window):
        if len(self.input_variables) < self.memory_size:
            self.input_variables.append(input_window)
            self.output_variables.append(output_window)
            self.learning_progress.append(0.0)
        else:
            if self.memory_update_strategy == 'Random':
                i = random.randrange(0, len(self.input_variables))
                self.input_variables[i] = input_window
                self.output_variables[i] = output_window
                self.learning_progress[i] = 0.0

    def update_learning_progress(self, model):
        for i in range(len(self.input_variables)):
            predictions = model.predict(np.array([self.input_variables[i]]))
            prediction_error = np.linalg.norm(predictions - np.array([self.output_variables[i]])) ** 2

            if self.learning_progress[i] == 0.0:
                self.learning_progress[i] = prediction_error
            else:
                self.learning_progress[i] = consolidation_threshold * self.learning_progress[i] + (
                            1 - consolidation_threshold) * prediction_error


# Online learning loop
stored_performances = []
r2_scores = []
memory_size=10
for task, (train_data, test_data) in enumerate(all_data):
    memory = Memory(memory_size, 'Random')  # You can change memory_update_strategy as needed
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_data):
            optimizer.zero_grad()
            if inputs is None:
                print(f"Warning: inputs is None in batch {i}")
            else:
                print(f"Batch {i}, Input Shape: {inputs.shape}")

            if labels is None:
                print(f"Warning: labels is None in batch {i}")

            outputs = model(inputs)

            if outputs is None:
                print(f"Warning: model(inputs) returned None in batch {i}")
            else:
                print(f"Batch {i}, Output Shape: {outputs.shape}")

            #outputs = model(inputs).view(-1)

            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
        for inputs, labels in test_data:
            outputs = model(inputs).view(-1)
            # Calculate and store performance metrics here
            predictions = outputs.detach().numpy()
            actuals = labels.detach().numpy()
            r2 = r2_score(actuals, predictions)
            r2_scores.append(r2)
            stored_performances.append(r2)
#
# # Save model and memory data
# model.save_weights('model_weights.h5')
# np.save('memory_input.npy', memory.input_variables)
# np.save('memory_output.npy', memory.output_variables)

# Calculate R-squared after training
final_performances = []
stored_performances = []
r2_scores = []
rmse_scores = []
forgetting_r2 = []
forgetting_rmse = []

for task, (train_data, test_data) in enumerate(all_data):  # Unpack the tuple here
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_data:  # Use test_data here
            inputs = inputs.float()  # Ensure inputs are float type
            labels = labels.float()  # Ensure labels are float type
            outputs = model(inputs).view(-1)
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2_final = r2_score(actuals, predictions)
    rmse_final = math.sqrt(mean_squared_error(actuals, predictions))
    print(f'Final R-squared for task {task + 1}: {r2_final}')
    print(f'Final RMSE for task {task + 1}: {rmse_final}')
    final_performances.append((r2_final, rmse_final))


# Compute forgetting
for task in range(num_tasks):
    forgetting_r2.append(stored_performances[task][0] - final_performances[task][0])
    forgetting_rmse.append(stored_performances[task][1] - final_performances[task][1])
    print(f'Forgetting for R-squared of task {task + 1}: {forgetting_r2[-1]}')
    print(f'Forgetting for RMSE of task {task + 1}: {forgetting_rmse[-1]}')

# Compute memory stability
memory_stability_r2 = 1 - np.mean(forgetting_r2)
memory_stability_rmse = 1 - np.mean(forgetting_rmse)
print(f'Memory stability for R-squared: {memory_stability_r2}')
print(f'Memory stability for RMSE: {memory_stability_rmse}')

