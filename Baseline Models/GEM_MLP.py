
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
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set_style("whitegrid")
import torch.optim as optim
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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten input if needed
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()  # This line flattens the output to have shape (batch_size)


input_dim = 7  
hidden_dim = 32
output_dim = 1
batch_size = 32
epochs = 200
#ewc_lambda = 500

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
    return DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True), DataLoader(
        MyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

df = pd.read_csv('data_file')
num_tasks = 10
all_tasks = train_test_split_tasks(df, num_tasks)
all_data = [(load_data(task)) for task in all_tasks]

model = MLP(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# GEM specific components
class GEMMemory:
    def __init__(self, capacity, input_dim, output_dim):
        self.memory_data = torch.zeros(capacity, input_dim)
        self.memory_targets = torch.zeros(capacity, output_dim)
        self.position = 0
        self.capacity = capacity

    def store(self, data, targets):
        batch_size = data.size(0)
        for i in range(batch_size):
            self.memory_data[self.position] = data[i]
            self.memory_targets[self.position] = targets[i]
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = torch.randint(0, self.capacity, (batch_size,))
        return self.memory_data[indices], self.memory_targets[indices]


#gem_loss
def gem_loss(predictions, targets, model, memory, margin=0.5):
    base_loss = criterion(predictions.squeeze(), targets)
    penalty = 0

    current_batch_size = targets.size(0)
    mem_data, mem_targets = memory.sample(current_batch_size)

    past_task_output = model(mem_data)
    if past_task_output.dim() == 2:
        past_task_output = past_task_output.squeeze(1)

    current_task_output = model(inputs)
    if current_task_output.dim() == 2:
        current_task_output = current_task_output.squeeze(1)

    # Ensure mem_targets and targets are 1D
    mem_targets = mem_targets.squeeze()
    targets = targets.squeeze()



    penalty += torch.dot(past_task_output - mem_targets, current_task_output - targets).clamp(min=0)
    return base_loss + margin * penalty


# Initialize GEM memory
memory_capacity = 1024  # You can adjust this value
gem_memory = GEMMemory(memory_capacity, input_dim, output_dim)


stored_performances = []
r2_scores = []
rmse_scores = []
forgetting_r2 = []
forgetting_rmse = []
# Training with GEM
for task, (train_data, test_data) in enumerate(all_data):
    # Train on this task
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_data):
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)

            # Store data in GEM memory
            gem_memory.store(inputs, labels)

            # Compute GEM loss
            loss = gem_loss(outputs, labels, model, gem_memory)

            loss.backward()
            optimizer.step()

    # Evaluate performance on this task
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            outputs = model(inputs).view(-1)
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2_scores.append(r2_score(actuals, predictions))
    rmse_scores.append(np.sqrt(mean_squared_error(actuals, predictions)))
    print(f'R-squared for task {task + 1}: {r2_scores[-1]}')
    #print(f'RMSE for task {task + 1}: {rmse_scores[-1]}')
    stored_performances.append((r2_scores[-1], rmse_scores[-1]))

    ###########

final_performances = []
r2_final_list=[]
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
    r2_final_list.append(r2_final)
    rmse_final = math.sqrt(mean_squared_error(actuals, predictions))
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

# R2_Score
plt.figure(figsize=(12, 6))
plt.plot(r2_scores, label='R2 Score')
plt.xlabel('Tasks')
plt.ylabel('Value')
plt.legend()
plt.title('R2 Score')
plt.show()
