###### This is the final code which uses EWC+LSTM on output and also plot

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
        return predictions.squeeze()  # This line flattens the output to have shape (batch_size)


input_dim = 7  # assuming  12 features
hidden_dim = 32
output_dim = 1
batch_size = 32
epochs = 200
ewc_lambda = 1000

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

df = pd.read_csv('us_infleunza_output.csv')
num_tasks = 10
all_tasks = train_test_split_tasks(df, num_tasks)
all_data = [(load_data(task)) for task in all_tasks]

model = LSTM(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


#######################
# compute_FIM_and_params
def compute_FIM_and_params(data, model):
    FIM = {}
    model_params = {}

    model.zero_grad()
    for x, y in data:
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                model_params[name] = param.data.clone()
                if name in FIM:
                    FIM[name] += param.grad.data.clone() ** 2
                else:
                    FIM[name] = param.grad.data.clone() ** 2
    return FIM, model_params

# ewc_loss
def ewc_loss(predictions, targets, ewc_lambda, model, old_params, FIM):
    base_loss = criterion(predictions.squeeze(), targets)
    penalty = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in old_params:
            penalty += (FIM[name] * (param - old_params[name]) ** 2).sum()
    # print("this is  panelty", penalty)
    # print("this is base loss", base_loss)
    # print("this is base loss + ewc*panelty", base_loss + ewc_lambda * penalty )
    total_loss= base_loss + ewc_lambda * penalty
    return total_loss

stored_performances = []
r2_scores = []
rmse_scores = []
forgetting_r2 = []
forgetting_rmse = []

parameters_with_ewc_values = {}
FIM, model_params = {}, {}

for name, param in model.named_parameters():
    if param.requires_grad:
        # Initialize the storage for each parameter with a list of zeros
        parameters_with_ewc_values[name] = [torch.zeros_like(param).view(-1) for _ in range(len(all_data))]

########### Training  ############
for task, (train_data, test_data) in enumerate(all_data):
    # Train on this task
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_data):
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)

            if task > 0:
                loss = ewc_loss(outputs, labels, ewc_lambda, model, model_params, FIM)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # Store the parameter values after each epoch for visualization
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Flatten the parameter values before storing them
                parameters_with_ewc_values[name][task] = param.data.clone().view(-1)

    # Compute the Fisher information matrix; save the parameters and the diagonal of the FIM
    FIM, model_params = compute_FIM_and_params(train_data, model)
    if task > 0:
        # Add the EWC penalty to the loss
        loss = ewc_loss(outputs, labels, ewc_lambda, model, model_params, FIM)

    # Evaluate performance on this task
    predictions = []
    actuals = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            ...
            outputs = model(inputs).view(-1)
            predictions.extend(outputs.numpy().flatten().tolist())
            actuals.extend(labels.numpy().flatten().tolist())

    r2_scores.append(r2_score(actuals, predictions))
    rmse_scores.append(math.sqrt(mean_squared_error(actuals, predictions)))
    print(f'R-squared for task {task + 1}: {r2_scores[-1]}')
    print(f'RMSE for task {task + 1}: {rmse_scores[-1]}')
    stored_performances.append((r2_scores[-1], rmse_scores[-1]))

    # # New code for plotting each task seperately
    # plt.figure(figsize=(10, 5))
    # plt.plot(actuals, label='Actual values')
    # plt.plot(predictions, label='Predicted values')
    # plt.title(f'Actual vs Predicted values for task {task + 1}')
    # plt.legend()
    # plt.show()

#############################################################################
#### After all tasks, re-evaluate performance on all tasks
final_performances = []

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



# R2_Score
plt.figure(figsize=(12, 6))
plt.plot(r2_scores, label='R2 Score')
plt.xlabel('Tasks')
plt.ylabel('Value')
plt.legend()
plt.title('R2 Score')
plt.show()

# RMSE
plt.figure(figsize=(12, 6))
plt.plot(rmse_scores, label='RMSE')
plt.xlabel('Tasks')
plt.ylabel('Value')
plt.legend()
plt.title('RMSE')
plt.show()

#Memory Forgetting
# R2 Forgetting
plt.figure(figsize=(12, 6))
plt.plot(memory_stability_rmse, label='memory_stability_rmse')
plt.plot(memory_stability_r2, label='memory_stability_r2')
plt.xlabel('Tasks')
plt.ylabel('Value')
plt.legend()
plt.title('Memory stability')
plt.show()

# R2 Forgetting
plt.figure(figsize=(12, 6))
plt.plot(forgetting_r2, label='forgetting_r2')
plt.plot(forgetting_r2, label='forgetting_r2')
plt.xlabel('Tasks')
plt.ylabel('Value')
plt.legend()
plt.title('R2 Forgetting')
plt.show()

# RMSE Forgetting
plt.figure(figsize=(12, 6))
plt.plot(forgetting_rmse, label='RMSE Forgetting')
plt.xlabel('Tasks')
plt.ylabel('Value')
plt.legend()
plt.title('RMSE Forgetting')
plt.show()
############################################################################
###Print the names and values of the parameters that took part in EWC
# print("Parameters with EWC:")
# for name, values in parameters_with_ewc_values.items():
#     print("these are parameters ")
#     print(name)
#     for i, value in enumerate(values):
#         print(f"Task {i + 1}: {value}")
# #Plot the changes in parameter values over different tasks
# for name, values in parameters_with_ewc_values.items():
#     plt.figure(figsize=(10, 5))
#     for task in range(len(values)):
#         plt.plot(values[task], label=f'Task {task + 1}')
#     plt.title(f'Parameter Value Changes for {name}')
#     plt.xlabel('Task')
#     plt.ylabel('Parameter Value')
#     plt.legend()
#     plt.show()
# for name, values in parameters_with_ewc_values.items():
#     shapes = [v.shape for v in values]
#     print(name, shapes)
#
# #
# # # Convert the lists of parameter values to numpy arrays
# for name, values in parameters_with_ewc_values.items():
#     parameters_with_ewc_values[name] = np.stack([v.numpy() for v in values])
#
# # Plot the heatmap for changes in parameter values over different tasks
# for name, values in parameters_with_ewc_values.items():
#     plt.figure(figsize=(10, 5))
#     heatmap = plt.imshow(values, cmap='coolwarm', aspect='auto')
#     plt.colorbar(heatmap, label='Parameter Value')
#     plt.xlabel('Task')
#     plt.ylabel('Parameter')
#     plt.title(f'Heatmap of {name} Value Changes')
#     plt.show()
