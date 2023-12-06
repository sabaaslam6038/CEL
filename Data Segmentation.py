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
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
sns.set_style("whitegrid")
from mystyle import *
set_custom_plot_settings()

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

def plottingData():
    plt.figure(figsize=(12, 8))
    plt.plot(df['value'], color='blue', linewidth=2)
    n = 45  # Display every 10th date label
    dates = df['Date'].dt.strftime('%d-%m-%Y').iloc[::n]
    plt.xticks(ticks=range(1, len(df['Date']) + 1, n), labels=dates, rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Date', fontsize=16, fontweight='bold', labelpad=12)
    plt.ylabel('New Cases', fontsize=16, fontweight='bold')
    plt.title('Mpox Disease outbreak Data', fontsize=18, fontweight='bold')
    plt.legend(['New Mpox Cases'], fontsize=14)
    plt.tight_layout()
    ax = plt.gca()
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.grid(True)
    plt.show()

def display_task_info(task_data, task_index):
    stats = task_data['value'].describe()   # Get statistics
    start_date = task_data['Date'].iloc[0]# Extract start and end dates
    end_date = task_data['Date'].iloc[-1]
    start_date_dt = datetime.strptime(start_date, '%d-%m-%y') # Convert start and end dates to datetime objects
    end_date_dt = datetime.strptime(end_date, '%d-%m-%y')
    num_days = (end_date_dt - start_date_dt).days + 1  # +1 to include both starting and ending dates, # Calculate the number of days in the task
    task_info_df.loc[task_index] = [task_index + 1, stats['count'], stats['mean'], stats['std'], stats['min'], stats['25%'], stats['50%'], stats['75%'], stats['max'], start_date, end_date, num_days]   # Append the extracted data to the DataFrame

def plottingVoilin():
    sns.set_style("whitegrid")
    # Create the violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(x="Context", y="Value", data=plot_data, inner="box", cut=1, box=True)
    plt.title('Value Distribution across Contexts', fontsize=18, fontweight='bold', color='black')
    plt.ylabel('Value', fontsize=16, color='black', fontweight='bold')
    plt.xlabel('Contexts', fontsize=16, color='black', fontweight='bold', labelpad=15)
    plt.tight_layout()
    # Changing the text color to black
    ax = plt.gca()
    ax.tick_params(axis='x', colors='black', labelsize=14)
    ax.tick_params(axis='y', colors='black', labelsize=14)
    ax.grid(True)
    plt.show()


######################### MAIN  #########################
df=pd.read_csv('data_file')
for i in range(1,8):
    df[f'value_lag_{i}']= df['value'].shift(i)
df.dropna(inplace=True)
df.to_csv('africa_mpox_output.csv', index=False)
# Convert the 'Date' column to datetime format if it's not already
df['Date'] = pd.to_datetime(df['Date'])
# plotting whole data
plottingData()


input_dim = 0  # assuming  12 features
hidden_dim = 32
output_dim = 1
batch_size = 32
epochs = 200
ewc_lambda = 10
num_tasks = 10

# Now reading the OUTPUT data
df = pd.read_csv('data_file.csv')

task_info_df = pd.DataFrame(columns=["Task", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Start Date", "End Date", "Number of Days"])
all_tasks = train_test_split_tasks(df, num_tasks)
# Loop through tasks and extract the info
for idx, task in enumerate(all_tasks):
    display_task_info(task, idx)
all_data = [(load_data(task)) for task in all_tasks]
task_info_df.to_csv("africa_mpox_task_info.csv", index=False)

##Prepare the data for violin plots
all_value_data = []
all_task_labels = []
for idx, task in enumerate(all_tasks):
    all_value_data.extend(task['value'].tolist())
    all_task_labels.extend([f'Context {idx + 1}'] * len(task['value']))
plot_data = pd.DataFrame({
    'Context': all_task_labels,
    'Value': all_value_data
})
plottingVoilin()



