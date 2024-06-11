import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Callable, Dict, Tuple
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.profiler import profile, record_function, ProfilerActivity
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'faster_kan'))

try:
    from fasterkan.fasterkan import FasterKAN, FasterKANvolver
    from torchkan import KANvolver
    from efficient_kan import KAN
except:
    from faster_kan.fasterkan.fasterkan import FasterKAN, FasterKANvolver
    from faster_kan.torchkan import KANvolver
    from faster_kan.efficient_kan import KAN

from torchsummary import summary

import optuna
from optuna.trial import TrialState

from mixer_model import MLPMixer, KANMixer

# Define transformations
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the full CIFAR-100 dataset
full_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
batch_size = 32
# Define the split sizes
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split the dataset indices
train_indices, test_val_indices = train_test_split(list(range(len(full_dataset))), test_size=(test_size + val_size), random_state=42)
val_indices, test_indices = train_test_split(test_val_indices, test_size=test_size, random_state=42)

# Create the train, val, and test sets
trainset = Subset(full_dataset, train_indices)
valset = Subset(full_dataset, val_indices)
testset = Subset(full_dataset, test_indices)

# Create DataLoaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], device: str):
        super().__init__()
        self.layer1 = nn.Linear(layers[0], layers[1], device=device)
        self.layer2 = nn.Linear(layers[1], layers[2], device=device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.sigmoid(x)
        return x
        
num_hidden = 128

# Count parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Define models
input_dim = 3072
bool_flag = False
num_classes = 100

model_0 = FasterKAN([input_dim, 1, num_hidden, num_hidden // 2, num_hidden // 4, num_classes], grid_min=-1.2, grid_max=1.2, num_grids=64, exponent=2, inv_denominator=0.5, train_grid=bool_flag, train_inv_denominator=bool_flag).to(device)
model_1 = MLP(layers=[input_dim, num_hidden * 5, num_classes], device=device)
model_2 = KAN([input_dim, num_hidden, num_classes], grid_size=5, spline_order=3).to(device)
model_3 = KANvolver([input_dim, num_hidden, num_classes], polynomial_order=2, base_activation=nn.ReLU).to(device)
model_4 = FasterKANvolver([num_hidden * 2, num_hidden, num_hidden // 2, num_hidden // 4, num_classes], grid_min=-1.2, grid_max=1.2, num_grids=8, exponent=2, inv_denominator=0.5, train_grid=bool_flag, train_inv_denominator=bool_flag, view=[-1, 3, 32, 32]).to(device)

# Ours :: S/16 Model
input_dim = 32 * 32 * 3  # CIFAR-1
# model_names = ["MLPMixer"] # 2x2 = 4 patches
hidden_dim = 512  # Hidden dimension for KANMixer-S/16
tokens_kan_dim = 256  # Tokens KAN dimension for KANMixer-S/16
channels_kan_dim = 2048  # Channels KAN dimension for KANMixer-S/16
num_classes = 100  # CIFAR-100 has 100 classes
num_patches = (32 // 16) * (32 // 16)  # 2x2 = 4 patches

# Create the model and move it to the desired device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_5 = KANMixer(input_dim, num_patches, hidden_dim, tokens_kan_dim, channels_kan_dim, num_classes).to(device)

# Baseline
input_dim = 32 * 32 * 3  # CIFAR-100 images are 32x32 with 3 color channels
num_patches = (32 // 16) * (32 // 16)  # 2x2 = 4 patches
hidden_dim = 512  # Hidden dimension for MLP-Mixer-S/16
tokens_mlp_dim = 256  # Tokens MLP dimension for MLP-Mixer-S/16
channels_mlp_dim = 2048  # Channels MLP dimension for MLP-Mixer-S/16
num_classes = 100  # CIFAR-100 has 100 classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_6 = MLPMixer(input_dim, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, num_classes).to(device)

models = [model_0, model_1, model_2, model_5, model_6]
model_names = ["FasterKAN", "MLP", "KAN", "KANMixer", "MLPMixer"]

# models = [model_6]
# model_names = ["MLPMixer"]

# Training and validation loop for each model
results = []
loss_trends = {name: {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []} for name in model_names}

epochs = 100

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

import os

os.makedirs('./images', exist_ok=True)

for model, model_name in zip(models, model_names):
    print(f"Training {model_name}...")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_epoch = 0
        train_accuracy_epoch = 0

        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, input_dim).to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_epoch += loss.item()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                train_accuracy_epoch += accuracy.item()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
        
        train_loss_epoch /= len(trainloader)
        train_accuracy_epoch /= len(trainloader)
        loss_trends[model_name]["train_loss"].append(train_loss_epoch)
        loss_trends[model_name]["train_accuracy"].append(train_accuracy_epoch)
        
        # Validation
        model.eval()
        val_loss_epoch = 0
        val_accuracy_epoch = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, input_dim).to(device)
                labels = labels.to(device)
                output = model(images)
                val_loss_epoch += criterion(output, labels).item()
                val_accuracy_epoch += (output.argmax(dim=1) == labels).float().mean().item()
        
        val_loss_epoch /= len(valloader)
        val_accuracy_epoch /= len(valloader)
        loss_trends[model_name]["val_loss"].append(val_loss_epoch)
        loss_trends[model_name]["val_accuracy"].append(val_accuracy_epoch)
        
        # Update learning rate
        scheduler.step(val_loss_epoch)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss_epoch}, Train Accuracy: {train_accuracy_epoch}, Val Loss: {val_loss_epoch}, Val Accuracy: {val_accuracy_epoch}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Write results to text file
        with open(f'results_cifar{num_classes}.txt', 'a') as f:
            f.write(f"Model: {model_name}, Epoch: {epoch + 1}, Train Loss: {train_loss_epoch}, Train Accuracy: {train_accuracy_epoch}, Val Loss: {val_loss_epoch}, Val Accuracy: {val_accuracy_epoch}\n")
        
        # # Early stopping
        # early_stopping(val_loss_epoch)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     f.write("Early stopping\n")
        #     break
    
    model.eval()
    test_loss_epoch = 0
    test_accuracy_epoch = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(-1, input_dim).to(device)
            labels = labels.to(device)
            output = model(images)
            test_loss_epoch += criterion(output, labels).item()
            test_accuracy_epoch += (output.argmax(dim=1) == labels).float().mean().item()
    
    test_loss_epoch /= len(testloader)
    test_accuracy_epoch /= len(testloader)
    print(f"Test Loss: {test_loss_epoch}, Test Accuracy: {test_accuracy_epoch}")
    
    with open(f'results_cifar{num_classes}.txt', 'a') as f:
        f.write(f"TEST :: Model: {model_name}, Test Loss: {test_loss_epoch}, Test Accuracy: {test_accuracy_epoch}\n")

    results.append((model_name, train_loss_epoch, train_accuracy_epoch, val_loss_epoch, val_accuracy_epoch, test_loss_epoch, test_accuracy_epoch))
    
    model.to('cpu')  # Move model to CPU after training

# Print loss trends and save plots
import matplotlib.pyplot as plt

# for model_name in model_names:
#     plt.plot(loss_trends[model_name]["train_loss"], label=f'{model_name} Train Loss')
#     plt.plot(loss_trends[model_name]["val_loss"], label=f'{model_name} Val Loss')
#     plt.title(f'{model_name} Loss Trend')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(f'./images/new_{model_name}_cifar{num_classes}.png')
#     plt.clf()  # Clear the current figure for the next plot

plt.figure(figsize=(10, 6))
for model_name in model_names:
    plt.plot(loss_trends[model_name]["train_loss"], label=f'{model_name} Train Loss')
plt.title('Combined Training Loss for All Models')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig(f'./images/new_combined_training_loss_cifar{num_classes}.png')
plt.clf()  # Clear the current figure
