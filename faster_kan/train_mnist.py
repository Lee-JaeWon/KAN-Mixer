# Train on MNIST
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

from torch.profiler import profile, record_function, ProfilerActivity
from efficient_kan import KAN
from torchkan import KANvolver

from fasterkan.fasterkan import FasterKAN, FasterKANvolver
from torchsummary import summary

import optuna
from optuna.trial import TrialState

# Define transformations
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
    transforms.Normalize((0.5,), (0.5,))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform_train
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform_val
)

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
        

batch_size = 64
num_hidden = 64

trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size = batch_size, shuffle=False)


# Count parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define model
# Calculate total and trainable parameters
bool_flag = True # True # False

model_0 = FasterKAN([28 * 28, num_hidden,  10], grid_min = -1.2, grid_max = 0.2, num_grids = 8, exponent = 2, inv_denominator = 0.5, train_grid = bool_flag, train_inv_denominator = bool_flag).to(device)
total_params, trainable_params = count_parameters(model_0)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

model_1 = MLP(layers=[28 * 28, num_hidden*5, 10], device=device)
total_params, trainable_params = count_parameters(model_1)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

model_2 = KAN([28 * 28, num_hidden, 10], grid_size=5, spline_order=3).to(device)
total_params, trainable_params = count_parameters(model_2)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


model_3 = KANvolver([28 * 28, num_hidden, 10], polynomial_order=2, base_activation=nn.ReLU).to(device)
total_params, trainable_params = count_parameters(model_3)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Define model
model_4 = FasterKANvolver([ num_hidden*2, num_hidden,num_hidden//2,num_hidden//4, 10], grid_min = -1.2, grid_max = 1.2, num_grids = 8, exponent = 2, inv_denominator = 0.5, train_grid = bool_flag, train_inv_denominator = bool_flag, view = [-1, 1, 28, 28]).to(device)
total_params, trainable_params = count_parameters(model_4)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

#print(summary(model,(1,28,28)))
#print(summary(model_1,(1,28,28)))
#print(summary(model_2,(1,28,28)))
#print(summary(model_3,(1,28,28)))
print(summary(model_4,(1,784)))

model_last = model = model_0
print(summary(model_0,(1,784)))
model_last.to(device)

epochs = 100

# Define early stopping class
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



# Define optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1, verbose=True)

# Define loss
criterion = nn.CrossEntropyLoss()

#early_stopping = EarlyStopping(patience=5, min_delta=0.01)

for epoch in range(epochs):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Start CUDA timing
            #start_time = time.time()
            
            optimizer.zero_grad()

            # Record forward pass
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            output = model(images)
            #output = model(images)
                   
            loss = criterion(output, labels)
            loss.backward()
            
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Stop timing
            #end_time = time.time()
            
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

            # Print profiler results every 10 batches
            #if i % 50 == 0:
            #    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
    #early_stopping(val_loss)
    #if early_stopping.early_stop:
    #    print("Early stopping")
    #    break
        
"""EXPERIMENT FOR OPTUNA PAUSE TILL 1ST WEEK OF JUNE NO TIME TO IMPLEMENT
# Define Optuna objective function
def objective(trial):
    # Hyperparameters to tune
    batch_size = trial.suggest_int('batch_size', 16, 256, log=True)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4)
    factor = trial.suggest_loguniform('factor', 0.5,  0.9 )
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    # Model instantiation
    model = model_last
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=0, verbose=True)
    
    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    for epoch in range(15):
        # Train
        model.train()
        with tqdm(trainloader) as pbar:
            for images, labels in pbar:
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
        
        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                val_loss += criterion(output, labels).item()
                val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return val_accuracy

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
#""
# Apply the best hyperparameters
best_batch_size = 256#trial.params['batch_size']
best_lr = 1.0e-3#trial.params['lr']
best_weight_decay = 1.0e-5#trial.params['weight_decay']
best_factor = 0.7#trial.params['factor']

trainloader = DataLoader(trainset, batch_size=best_batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=best_batch_size, shuffle=False)

# Final model training with best hyperparameters
model = model_last
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=best_factor, patience=0, verbose=True)
criterion = nn.CrossEntropyLoss()

early_stopping = EarlyStopping(patience=5, min_delta=0.01)

for epoch in range(epochs):
    model.train()
    with tqdm(trainloader) as pbar:
        for _, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
    
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break


"""
#"""
#"""