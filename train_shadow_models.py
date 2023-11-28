import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
from model_architecture import ShadowNet
from get_data import get_shadow_data

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 10
num_samples_per_shadow = 25000
batch_size = 64
learning_rate = 0.001
epochs = 10

train_loaders, val_loaders, test_loaders = get_shadow_data()

# Function to train a shadow model
def train_shadow_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    model.to(device)  # Move the model to the GPU if available

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted_train = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_train.eq(labels).sum().item()

        scheduler.step()

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                _, predicted_val = val_outputs.max(1)
                total_val += val_labels.size(0)
                correct_val += predicted_val.eq(val_labels).sum().item()


        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Accuracy: {100 * correct_train / total_train:.2f}%, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Accuracy: {100 * correct_val / total_val:.2f}%')

# Train three shadow models
#get dataloaders
train_loaders, val_loaders, test_loaders = get_shadow_data()

for i, (shadow_train_loader, shadow_val_loader) in enumerate(zip(train_loaders, val_loaders)):
    print(f"Training Shadow Model {i+1}")

    shadow_model = ShadowNet(num_classes=num_classes, use_batchnorm=True, use_dropout=True)
    shadow_model.to(device)

    # Train the shadow model
    train_shadow_model(shadow_model, shadow_train_loader, shadow_val_loader, epochs=epochs)

    torch.save(shadow_model.state_dict(), f'shadow_model_{i+1}.pth')

print("Training of Shadow Models complete.")
