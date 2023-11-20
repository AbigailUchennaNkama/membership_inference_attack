import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

#Create shadow model dataset

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 10
num_samples_per_shadow = 25000
batch_size = 64
learning_rate = 0.001
epochs = 10

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
cifar_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cifar_test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


# Create train, validation, and test datasets
train_datasets = []
val_datasets = []
test_datasets = []

for _ in range(3):
    # Train and validation split for shadow model
    train_data, val_data = random_split(cifar_dataset, [num_samples_per_shadow, (len(cifar_dataset) - num_samples_per_shadow)])
    test_data, _ = random_split(cifar_test_dataset, [5000, len(cifar_test_dataset) - 5000])

    train_datasets.append(train_data)
    val_datasets.append(val_data)
    test_datasets.append(test_data)

# Create dataloaders for train, validation, and test datasets
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in train_datasets]
val_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in val_datasets]
test_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in test_datasets]


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
# Train three shadow models
shadow_models = []
for i, (shadow_train_loader, shadow_val_loader) in enumerate(zip(train_loaders, val_loaders)):
    print(f"Training Shadow Model {i+1}")

    # Create an instance of the ShadowNet
    shadow_model = ShadowNet(num_classes=num_classes, use_batchnorm=True, use_dropout=True)
    shadow_model.to(device)

    # Train the shadow model
    train_shadow_model(shadow_model, shadow_train_loader, shadow_val_loader, epochs=epochs)

    # Save the trained shadow model
    torch.save(shadow_model.state_dict(), f'shadow_model_{i+1}.pth')

    shadow_models.append(shadow_model)

print("Training of Shadow Models complete.")
