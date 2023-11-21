import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from tqdm.auto import tqdm
from models import ShadowNet
import pandas as pd
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
from model_architecture import load_model, ShadowNet

#Create shadow model dataset
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

# Function to get shadow data
def get_shadow_data():
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

        return train_loaders, val_loaders, test_loaders




"""
Create datasets for the attack model using prediction posteriors from shadow models training and test data
"""

# Function to create an attack dataset using predictions posteriors from the shadow model
def create_attack_dataset(shadow_model, dataloader, is_member=True):
    shadow_model.to(device)
    shadow_model.eval()
    all_predictions = []
    labels = []

    with torch.no_grad():
        for inputs, lbl in tqdm(dataloader):
            inputs, lbl = inputs.to(device), lbl.to(device)
            logits = shadow_model(inputs)
            predictions = torch.softmax(logits, dim=1)
            top_k_values = torch.topk(predictions, k=3).values
            all_predictions.extend(top_k_values.cpu().detach().numpy())
            labels.extend([1 if is_member else 0] * len(predictions))

    return np.round(np.array(all_predictions), 7), np.array(labels)


def get_attack_data():
        #get shadow models
        shadow_model_1 = load_model('./models/shadow_model_1.pth', ShadowNet, num_classes=10)
        shadow_model_2 = load_model('./models/shadow_model_2.pth', ShadowNet, num_classes=10)
        shadow_model_3 = load_model('./models/shadow_model_3.pth', ShadowNet, num_classes=10)

        #get shadow train and test data
        train_loaders, _, test_loaders = get_shadow_data()

        # Create attack dataset for members
        print(f"Create attack dataset for members")
        predictions_members_1, labels_members_1 = create_attack_dataset(shadow_model_1, train_loaders[0], is_member=True)
        predictions_members_2, labels_members_2 = create_attack_dataset(shadow_model_2, train_loaders[1], is_member=True)
        predictions_members_3, labels_members_3 = create_attack_dataset(shadow_model_3, train_loaders[2], is_member=True)

        # Create attack dataset for non-members
        print(f"Create attack dataset for non-members")
        predictions_non_members_1, labels_non_members_1 = create_attack_dataset(shadow_model_1, test_loaders[0], is_member=False)
        predictions_non_members_2, labels_non_members_2 = create_attack_dataset(shadow_model_2, test_loaders[1], is_member=False)
        predictions_non_members_3, labels_non_members_3 = create_attack_dataset(shadow_model_3, test_loaders[2], is_member=False)

        # Combine the member and non-member datasets
        attack_prob_input = np.concatenate(
            [predictions_members_1, predictions_members_2, predictions_members_3,
            predictions_non_members_1, predictions_non_members_2, predictions_non_members_3],
            axis=0
        )
        attack_labels = np.concatenate(
            [labels_members_1, labels_members_2, labels_members_3,
            labels_non_members_1, labels_non_members_2, labels_non_members_3],
            axis=0
        )

        # Shuffle the data
        shuffle_indices = np.random.permutation(len(attack_labels))
        attack_inputs = attack_prob_input[shuffle_indices]
        attack_labels = attack_labels[shuffle_indices]

        # Create a DataFrame with column names p1, p2, ..., p10
        column_names = [f'p{i}' for i in range(1, attack_inputs.shape[1] + 1)]
        attack_data = pd.DataFrame(attack_inputs, columns=column_names)
        attack_data['labels'] = attack_labels


        # Convert to PyTorch tensors
        attack_inputs_tensor = torch.FloatTensor(attack_inputs)
        attack_labels_tensor = torch.LongTensor(attack_labels)

        #print(len(attack_loader))
        print(f'All done!\nlength of attack dataset: {len(attack_inputs)}')

        print(attack_data.head())


        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            attack_inputs_tensor, attack_labels_tensor, test_size=0.2, random_state=42
        )

        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Create DataLoader for training
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Create DataLoader for validation
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Create DataLoader for testing
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")
        print(f"Number of testing samples: {len(test_loader.dataset)}")

        return train_loader, val_loader, test_loader
