import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

# Function to create an attack dataset using predictions from the shadow model
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

#get shadow models
shadow_model_1 = load_model('./shadow_model_1.pth', ShadowNet, num_classes=10)
shadow_model_2 = load_model('./shadow_model_2.pth', ShadowNet, num_classes=10)
shadow_model_3 = load_model('./shadow_model_3.pth', ShadowNet, num_classes=10)

#get shadow models

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

# Create DataLoader for training
attack_dataset = torch.utils.data.TensorDataset(attack_inputs_tensor, attack_labels_tensor)
attack_loader = torch.utils.data.DataLoader(attack_dataset, batch_size=64, shuffle=True)
#print(len(attack_loader))
print(f'All done!\nlength of attack dataset: {len(attack_inputs)}')

attack_data.head()


# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    attack_inputs, attack_labels, test_size=0.2, random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create DataLoader for validation
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create DataLoader for testing
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of testing samples: {len(test_loader.dataset)}")



class BinaryClassifier(nn.Module):
    def __init__(self, input_size=3):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    scheduler.step()

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 10):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           scheduler=scheduler)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        torch.save(model.state_dict(), "attack_model.pth")
    # 6. Return the filled results at the end of the epochs
    return results


# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 10

binary_model = BinaryClassifier()
binary_model = binary_model.to(device)

loss_fn = nn.CrossEntropyLoss()
# Create an optimizer
#optimizer = torch.optim.SGD(params=binary_model.parameters(),
                            #lr=0.001)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(binary_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model
model_results = train(model=binary_model,
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS
                      )

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
