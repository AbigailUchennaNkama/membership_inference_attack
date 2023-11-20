from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt



class BinaryClassifier(nn.Module):
    def __init__(self, input_size=3):  # Adjust input_size to match the data size
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def load_attack_model():
    local_path = './attack_model.pth'
    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    # Initialize the model
    model = BinaryClassifier()
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()
    return model

# Move values to device
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.
    """
    loss, acc = 0, 0
    true_labels = []
    predicted_scores = []

    model.eval()
    with torch.inference_mode():
        for inputs, labels in data_loader:
            # Send data to the target device
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            predictions = torch.softmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_scores.extend(predictions.cpu().numpy())
            loss += loss_fn(logits, labels)
            acc += accuracy_fn(y_true=labels, y_pred=predictions.argmax(dim=1))

        # Scale loss and acc
        loss /= len(data_loader)
        acc /= len(data_loader)
    return np.array(true_labels), np.array(predicted_scores), {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc
        }

# Calculate model 1 results with device-agnostic code
true_labels, predicted_scores, model_results = eval_model(model=binary_model, data_loader=test_loader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn,
    device=device
)
model_results


#plot confusion metrix
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# plot roc curve
def plot_roc_curve(y_true, y_scores, title='Receiver Operating Characteristic (ROC) Curve'):
    plt.figure(figsize=(8, 6))


    fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2,color='darkorange', label=f'Class (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([0, 0, 1], [0, 1, 1],
         linestyle = ':',
         color='gray'
         )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

class_names = ['non-member', 'member']

# Get true labels and predicted probabilities for the test set
#true_labels, predicted_scores = get_true_labels_and_scores(binary_model, test_loader)

# Convert predicted probabilities to predicted class labels
predicted_labels = np.argmax(predicted_scores, axis=1)

# Plot confusion matrix
plot_confusion_matrix(true_labels, predicted_labels, class_names)

# Plot ROC curve
plot_roc_curve(true_labels, predicted_scores)
