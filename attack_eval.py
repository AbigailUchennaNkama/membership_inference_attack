import torch
from torch import nn
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from model_architecture import BinaryClassifier
import numpy as np
from dataset import get_attack_data

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move values to device
torch.manual_seed(42)

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):

    """Evaluates a the attack model on the test dataset.
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

def eval():
    binary_model = BinaryClassifier()
    binary_model = binary_model.to(device)
    _, _, test_loader = get_attack_data()
    true_labels, predicted_scores, model_results = eval_model(model=binary_model,
                                                              data_loader=test_loader,
                                                              loss_fn=loss_fn, accuracy_fn=accuracy_fn,
                                                              device=device
                                                              )
    print(model_results)

    class_names = ['non-member', 'member']
    predicted_labels = np.argmax(predicted_scores, axis=1)

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, class_names)

    # Plot ROC curve
    plot_roc_curve(true_labels, predicted_scores)

if __name__ == "__main__":
    eval()
