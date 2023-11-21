import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from alive_progress import alive_bar
import os
import shutil
from pathlib import Path
import random
from torchvision.datasets.utils import download_url
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = torch.Generator().manual_seed(42)

from test_loss import setup_model
model = setup_model()

def membership_inference(net, data_paths, label, train_data_paths, transform):
    """Compute membership inference per sample"""
    criterion = nn.CrossEntropyLoss(reduction="none")

    member_probs = []
    non_member_probs = []

    member_correct_pred = 0
    wrong_member_pred = 0
    non_member_correct_pred = 0
    wrong_non_member_pred = 0
    member_total = 0
    non_member_total = 0

    for path, true_label in zip(data_paths, label):
        from PIL import Image
        img = Image.open(path)  # Convert PosixPath to string
        transformed_image = transform(img).unsqueeze(0).to(DEVICE)
        net.eval()
        with torch.inference_mode():
            logits = net(transformed_image)
            pred_probs = torch.softmax(logits, dim=1)
            losses = criterion(logits, torch.tensor([true_label]).to(DEVICE))

        for idx, loss in enumerate(losses):
            if loss <= 0.000635:  # Threshold for membership
                member_total += 1
                if str(path) in [str(train_path) for train_path in train_data_paths]:
                    member_correct_pred += 1
                else:
                    wrong_member_pred += 1
            else:
                non_member_total += 1
                if str(path) not in [str(train_path) for train_path in train_data_paths]:
                    non_member_correct_pred += 1
                else:
                    wrong_non_member_pred += 1

            if str(path) in [str(train_path) for train_path in train_data_paths]:
                member_probs.append(pred_probs[0, true_label].item())
            else:
                non_member_probs.append(pred_probs[0, true_label].item())

    member_precision = (member_correct_pred / member_total) * 100 if member_total > 0 else 0
    non_member_precision = (non_member_correct_pred / non_member_total) * 100 if non_member_total > 0 else 0
    overall_accuracy = ((member_correct_pred + non_member_correct_pred) / (member_total + non_member_total)) * 100 if (member_total + non_member_total) > 0 else 0

    return member_total, non_member_total, \
        member_correct_pred, non_member_correct_pred, \
        wrong_non_member_pred, wrong_member_pred, \
        member_precision, non_member_precision, \
        overall_accuracy, member_probs, non_member_probs

def main():

    print("Running on device:", DEVICE.upper())

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # download and pre-process CIFAR10
    print("Creating cifar10 dataset...")
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    with tarfile.open("./cifar10.tgz", "r:gz") as tar:
        tar.extractall(path="./data")

    data_dir = "/content/data/cifar10"

    train_set = ImageFolder(data_dir + '/train', transform=transform)
    test_set = ImageFolder(data_dir + '/test', transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # create data paths
    test_data_paths = random.sample(list(Path(data_dir, "test").glob("*/*.png")), 500)
    train_data_paths = random.sample(list(Path(data_dir, "train").glob("*/*.png")), 500)

    train_test_paths = test_data_paths + train_data_paths
    random.shuffle(train_test_paths)


    class_to_label = {
        label: idx for idx, label in enumerate(
            sorted(set(path.parent.stem for path in train_test_paths))
            )
        }

    train_test_labels = [
        class_to_label[path.parent.stem] for path in train_test_paths
        ]

    print(f'Total train: {len(train_data_paths)}, Total test: {len(test_data_paths)}')
    print('---'*45)

    model = setup_model()
    model.eval()

    member_total, non_member_total, \
    member_correct_pred, non_member_correct_pred, \
    wrong_non_member_pred, wrong_member_pred, \
    member_precision, non_member_precision, \
    overall_accuracy, member_probs, non_member_probs = membership_inference(model, train_test_paths, train_test_labels, train_data_paths, transform)

    print(f'Number of train_member predicted: {member_total}')
    print(f'Number of non_member predicted: {non_member_total}')
    print(f'correct_member (True positive): {member_correct_pred}')
    print(f'correct_non_member (True negative): {non_member_correct_pred}')
    print(f'wrong_member (False positive): {wrong_member_pred}')
    print(f'wrong_non_member (False negative): {wrong_non_member_pred}')
    print(f'Member precision: {member_precision:.2f}%')
    print(f'Non-Member precision: {non_member_precision:.2f}%')
    print(f'Overall Accuracy: {overall_accuracy:.2f}%')

    # make confusion metrix
    y_true = np.array([1] * member_correct_pred + [0] * non_member_correct_pred +
                      [1] * wrong_member_pred + [0] * wrong_non_member_pred)
    y_pred = np.array([1] * member_correct_pred + [0] * non_member_correct_pred +
                      [0] * wrong_member_pred + [1] * wrong_non_member_pred)

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Non-Member', 'Member'],
            yticklabels=['Non-Member', 'Member'])

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print('--'*45)
    print('Membership inference based on prediction probabilities')
    # Plot ROC curve
    plt.figure()
    fpr, tpr, _ = roc_curve([1] * len(member_probs) + [0] * len(non_member_probs),
                            member_probs + non_member_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()



if __name__ == "__main__":
    main()  # Retrieve model and train_test_dataloader
