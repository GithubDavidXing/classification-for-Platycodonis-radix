import os
os.environ["OMP_NUM_THREADS"] = "12"
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from process import *
import imgvision as iv

import copy
from network import NET

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def check_labels(labels, num_classes):
    assert labels.max() < num_classes, f"标签值超出范围: {labels.max()} >= {num_classes}"
    print(f"标签范围: {labels.min()} - {labels.max()}")

def train(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    printed = False  # 只打印一次图片 shape

    for images, labels, filename in train_loader:
        if not printed:
            # print(f"[Train] image shape: {images.shape}")
            printed = True

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total_samples
    average_loss = total_loss / len(train_loader)

    return accuracy, average_loss

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    val_name = []
    valt_labels = []
    valp_labels = []
    with torch.no_grad():
        for images, labels, filename in data_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_name.extend(filename)
            valt_labels.extend(labels.detach().cpu().numpy())
            valp_labels.extend(predicted.detach().cpu().numpy())

    print(f'验证集:{val_name}')
    print(f'真实类别:{valt_labels};预测类别:{valp_labels}')

    accuracy = correct / total_samples
    average_loss = total_loss / len(data_loader)

    return accuracy, average_loss


def plot_confusion_matrix(y_true, y_pred, class_names, fold, save_dir='./', band=0):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, ha='center')
    plt.yticks(tick_marks, class_names, va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', verticalalignment='center', color='black')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f'confusion_matrix_fold{fold}_band{band}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed = 1000
    setup_seed(seed)
    band = 184
    num_classes = 4

    log_path = './log_selectbandd/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + 'log_vgg_longepoch-band' + str(band) + '-' + 'seed' + str(seed) + '-' + time.strftime(
        "%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)

    print('随机数种子', seed)
    print(f'Using device: {device}')

    ave_acc = []
    batch_size = 32
    all_data = get_data('data', printf=False, arg=False, mode=1)

    labels = all_data.get_labels()
    print(f"数据集中的标签最大值: {np.array(labels).max()}")
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True)

    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold + 1}")
        print(f'label数量：{len(labels)}')
        train_labels = np.array(labels)[train_index]
        print("训练集序号:", train_index)
        print("训练集标签:", train_labels)
        test_labels = np.array(labels)[test_index]
        print("测试集序号:", test_index)
        print("测试集标签:", test_labels)
        print('-' * 30)

        train_subset = Subset(all_data, train_index)
        test_subset = Subset(all_data, test_index)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        val_loader = test_loader


        model = NET(band=band)
        model.to(device)

        for param in model.features.parameters():
            param.require_grad = False
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.5e-5, weight_decay=1e-4)


        best_valid_accuracy = 0.0
        best_model_state_dict = copy.deepcopy(model.state_dict())
        best_epoch = -1
        num_epochs = 1600

        for epoch in range(num_epochs):
            train_accuracy, train_loss = train(model, train_loader, criterion, optimizer, device, num_classes)
            valid_accuracy, valid_loss = evaluate(model, val_loader, criterion, device)
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy * 100:.2f}%, Train Loss: {train_loss:.4f}, '
                f'Validation Accuracy: {valid_accuracy * 100:.2f}%, Validation Loss: {valid_loss:.4f}')
            print('-' * 30)

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_epoch = epoch
                best_model_state_dict = copy.deepcopy(model.state_dict())
        print(f'Best Validation Accuracy: {best_valid_accuracy * 100:.2f}% at Epoch {best_epoch + 1}')

        model.load_state_dict(best_model_state_dict)
        test_accuracy, test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}')

        torch.save(model.state_dict(), 'best_model.pth')

        true_labels = []
        predicted_labels = []
        name = []

        model.eval()
        with torch.no_grad():
            for images, labels, filename in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                name.extend(filename)
                true_labels.extend(labels.detach().cpu().numpy())
                predicted_labels.extend(predicted.detach().cpu().numpy())

        class_names = ["Anhui", "Hebei", "Inner Mongolia", "Zhejiang"]
        plot_confusion_matrix(true_labels, predicted_labels, class_names, fold=fold + 1, save_dir='./cm/',
                              band=band)

        print('测试集真实类别和预测类别：')
        print(name)
        print(true_labels)
        print(predicted_labels)
        print('Training, evaluation, and confusion matrix plotting finished.')
        ave_acc.append(test_accuracy)
        labels = all_data.get_labels()

    average_accuracy = np.mean(ave_acc)
    print(f'分层交叉验证（Stratified K-Fold）平均准确率: {average_accuracy * 100:.2f}%')