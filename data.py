import numpy as np
import tensorflow as tf
import random
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader


def normalize(x):
    return (x - x.min()) * (2 * np.pi / (x.max() - x.min()))

def data_load_and_process(dataset="kmnist", reduction_sz: int = 4):
    data_path = "/Users/jwheo/Desktop/Y/OtherCodesForStudy/QCDS/data"
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == "kmnist":
        # Path to training images and corresponding labels provided as numpy arrays
        kmnist_train_images_path = f"{data_path}/kmnist-train-imgs.npz"
        kmnist_train_labels_path = f"{data_path}/kmnist-train-labels.npz"

        # Path to the test images and corresponding labels
        kmnist_test_images_path = f"{data_path}/kmnist-test-imgs.npz"
        kmnist_test_labels_path = f"{data_path}/kmnist-test-labels.npz"

        x_train = np.load(kmnist_train_images_path)["arr_0"]
        y_train = np.load(kmnist_train_labels_path)["arr_0"]

        # Load the test data from the corresponding npz files
        x_test = np.load(kmnist_test_images_path)["arr_0"]
        y_test = np.load(kmnist_test_labels_path)["arr_0"]

    x_train, x_test = (
        x_train[..., np.newaxis] / 255.0,
        x_test[..., np.newaxis] / 255.0,
    )
    train_filter_tf = np.where((y_train == 0) | (y_train == 1))
    test_filter_tf = np.where((y_test == 0) | (y_test == 1))

    x_train, y_train = x_train[train_filter_tf], y_train[train_filter_tf]
    x_test, y_test = x_test[test_filter_tf], y_test[test_filter_tf]

    x_train = tf.image.resize(x_train, (256, 1)).numpy()
    x_test = tf.image.resize(x_test, (256, 1)).numpy()
    x_train, x_test = tf.squeeze(x_train).numpy(), tf.squeeze(x_test).numpy()

    X_train = PCA(reduction_sz).fit_transform(x_train)
    X_test = PCA(reduction_sz).fit_transform(x_test)

    X_train_sliced = X_train[:400]
    X_val_sliced = X_train[400:500]
    X_test_sliced = X_test[:100]

    y_train_sliced = y_train[:400]
    y_val_sliced = y_train[400:500]
    y_test_sliced = y_test[:100]

    x_train, x_val, x_test = [], [], []
    for x in X_train_sliced:
        x = normalize(x)
        x_train.append(x)
    for x in X_val_sliced:
        x = normalize(x)
        x_val.append(x)
    for x in X_test_sliced:
        x = normalize(x)
        x_test.append(x)

    return x_train, x_val, x_test, y_train_sliced, y_val_sliced, y_test_sliced


class PairDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data  # 전처리된 데이터 (list 또는 numpy array)
        self.targets = targets  # 해당 라벨

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 첫 번째 샘플: index 사용
        x1 = self.data[index]
        y1 = self.targets[index]
        # 두 번째 샘플: 무작위 선택 (동일 샘플이 선택될 수도 있음)
        idx2 = random.randint(0, len(self.data) - 1)
        while idx2 == index:
            idx2 = random.randint(0, len(self.data) - 1)
        x2 = self.data[idx2]
        y2 = self.targets[idx2]
        # similarity label: 두 라벨이 같으면 1, 아니면 0
        label = 1 if y1 == y2 else 0
        return x1, x2, label


def NQEDataLoaders(batch_size=64, test_batch_size=1000, dataset="kmnist", reduction_sz=4):
    X_train, X_val, X_test, y_train, y_val, y_test = data_load_and_process(dataset=dataset, reduction_sz=reduction_sz)

    train = PairDataset(X_train, y_train)
    val = PairDataset(X_val, y_val)
    test = PairDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test, batch_size=test_batch_size, shuffle=True, pin_memory=True)

    return train_loader, val_loader, test_loader