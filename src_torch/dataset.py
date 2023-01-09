import csv
import numpy as np
import os
import torch.utils.data as data

from PIL import Image
from torchvision import datasets

import h5py

class CelebA_attr(data.Dataset):
    def __init__(self, data_root, train, transforms):
        self.split = 'train' if train else 'test'
        self.dataset = datasets.CelebA(root=data_root, split=self.split,
                                       target_type='attr', download=False)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1)\
                    + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


class GTSRB(data.Dataset):
    def __init__(self, data_root, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(data_root, 'GTSRB/Train')
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(data_root, 'GTSRB/Test')
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + '/' + format(c, '05d') + '/'
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = np.array(images)[indices]
        labels = np.array(labels)[indices]
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, 'GT-final_test.csv')
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + '/' + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CustomGTSRBDataSet(data.Dataset):
    def __init__(self, data_file, is_train=False, transform=False):
        self.is_train = is_train
        self.transform = transform

        dataset = load_dataset_h5(data_file + '/gtsrb.h5', keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

        x_train = dataset['X_train']
        y_train = np.argmax(dataset['Y_train'], axis=1)
        x_test = dataset['X_test']
        y_test = np.argmax(dataset['Y_test'], axis=1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        if is_train:
            self.x = x_train
            self.y = y_train
        else:
            self.x = x_test
            self.y = y_test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_dataset_h5(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset
