from torchvision import datasets, transforms
import torch
from skimage import io
from torch.utils.data import Dataset
from base.base_data_loader import BaseDataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


class PlantTraitsDataLoader(BaseDataLoader):
    """
    PlantTraits data loading using BaseDataLoader
    """

    def __init__(self,
                 tabular_data_dir,
                 image_data_dir,
                 batch_size,
                 img_size,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1,
                 training=True):

        # initalize input attributes
        self.img_size = self._is_tuple(img_size)
        self.tabular_data_dir = tabular_data_dir
        self.image_data_dir = image_data_dir
        self.is_training = training

        # class names
        self.class_names = [
            'X4_mean',
            'X11_mean',
            'X18_mean',
            'X26_mean',
            'X50_mean',
            'X3112_mean',
        ]
        self.aux_class_names = list(
            map(lambda x: x.replace("mean", "sd"), self.class_names))

        # initalize attributes
        self.features = None
        self.dataset = None
        self.img_transform = None

        self._img_transformation()
        self.create_dataset()

        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)

    def create_dataset(self):
        tabular_data = self.load_tabular_data()
        self.dataset = PlantDataset(tabular_data,
                                    self.features,
                                    self.class_names,
                                    self.aux_class_names,
                                    self.img_size,
                                    img_transform=self.img_transform,
                                    with_labels=self.is_training)

    def load_tabular_data(self):

        if self.is_training:
            train_data = self._read_csv('train')
            train_aux_mean = train_data.loc[:, self.aux_class_names].mean()
            train_data.loc[:, self.
                        aux_class_names] = train_data.loc[:, self.
                                                            aux_class_names].fillna(
                                                                train_aux_mean)
            self.features = train_data.drop(columns=self.class_names + self.aux_class_names + ['id'] + ['image_path']).columns
            return train_data
        
        else:
            test_data = self._read_csv('test')
            self.features = test_data.columns[1:-1]
            return test_data


    def _read_csv(self, type):
        path = os.path.join(self.tabular_data_dir, type + '.csv')
        tabular_data = pd.read_csv(path)
        tabular_data[
            'image_path'] = f'{self.image_data_dir}/{type}_images/' + tabular_data[
                'id'].astype(str) + '.jpeg'
        return tabular_data

    def _is_tuple(self, img_size):
        if isinstance(img_size, (tuple, list)):
            return img_size
        return (img_size, img_size)

    def _img_transformation(self):
        if self.is_training:
            self.img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=(self.img_size[0],
                                                   self.img_size[1]),
                                             scale=(0.05, 0.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.img_transform = None


class PlantDataset(Dataset):

    def __init__(self,
                 df,
                 feature_names,
                 class_names,
                 aux_class_names,
                 image_size,
                 img_transform=None,
                 with_labels=False):

        self.df = df
        self.class_names = class_names
        self.aux_class_names = aux_class_names
        self.img_transform = img_transform
        self.feature_names = feature_names
        self.image_size = image_size
        self.with_labels = with_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = torch.tensor(self.df.loc[idx, self.feature_names], dtype=torch.float32)

        labels = torch.tensor([0], dtype=torch.float32)
        aux_labels = torch.tensor([0], dtype=torch.float32)

        if self.with_labels:
            labels = self.df.loc[idx, self.class_names]
            aux_labels = self.df.loc[idx, self.aux_class_names]
            labels = torch.tensor(labels)
            aux_labels = torch.tensor(aux_labels)

        img_name = self.df.loc[idx, 'image_path']

        image = io.imread(img_name)

        if self.img_transform:
            image = self.img_transform(image)
        else:
            image = self._image_decoder(image)

        sample = {
            'image': image,
            'features': features,
            'labels': labels,
            'aux_labels': aux_labels
        }

        return sample

    def _image_decoder(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        return transform(image)
