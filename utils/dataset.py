
import os
import pickle

import torch
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, dataset_path, image_files, labels, transform=None):
        super(BaseDataset, self).__init__()
        self.dataset_path = dataset_path
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_file = self.image_files[idx]
        image_file = os.path.join(self.dataset_path, image_file)
        image = Image.open(image_file)
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class UNIDataloader:
    def __init__(self, cfg, transform=None):
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.transform = transform
        with open(cfg.DATASET.PKL, 'rb') as f:
            self.info = pickle.load(f)

        self.seenclasses = self.info['seenclasses'].to(cfg.DEVICE)
        self.unseenclasses = self.info['unseenclasses'].to(cfg.DEVICE)

        self.classnames = self.info['classnames']
        self.attributes = self.info['attributes']
        self.att2cls = torch.from_numpy(self.info['att2cls'])

        (self.train_set,
         self.test_seen_set,
         self.test_unseen_set) = self.torch_dataset()

        self.train_loader = DataLoader(self.train_set,
                                       batch_size=cfg.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=cfg.WORKERS)
        self.test_seen_loader = DataLoader(self.test_seen_set,
                                           batch_size=cfg.BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=cfg.WORKERS)
        self.test_unseen_loader = DataLoader(self.test_unseen_set,
                                             batch_size=cfg.BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=cfg.WORKERS)

    def torch_dataset(self):
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.Resize(self.cfg.INPUT_SIZE),
                transforms.CenterCrop(self.cfg.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        baseset = BaseDataset(self.cfg.DATASET.PATH,
                              self.info['image_files'],
                              self.info['labels'],
                              self.transform)

        train_set = Subset(baseset, self.info['trainval_loc'])
        test_seen_set = Subset(baseset, self.info['test_seen_loc'])
        test_unseen_set = Subset(baseset, self.info['test_unseen_loc'])

        return train_set, test_seen_set, test_unseen_set
