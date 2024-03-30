import sys

from cv2 import line
sys.path.append('/MACS Winter 22/GitHub Repositories/FedDG-Extension')
import os
import torch
from data.meta_dataset import MetaDataset, GetDataLoaderDict
from configs.default import vlcs_path
from torchvision import transforms

transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness= 0.3,contrast=0.3,saturation=0.3,hue=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_test = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

vlcs_name_dict = {
    'c': 'CALTECH',
    'l': 'LABELME',
    'v': 'PASCAL',
    's': 'SUN',
}

split_dict = {
    'train': 'train',
    'val': 'crossval',
    'total': 'test',
}


class VLCS_SingleDomain():
    def __init__(self, root_path=vlcs_path, domain_name='v', split='total', train_transform=None):
        if domain_name in vlcs_name_dict.keys():
            self.domain_name = vlcs_name_dict[domain_name]
            self.domain_label = list(vlcs_name_dict.keys()).index(domain_name)
        else:
            raise ValueError('domain_name should be in VLCS')
        self.root_path = root_path
        # self.root_path = os.path.join(root_path, 'raw_images')
        self.split = split
        # self.split_file = root_path + '/split_files/' + f'{self.domain_name}_{split_dict[self.split]}_kfold' + '.txt'
        # self.split_file = os.path.join(root_path, 'split_files', f'{self.domain_name}_{split_dict[self.split]}_kfold' + '.txt')

        if train_transform is not None:
            self.transform = train_transform
        else:
            self.transform = transform_test
        # print("Root path: ",self.root_path)
        imgs, labels = VLCS_SingleDomain.get_dataset(self.root_path, self.split)
        self.dataset = MetaDataset(imgs, labels, self.domain_label, self.transform)

    @staticmethod
    def get_dataset(root_path, split):
        imgs = []  # List to store class labels
        labels = []  # List to store file paths
        domain_names = os.listdir(root_path)  # Get the domain names from the directory

        for domain_name in domain_names:
            domain_path = root_path +'/' +domain_name
            # print("Domain Path: ",domain_path)
            split_names = os.listdir(domain_path)
            for split_data in split_names:
                if(split_data == split):
                    split_path = domain_path +'/' +split_data
                    # print("Split Path: ",split_path)
                    # Get the class names from the directory
                    class_names = os.listdir(split_path)
                    for class_name in class_names:
                        class_path = split_path + '/' +class_name
                        # print("Class Path: ", class_path)
                        if os.path.isdir(class_path):
                            files = os.listdir(class_path)
                            for file in files:
                                file_path = class_path +'/' +file
                                if os.path.isfile(file_path):
                                    imgs.append(file_path)
                                    labels.append(int(class_name))
        return imgs, labels

class VLCS_FedDG():
    def __init__(self, test_domain='v', batch_size=16):
        self.batch_size = batch_size
        self.domain_list = list(vlcs_name_dict.keys())
        self.test_domain = test_domain
        self.train_domain_list = self.domain_list.copy()
        self.train_domain_list.remove(self.test_domain)

        self.site_dataset_dict = {}
        self.site_dataloader_dict = {}
        for domain_name in self.domain_list:
            self.site_dataloader_dict[domain_name], self.site_dataset_dict[domain_name] = VLCS_FedDG.SingleSite(domain_name, self.batch_size)


        self.test_dataset = self.site_dataset_dict[self.test_domain]['test']
        self.test_dataloader = self.site_dataloader_dict[self.test_domain]['test']


    @staticmethod
    def SingleSite(domain_name, batch_size=16):
        dataset_dict = {
            'train': VLCS_SingleDomain(domain_name=domain_name, split='train', train_transform=transform_train).dataset,
            'val': VLCS_SingleDomain(domain_name=domain_name, split='val').dataset,
            'test': VLCS_SingleDomain(domain_name=domain_name, split='test').dataset,
        }
        dataloader_dict = GetDataLoaderDict(dataset_dict, batch_size)
        return dataloader_dict, dataset_dict

    def GetData(self):
        return self.site_dataloader_dict, self.site_dataset_dict
