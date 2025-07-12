#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.sampling import *
from utils.dataset_utils import separate_data, read_record
from utils.FEMNIST import FEMNIST
from utils.ShakeSpeare import ShakeSpeare
from utils import mydata
from torch.autograd import Variable
import torch.nn.functional as F
import os
import json

def get_dataset(args):
    # CPU optimization: Force single-threaded data loading
    args.num_workers = 0
    args.pin_memory = False

    file = os.path.join("data", args.dataset + "_" + str(args.num_users))
    if args.iid:
        file += "_iid"
    else:
        file += "_noniidCase" + str(args.noniid_case)

    if args.noniid_case > 4:
        file += "_beta" + str(args.data_beta)

    file += ".json"
    
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.generate_data:
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        else:
            dict_users = read_record(file)
            
    elif args.dataset == 'cifar10':
        # CPU-optimized transforms (same as original but ensuring consistency)
        trans_cifar10_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
            
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset_train = mydata.CIFAR100_coarse('../data/cifar100_coarse', train=True, download=True,
                                               transform=trans_cifar100)
        dataset_test = mydata.CIFAR100_coarse('../data/cifar100_coarse', train=False, download=True,
                                              transform=trans_cifar100)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
            
    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist/', train=False, download=True, transform=trans)
        if args.generate_data:
            if args.iid:
                dict_users = fashion_mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = fashion_mnist_noniid(dataset_train, args.num_users, case=args.noniid_case)
        else:
            dict_users = read_record(file)
            
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(True)
        dataset_test = FEMNIST(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        
    elif args.dataset == 'ShakeSpeare':
        dataset_train = ShakeSpeare(True)
        dataset_test = ShakeSpeare(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        
    else:
        exit('Error: unrecognized dataset')

    if args.generate_data:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        with open(file, 'w') as f:
            dataJson = {
                "dataset": args.dataset,
                "num_users": args.num_users,
                "iid": args.iid,
                "noniid_case": args.noniid_case,
                "data_beta": args.data_beta,
                "train_data": dict_users
            }
            json.dump(dataJson, f)

    return dataset_train, dataset_test, dict_users


def get_cifar10_dataset(num_users, iid=True, alpha=0.5):
    """
    Get CIFAR-10 dataset with CPU optimizations for Decentralized GitFL
    
    Args:
        num_users: Number of users (not used in this simple version)
        iid: Whether to use IID distribution
        alpha: Dirichlet alpha parameter for non-IID data
    
    Returns:
        train_loader, test_loader, dataset_train, dataset_test
    """
    # CPU-optimized transforms (consistent with original GitFL)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create datasets
    dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
    dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    
    # CPU-optimized DataLoader settings
    train_loader = DataLoader(
        dataset_train, 
        batch_size=32,      # Reduced for CPU optimization
        shuffle=True, 
        num_workers=0,      # Single-threaded for CPU
        pin_memory=False    # Disable for CPU
    )
    
    test_loader = DataLoader(
        dataset_test, 
        batch_size=64,      # Slightly larger for evaluation
        shuffle=False, 
        num_workers=0,      # Single-threaded for CPU
        pin_memory=False    # Disable for CPU
    )
    
    return train_loader, test_loader, dataset_train, dataset_test


def get_cpu_optimized_dataloader(dataset, batch_size=32, shuffle=True, drop_last=False):
    """
    Helper function to create CPU-optimized DataLoader
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size (default 32 for CPU optimization)
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader with CPU optimizations
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      # Single-threaded
        pin_memory=False,   # Disable for CPU
        drop_last=drop_last,
        persistent_workers=False  # Don't keep workers alive
    )