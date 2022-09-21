'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from loader import Loader2
from utils import progress_bar
from models import *
from load_cifar import load_cifar10

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
import yaml
import random

random_seed = 9999
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config_path', default='yml/debug.yaml')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

log_name = 'lr_' + str(config['optimizer']['lr']) + 'momentum_' + str(config['optimizer']['momentum']) + \
           'batch_' + str(config['batch_size'])
log_path = os.path.join('log', log_name, config['name'])
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(level=logging.ERROR, filename=log_path+'/result.txt', format='%(message)s')
logging.error('main.py')

best_acc = 0
start_epoch = 0

# Data
train_x, train_y, test_x, test_y, train_path, test_path = load_cifar10('data/cifar-10-batches-py')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# we can also use PyTorch official CIFAR10 dataset if torchvision.datasets.CIFAR10 is imported
# testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testset = Loader2(test_x, test_y, test_path, is_train=False, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=config['num_workers'], pin_memory=True,)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net.linear = nn.Linear(512, 4)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(log_path+'/checkpoint/rotation.pth')['net'])
net.linear = nn.Linear(512, 10)
net = net.to(device)


# Training
def train(net, criterion, optimizer, epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    logging.error(f'train_loss: {train_loss/(batch_idx+1)} train_acc: {100.*correct/total}')
    return correct/total

def test(net, criterion, epoch, cycle):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    logging.error(f'test_loss: {test_loss/(batch_idx+1)} test_acc: {100.*correct/total}')
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(log_path+'/checkpoint'):
            os.makedirs(log_path+'/checkpoint', exist_ok=True)
        torch.save(state, log_path+f'/checkpoint/main_{cycle}.pth')
        best_acc = acc

    return correct/total

# class-balanced sampling (pseudo labeling)
def get_plabels(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(10)]

    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True,)

    # overflow goes into remaining
    remaining = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if len(class_dict[predicted.item()]) < 100:
                class_dict[predicted.item()].append(samples[idx])
            else:
                remaining.append(samples[idx])
            progress_bar(idx, len(ploader))

    sample1k = []
    for items in class_dict.values():
        if len(items) == 100:
            sample1k.extend(items)
        else:
            # supplement samples from remaining 
            sample1k.extend(items)
            add = 100 - len(items)
            sample1k.extend(remaining[:add])
            remaining = remaining[add:]
    
    return sample1k

# confidence sampling (pseudo labeling)
def get_plabels2(net, samples, cycle):
    # dictionary with 10 keys as class labels
    class_dict = {}
    [class_dict.setdefault(x,[]) for x in range(10)]

    sample1k = []
    sub5k = Loader2(train_x, train_y, train_path, is_train=True,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            scores, predicted = outputs.max(1)
            # save top1 confidence score 
            outputs = F.normalize(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            top1_scores.append(probs[0][predicted.item()].cpu())
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[:1000]]

# entropy sampling
def get_plabels3(net, samples, cycle):
    sample1k = []
    sub5k = Loader2(is_train=False,  transform=transform_test, path_list=samples)
    ploader = torch.utils.data.DataLoader(sub5k, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True,)

    top1_scores = []
    net.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(ploader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            e = -1.0 * torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1)
            top1_scores.append(e.view(e.size(0)))
            progress_bar(idx, len(ploader))
    idx = np.argsort(top1_scores)
    samples = np.array(samples)
    return samples[idx[-1000:]]


if __name__ == '__main__':
    labeled = []
    accuracy = []
    datas = []

    def weight_reset(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    CYCLES = 10
    for cycle in range(CYCLES):
        logging.error(f'{cycle} iteration')
        criterion = nn.CrossEntropyLoss()
        def make_optimizer(params, name, **kwargs):
            return optim.__dict__[name](params, **kwargs)
        optimizer = make_optimizer(net.parameters(),**config['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        best_acc = 0
        print('Cycle ', cycle)

        with open(log_path+f'/loss/batch_{cycle}.txt', 'r') as f:
            samples = f.readlines()
            
        if cycle > 0:
            print(f'load pre-best weights')
            net.load_state_dict(torch.load(log_path+f'/checkpoint/main_{cycle-1}.pth')['net'])
            sample1k = get_plabels2(net, samples, cycle)
        
        else:
            samples = np.array(samples)
            sample1k = samples[[j*5 for j in range(1000)]]
            
        # add 1k samples to labeled set
        labeled.extend(sample1k)
        print(f'>> Labeled length: {len(labeled)}')
        trainset = Loader2(train_x, train_y, train_path, is_train=True, transform=transform_train, path_list=labeled)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True,)

        # train DL model with SGD optimizer
        for epoch in range(200):
            ACC = train(net, criterion, optimizer, epoch, trainloader)
            acc = test(net, criterion, epoch, cycle)
            data = (len(trainset))
            scheduler.step()

        accuracy.append(acc)
        datas.append(data)

        # histgram of queried data
        hist = [0]*10
        for _, label in trainloader:
            for i in label:
                hist[i] += 1
        os.makedirs(log_path+'/hist', exist_ok=True)
        fig = plt.figure()
        plt.bar(classes, hist, width=0.9)
        plt.xlabel('classes')
        plt.ylabel('number of queried data')
        fig.savefig(os.path.join(log_path, 'hist', str(cycle)+'.png'))
            
    with open(log_path+f'/test.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(accuracy)

    fig = plt.figure()
    plt.plot(accuracy)
    plt.ylim(0, 1)
    plt.title('Model accuracy')
    plt.ylabel('test accuracy')
    plt.xlabel('AL Iterations')
    plt.grid(True)
    fig.savefig(log_path+'/acc.png')