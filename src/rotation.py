'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from loader import RotationLoader
from models import *
from utils import progress_bar
from load_cifar import load_cifar10
import numpy as np
import argparse
import logging
import yaml
import os
import random
# SSLの学習パート

# fix seed
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

# configs
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config_path', default='yml/debug.yaml')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

# log
log_name = 'lr_' + str(config['optimizer']['lr']) + 'momentum_' + str(config['optimizer']['momentum']) + \
           'batch_' + str(config['batch_size'])
log_path = os.path.join('log', log_name, config['name'])
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(level=logging.ERROR, filename=log_path+'/result.txt', format='%(message)s')

if(os.path.exists(log_path+f'/checkpoint/rotation.pth')):

    print('ssl-weight already exists')
    pass

else:

    best_acc = 0
    start_epoch = 0

    # Prepare CIFAR10 Data
    tmp = CIFAR10(root='./data', train=True, download=True, transform=None)
    tmp = CIFAR10(root='./data', train=False, download=True, transform=None)
    train_x, train_y, test_x, test_y, train_path, test_path= load_cifar10('data/cifar-10-batches-py')

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # dataset
    trainset = RotationLoader(train_x, train_y, train_path, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['rot_batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True,)
    testset = RotationLoader(test_x, test_y, test_path, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['rot_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True,)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    # modify the architecture to fit pretext tasks
    print('==> Building model..')
    net = ResNet18()
    net.linear = nn.Linear(512, 4)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    def make_optimizer(params, name, **kwargs):
        return optim.__dict__[name](params, **kwargs)
    optimizer = make_optimizer(net.parameters(),**config['rot_optimizer'])

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(trainloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
            optimizer.zero_grad()
            outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)
            loss1, loss2, loss3, loss4 = criterion(outputs, targets), criterion(outputs1, targets1), criterion(outputs2, targets2), criterion(outputs3, targets3)
            loss = (loss1+loss2+loss3+loss4)/4.
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            total += targets.size(0)*4

            correct += predicted.eq(targets).sum().item()
            correct += predicted1.eq(targets1).sum().item()
            correct += predicted2.eq(targets2).sum().item()
            correct += predicted3.eq(targets3).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        logging.error(f'train_loss: {train_loss/(batch_idx+1)}, train_acc: {100.*correct/total}')

        return correct/total


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
                inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
                outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)
                loss1, loss2, loss3, loss4 = criterion(outputs, targets), criterion(outputs1, targets1), criterion(outputs2, targets2), criterion(outputs3, targets3)
                loss = (loss1+loss2+loss3+loss4)/4.
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                _, predicted3 = outputs3.max(1)
                total += targets.size(0)*4

                correct += predicted.eq(targets).sum().item()
                correct += predicted1.eq(targets1).sum().item()
                correct += predicted2.eq(targets2).sum().item()
                correct += predicted3.eq(targets3).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            logging.error(f'test_loss: {test_loss/((batch_idx+1))} test_acc: {100.*correct/total}')

        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(log_path+'/checkpoint'):
                os.mkdir(log_path+'/checkpoint')
            torch.save(state, log_path+f'/checkpoint/rotation.pth')
            best_acc = acc

    for epoch in range(start_epoch, start_epoch+120):
        acc = train(epoch)
        test(epoch)
        scheduler.step()
        if(acc>0.99):
            break
        
    with open(log_path+f'/main_best.txt', 'a') as f:
        f.write('rot_result'+str(acc)+'\n')