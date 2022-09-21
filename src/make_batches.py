'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from loader import RotationLoader
from utils import progress_bar
from models import *
from load_cifar import load_cifar10
import argparse
import numpy as np
import logging
import yaml
import os
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

if(os.path.exists(log_path + '/loss/batch_9.txt')):
    print('split batches already exist')
    pass

else:
    best_acc = 0
    start_epoch = 0

    train_x, train_y, test_x, test_y, train_path, test_path = load_cifar10('data/cifar-10-batches-py')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = RotationLoader(train_x, train_y, train_path, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,)

    logging.basicConfig(level=logging.ERROR, filename=log_path + '/result.txt', format='%(message)s')
    logging.error('make_batches.py')

    net = ResNet18()
    net.linear = nn.Linear(512, 4)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(log_path+'/checkpoint/rotation.pth')['net'])

    criterion = nn.CrossEntropyLoss()

    def test():
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(trainloader):
                inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
                outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)
                loss1, loss2, loss3, loss4 = criterion(outputs, targets), criterion(outputs1, targets1), criterion(outputs2, targets2), criterion(outputs3, targets3)
                loss = (loss1+loss2+loss3+loss4)/4.
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                _, predicted = outputs.max(1)
                _, predicted1 = outputs1.max(1)
                _, predicted2 = outputs2.max(1)
                _, predicted3 = outputs3.max(1)
                total += targets.size(0)*4

                correct += predicted.eq(targets).sum().item()
                correct += predicted1.eq(targets1).sum().item()
                correct += predicted2.eq(targets2).sum().item()
                correct += predicted3.eq(targets3).sum().item()

                loss = loss.item()

                # save loss and path
                s = '{0:10.20f}'.format(loss)
                s = s + '_' + str(path[0]) + "\n"
                with open(log_path+'/rotation_loss.txt', 'a') as f:
                    f.write(s)

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            logging.error(f'loss: {test_loss/(batch_idx+1)}, accuracy: {100.*correct/total}')

    if __name__ == "__main__":
        test()
    
        # read pretext tasks' loss
        with open(log_path+'/rotation_loss.txt', 'r') as f:
            losses = f.readlines()
        loss_1 = []
        name_2 = []
        for j in losses:
            loss_1.append(j[:-1].split('_')[0])
            Name = ''
            name = (j[:-1].split('_'))
            for k in name[1:]:
                Name += k + '_'
            Name = Name.rstrip('_')
            name_2.append(Name)

        # sort loss
        s = np.array(loss_1)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=15)
        sort_index = np.argsort(s)
        x = sort_index.tolist()
        x.reverse()
        sort_index = np.array(x)

        os.makedirs(log_path + '/loss', exist_ok=True)

        # split data into batches
        # the number of batches is the same as the AL cycles
        for i in range(10):
            sample5000 = sort_index[i*5000:(i+1)*5000]
            s = log_path + '/loss/batch_' + str(i) + '.txt'
            with open(s, 'w') as f:
                    f.close
            for k in sample5000:
                with open(s, 'a') as f:
                    f.write(name_2[k]+'\n')