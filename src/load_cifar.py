import pickle
import os
import numpy as np
from PIL import Image

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def load_cifar10(folder):

    print('==> Preparing data..')
    # ファイルの読み込み
    for i in range(1, 6):

        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)

        if i == 1:
            train_X = data_dict[b'data']
            train_y = data_dict[b'labels']
            train_path = data_dict[b'filenames']

        else:
            train_X = np.vstack((train_X, data_dict[b'data']))
            train_y = np.hstack((train_y, data_dict[b'labels']))
            train_path = np.hstack((train_path, data_dict[b'filenames']))

    # 辞書の連結
    data_dict = unpickle(os.path.join(folder, 'test_batch'))

    # テスト用ファイルの読み込み
    test_X = data_dict[b'data']
    test_y = np.array(data_dict[b'labels'])
    test_path = data_dict[b'filenames']

    # 3*32*32のndarrayに直し，PyTorchから持ってくるときの32*32*3の形に戻す
    train_X = [Image.fromarray(x.reshape([3, 32, 32]).transpose([1, 2, 0]), 'RGB') for x in train_X]
    test_X = [Image.fromarray(x.reshape([3, 32, 32]).transpose([1, 2, 0]), 'RGB') for x in test_X]
    
    return train_X, train_y, test_X, test_y, train_path, test_path
    
if(__name__ == '__main__'):
    pass