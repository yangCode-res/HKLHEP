import os
import pickle

import torch
import numpy as np

from preprocess import load_sparse


def load_adj(path, device=torch.device('cpu')):
    filename = os.path.join(path, 'code_adj.npz')
    adj = torch.from_numpy(load_sparse(filename)).to(device=device, dtype=torch.float32)
    return adj
def load_cate_adj(path, device=torch.device('cpu')):
    filename = os.path.join(path, 'cate_adj.npz')
    adj = torch.from_numpy(load_sparse(filename)).to(device=device, dtype=torch.float32)
    return adj
def load_cate_dict(path,dataset_path):
    if dataset_path=='mimic3':
        filename=os.path.join(path, 'encode_icd9tocatecory.pkl')
        dict=pickle.load(open(filename,'rb'))
    else:
        filename = os.path.join(path, 'encode_icd9tocatecory2.pkl')
        dict = pickle.load(open(filename, 'rb'))
    return dict
def load_cate__parent_dict(path,dataset_path):
    if dataset_path=='mimic3':
        filename=os.path.join(path, 'parent_catecory.pkl')
        dict=pickle.load(open(filename,'rb'))
    else:
        filename = os.path.join(path, 'parent_catecory.pkl')
        dict = pickle.load(open(filename, 'rb'))
    return dict
def load_subjectdict(path,dataset_path):
    if dataset_path=='mimic3':
        filename=os.path.join(path, 'text_full_dict.pkl')
        #filename=os.path.join(path, 'sum_full_text_mimic3.pkl')
        dict=pickle.load((open(filename,'rb')))
    else:
        filename = os.path.join(path, 'sum_full_text.pkl')
        dict = pickle.load((open(filename, 'rb')))
    return dict
class EHRDataset:
    def __init__(self, data_path, label='m', batch_size=32, shuffle=True, device=torch.device('cpu')):
        super().__init__()
        self.path = data_path
        self.code_x, self.visit_lens, self.y, self.divided, self.neighbors,self.user_data_feature,self.cate_features,self.text_features,self.admission_times = self._load(label)

        self._size = self.code_x.shape[0]
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def _load(self, label):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        user_data_features=load_sparse(os.path.join(self.path,'data_feature.npz'))
        cate_features=load_sparse(os.path.join(self.path,'cate_feature.npz'))
        text_features=load_sparse(os.path.join(self.path,'text_feature.npz'))
        admission_times=pickle.load(open(os.path.join(self.path,'admission_times.pkl'),'rb'))
        if label == 'm':
            y = load_sparse(os.path.join(self.path, 'code_y.npz'))
        elif label == 'h':
            y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        elif label=='e':
            y=np.load(os.path.join(self.path,'eh_y.npz'))['eh_y']
        else:
            raise KeyError('Unsupported label type')
        divided = load_sparse(os.path.join(self.path, 'divided.npz'))
        neighbors = load_sparse(os.path.join(self.path, 'neighbors.npz'))
        return code_x, visit_lens, y, divided, neighbors,user_data_features,cate_features,text_features,admission_times

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        return self.y

    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        divided = torch.from_numpy(self.divided[slices]).to(device)
        neighbors = torch.from_numpy(self.neighbors[slices]).to(device)
        user_features=torch.from_numpy(self.user_data_feature[slices]).to(device)
        cate_features=torch.from_numpy(self.cate_features[slices]).to(device)
        text_features=torch.from_numpy(self.text_features[slices]).to(device)

        admission_times=torch.from_numpy(self.admission_times[slices]).to(device)
        return code_x, visit_lens, divided, y, neighbors,user_features,cate_features,text_features,admission_times


class MultiStepLRScheduler:
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_lr = init_lr
        self.lrs = self._generate_lr(milestones, lrs)
        self.current_epoch = 0

    def _generate_lr(self, milestones, lrs):
        milestones = [1] + milestones + [self.epochs + 1]
        lrs = [self.init_lr] + lrs
        lr_grouped = np.concatenate([np.ones((milestones[i + 1] - milestones[i], )) * lrs[i]
                                     for i in range(len(milestones) - 1)])
        return lr_grouped

    def step(self):
        lr = self.lrs[self.current_epoch]
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0


def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str
