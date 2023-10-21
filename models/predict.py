import os
import random

import torch
import numpy as np
from preprocess.auxiliary import generate_neighbors,divide_middle
from models.model2 import Model
from models2.utils import load_adj, EHRDataset
from metrics2 import evaluate_codes, evaluate_hf
state_dict=torch.load(r'D:\pythonprojets\Chet-master\data\params\mimic3\m\58.pt')
seed = 6669
dataset = 'mimic3'  # 'mimic3' or 'eicu'
task = 'm'  # 'm' or 'h'
use_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

code_size = 48  # mn的嵌入大小s
graph_size = 32  # R的嵌入大小32
hidden_size = 150  # rnn hidden size
t_attention_size = 32
t_output_size = hidden_size
batch_size = 32
epochs = 200

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset_path = os.path.join('data', dataset, 'standard')
train_path = os.path.join(dataset_path, 'train')
valid_path = os.path.join(dataset_path, 'valid')
test_path = os.path.join(dataset_path, 'test')

code_adj = load_adj(dataset_path, device=device)
code_num = len(code_adj)
print('loading train data ...')
train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
print('loading valid data ...')
valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
print('loading test data ...')
def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result
test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

task_conf = {
    'm': {
        'dropout': 0.45,
        'output_size': code_num,
        'evaluate_fn': evaluate_codes,
        'lr': {
            'init_lr': 0.01,
            'milestones': [20, 30],
            'lrs': [1e-3, 1e-5]
        }
    },
    'h': {
        'dropout': 0.0,
        'output_size': 1,
        'evaluate_fn': evaluate_hf,
        'lr': {
            'init_lr': 0.01,
            'milestones': [2, 3, 20],
            'lrs': [1e-3, 1e-4, 1e-5]
        }
    }
}
output_size = task_conf[task]['output_size']
activation = torch.nn.Sigmoid()
loss_fn = torch.nn.BCELoss()
evaluate_fn = task_conf[task]['evaluate_fn']
dropout_rate = task_conf[task]['dropout']

param_path = os.path.join('data', 'params', dataset, task)
if not os.path.exists(param_path):
    os.makedirs(param_path)

model = Model(code_num=code_num, code_size=code_size,
              adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
              t_output_size=t_output_size,
              output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
model.load_state_dict(state_dict)
def predict(code_ids,lens):
    code_x_list=[]
    neighbors=[]
    for len in lens:
        code_x = torch.zeros((code_num))
        for code_id in code_ids:
            code_x[code_id]=1
        code_x_list.append(code_x)
    neighbors=generate_neighbors(code_x_list,lens,adj=code_adj)
    divided=divide_middle(code_x_list,neighbors,lens)

    valid_loss, f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
    output = model(code_x_list, divided, neighbors, lens, user_features, cate_features)