import os
import random
import sys
import time

import torch
import numpy as np

from models.model2 import Model
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler,load_cate_adj,load_cate_dict,load_cate__parent_dict
from metrics2 import evaluate_codes, evaluate_hf
# from prettytable import PrettyTable



def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result

def getModelSize(model):
    param_size = 0
    param_sum = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    #     param_sum += param.nelement()
    # buffer_size = 0
    # buffer_sum = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #     buffer_sum += buffer.nelement()
    # all_size = (param_size + buffer_size) / 1024 / 1024
    # print('模型总大小为：{:.3f}MB'.format(all_size))
    # #return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    # table = PrettyTable(['Modules', 'Parameters'])
    # total_params = 0
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad: continue
    #     params = parameter.numel()
    #     table.add_row([name, params])
    #     total_params += params
    # print(table)
    # print(f'Total Trainable Params: {total_params/(1024*1024)}M')
    # name = type(model).__name__
    # result = '-------------------%s---------------------\n' % name
    # total_num_params = 0
    # for i, (name, child) in enumerate(model.named_children()):
    #     num_params = sum([p.numel() for p in child.parameters()])
    #     total_num_params += num_params
    #     for i, (name, grandchild) in enumerate(child.named_children()):
    #         num_params = sum([p.numel() for p in grandchild.parameters()])
    # result += '[Network %s] Total number of parameters : %.3f M\n' % (name, total_num_params / (1024 * 1024))
    # result += '-----------------------------------------------\n'
    # print(result)



if __name__ == '__main__':

    seed = 6669
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    task = 'm'  # 'm' or 'h' or'e'
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(device)
    code_size = 48#mn的嵌入大小s
    graph_size = 32#R的嵌入大小32
    hidden_size = 150  # rnn hidden size
    t_attention_size = 32
    t_output_size = hidden_size
    batch_size = 32
    epochs = 100
    torch.cuda.current_device()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)
    cate_adj=load_cate_adj(dataset_path,device=device)
    cate_dict=load_cate_dict(dataset_path,dataset)
    parent_dict=load_cate__parent_dict(dataset_path,dataset)
    code_num = len(code_adj)
    print('this is code num',code_num)
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print(test_data)
    valid_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-4]
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
        },
        'e': {
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
                  t_output_size=t_output_size,cate_adj=cate_adj,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation,cate_dict=cate_dict,parent_dict=parent_dict).to(device)
    getModelSize(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                     task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('模型大小-:'+str(pytorch_total_params/1024/1024)+'M')
    best_f1=0
    best_f1_epoch=0
    best_recall_10=0
    best_recall_20=0
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, divided, y, neighbors,user_features, cate_features,text_features,admission_times= train_data[step]
            output = model(code_x, divided, neighbors, visit_lens,user_features,cate=cate_features,text_features=text_features,admission_times=admission_times).squeeze()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        valid_loss, f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical)
        valid_loss1, f1_score1 = evaluate_fn(model, test_data, loss_fn, output_size, test_historical)
        if f1_score>best_f1:
            best_f1=f1_score
            best_f1_epoch=epoch

        print("最好的f1值",str(best_f1),str(best_f1_epoch+1))
        torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
