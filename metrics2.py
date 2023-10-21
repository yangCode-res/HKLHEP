import numpy
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import pickle

def f1(y_true_hot, y_pred, metrics='weighted'):
    #print(y_true_hot[0])
   # print(y_pred[0])
    #493个测试结果
    pred_true_metrics=[]
    pred_false_metrics=[]
    y_trueNotpred_metrics=[]

    result = np.zeros_like(y_true_hot)

    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)#本次总共有多少种疾病
        result[i][y_pred[i][:true_number]] = 1
        true=set(np.sort(np.argwhere(y_true_hot[i]==1).reshape(-1),axis=0))
        pred=set(np.sort(y_pred[i][:true_number],axis=0))

        pred_true=true.intersection(pred)
        pred_false=true.symmetric_difference(pred)
        y_realNotpred = true.symmetric_difference(pred_true)
        y_trueNotpred_metrics.append(y_realNotpred)
        pred_true_metrics.append(pred_true)

       # print(result[i])

    # print(y_trueNotpred_metrics)
    # print(pred_true_metrics)
    # pickle.dump(y_trueNotpred_metrics,open("y_trueNotPred.pkl",'wb'))
    # pickle.dump(pred_true_metrics,open("pred_true_metrics.pkl",'wb'))
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            # r[i] += len(it) / min(k, len(t))
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)


def calculate_occurred(historical, y, preds, ks):
    # y_occurred = np.sum(np.logical_and(historical, y), axis=-1)
    # y_prec = np.mean(y_occurred / np.sum(y, axis=-1))
    r1 = np.zeros((len(ks), ))
    r2 = np.zeros((len(ks),))
    n = np.sum(y, axis=-1)
    for i, k in enumerate(ks):
        # n_k = np.minimum(n, k)
        n_k = n
        pred_k = np.zeros_like(y)
        for T in range(len(pred_k)):
            pred_k[T][preds[T][:k]] = 1
        # pred_occurred = np.sum(np.logical_and(historical, pred_k), axis=-1)
        pred_occurred = np.logical_and(historical, pred_k)
        pred_not_occurred = np.logical_and(np.logical_not(historical), pred_k)
        pred_occurred_true = np.logical_and(pred_occurred, y)
        pred_not_occurred_true = np.logical_and(pred_not_occurred, y)
        r1[i] = np.mean(np.sum(pred_occurred_true, axis=-1) / n_k)
        r2[i] = np.mean(np.sum(pred_not_occurred_true, axis=-1) / n_k)
    return r1, r2


def evaluate_codes(model, dataset, loss_fn, output_size, historical=None):
    # flag=0
    # with open('addcate.txt','w') as f:
    #     import torch.nn.functional as F
        np.set_printoptions(threshold=np.inf)
        model.eval()
        total_loss = 0.013
        labels = dataset.label()
        preds = []
        np.set_printoptions(threshold=np.inf)
        for step in range(len(dataset)):
            code_x, visit_lens, divided, y, neighbors,user_features ,cate_features,text_features,admission_times= dataset[step]
            output = model(code_x, divided, neighbors, visit_lens,user_features,cate_features,text_features,admission_times)
            pred = torch.argsort(output, dim=-1, descending=True)
            # output2 = output.cpu().detach().numpy()
            # for count,i in enumerate(output2):
            #
            #     new_list=[]
            #     max=numpy.max(i)
            #     min=numpy.min(i)
            #     for x in i:
            #         x = float(x - min) / (max - min)
            #         new_list.append(x)
            #     new_list=numpy.array(new_list)
            #     temp=new_list[np.where(y[count].cpu()!=0)]
            #     f.writelines(str(flag)+':  '+str(temp)+'\n')
            #     if flag==204:
            #         print(np.where(y[count].cpu()!=0))
            #     flag+=1

            # print(output.shape)
            # print('this is P',output[0])
            # print('this is pred',pred[0])
            # print('this is y',np.where(y[0].cpu()!=0))#第11轮比较适合
            # print('this is label_p',output[0][np.where(y[0].cpu()!=0)])
            # soft_output=torch.nn.Softmax(output)
            # print(soft_output)
           # soft_output=F.normalize(output,dim=-1)
            #print(soft_output)
            # print(soft_output[0][np.where(y[0].cpu()!=0)])
            #print(torch.nn.Softmax(output)[0][np.where(y[0].cpu()!=0)])
            preds.append(pred)
            loss = loss_fn(output, y)
            total_loss += loss.item() * output_size * len(code_x)
            print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
        avg_loss = total_loss / dataset.size()
        preds = torch.vstack(preds).detach().cpu().numpy()
        f1_score = f1(labels, preds)
        prec, recall = top_k_prec_recall(labels, preds, ks=[10, 20, 30, 40])

        #print('this is preds',preds)
        #print(labels,preds)
        if historical is not None:
            r1, r2 = calculate_occurred(historical, labels, preds, ks=[10, 20, 30, 40])
            print('\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f  --- occurred: %.4f, %.4f, %.4f, %.4f  --- not occurred: %.4f, %.4f, %.4f, %.4f'
                  % (avg_loss, f1_score, recall[0], recall[1], recall[2], recall[3], r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2], r2[3]))
        else:
            print('\r    Evaluation: loss: %.4f --- f1_score: %.4f --- top_k_recall: %.4f, %.4f, %.4f, %.4f'
                  % (avg_loss, f1_score, recall[0], recall[1], recall[2], recall[3]))
        return avg_loss, f1_score


def evaluate_hf(model, dataset, loss_fn, output_size=1, historical=None):
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    outputs = []
    preds = []
    for step in range(len(dataset)):
        code_x, visit_lens, divided, y, neighbors, user_features, cate_features, text_features, admission_times = dataset[step]
        output = model(code_x, divided, neighbors, visit_lens, user_features, cate_features, text_features,admission_times).squeeze()
        loss = loss_fn(output, y)
        total_loss += loss.item() * output_size * len(code_x)
        output = output.detach().cpu().numpy()
        outputs.append(output)
        pred = (output > 0.5).astype(int)
        preds.append(pred)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    print(dataset.size())
    avg_loss = total_loss / dataset.size()
    outputs = np.concatenate(outputs)
    preds = np.concatenate(preds)
    auc = roc_auc_score(labels, outputs)
    f1_score_ = f1_score(labels, preds)
    print('\r    Evaluation: loss: %.4f --- auc: %.4f --- f1_score: %.4f' % (avg_loss, auc, f1_score_))
    return avg_loss, f1_score_
