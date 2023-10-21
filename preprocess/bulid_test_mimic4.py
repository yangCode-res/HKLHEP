import os
import pickle
import time

import numpy as np
import tqdm

from preprocess.parse_csv import EHRParser
from utils import load_subjectdict


def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    np.random.seed(seed)
    common_pids = set()
    for i, code in enumerate(code_map):
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission[EHRParser.adm_id_col]]
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids)

    valid_num = len(patient_admission) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, file,patient_admission, admission_codes_encoded, max_admission_num, code_num):
    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)
    y = np.zeros((n, code_num), dtype=int)
    lens = np.zeros((n,), dtype=int)
    admissions_type = np.zeros((n, max_admission_num, 1), dtype=int)
    # admissions_time=[['1']*max_admission_num for i in range(n)]
    admissions_time = [[] for i in range(n)]
    print(len(admissions_time[0]))
    pid_list=[]
    for i, pid in enumerate(pids):
        pid_list.append(pid)
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        admission_time_pro = time.strptime(str(admissions[-2][EHRParser.adm_time_col]), '%Y-%m-%d %H:%M:%S')
        for k, admission in enumerate(admissions[:-1]):
            #pid_list.append(admission[EHRParser.adm_id_col])
            # print("new patient")
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            x[i, k, codes] = 1
            #admission_type = admission[EHRParser.admission_type_col]

            # print(admission_time_pro)
            admission_time_tail = time.strptime(str(admissions[k][EHRParser.adm_time_col]), '%Y-%m-%d %H:%M:%S')
            # print(admission_time_tail)
            timestamp = (time.mktime(admission_time_tail) - time.mktime(admission_time_pro))
            # print(timestamp)

            #admissions_type[i, k, 0] = admission_type
            # admissions_time[i,k,0]=timestamp
            admissions_time[i].append(timestamp)
        codes = np.array(admission_codes_encoded[admissions[-1][EHRParser.adm_id_col]])
        y[i, codes] = 1
        lens[i] = len(admissions) - 1
    # if file=='train':
    #     pickle.dump(pid_list,open('train_pid.pkl','wb'))
    # if file == 'test':
    #     pickle.dump(pid_list,open('test_pid.pkl', 'wb'))
    # if file == 'valid':
    #     pickle.dump(pid_list,open('valid_pid.pkl', 'wb'))

    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens,admissions_time,admissions_type
def build_times(list,max_admission_num):

    n=len(list)
    x=np.zeros((n,max_admission_num,1))
    for index,i in enumerate(list):
        for index_k,j in enumerate(i):
            x[index,index_k,0]=j
    return x
def build_text(pids, max_admission_num,patient_admission):

    dataset = 'mimic4'
    dataset_path = os.path.join('data', dataset, 'standard')
    dict=load_subjectdict(dataset_path,dataset_path)
    error_count=0
    n=len(pids)
    data_feature = np.zeros((n,max_admission_num, 300), dtype=float)
    #print(dict)
    for i,pid in tqdm.tqdm(enumerate(pids)):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k,admission in enumerate(admissions[:-1]):
            try:
                data_feature[i,k,:]=dict[str(pid)][str(admissions[k][EHRParser.adm_id_col])].cpu()
            except:
                error_count+=1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    print('this is error count',error_count)
    return data_feature
def build_cate(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    dataset_path = os.path.join('preprocess', 'encode_icd9tocatecory2.pkl')
    cate_dict = pickle.load(open(dataset_path, 'rb'))
    #cate_dict=pickle.load(open('D:\chet_models\Chet-master\data\mimic4\encoded\encode_icd9tocatecory2.pkl','rb'))
    n = len(pids)
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            cate_list=[]
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            for code in codes:
                cate_list.append(cate_dict[code])

            try:
                x[i, k, cate_list] = 1
            except:
                print('this is i',i)
                print('this is k',k)
                print(cate_list)
                quit()
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x
def build_heart_failure_y(hf_prefix, codes_y, code_map):
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y
def build_user_data_feature(pids,user_data):
    age_dict={}
    print(user_data)
    n=len(pids)
    data_feature=np.zeros((n,3),dtype=int)
    lens=np.zeros((n,),dtype=int)
    for i,pid in enumerate(pids):
        if user_data[pid][0]=='F':
            data_feature[i][0]=0
        elif user_data[pid][0]=='M':
            data_feature[i][0]=1
        data_feature[i][1]=user_data[pid][1]
        data_feature[i][2]=user_data[pid][2]
        age=int(data_feature[i][1])
        if age not in age_dict:
            age_dict[age]=0
        age_dict[age]+=1
    print(age_dict)
    return data_feature,lens