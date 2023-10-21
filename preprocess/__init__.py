import os
import pickle

import numpy as np


def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)


def load_sparse(path):
    data = np.load(path)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat


def save_data(path, code_x, visit_lens, codes_y, hf_y, divided, neighbors,data_feature,cate_feature,text,event_types,admission_times,eh_y):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    np.savez(os.path.join(path, 'eh_y'), eh_y=eh_y)
    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)
    save_sparse(os.path.join(path,'data_feature'),data_feature)
    save_sparse(os.path.join(path,'cate_feature'),cate_feature)
    save_sparse(os.path.join(path, 'text_feature'), text)
    save_sparse(os.path.join(path,'event_types'),event_types)
    pickle.dump(admission_times,open(os.path.join(path,'admission_times.pkl'),'wb'))
def save_data_mimic4(path, code_x, visit_lens, codes_y, hf_y, divided, neighbors,data_feature,cate_feature,text,admission_times):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)
    save_sparse(os.path.join(path,'data_feature'),data_feature)
    save_sparse(os.path.join(path,'cate_feature'),cate_feature)
    save_sparse(os.path.join(path, 'text_feature'), text)
    pickle.dump(admission_times,open(os.path.join(path,'admission_times.pkl'),'wb'))