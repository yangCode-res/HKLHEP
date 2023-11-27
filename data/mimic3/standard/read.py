import torch
import matplotlib.pyplot as plt
from preprocess import load_sparse
import numpy as np
code_x=load_sparse('./code_x.npz')
visit_lens = np.load('./visit_lens.npz')['lens']
divided = load_sparse('./divided.npz')
x_list=[]
label_f=[]
label_n=[]
for lens_i,code_x_i,divided_i in zip(visit_lens,code_x,divided):
    x=code_x_i[lens_i-1]
    x=torch.tensor(x)
    d=torch.tensor(divided_i[lens_i-1])
    m1, m2, m3 = d[:, 0], d[:, 1], d[:, 2]  # m1是persistent diseases
    m1_index = torch.where(m1 > 0)[0]  # m2是emerging neighbors
    m2_index = torch.where(m2 > 0)[0]  # m3是emerging unrelated diseases
    m3_index = torch.where(m3 > 0)[0]
    x_list.append(torch.where(x>0)[0])
    label_f.append(list(m1_index))
    label_n.append(list(m2_index)+list(m3_index))
x_norm=[]
y_norm=[]
color=[]
for index,f_i in enumerate(label_f):
    for f in f_i:
        x_norm.append(int(f))
        y_norm.append(index)
        color.append('#FFD700')
for index,n_i in enumerate(label_n):
    for n in n_i:
        print(int(n))
        x_norm.append(int(n))
        y_norm.append(index)
        color.append('#008000')
plt.scatter(x_norm[:], y_norm[:],s=0.1,color=color)
plt.show()
# print(x_norm)
# print(y_norm)
# print(label_n)