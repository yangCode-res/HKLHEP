import  pickle
from collections import OrderedDict

import torch

data=pickle.load(open('analys/dict_summary.pkl','rb'))
code_map=pickle.load(open('code_map.pkl','rb'))
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering,MeanShift
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
cnames = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen':           '#90EE90',
    'lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#FAA460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}

#print(code_map)
catelogy=open('cate.txt','r')
catelogydict={}
diapict={}
count_dia=1
for line in catelogy.readlines():
    splitline=line.split(':')
    key=splitline[0]
    value=splitline[1].replace('\n','')
    #print(value)
    if value not in diapict:
        diapict[value]=count_dia
        count_dia+=1
    if key not in catelogydict:
        catelogydict[key]=diapict[value]
#print(diapict)
#print(catelogydict)
def dict_flip(dict_source):
    dict_flipped={}
    for key,value in dict_source.items():
        if value not in dict_flipped:
            dict_flipped[value]=key
        else:
            dict_flipped[value]=key
            print("error")
    return dict_flipped
#flit_code_map=dict_flip(code_map)
print(len(data))
import csv
print('lendata',len(catelogydict))
with open('patient.csv', mode='w',newline='') as employee_file:

    employee_writer = csv.writer(employee_file)
    hearder_list=[]
    hearder_list.append('user-id')
    for i in range(1,158):
        hearder_list.append(i)
    employee_writer.writerow(hearder_list)
    for user_id in data:
        temp_list = [0 for i in range(0, 158)]
        temp_list[0]=user_id
        for code in data[user_id]:
            i=code
            #print(i)
            temp=i.split('.')[0]
            temp_list[catelogydict[temp]]+=1
        employee_writer.writerow(temp_list)
#data_pre=pd.read_csv('employee_file.csv',usecols=[i for i in range(1,158)])
data_pre=pd.read_csv('employee_file.csv')
data=torch.tensor(data_pre.iloc[:,1:].values)
data_pre2=pd.read_csv('employee_file.csv',names=['user-id'])
print(data_pre.head(5))

#pca=PCA()
#data=pca.fit_transform(data_pre)
#print(data)
estimator=KMeans(n_clusters=20)#12
#estimator=MeanShift()
estimator.fit(data)
predict=estimator.predict(data)
#目前最好的效果是PCA()，同时n-cluster=10


##
# gmm=GMM(n_components=10).fit(data)
# predict=gmm.predict(data)
print('predict length',len(predict))
dict={}
clusters={}
for index,i in enumerate(data_pre2.iterrows()):
    if index==0:
        continue
    if int(i[0][0]) not in clusters:
        clusters[int(i[0][0])]=0

for index,item  in enumerate(clusters.keys()):
    clusters[int(item)]=predict[index]
print(clusters)
pickle.dump(clusters,open('user_clusters.pkl','wb'))
small_c=9
small=1000
for i in predict:
    if i not in dict:
        dict[i]=0
    dict[i]+=1
#print(dict)
for i in dict:
    if dict[i]<small:
        small_c=i
        small=dict[i]
print(small_c)
#print(clusters)
plt.figure(figsize=(20,8))
# 建立四个颜色的列表
colored = ['orange', 'green', 'blue', 'purple','red','black','pink','yellow','purple','grey','indigo','coral','peru','skyblue','palegreen']
colored=list(cnames.keys())
colr = [colored[i] for i in predict]
plt.scatter(data[:,1], data[:,0], color=colr)
plt.show()
for i in clusters:
    if clusters[i]==small_c:
        print(i)



#data_predict=estimator.fit_predict(data)
print("SC",silhouette_score(data,predict))
#print(flit_code_map)
