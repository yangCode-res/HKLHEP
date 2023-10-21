import os
import _pickle as pickle
import sys

from preprocess import save_sparse, save_data_mimic4
from preprocess.parse_csv_mimic4 import Mimic3Parser, Mimic4Parser, EICUParser
from preprocess.encode import encode_code
from preprocess.bulid_test_mimic4 import split_patients, build_code_xy, build_heart_failure_y, build_user_data_feature, \
    build_cate, build_times, build_text
from preprocess.auxiliary import generate_code_code_adjacent, generate_neighbors, normalize_adj, divide_middle, generate_code_levels

if __name__ == '__main__':
    conf = {
        'mimic3': {
            'parser': Mimic3Parser,
            'train_num': 6000,
            'test_num': 1000,
            'threshold': 0.01
        },
        'mimic4': {
            'parser': Mimic4Parser,
            'train_num': 8000,
            'test_num': 1000,
            'threshold': 0.01,
            'sample_num': 10000
        },
        'eicu': {
            'parser': EICUParser,
            'train_num': 8000,
            'test_num': 1000,
            'threshold': 0.01
        }
    }
    from_saved = True
    data_path = 'data'
    dataset = 'mimic4'  # mimic3, eicu, or mimic4
    dataset_path = os.path.join(data_path, dataset)
    raw_path = os.path.join(dataset_path, 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/%s/raw`' % dataset)
        exit()
    parsed_path = os.path.join(dataset_path, 'parsed')
    if from_saved:
        parser = conf[dataset]['parser'](raw_path)
        patient_admission = pickle.load(open(os.path.join(parsed_path, 'patient_admission.pkl'), 'rb'))
        admission_codes = pickle.load(open(os.path.join(parsed_path, 'admission_codes.pkl'), 'rb'))
        user_data = parser.parse2()
       # pickle.dump(user_data, open(os.path.join(parsed_path, 'user_data.pkl'), 'wb'))
        #user_data = pickle.load(open(os.path.join(parsed_path, 'user_data.pkl'), 'rb'))
    else:
        parser = conf[dataset]['parser'](raw_path)
        sample_num = conf[dataset].get('sample_num', None)
        patient_admission, admission_codes ,user_data= parser.parse(sample_num)

        print('saving parsed data ...')
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)
        pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
        pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))
        pickle.dump(user_data, open(os.path.join(parsed_path, 'user_data.pkl'), 'wb'))
    patient_num = len(patient_admission)
    max_admission_num = max([len(admissions) for admissions in patient_admission.values()])
    avg_admission_num = sum([len(admissions) for admissions in patient_admission.values()]) / patient_num
    max_visit_code_num = max([len(codes) for codes in admission_codes.values()])
    avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(admission_codes)
    print('patient num: %d' % patient_num)
    print('max admission num: %d' % max_admission_num)
    print('mean admission num: %.2f' % avg_admission_num)
    print('max code num in an admission: %d' % max_visit_code_num)
    print('mean code num in an admission: %.2f' % avg_visit_code_num)

    print('encoding code ...')
    admission_codes_encoded, code_map = encode_code(patient_admission, admission_codes)
    code_num = len(code_map)
    print('There are %d codes' % code_num)

    code_levels = generate_code_levels(data_path, code_map)
    pickle.dump({
        'code_levels': code_levels,
    }, open(os.path.join(parsed_path, 'code_levels.pkl'), 'wb'))

    # train_pids, valid_pids, test_pids = split_patients(
    #     patient_admission=patient_admission,
    #     admission_codes=admission_codes,
    #     code_map=code_map,
    #     train_num=conf[dataset]['train_num'],
    #     test_num=conf[dataset]['test_num']
    # )
    pid_pre = pickle.load(open('pids.pkl', 'rb'))
    train_pids, valid_pids, test_pids = pid_pre['train_pids'], pid_pre['valid_pids'], pid_pre['test_pids']
    print('There are %d train, %d valid, %d test samples' % (len(train_pids), len(valid_pids), len(test_pids)))
    code_adj = generate_code_code_adjacent(pids=train_pids, patient_admission=patient_admission,
                                           admission_codes_encoded=admission_codes_encoded,
                                           code_num=code_num, threshold=conf[dataset]['threshold'])

    common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num]
    print('building train codes features and labels ...')
    (train_code_x, train_codes_y, train_visit_lens, train_times, train_types) = build_code_xy(train_pids,'train' ,*common_args)

    print('building valid codes features and labels ...')
    (valid_code_x, valid_codes_y, valid_visit_lens, valid_times, valid_types) = build_code_xy(valid_pids,'valid', *common_args)
    print('building test codes features and labels ...')
    (test_code_x, test_codes_y, test_visit_lens, test_times, test_types) = build_code_xy(test_pids, 'test',*common_args)

    sum_time_list=[]
    for i in  train_times:
        for j in i:
            sum_time_list.append(j)
    for i in  valid_times:
        for j in i:
            sum_time_list.append(j)
    for i in  test_times:
        for j in i:
            sum_time_list.append(j)

    max=max(sum_time_list)
    min=min(sum_time_list)
    max_min=max-min

    for index_i,i in enumerate(train_times):
        for index_j,j in enumerate(i):
            train_times[index_i][index_j]=(j - min)/max_min
    for index_i, i in enumerate(valid_times):
        for index_j, j in enumerate(i):
            valid_times[index_i][index_j] = (j - min)/max_min
    for index_i, i in enumerate(test_times):
        for index_j, j in enumerate(i):
            test_times[index_i][index_j] = (j - min)/max_min
    train_times=build_times(train_times,max_admission_num)
    test_times=build_times(test_times,max_admission_num)
    valid_times=build_times(valid_times,max_admission_num)
    (train_cate_x) = build_cate(train_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num=157)
    print('building test cate features ...')
    (test_cate_x) = build_cate(test_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num=157)
    print('building valid cate features ...')
    (valid_cate_x) = build_cate(valid_pids, patient_admission, admission_codes_encoded, max_admission_num, code_num=157)
    print('buliding train user_data_features ...')
    (train_data_feature,train_feature_lens)=build_user_data_feature(train_pids,user_data)
    print('buliding valid user_data_features ...')
    (vaild_data_feature,valid_feature_lens)=build_user_data_feature(valid_pids,user_data)
    print('buliding test user_data_features ...')
    (test_data_feature, test_feature_lens) = build_user_data_feature(test_pids, user_data)
    print('generating train neighbors ...')
    train_neighbors = generate_neighbors(train_code_x, train_visit_lens, code_adj)
    print('generating valid neighbors ...')
    valid_neighbors = generate_neighbors(valid_code_x, valid_visit_lens, code_adj)
    print('generating test neighbors ...')
    test_neighbors = generate_neighbors(test_code_x, test_visit_lens, code_adj)

    print('generating train middles ...')
    train_divided = divide_middle(train_code_x, train_neighbors, train_visit_lens)
    print('generating valid middles ...')
    valid_divided = divide_middle(valid_code_x, valid_neighbors, valid_visit_lens)
    print('generating test middles ...')
    test_divided = divide_middle(test_code_x, test_neighbors, test_visit_lens)

    print('building train heart failure labels ...')
    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    print('building valid heart failure labels ...')
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    print('building test heart failure labels ...')
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)
    text_feature_train = build_text(train_pids, max_admission_num, patient_admission)

    print('buliding valid text_features ...')
    text_feature_valid = build_text(valid_pids, max_admission_num, patient_admission)

    print('buliding test text_features ...')
    text_feature_test = build_text(test_pids, max_admission_num, patient_admission)
    encoded_path = os.path.join(dataset_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    print('saving encoded data ...')
    pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
    pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    pickle.dump({
        'train_pids': train_pids,
        'valid_pids': valid_pids,
        'test_pids': test_pids
    }, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

    print('saving standard data ...')
    standard_path = os.path.join(dataset_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        os.makedirs(valid_path)
        os.makedirs(test_path)

    print('\tsaving training data')
    save_data_mimic4(train_path, train_code_x, train_visit_lens, train_codes_y, train_hf_y, train_divided, train_neighbors,train_data_feature,train_cate_x,text_feature_train,train_times)
    print('\tsaving valid data')
    save_data_mimic4(valid_path, valid_code_x, valid_visit_lens, valid_codes_y, valid_hf_y, valid_divided, valid_neighbors,vaild_data_feature,valid_cate_x,text_feature_valid,valid_times)
    print('\tsaving test data')
    save_data_mimic4(test_path, test_code_x, test_visit_lens, test_codes_y, test_hf_y, test_divided, test_neighbors,test_data_feature,test_cate_x,text_feature_test,test_times)

    code_adj = normalize_adj(code_adj)
    save_sparse(os.path.join(standard_path, 'code_adj'), code_adj)