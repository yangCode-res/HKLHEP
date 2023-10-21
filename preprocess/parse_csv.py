import os
import pickle
from datetime import datetime
from collections import OrderedDict

import pandas
import pandas as pd
import numpy as np
from utils import load_subjectdict

class EHRParser:
    pid_col = 'pid'
    adm_id_col = 'adm_id'
    adm_time_col = 'adm_time'
    cid_col = 'cid'
    gender_col='gender'
    age_col='age'
    cluster_col='cluster'
    text_col='text'
    admission_type_col='eventType'
    def __init__(self, path):
        self.path = path

        self.skip_pid_check = False

        self.patient_admission = None
        self.admission_codes = None
        self.admission_procedures = None
        self.admission_medications = None
        self.users_data=None
        self.texts_data=None
        self.parse_fn = {'d': self.set_diagnosis}

    def set_admission(self):
        raise NotImplementedError

    def set_diagnosis(self):
        raise NotImplementedError
    def set_user(self):
        raise NotImplementedError
    def set_text(self):
        raise NotImplementedError
    @staticmethod
    def to_standard_icd9(code: str):
        raise NotImplementedError
    @staticmethod
    def gendertoId(gender:str):
        raise NotImplementedError
    @staticmethod
    def type_toid(event:str):
        raise NotImplementedError
    def parse_admission(self):
        print('parsing the csv file of admission ...')
        filename, cols, converters = self.set_admission()
        print(filename,cols,converters)
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        admissions = self._after_read_admission(admissions, cols)
        all_patients = OrderedDict()
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(admissions)), end='')
            pid, adm_id, adm_time,type = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]],row[cols[self.admission_type_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            admission = all_patients[pid]
            admission.append({self.adm_id_col: adm_id, self.adm_time_col: adm_time,self.admission_type_col:type})
        print('\r\t%d in %d rows' % (len(admissions), len(admissions)))

        patient_admission = OrderedDict()
        for pid, admissions in all_patients.items():
            if len(admissions) >= 2:
                patient_admission[pid] = sorted(admissions, key=lambda admission: admission[self.adm_time_col])

        self.patient_admission = patient_admission

    def parse_user(self):
        print('parsing the csv file of user ...')
        filename, cols, converters = self.set_user()
        print(filename,cols,converters)
        users=pd.read_csv(os.path.join(self.path,filename),usecols=list(cols.values()),converters=converters)
        #print(users)
        all_users=OrderedDict()
        for i ,row in users.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(users)), end='')
            pid,gender,age,cluster=row[cols[self.pid_col]],row[cols[self.gender_col]],row[cols[self.age_col]],row[cols[self.cluster_col]]
            if pid not in all_users:
                all_users[pid]=[]
            all_users[pid].append(gender)
            all_users[pid].append(age)
            all_users[pid].append(cluster)
        print('\r\t%d in %d rows' % (len(users), len(users)))
        self.users_data=all_users
    def parse_text(self):
        print('parsing the csv file of text ...')
        filename, cols, converters = self.set_text()
        print(filename,cols,converters)
        texts=pd.read_csv(os.path.join(self.path,filename),usecols=list(cols.values()),converters=converters)
        print(texts)
        all_texts=OrderedDict()
        for i,row in texts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(texts)), end='')
            pid,had_id,text=row[cols[self.pid_col]],row[cols[self.adm_id_col]],row[cols[self.text_col]]
            if pid not in all_texts:
                all_texts[pid]={}
            all_texts[pid]={had_id:text}
        self.texts_data=all_texts
    def _after_read_admission(self, admissions, cols):

        return admissions

    def _parse_concept(self, concept_type):
        assert concept_type in self.parse_fn.keys()
        filename, cols, converters = self.parse_fn[concept_type]()
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        concepts = self._after_read_concepts(concepts, concept_type, cols)
        result = OrderedDict()
        for i, row in concepts.iterrows():
            if i % 100 == 0:
                print('\r\t%d in %d rows' % (i + 1, len(concepts)), end='')
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
                if code == '':
                    continue
                if adm_id not in result:
                    result[adm_id] = []
                codes = result[adm_id]
                codes.append(code)
        print('\r\t%d in %d rows' % (len(concepts), len(concepts)))
        return result

    def _after_read_concepts(self, concepts, concept_type, cols):
        return concepts

    def parse_diagnoses(self):
        print('parsing csv file of diagnosis ...')
        self.admission_codes = self._parse_concept('d')

    def calibrate_patient_by_admission(self):
        dataset = 'mimic3'
        dataset_path = os.path.join('data', dataset, 'standard')
        #dict = pickle.load(open(r'D:\chet_models\Chet-master\thisisfull_text_hadm_id.pkl','rb'))
       # print(len(dict))
        print('calibrating patients by admission ...')
        del_pids = set()
        del_pids2=[]
        for pid, admissions in self.patient_admission.items():

            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    break
                # if str(pid) not in dict:
                #     break
                # if str(adm_id) not in dict[str(pid)]:
                #     break
            else:
                continue
            del_pids.add(pid)
        # for pid, admissions in self.patient_admission.items():
        #     if str(self.patient_admission[pid][-2][self.adm_id_col]) not in dict:
        #         del_pids.add(pid)

        # for pid ,admissions in self.patient_admission.items():
        #
        #     for admission in admissions:
        #         adm_id=admission[self.adm_id_col]
        #         if str(pid) not in dict:
        #             break
        #         if str(adm_id) not in dict[str(pid)]:
        #             break
        #     else:
        #         continue

            #del_pids.append(pid)

        for pid in del_pids:

            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]
            del self.patient_admission[pid]


    def calibrate_admission_by_patient(self):
        print('calibrating admission by patients ...')
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        del_adm_ids = [adm_id for adm_id in self.admission_codes if adm_id not in adm_id_set]
        for adm_id in del_adm_ids:
            del self.admission_codes[adm_id]

    def sample_patients(self, sample_num, seed):
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes

    def parse(self, sample_num=None, seed=6669):
        self.parse_admission()
        self.parse_user()
        self.parse_diagnoses()
        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        self.parse_user()
        self.parse_text()
        if sample_num is not None:
            self.sample_patients(sample_num, seed)
        return self.patient_admission, self.admission_codes,self.users_data,self.texts_data

    def parse_user_only(self, sample_num=None, seed=6669):
        self.parse_user()
        return self.users_data

class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.adm_time_col: 'ADMITTIME',self.admission_type_col:'ADMISSION_TYPE'}
        converter = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S'),
            'ADMISSION_TYPE':Mimic3Parser.type_toid
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converter = {'SUBJECT_ID': int, 'HADM_ID': int, 'ICD9_CODE': Mimic3Parser.to_standard_icd9}
        return filename, cols, converter
    def set_user(self):
        filename='PATIENTS2.csv'
        cols={self.pid_col:'SUBJECT_ID',self.gender_col:'GENDER',self.age_col:'AGE',self.cluster_col:'CLUSTERS'}
        converter={
            'SUBJECT_ID': int,
            'GENDER':Mimic3Parser.gendertoId,
            "DOB":int,

        }
        return filename,cols,converter
    def set_text(self):
        filename='text_full.csv'
        cols={self.pid_col:'SUBJECT_ID',self.adm_id_col:'HADM_ID',self.text_col:'TEXT'}
        converter={'SUBJECT_ID': int, 'HADM_ID': int, 'text': str}
        return filename,cols,converter
    @staticmethod
    def type_toid(event:str):
        if event=='EMERGENCY':
            return 0
        elif event=='ELECTIVE':
            return 1
        elif event=='NEWBORN':
            return 2
        elif event=='URGENT':
            return 3
    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code
    @staticmethod
    def gendertoId(gender: str):
        if gender=='F':
            return 0
        if gender=='M':
            return 1


class Mimic4Parser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.icd_ver_col = 'icd_version'
        self.icd_map = self._load_icd_map()
        self.patient_year_map = self._load_patient()

    def _load_icd_map(self):
        print('loading ICD-10 to ICD-9 map ...')
        filename = 'icd10-icd9.csv'
        cols = ['icd10cm', 'icd9cm']
        converters = {'icd10cm': str, 'icd9cm': str}
        icd_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        icd_map = {row['icd10cm']: row['icd9cm'] for _, row in icd_csv.iterrows()}
        return icd_map

    def _load_patient(self):
        print('loading patients anchor year ...')
        filename = 'patients.csv'
        cols = ['subject_id', 'anchor_year', 'anchor_year_group']
        converters = {'subject_id': int, 'anchor_year': int, 'anchor_year_group': lambda cell: int(str(cell)[:4])}
        patient_csv = pandas.read_csv(os.path.join(self.path, filename), usecols=cols, converters=converters)
        patient_year_map = {row['subject_id']: row['anchor_year'] - row['anchor_year_group']
                            for i, row in patient_csv.iterrows()}
        return patient_year_map

    def set_admission(self):
        filename = 'admissions.csv'
        cols = {self.pid_col: 'subject_id', self.adm_id_col: 'hadm_id', self.adm_time_col: 'admittime'}
        converter = {
            'subject_id': int,
            'hadm_id': int,
            'admittime': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnoses_icd.csv'
        cols = {
            self.pid_col: 'subject_id',
            self.adm_id_col: 'hadm_id',
            self.cid_col: 'icd_code',
            self.icd_ver_col: 'icd_version'
        }
        converter = {'subject_id': int, 'hadm_id': int, 'icd_code': str, 'icd_version': int}
        return filename, cols, converter

    def _after_read_admission(self, admissions, cols):
        print('\tselecting valid admission ...')
        valid_admissions = []
        n = len(admissions)
        for i, row in admissions.iterrows():
            if i % 100 == 0:
                print('\r\t\t%d in %d rows' % (i + 1, n), end='')
            pid = row[cols[self.pid_col]]
            year = row[cols[self.adm_time_col]].year - self.patient_year_map[pid]
            if year > 2012:
                valid_admissions.append(i)
        print('\r\t\t%d in %d rows' % (n, n))
        print('\t\tremaining %d rows' % len(valid_admissions))
        return admissions.iloc[valid_admissions]

    def _after_read_concepts(self, concepts, concept_type, cols):
        print('\tmapping ICD-10 to ICD-9 ...')
        n = len(concepts)
        if concept_type == 'd':
            def _10to9(i, row):
                if i % 100 == 0:
                    print('\r\t\t%d in %d rows' % (i + 1, n), end='')
                cid = row[cid_col]
                if row[icd_ver_col] == 10:
                    if cid not in self.icd_map:
                        code = self.icd_map[cid + '1'] if cid + '1' in self.icd_map else ''
                    else:
                        code = self.icd_map[cid]
                    if code == 'NoDx':
                        code = ''
                else:
                    code = cid
                return Mimic4Parser.to_standard_icd9(code)

            cid_col, icd_ver_col = cols[self.cid_col], self.icd_ver_col
            col = np.array([_10to9(i, row) for i, row in concepts.iterrows()])
            print('\r\t\t%d in %d rows' % (n, n))
            concepts[cid_col] = col
        return concepts

    @staticmethod
    def to_standard_icd9(code: str):
        return Mimic3Parser.to_standard_icd9(code)


class EICUParser(EHRParser):
    def __init__(self, path):
        super().__init__(path)
        self.skip_pid_check = True

    def set_admission(self):
        filename = 'patient.csv'
        cols = {
            self.pid_col: 'patienthealthsystemstayid',
            self.adm_id_col: 'patientunitstayid',
            self.adm_time_col: 'hospitaladmitoffset'
        }
        converter = {
            'patienthealthsystemstayid': int,
            'patientunitstayid': int,
            'hospitaladmitoffset': lambda cell: -int(cell)
        }
        return filename, cols, converter

    def set_diagnosis(self):
        filename = 'diagnosis.csv'
        cols = {self.pid_col: 'diagnosisid', self.adm_id_col: 'patientunitstayid', self.cid_col: 'icd9code'}
        converter = {'diagnosisid': int, 'patientunitstayid': int, 'icd9code': EICUParser.to_standard_icd9}
        return filename, cols, converter

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        code = code.split(',')[0]
        c = code[0].lower()
        dot = code.find('.')
        if dot == -1:
            dot = None
        if not c.isalpha():
            prefix = code[:dot]
            if len(prefix) < 3:
                code = ('%03d' % int(prefix)) + code[dot:]
            return code
        if c == 'e':
            prefix = code[1:dot]
            if len(prefix) != 3:
                return ''
        if c != 'e' or code[0] != 'v':
            return ''
        return code

    def parse_diagnoses(self):
        super().parse_diagnoses()
        t = OrderedDict.fromkeys(self.admission_codes.keys())
        for adm_id, codes in self.admission_codes.items():
            t[adm_id] = list(set(codes))
        self.admission_codes = t
if __name__ == '__main__':
    a=EHRParser(r'D:\pythonprojets\Chet-master\data\mimic3\raw')
    a.parse_user()