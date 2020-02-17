import os
import re
import csv
import random
import datetime
import argparse
import numpy as np
from scipy.io import loadmat


data_dir_amc_v1 = 'dataset/amc_v1'
data_dir_amc_v2 = 'dataset/amc_v2'


def load_ecg_list_local_v1():
    r = list()
    summary = list()
    summary.append(['case', 'total', 'sinus', 'afib', 'vpc', 'noise', 'others'])
    case_re = re.compile('([B-L])-[0-9]{2}_[0-9]{6}_[0-9]{2}')
    ecg_re = re.compile('([B-L])-[0-9]{2}_[0-9]{6}_[0-9]{2}_[0-9]{14}_0')
    for label in os.listdir(data_dir_amc_v1):
        ecgs = os.listdir(os.path.join(data_dir_amc_v1, label))
        for ecg in ecgs:
            split_ecg = os.path.splitext(ecg)
            if split_ecg[1] == '.csv' and ecg_re.search(split_ecg[0]):
                case = case_re.search(split_ecg[0]).group(0)
                dt = datetime.datetime.strptime(split_ecg[0][-16:-2], '%Y%m%d%H%M%S')
                if case == 'F-05_170803_07':
                    print(dt)



def get_ecg_list_local_v2(sinus_select_ratio=0.1):
    r = list()
    summary = list()
    summary.append(['case', 'total', 'afib'])
    total_total = 0
    total_afib = 0
    case_re = re.compile('([B-L])-[0-9]{2}_[0-9]{6}_[0-9]{2}')
    for op in os.listdir(data_dir_amc_v2):
        if case_re.match(op):
            case_total = 0
            case_afib = 0
            afib = set()
            if os.path.isdir(os.path.join(data_dir_amc_v2, op, 'AF')):
                for file in os.listdir(os.path.join(data_dir_amc_v2, op, 'AF')):
                    afib.add(os.path.splitext(file)[0])
            for file in os.listdir(os.path.join(data_dir_amc_v2, op)):
                if os.path.splitext(file)[1] == '.csv':
                    if random.random() < sinus_select_ratio or os.path.splitext(file)[0] in afib:
                        case_total += 1
                        case_afib += 1 if os.path.splitext(file)[0] in afib else 0
                        with open(os.path.join(data_dir_amc_v2, op, file), newline='') as csvfile:
                            csvreader = csv.DictReader(csvfile, delimiter=',')
                            tmp_x = list()
                            for row in csvreader:
                                tmp_x.append(float(row['EKG']))
                        r.append([file, 1 if os.path.splitext(file)[0] in afib else 0, tmp_x])
            summary.append([op, case_total, case_afib])
            total_total += case_total
            total_afib += case_afib
    summary.append(['total', total_total, total_afib])

    with open('summary.txt', 'w', newline='') as summary_file:
        csvwriter = csv.writer(summary_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in summary:
            csvwriter.writerow(row)

    return r


load_ecg_list_local_v1()
