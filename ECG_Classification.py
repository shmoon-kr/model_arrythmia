import os
import re
import csv
import random
import argparse
import numpy as np
from scipy.io import loadmat

#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Apply CNN-LSTM model to datasets.')
parser.add_argument('-d', help='Specify dataset : choose one from cic v1 and v2.')

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import TimeDistributed, Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from statistics import harmonic_mean
from keras.datasets import imdb

data_dir_cic_2017 = 'dataset/cic_2017/'
data_dir_amc_v1 = 'dataset/amc_v1'
data_dir_amc_v2 = 'dataset/amc_v2'


def get_ecg_list_cic():
    with open(os.path.join(data_dir_cic_2017, 'REFERENCE.csv'), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        r = list()
        for row in csvreader:
            d = loadmat(os.path.join(data_dir_cic_2017, row[0]+'.mat'))
            if row[1] == 'N':
                y = 0
            elif row[1] == 'A':
                y = 1
            elif row[1] == 'O':
                y = 2
            elif row[1] == '~':
                y = 3
            else:
                assert False, 'Unknown class %s.' % row[0]
            r.append([row[0], y, d['val'][0]])
        return r


def get_ecg_list_local_v1(sinus_select_ratio=0.5):
    r = list()
    summary = list()
    summary.append(['case', 'total', 'sinus', 'afib', 'vpc', 'noise', 'others'])
    cases = dict()
    case_re = re.compile('([B-L])-[0-9]{2}_[0-9]{6}_[0-9]{2}')
    ecg_re = re.compile('([B-L])-[0-9]{2}_[0-9]{6}_[0-9]{2}_[0-9]{14}')
    for label in os.listdir(data_dir_amc_v1):
        ecgs = os.listdir(os.path.join(data_dir_amc_v1, label))
        for ecg in ecgs:
            split_ecg = os.path.splitext(ecg)
            if split_ecg[1] == '.csv' and ecg_re.search(split_ecg[0]):
                case = case_re.search(split_ecg[0]).group(0)
                if case not in cases:
                    cases[case] = [0] * 5
                if label == 'Normal':
                    pos = 0
                elif label == 'Afib':
                    pos = 1
                elif label == 'VPC':
                    pos = 2
                elif label == 'Noise':
                    pos = 3
                elif label == 'Other_Arr':
                    pos = 4
                else:
                    assert False
                cases[case][pos] += 1
                if random.random() < sinus_select_ratio or label != 'Normal':
                    with open(os.path.join(data_dir_amc_v1, label, ecg), newline='') as csvfile:
                        csvreader = csv.DictReader(csvfile, delimiter=',')
                        tmp_x = list()
                        for row in csvreader:
                            tmp_x.append(float(row['EKG']))
                    if len(tmp_x) == 2500:
                        r.append([case, pos, tmp_x])

    with open('summary_v1.txt', 'w', newline='') as summary_file:
        csvwriter = csv.writer(summary_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['case', 'Normal', 'Afib', 'VPC', 'Noise', 'Others'])
        total = [0] * 5
        distict_afib = 0
        distict_vpc = 0
        for key, val in cases.items():
            for i in range(5):
                total[i] += val[i]
            if val[1]:
                distict_afib += 1
            if val[2]:
                distict_vpc += 1
            csvwriter.writerow([key] + val + [sum(val)])
        csvwriter.writerow(['total'] + total + [sum(total)])
        print(distict_afib, distict_vpc)

    return r


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


def split_dataset_local(ecg_list, training_ratio=0.8):
    training_x = list()
    training_y = list()
    test_x = list()
    test_y = list()
    for row in ecg_list:
        x = list()
        for i in range(24):
            timedistributed_x = list()
            for j in range(200):
                timedistributed_x.append([row[2][i*100+j]])
            x.append(timedistributed_x)
        if random.random() > training_ratio:
            test_x.append(x)
            test_y.append(row[1])
        else:
            training_x.append(x)
            training_y.append(row[1])
    training_x = np.array(training_x)
    training_y = np.array(training_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return training_x, training_y, test_x, test_y


def split_dataset_cic(ecg_list, training_ratio=0.8):
    training_x = list()
    training_y = list()
    test_x = list()
    test_y = list()
    for row in ecg_list:
        if len(row[2]) == 9000:# 5977/8528 data have length 9000 (30 seconds)
            x = list()
            for i in range(15):
                timedistributed_x = list()
                for j in range(600):
                    timedistributed_x.append([row[2][i*600+j]])
                x.append(timedistributed_x)
            if random.random() > training_ratio:
                test_x.append(x)
                test_y.append(row[1])
            else:
                training_x.append(x)
                training_y.append(row[1])
    training_x = np.array(training_x)
    training_y = np.array(training_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return training_x, training_y, test_x, test_y


def get_cic_model():
    ecg_list = get_ecg_list_cic()
    training_x, training_y, test_x, test_y = split_dataset_cic(ecg_list)
    training_y = to_categorical(training_y, num_classes=4)
    categorial_test_y = to_categorical(test_y, num_classes=4)

    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=5, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=5, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=5, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=5, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=5, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=512, kernel_size=3, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(GlobalAveragePooling1D()))
    model.add(BatchNormalization())
    # model.add(Dropout(0.25))
    # model.add(LSTM(128, return_sequences=True))
    # model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(LSTM(128))
    model.add(Dense(4, activation='softmax'))

    model.build((None, 15, 600, 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(training_x, training_y, epochs=100, batch_size=1024)
    loss, acc = model.evaluate(test_x, categorial_test_y, batch_size=1024)
    pred_y = np.argmax(model.predict(test_x), axis=1)

    print(loss, acc)
    res = np.zeros((4, 4))

    for i in range(len(pred_y)):
        res[test_y[i], pred_y[i]] += 1

    print(res)
    score = list()

    for i in range(4):
        precision = res[i, i] / sum(res[:, i])
        recall = res[i, i] / sum(res[i, :])
        f1 = harmonic_mean([precision, recall])
        score.append([precision, recall, f1])

    print(np.array(score))
    model.save('model/ecg_cnn_lstm.h5')


def get_local_model(v=1):
    if v == 1:
        ecg_list = get_ecg_list_local_v1(sinus_select_ratio=0.5)
        num_classes = 5
    elif v == 2:
        ecg_list = get_ecg_list_local_v2(sinus_select_ratio=0.1)
        num_classes = 2
    else:
        assert False

    training_x, training_y, test_x, test_y = split_dataset_local(ecg_list)
    training_y = to_categorical(training_y, num_classes=num_classes)
    categorial_test_y = to_categorical(test_y, num_classes=num_classes)

    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=3, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=2, activation='relu', padding='valid', strides=1)))
    model.add(TimeDistributed(GlobalAveragePooling1D()))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))

    model.build((None, 24, 200, 1))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(training_x, training_y, epochs=100, batch_size=1024)
    loss, acc = model.evaluate(test_x, categorial_test_y, batch_size=1024)
    pred_y = np.argmax(model.predict(test_x), axis=1)

    print(loss, acc)

    res = np.zeros((num_classes, num_classes))
    for i in range(len(pred_y)):
        res[test_y[i], pred_y[i]] += 1

    print(res)
    score = list()

    for i in range(num_classes):
        precision = res[i, i] / sum(res[:, i])
        recall = res[i, i] / sum(res[i, :])
        f1 = harmonic_mean([precision, recall])
        score.append([precision, recall, f1])

    print(np.array(score))
    model.save('model/ecg_cnn_lstm_local.h5')

#ecg_list = get_ecg_list_local()
#get_local_model(v=1)
#get_local_model(v=2)


if __name__ == "__main__":
    # execute only if run as a script
    args = parser.parse_args()

    if args.d is None:
        assert False, 'dataset was not specified. choose one of cic, v1, and v2'
    if args.d == 'cic':
        get_cic_model()
    elif args.d == 'v1':
        get_local_model(v=1)
    elif args.d == 'v2':
        get_local_model(v=2)
    else:
        assert False, 'invalid dataset %s. choose one of cic, v1, and v2' % args.d
