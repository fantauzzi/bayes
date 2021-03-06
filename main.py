import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB

def read_telemetry(fname):
    telemetry = []
    with open(fname) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            numeric_line = [float(item) for item in line]
            telemetry.append(numeric_line)
    return telemetry

def read_labels(fname):
    labels = []
    with open(fname) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            labels.append(line[0])
    return labels

def prep_data(telemetry):
    # s, d, s_dot, d_dot
    res=[]
    for line in telemetry:
        s, d, s_dot, d_dot = line
        if 2<d<6:
            d-=4
        elif d>6:
            d-=8
        res.append([d, d_dot])  # acc .848
    return res

dataset_dir = './nd013_pred_data'
telemetry_fname = 'train_states.txt'
telemetry = read_telemetry(dataset_dir+'/'+telemetry_fname)
print('Read', len(telemetry), 'lines from input csv file', telemetry_fname)

labels_fname = 'train_labels.txt'
labels = read_labels(dataset_dir+'/'+labels_fname)
assert len(telemetry) == len(labels)

clf = GaussianNB()
telemetry_prepped=prep_data(telemetry)
clf.fit(telemetry_prepped, labels)

telemetry_test_fname = 'test_states.txt'
telemetry_test = read_telemetry(dataset_dir+'/'+telemetry_test_fname)
print('Read', len(telemetry_test), 'lines from input csv file', telemetry_test_fname)

labels_test_fname='test_labels.txt'
labels_test=read_labels(dataset_dir+'/'+labels_test_fname)
assert len(telemetry_test) == len(labels_test)

telemetry_test_prepped= prep_data(telemetry_test)
acc= clf.score(telemetry_test_prepped, labels_test)

print("Overall accuracy is",acc)
