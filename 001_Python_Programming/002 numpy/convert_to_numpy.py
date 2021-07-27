import os, glob
import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split

'''
과제 요구사항을 제대로 이해하지 못했을 수 있어서 잘 못된 것이 있다면 알려주시면 감사하겠습니다. 그에 맞게 수정하겠습니다.
'''


os.chdir("/home/gon/Desktop/MLCS-Internship-Program/001 Python Programming/001 csv")

path_to_train_2d_datasets1 = os.path.join('datasets/outputs/train1/*.csv')
train_2d_files1 = glob.glob(path_to_train_2d_datasets1)
train1 = np.empty((len(train_2d_files1), 28, 28))
train_label1 = np.empty((len(train_2d_files1)))

for data_idx, data_path in enumerate(train_2d_files1):
    train_label1[data_idx] = list(data_path)[-8]
    # data_path를 list화 해서 label부분을 직접 참조해 값을 넣어줌. 다른 방법이 있는지 생각해보자. 파일마다 숫자의 크기가 달라서 참조해야하는 위치가 다를땐??
    path_to_train_2d_data = os.path.join(data_path)
    tr = open(path_to_train_2d_data, 'r')
    csv_data = csv.reader(tr)
    list_csv_data = list(csv_data)
    for i in range(28):
        for j in range(28):
            train1[data_idx,i,j] = list_csv_data[i][j]
    tr.close()
print("train datasets1 label:",train_label1)
#for loop을 이용해 empty numpy arrays를 loaded 2D MNIST data와 label로 채움
#train1 directory 처리 완료

path_to_train_2d_datasets2 = os.path.join('datasets/outputs/train2/*.csv')
train_2d_files2 = glob.glob(path_to_train_2d_datasets2)
train2 = np.empty((len(train_2d_files2), 28, 28))
train_label2 = np.empty((len(train_2d_files2)))

for data_idx, data_path in enumerate(train_2d_files2):
    train_label2[data_idx] = list(data_path)[-8]
    path_to_train_2d_data = os.path.join(data_path)
    tr = open(path_to_train_2d_data, 'r')
    csv_data = csv.reader(tr)
    list_csv_data = list(csv_data)
    for i in range(28):
        for j in range(28):
            train2[data_idx,i,j] = list_csv_data[i][j]
    tr.close()
print("train datasets2 label:", train_label2)
#train2 directory 처리 완료

path_to_test_2d_datasets = os.path.join('datasets/outputs/test/*.csv')
test_2d_files = glob.glob(path_to_test_2d_datasets)
test = np.empty((len(test_2d_files), 28, 28))
test_label = np.empty((len(test_2d_files)))

for data_idx, data_path in enumerate(test_2d_files):
    test_label[data_idx] = list(data_path)[-8]
    path_to_test_2d_data = os.path.join(data_path)
    tr = open(path_to_test_2d_data, 'r')
    csv_data = csv.reader(tr)
    list_csv_data = list(csv_data)
    for i in range(28):
        for j in range(28):
            test[data_idx,i,j] = list_csv_data[i][j]
    tr.close()
print("test datasets label:", test_label)
#test directory 처리 완료

tot_temp = np.concatenate((train1, train2, test), axis=0)
tot_label_temp = np.concatenate((train_label1, train_label2, test_label), axis=0)
#concat 완료

joined = list(zip(tot_temp, tot_label_temp))
random.shuffle(joined)
tot_temp, tot_label_temp = zip(*joined)
tot = list(tot_temp)
tot_label = list(tot_label_temp)
#두개의 list(image data와 label)를 같은 order로 shuffle 후 다시 list화
#train_test_split module을 쓰면 의미가 없어지긴 한다

'''split module의 경우 비율을 입력하여 나누는 방법을 찾지 못해 train_test_split module을 사용했습니다.'''
x_train_and_test, x_val, y_train_and_test, y_val = train_test_split(tot, tot_label, test_size=0.2, shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(x_train_and_test, y_train_and_test, test_size=0.125, shuffle=False)
#조사해보니 shuffle=False로 하지 않았을 때, random하게 shuffle해서 비율을 나누는데 같은 순서로 shuffle됨. 69~75행에서 shuffle을 할 필요 x
#먼저 val을 따로 나눠주고, train과 test를 비율에 맞게 나눠줌

path_to_save = os.path.join('dataset.npz')
np.savez(path_to_save, train_x=x_train, train_y=y_train, val_x=x_val, val_y=y_val, test_x=x_test, test_y=y_test)
#npz파일로 저장 완료
