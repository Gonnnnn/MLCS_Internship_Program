"""
1. choose topic: regression / classification 직접 데이터 셋을 선정
2. collect data 원본 데이터를 읽고
3. Class: 원하는 형태의 결과로 저장 및 return

Dataloader for regression using table data (ex. stock price prediction, COVID-19, population)
structure
code column은 제거하고 나머지 데이터를 이용하여
{'Seoul': [{'temperature': [10, 20, 30]}, {'humidity': [0.1, 0.2]}]}
도시별로 정리하여 dictionary를 만든다

도시 -> 항목 -> 데이터 순으로 누적시킨다
"""

import os
import glob
import csv
import numpy as np
import random


class DataLoader:
    def __init__(self):
        self.variables = list()  # list for saving variable names
        self.data = dict()  # dictionary: key = variable name , value = dictionary data
        self.read_data = dict()
        self.process_data = dict()

    def load(self, file_name):
        csv_files = glob.glob(os.path.join(f'{file_name}.csv'))
        random.shuffle(csv_files)
        for file in csv_files:
            with open(file, newline='') as f:
                rows = csv.reader(f)
                done = False
                for row in rows:
                    while not done:
                        for v in row:
                            self.variables.append(v)  # 변수 이름 긁어오기
                        done = True
                    row = row[1:]  # province를 제외한 나머지
                    if row[0] == 'province':
                        pass
                    else:
                        self.data[row[0]] = list()
                        for i in range(len(row) - 1):  # 9EA -> range(8)
                            tmp_dict = dict()
                            tmp_dict[self.variables[i + 2]] = [row[i + 1]]
                            if not self.data[row[0]]:
                                self.data[row[0]] = [tmp_dict]
                            else:
                                self.data[row[0]].append(tmp_dict)
                                # self.data[row[0]][i][self.variables[i + 2]].append(row[i + 1])
                                # key - list index - inner key - list : append
                                # 'Seoul' -> ['date', 'avg_temp', ...] 8EA ->

        return print(self.data.keys())

    def process(self, target_name):
        Data = self.data
        # target name 에 맞게 본다, 이 데이터의 경우 도시 이름
        for d in Data[target_name]:
            for k in d.keys():
                self.process_data[k] = d[k]
        return print(self.process_data)

    def save(self, path_name, city_name):
        if self.data:
            np.savez(os.path.join(f'{path_name}_of_{city_name}.npz'), **self.process_data)
        else:
            print("There are no data in the base.")

    def read(self, r_pathname):
        dataset = np.load(f'{r_pathname}.npz')
        for k in dataset.keys():
            self.read_data[k] = dataset[k]
        return print(self.read_data.keys())

    def keys(self):
        print(self.data.keys())
        return self.data.keys()

    def values(self):
        print(self.data.values())
        return self.data.values()

    def read_keys(self):
        print(self.read_data.keys())
        return self.read_data.keys()

    def read_values(self, key):
        print(self.read_data[key])
        return self.read_data[key]

    def remove(self, date):
        remove_list = []
        for i in range(len(self.read_data['date'])):
            if self.read_data['date'][i] == date:
                for k in self.read_data.keys():
                    remove_list.append(self.read_data[k][i])
                    self.read_data[k][i] = None
        print(remove_list)

    def edit(self, category, old_data, new_data):
        edit_list = []
        for i in range(len(self.read_data[category])):
            if self.read_data[category][i] == old_data:
                edit_list.append(self.read_data[category][i])
                self.read_data[category][i] = new_data
        print(edit_list)


filename = 'weather'
pathname = 'weather'
cityname = 'Seoul'
dl = DataLoader()
dl.load(filename)
dl.process(cityname)
dl.save(pathname, cityname)
read_pathname = f'{filename}_of_{cityname}'
dl.read(read_pathname)
dl.keys()
dl.read_keys()
dl.read_values('date')
dl.edit('date', '2020/06/29', '2021/07/13')
dl.remove('2021/07/13')