import numpy as np
import os, csv, glob


class Dataloader:
    def __init__(self, path, dataset_name, batch_size):
        self.path = path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.date = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.adj = []  # 주식의 분할, 배분 등이 고려된 종가
        self.volume = []  # 거래량

    def normalize(self, arr):
        arr = arr - arr.mean()
        arr = arr / arr.max()

    def how_many(self, csvs):
        # batch_size만큼의 데이터를 불러오기 위해 필요한 csv 파일의 수 구하기

        num_of_rows = 0
        num_of_csv  = 0
        rows_to_cut = 0
        type = 0
        for i in range(len(csvs)):
            with open(csvs[-1-i], newline='') as csv_file:
                reader = csv.reader(csv_file)

                num_of_rows += len(list(reader)) - 1
                num_of_csv += 1

                if num_of_rows < self.batch_size:
                    continue
                elif num_of_rows == self.batch_size:
                    type = 1
                    break
                elif num_of_rows > self.batch_size:
                    rows_to_cut = num_of_rows % self.batch_size
                    type = 2
                    break

        return type, rows_to_cut, num_of_csv

    def creat(self):
        csvs = glob.glob(os.path.join(self.path, '*.csv'))
        type, rows_to_cut, num_of_csv = self.how_many(csvs) # batch size만큼의 데이터를 불러오기 위한 csv 파일의 수


        for i in range(num_of_csv): # 필요한 파일의 수만큼 iteration하며 데이터 정리
            with open(csvs[-1 * num_of_csv + i], newline='') as csv_file:
                reader = csv.reader(csv_file)

                if i == 0:
                    if type == 2:
                        for index, row in enumerate(reader):
                            # rows_to_cut 부분 제외하기
                            # batch size를 맞추기 위함
                            if index > rows_to_cut:
                                row[0] = row[0].replace('-', '')  # '-' 삭제
                                self.date.append(int(row[0]))
                                self.open.append(float(row[1]))
                                self.high.append(float(row[2]))
                                self.low.append(float(row[3]))
                                self.close.append(float(row[4]))
                                self.adj.append(float(row[5]))
                                self.volume.append(float(row[6]))

                    else:
                        for index, row in enumerate(reader):
                            # reader[0]제외
                            if index == 0:
                                continue
                            row[0] = row[0].replace('-', '')  # '-' 삭제
                            self.date.append(int(row[0]))
                            self.open.append(float(row[1]))
                            self.high.append(float(row[2]))
                            self.low.append(float(row[3]))
                            self.close.append(float(row[4]))
                            self.adj.append(float(row[5]))
                            self.volume.append(float(row[6]))

                else:
                    for index, row in enumerate(reader):
                        if index == 0:
                            continue
                        row[0] = row[0].replace('-', '')  # '-' 삭제
                        self.date.append(int(row[0]))
                        self.open.append(float(row[1]))
                        self.high.append(float(row[2]))
                        self.low.append(float(row[3]))
                        self.close.append(float(row[4]))
                        self.adj.append(float(row[5]))
                        self.volume.append(float(row[6]))

        np.savez(self.path+'/'+self.dataset_name+'.npz',
                 date = self.date,
                 open = self.open,
                 high = self.high,
                 low = self.low,
                 close = self.close,
                 adj = self.adj,
                 volume = self.volume)

    def load(self):
        d = np.load(self.path+'/'+self.dataset_name+'.npz')

        self.date = d['date']
        self.open = d['open']
        self.high = d['high']
        self.low = d['low']
        self.close = d['close']
        self.adj = d['adj']
        self.volume = d['volume']

        return {
            'da': d['date'],
            'op': d['open'],
            'hi': d['high'],
            'lo': d['low'],
            'cl': d['close'],
            'ad': d['adj'],
            'vo': d['volume']
        }


os.chdir("/home/gon/Desktop/MLCS-Internship-Program/001_Python_Programming")
path = '004 Custom Dataset'
dataset_name = 'samsung'
batch_size = 40

if __name__ == "__main__":
    dl = Dataloader(path, dataset_name, batch_size)
    dl.creat()
    dl.load()