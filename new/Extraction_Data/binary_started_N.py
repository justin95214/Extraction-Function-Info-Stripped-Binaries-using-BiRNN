from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import warnings
import collections
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
# status initialize

import os

def check_func():
    gcc_version = 7
    package = 'core'
    unit = 0
    input = 6
    for oo in range(0,4):
        path = 'F:/except_binary/'
        file = '' + str(oo) + '_' + package + str(gcc_version) +'.csv'
        print("input:", input)
        print("unit:", unit)
        print("path:", path)
        print("file:", path + file)
        save_path = 'F:/except_binary/anaylsis/' + file
        print("save_path:", save_path)

        # 파일읽기
        bin8_0 = pd.read_csv(path + file, index_col=0)
        print(bin8_0.shape)

        # reset_index (hex processing 하면서 값이 빠졌으니까 + n_gram 에서 index를 다루기 때문에)
        bin8_0.reset_index(inplace=True, drop=True)

        print('reset_index 완료')
        print('input data shape')
        bin8_0.head()

        # In[4]:

        # (2-1) 데이터체크 1 - hex(16진수)가 256 label을 가져야 dummies 변환 가능
        # 16진수 256개 종류가 있어서 pd.get_dummies 사용 가능.
        print(len(bin8_0['bin'].unique()))

        # (2-2) 데이터 체크 2 - 1, 0 비율 ==> 1이 함수의 갯수를 뜻함
        # 정답 데이터 1, 0 비율 확인  ==> 1이 함수의 갯수를 뜻함
        print(bin8_0['label'].value_counts())

        # ## (3) N Byte씩 자르기

        # In[5]:

        idx_bin = bin8_0[bin8_0['label'] == 1].index  # 407, 474 ...
        ls_bin = list(idx_bin)
        #print(ls_bin)
        print("length:",len(ls_bin))
        binary_box = [[0 for col in range(2)] for row in range(0, len(ls_bin))]
        for idx,i in zip(range(0,len(ls_bin)), ls_bin):
            list_temp =[]

            for j in range(0,5):
                list_temp.append(bin8_0['bin'][i+j])
            binary_box[idx][0] = tuple(list_temp)

        cols_name = ['binary-start chunk','anaylsis']
        pred_df = pd.DataFrame(binary_box,columns=cols_name)
        #print(pred_df['binary-start chunk'].to_numpy())
        how_many = set(pred_df['binary-start chunk'])
        for k in range(0,len(how_many)):
            pred_df['anaylsis'][k] = np.array(list(how_many),dtype=object)[k]

        print(type(how_many))
        #print(how_many)
        pred_df.to_csv(save_path, sep=",")
        print('save 완료')

if __name__ == "__main__":
    check_func()