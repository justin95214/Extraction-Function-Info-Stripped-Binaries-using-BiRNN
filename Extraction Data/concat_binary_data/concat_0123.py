import sys
import os
import glob
import pandas as pd
import numpy as np

def atone():
    #input_file = r'C:\Users\82109\Desktop\answer\concat_answer'
    #output_file = r'C:\Users\82109\Desktop\answer\concat_answer\bin+core_answer_3.csv'

    #file_name =  str(i) +'_bin_core7'
    input_file = r'F:\except_binary\o3_core7'
    output_file = r'F:\except_binary\ ' + 'gcc7_0123' +  '.csv'
    print(input_file)
    print(output_file)

    allFile_list = glob.glob(os.path.join(input_file, '*.csv'))
    print(allFile_list)

    allData = []

    for file in allFile_list:
        df = pd.read_csv(file, engine='python')
        allData.append(df)

    dataCombine = pd.concat(allData, axis=0, ignore_index=True)
    print("next")
    print("type",type(dataCombine))
    ndarray = np.array(dataCombine)
    print(len(dataCombine['label']))
    dataCombine.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("next1")


if __name__ == '__main__':
    atone()
    print("\ncomplete!!\n")