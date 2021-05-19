import numpy as np
from tkinter import filedialog
from elftools.elf.elffile import ELFFile
import os
import pandas as pd


#C:/Users/82109/Desktop/answer/binutils/0
#'C:/Users/82109/Desktop/answer/pred_mine/0'
if __name__ == '__main__':

    path_dir = 'C:/Users/82109/Desktop/answer/binutils/3'
    output_file =path_dir[:-1]+"concat_"+path_dir[39:]+"_50_80.csv"
    print(output_file)
    file_list = os.listdir(path_dir)
    print(file_list)

    allData=[]
    for file in file_list:
        df = pd.read_csv(path_dir+"/"+file,engine ='python')
        print(path_dir+"/"+file)
        allData.append(df)

    dataCombine = pd.concat(allData, axis=0, ignore_index =True)
    value_count = dataCombine.loc[dataCombine['label'] == 1, "bin"].value_counts()
    print(value_count)
    data_count = pd.DataFrame(value_count)
    print("next")

    path_dir1 = 'C:/Users/82109/Desktop/answer/pred_mine/3 _50_80'
    output_file1 =path_dir1[:-1]+"concat_"+path_dir1[40:]+".csv"
    print(output_file1)
    file_list1 = os.listdir(path_dir1)
    print(file_list1)

    allData1=[]
    for file in file_list1:
        df1 = pd.read_csv(path_dir1+"/"+file,engine ='python')
        print(path_dir1+"/"+file)
        allData1.append(df1)

    dataCombine1 = pd.concat(allData1, axis=0, ignore_index =True)
    value_count1 = dataCombine1.loc[dataCombine1['predict'] == 1, "binary"].value_counts()
    print(value_count1)
    data_count1 = pd.DataFrame(value_count1)
    print("next")

#############################################################
    count_concat = pd.concat([data_count,data_count1], axis=1)
    count_concat.to_csv(output_file, encoding='utf-8-sig')
    print("next1")