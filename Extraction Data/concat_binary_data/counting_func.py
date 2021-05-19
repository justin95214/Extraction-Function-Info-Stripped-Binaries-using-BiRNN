import numpy as np
from tkinter import filedialog
from elftools.elf.elffile import ELFFile
import os
import pandas as pd


#C:/Users/82109/Desktop/answer/binutils/0
#'C:/Users/82109/Desktop/answer/pred_mine/0'
if __name__ == '__main__':

    path_dir = 'F:/except_binary/gcc9/o0123'
    #path_dir = 'C:/Users/82109/Desktop/answer/check/2_answer'
    #path_dir = 'C:/Users/82109/Desktop/answer/binutils/0'
    file_list = os.listdir(path_dir)
    print(file_list)

    answer_box = [[0 for col in range(2)] for row in range((len(file_list)))]
    count_value_box = [[0 for col in range(19)] for row in range(50)]
    for i in range(0,len(file_list)):
        print(path_dir+"/"+file_list[i])
        print("파일분석 :",file_list[i])
        a=  pd.read_csv(path_dir+"/"+file_list[i], index_col=0)
        print("길이",len(a['label']))
        value_count = a.loc[a['label']==1,"bin"].value_counts()
        value_count = value_count.to_frame().reset_index()



        print(value_count)
        #print(a['predict'])
        count = 0
        #mine >>predict ///
        # for j in range(0,len(a['label'])):
        #     #array[i][0] = str(array[i][0]).replace('.0"', "")
        #     if int(a['label'][j]) == 1:
        #         count = count+1


        answer_box[i][0] = file_list[i]
        answer_box[i][1] = count

    print(answer_box)
    cols_name = ['file', 'count']
    df = pd.DataFrame(answer_box, columns=cols_name)
    df.to_csv("F:/except_binary/gcc9"+"/"+'bincore_0123' + "_answer.csv")
    """
    ##########################################################################33
    path_dir = 'C:/Users/82109/Desktop/answer/pred_mine/core_predict'
    file_list = os.listdir(path_dir)
    print(file_list)

    answer_box = [[0 for col in range(2)] for row in range((len(file_list)))]
    for i in range(0,len(file_list)):
        print(path_dir+"/"+file_list[i])
        a=  pd.read_csv(path_dir+"/"+file_list[i], index_col=0)
        #print(a['predict'])
        count = 0
        #mine >>predict ///
        for j in range(0,len(a['predict'])):
            #array[i][0] = str(array[i][0]).replace('.0"', "")
            if int(a['predict'][j]) == 1:
                count = count+1

        print(file_list[i],'func count =',count)
        answer_box[i][0] = file_list[i]
        answer_box[i][1] = count
        value_count = a.loc[a['predict']==1,"binary"].value_counts()
        value_count = value_count.to_frame().reset_index()
        print(value_count)

    print(answer_box)
    cols_name = ['file', 'count']
    df = pd.DataFrame(answer_box, columns=cols_name)
    df.to_csv("C:/Users/82109/Desktop"+"/"+'bin_0' + "_analysis.csv")
    """