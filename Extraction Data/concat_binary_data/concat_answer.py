import sys
import os
import glob
import pandas as pd


def atone():
    #input_file = r'C:\Users\82109\Desktop\answer\concat_answer'
    #output_file = r'C:\Users\82109\Desktop\answer\concat_answer\bin+core_answer_3.csv'

    for i in range(0,4):
        print("최적화:",i)
        file_name =  str(i) +'_bin7'
        input_file = r'F:\except_binary\o'+ file_name
        output_file = r'F:\except_binary\ ' + file_name +  '.csv'
        print(input_file)
        print(output_file)

        allFile_list = glob.glob(os.path.join(input_file, '*.csv'))
        print(allFile_list)

        allData = []
        for data in allFile_list:
            data = data[len(input_file)+1:-9]
            print(data)


        for file in allFile_list:
            df = pd.read_csv(file, engine='python')
            allData.append(df)

        dataCombine = pd.concat(allData, axis=0, ignore_index=True)
        print("next")
        print(len(dataCombine['label']))
        dataCombine.to_csv(output_file, index=False, encoding='utf-8-sig')
        print("next1")


if __name__ == '__main__':
    atone()
    print("\ncomplete!!\n")