import sys


import pandas as pd
import numpy as np
from tkinter import filedialog
from elftools.elf.elffile import ELFFile
import os
from functools import partial
from sklearn.preprocessing import LabelEncoder

# input model sequence length
global file_name


def get_bytes(filename, functions):

    with open(filename, 'rb') as f:
        f.seek(functions['value'])
        list1 = []
        for b in f.read(functions['size']):
            list1.append(b)
        list.append(list1)
        # print('0x%x %d %d' % (functions['value'],functions['value'], functions['size']),list1)#,['%02x' % b for b in f.read(functions['size'])])
        array = [[0 for col in range(3)] for row in range((len(list1)))]
        print("길이", len(list1))
        for i in range(0, len(list1)):
            array[i][1] = list1[i]
            array[i][0] = functions['value']
            functions['value'] = functions['value'] + 1

        Dataframe = pd.DataFrame(array)
        #print(Dataframe)

        get_bytes_answer(filename, functions_s)
        print("Extracting Executable File.  Wait a moment please.")
        for i in range(0, len(list1)):
            for j in listA:
                if j == array[i][0]:
                    array[i][2] = 1

        Dataframe = pd.DataFrame(array, columns=['addr', 'bin', 'label'])
        

        # Dataframe.to_csv("C:/Users/82109/Desktop/answer/check/1_answer/"+filename[41:]+"_test.csv",index=False)
        # Dataframe.to_csv("C:/Users/82109/Desktop/answer/check/c_3_answer/" + filename[len(path_dir)+1:] + "_test.csv", index=False)
        print("csv파일 저장:",filename + "_test.csv")
        Dataframe.to_csv(filename + "_test.csv", index=False)

def get_bytes_answer(filename, functions):

    with open(filename, 'rb') as M:
        for func in functions.keys():
            M.seek(functions[func]['value'])
            listA.append(functions[func]['value'])
            list12 = []

def process_file_answer(filename):
    print('Processing file:', filename)

    functions_s = {}
    with open(filename, 'rb') as W:
        p_vaddr = []
        p_offset = []
        elffile = ELFFile(W)
        for seg in elffile.iter_segments():
            if seg.header.p_type == 'PT_LOAD':
                p_vaddr.append(seg.header.p_vaddr)
                p_offset.append(seg.header.p_offset)

        
        section = elffile.get_section_by_name('.symtab')
        if not section:
            print('No symbol table found. Perhaps this ELF has been stripped?')
            return

        for symb in section.iter_symbols():
            if symb.entry.st_info.type == 'STT_FUNC':
                if symb.entry.st_shndx == 'SHN_UNDEF':
                    continue
                if symb.entry.st_size == 0:
                    continue
                st_value = symb.entry.st_value
                # print(symb.entry)
                for i in range(len(p_vaddr) - 1, -1, -1):
                    # print('symb.entry.st_value',symb.entry.st_value ,p_vaddr[i])
                    if (symb.entry.st_value >= p_vaddr[i]):
                        st_value = symb.entry.st_value - p_vaddr[i] + p_offset[i]
                # print(symb.name, st_value, symb.entry.st_size) #, symb.entry.st_shndx)
                functions_s[symb.name] = {'value': st_value, 'size': symb.entry.st_size}

    # print(functions)
    return functions_s

def process_file(filename):
    print('Processing file:', filename)

    functions = {}
    with open(filename, 'rb') as f:
        p_vaddr = []
        p_offset = []
        p_filesz = []
        p_memsz = []
        elffile = ELFFile(f)

        for seg in elffile.iter_segments():
            if seg.header.p_type == 'PT_LOAD':
                #print(seg.header)
                p_vaddr.append(seg.header.p_vaddr)
                p_offset.append(seg.header.p_offset)
                p_memsz.append(seg.header.p_memsz)
                p_filesz.append(seg.header.p_filesz)
                # print("p_offset : %d p_vaddr : %d p_memsz : %d p_filesz : %d" %(p_offset,p_vaddr,p_memsz,p_filesz))

        section_inter = elffile.get_section_by_name('.interp')
        section_text = elffile.get_section_by_name('.text')
        section_fini = elffile.get_section_by_name('.fini')

        mem = 0
        print(elffile['e_shnum'])
        for i in range(1, elffile['e_shnum']):
            #print(elffile.get_section(i))

            section_numbering = elffile.get_section(i).header.sh_size
            mem = mem + section_numbering

        if not section_text:
            print('Is it rightA?')
            return
        if not section_fini:
            print('Is it right B?')
            return

        #print(section_text)
        #print(elffile.header)

        #print("All section - :", hex(section_inter.header.sh_offset))
        #print("All section -  size :", mem)
        #print("section - text:", hex(section_text.header.sh_offset))
        #print("section - text size :", hex(section_text.header.sh_size))
        #print("section - fini:", hex(section_fini.header.sh_offset))
        #print("section - fini size :", hex(section_fini.header.sh_size))

        functions = {'value': section_inter.header.sh_offset, 'size': mem}
        #print(hex(section_text.header.sh_offset), section_text.header.sh_entsize, section_text.header.sh_size)
    # print(functions)
    return functions

    import pandas as pd



def load_trained_model():
    from tensorflow.keras.models import load_model
    #C:/Users/82109/Desktop/model/
    model = load_model(model_name)
    print('model load 완료')

    return model

def eval_load():
    # data load
    print('1. test data_load start')
    print(filename)
    #인자로 받아온 파일명
    dir0 = filename
    print(dir0)
    #실행파일을 csv로 바꾼 것
    file_n =  dir0+"_test.csv"
    print(file_n)
    #csv파일 읽기 (여기부터 수정하면됨)
    data = pd.read_csv(file_n, index_col=0)
    data.reset_index(inplace=True, drop=True)
    print('2. input data load and reset_index finish')
    print("model_sequence_length :",model_sequence_length)
    # input sequence 로 나눔, 맨 뒤 나머지 빼기
    rest = len(data) % (model_sequence_length)
    print('3. rest data : ', rest)
    data_sub_rest = data[:len(data) - rest]
    print('4. subtracting the rest finish', data_sub_rest.shape)

    # one hot encoding
    data_onehot = pd.get_dummies(data_sub_rest['bin'])
    print('5. one hot encoding finish', data_onehot.shape)

    data_ndarray = data_onehot.to_numpy() # data numpy 변환
    print('6. Transforming ndarray data finish', data_ndarray.shape)

    print('7. before reshape data shape', data_ndarray.shape)
    data_reshape = data_ndarray.reshape(-1, model_sequence_length, data_ndarray.shape[1])
    print('8. after reshape data shape', data_reshape.shape)
    print('9. final return data',data_reshape.shape)

    #label_testfile["text"] = filename +' load 완료'
    print('10. test file load 완료')
    # reshape 된 data return
    return data_reshape
    #eval_load해서 anaylsis()함수에서 test_data가 predict에 바로 넣을수있게

def anaylsis():


    print("anaylsis")
    global test_model
    test_model = load_trained_model()  # model load

    global test_data
    test_data = eval_load()

    dir0 = filename
    print(dir0)
    #실행파일을 csv로 바꾼 것
    file_n =  dir0+"_test.csv"
    data0 = pd.read_csv(file_n, index_col=0)
    data0.reset_index(inplace=True, drop=True)
    check_list=[]
    for i in data0['bin']:
        check_list.append(i)

    check =set(check_list)
    #print(check)

    print("Analyzing the binary... ")
    print(test_data.shape)

    pr = test_model.predict(test_data)
    print(type(pr))
    print(pr.shape)
    #print(hex)
    pred = np.round(np.array(pr).flatten().tolist())

    dir0 = filename

    print(dir0)
    file_name =  dir0+"_test.csv"
    print("pred:",file_name)
    hex_data = pd.read_csv(file_name, index_col=0)
    
    answer_box = [[0 for col in range(3)] for row in range((len(pred)))]
    for i in range(0,len(pred)):
        answer_box[i][0]= data0['bin'][i]
        answer_box[i][1] = hex(int(data0['bin'][i]))
        answer_box[i][2]= pred[i]

    start_list =[]
    count = 0
    for i in range(0, len(pred)):
        if answer_box[i][2] ==1:
            count = count + 1
            start_list.append(answer_box[i][1])

    for j in start_list:
        if j != answer_box[i][1]:
            start_list.append(answer_box[i][1])
    
    
    cols_name = ['binary','Hex', 'predict']
    pred_df = pd.DataFrame(answer_box, columns=cols_name)
    pred_df.to_csv(filename+"_pred.csv", sep=",")
    #예측결과
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


    # 테스트 predict 결과들 비교 (평가지표 보기위함)

    y_test = np.array(data0['label'][:len(pred)]).flatten().tolist()

    # 13. 평가지표들 출력
    ## 평가지표들
    k_accuracy = float(accuracy_score(y_test, pred))
    k_recall = float(recall_score(y_test, pred))
    k_precision = float(precision_score(y_test, pred))
    k_f1_score = float(f1_score(y_test, pred))
    # k_cm = float(confusion_matrix(y_test, pred))

    print('accuracy_score', k_accuracy)
    print('recall_score', k_recall)
    print('precision_score', k_precision)
    print('f1_score', k_f1_score)

if __name__ == '__main__':
    #명령어 실행시 인자로 첫번째 : gcc컴파일러버전  / 두번째 : 분석할 실행파일(절대경로)
    print("gcc Ver :  ")
    gcc_ver = input()
    gcc_ver = int(gcc_ver)
    print("directory (linux):  ")
    filename = input()
    print("filename:",filename)
    print("gcc Ver :",gcc_ver)
    #컴파일러 버전 선택(sequence길이와 모델h5이 바뀜)
    #h5모델 수정해서 적용하면 됨
    if gcc_ver == 6 or 8:
        model_sequence_length = 96
        print("model_sequence_length :",model_sequence_length)
        if gcc_ver == 6:
            print('gcc6_bin_core_s96_h144_o0123.h5')
            model_name = 'gcc6_bin_core_s96_h144_o0123.h5'
        elif gcc_ver == 8:
            print('gcc8_bin_core_s96_h144_o0123.h5')
            model_name = 'gcc8_bin_core_s96_h144_o0123.h5'

    if gcc_ver == 7 or 9:
        model_sequence_length = 192
        print("model_sequence_length :", model_sequence_length)
        if gcc_ver == 7:
            print('gcc7_bin_core_s192_h288_o0123.h5')
            model_name = 'gcc7_bin_core_s192_h288_o0123.h5'
        elif gcc_ver == 9:
            print('gcc9_bin_core_s192_h288_o0123.h5')
            model_name = 'gcc9_bin_core_s192_h288_o0123.h5'

    list = []
    listA = []

    functions_s = process_file_answer(filename)
    functions = process_file(filename)
    get_bytes(filename, functions)
    a = pd.DataFrame(list)
    anaylsis()

    os.system('pause')
