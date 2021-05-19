import sys
from tkinter import *
import time
import pandas as pd
import numpy as np
from tkinter import filedialog
from elftools.elf.elffile import ELFFile
import os
from functools import partial
from sklearn.preprocessing import LabelEncoder

# input model sequence length
global file_name
model_sequence_length = 50
# file 이름
#filename = 'gcc6_2_.csv'
#model_name = 'testmodel0.h5'
#model_name = 'testmodel_R0_03.h5'
dir_name = []
dir_name0 = []



print(sys.executable)
print(os.path.dirname(sys.executable))



def get_bytes(filename, functions):
    list = []
    with open(filename, 'rb') as f:
        count = 0
        for func in functions.keys():
            f.seek(functions[func]['value'])

            list1 = []
            for b in f.read(functions[func]['size']):
                list1.append(b)
            print(list1)
            list.append(list1)
            count = count + 1


    with open(filename, 'rb') as F:
        for func in functions.keys():
            F.seek(functions[func]['value'])
            print('%s 0x%x %d' % (func, functions[func]['value'], functions[func]['size']),
            ['%02x %d' % (b,b) for b in F.read(functions[func]['size'])])

    print(count)
    return list

def process_file(filename):
    print('Processing file:', filename)
    print("###############################################################")
    functions = {}
    with open(filename, 'rb') as f:
        p_vaddr = []
        p_offset = []
        elffile = ELFFile(f)
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
                for i in range(len(p_vaddr) - 1, -1, -1):
                    if (symb.entry.st_value >= p_vaddr[i]):
                        st_value = symb.entry.st_value - p_vaddr[i] + p_offset[i]
                # print(symb.name, st_value, symb.entry.st_size) #, symb.entry.st_shndx)
                functions[symb.name] = {'value': st_value, 'size': symb.entry.st_size}
                print(st_value, symb.entry.st_size)



    # print(functions)
    return functions

import pandas as pd

if __name__ == '__main__':
    path_dir = 'C:/Users/82109/Desktop/answer/file/2'
    file_list = os.listdir(path_dir)
    print(file_list)
    for m in file_list:
        dir0 = path_dir+"/"+m
        # Test_data = pd.read_csv(dir, index_col=0)
        print(dir0)
        filename = dir0  # sys.argv[1]
        functions = process_file(filename)
        get = get_bytes(filename, functions)
        a = pd.DataFrame(get)
        a.to_csv(filename + "_exe.txt", index=False, header=None, sep="\t")

        mid_filename = filename + "_exe.txt"
        print(filename)

        f = open(mid_filename, 'r')
        # 함수명 텍스트
        line = f.readline()
        list = []
        label_list = []

        while line:
            split_line = line.split()
            # split_line = split_line[1:]
            # print(split_line)

            for i in range(0, len(split_line)):
                list.append(split_line[i])
                if i == 0:
                    label_list.append("1")
                else:
                    label_list.append("0")

            line = f.readline()

        print(len(label_list))
        print(len(list))

        array = [[0 for col in range(2)] for row in range((len(list)))]
        for i in range(0, len(array)):
            array[i][0] = list[i]
            # print(list[i])
            # print(type(list[i]))
            array[i][1] = int(label_list[i])
            if array[i][0] == '"':
                array[i][0] = np.NaN
            array[i][0] = str(array[i][0]).replace('.0"', "")
            array[i][0] = str(array[i][0]).replace('"', "")
            if array[i][0] == "nan":
                array[i][0] = np.NaN

        cols_name = ['bin', 'label']
        df = pd.DataFrame(array, columns=cols_name)
        df = df.dropna(axis=0)
        df.to_csv( 'C:/Users/82109/Desktop/answer/binutils/2/'+ m + "_answer.csv")