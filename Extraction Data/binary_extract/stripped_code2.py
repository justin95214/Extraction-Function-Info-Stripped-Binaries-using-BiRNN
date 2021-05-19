import sys
from elftools.elf.elffile import ELFFile

# pip install pyelftools



def get_bytes(filename, functions):

    with open(filename, 'rb') as f:
        f.seek(functions['value'])
        list1 = []
        for b in f.read(functions['size']):
            list1.append(b)
        list.append(list1)
        #print('0x%x %d %d' % (functions['value'],functions['value'], functions['size']),list1)#,['%02x' % b for b in f.read(functions['size'])])
        array = [[0 for col in range(3)] for row in range((len(list1)))]
        print("길이",len(list1))
        for i in range(0,len(list1)):
            array[i][1] = list1[i]
            array[i][0] = functions['value']
            functions['value'] = functions['value'] + 1

        Dataframe= pd.DataFrame(array)
        print(Dataframe)

        get_bytes_answer(filename,functions_s)
        print(listA)
        for i in range(0, len(list1)):
            for j in listA:
                if j == array[i][0]:
                    array[i][2] = 1

        Dataframe = pd.DataFrame(array,columns=['addr','bin','label'])
        print(Dataframe)

        #Dataframe.to_csv("C:/Users/82109/Desktop/answer/check/1_answer/"+filename[41:]+"_test.csv",index=False)
        #Dataframe.to_csv("C:/Users/82109/Desktop/answer/check/c_3_answer/" + filename[len(path_dir)+1:] + "_test.csv", index=False)
        Dataframe.to_csv("F:/except_binary/o2_core6/" + filename[len(path_dir) + 1:] + "_test.csv",index=False)

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

        print(">>>>>>>>>>>>>>>>>>>>",p_vaddr,p_offset)
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
                #print(symb.entry)
                for i in range(len(p_vaddr) - 1, -1, -1):
                    #print('symb.entry.st_value',symb.entry.st_value ,p_vaddr[i])
                    if (symb.entry.st_value >= p_vaddr[i]):
                        st_value = symb.entry.st_value - p_vaddr[i] + p_offset[i]
                # print(symb.name, st_value, symb.entry.st_size) #, symb.entry.st_shndx)
                functions_s[symb.name] = {'value': st_value, 'size': symb.entry.st_size}

    # print(functions)
    return functions_s

def process_file(filename):
    print('Processing file:', filename)
    print("###############################################################")
    functions = {}
    with open(filename, 'rb') as f:
        p_vaddr = []
        p_offset = []
        p_filesz= []
        p_memsz= []
        elffile = ELFFile(f)

        for seg in elffile.iter_segments():
            if seg.header.p_type == 'PT_LOAD':

                print(seg.header)
                p_vaddr.append(seg.header.p_vaddr)
                p_offset.append(seg.header.p_offset)
                p_memsz.append(seg.header.p_memsz)
                p_filesz.append(seg.header.p_filesz)
                #print("p_offset : %d p_vaddr : %d p_memsz : %d p_filesz : %d" %(p_offset,p_vaddr,p_memsz,p_filesz))

        section_inter = elffile.get_section_by_name('.interp')
        section_text = elffile.get_section_by_name('.text')
        section_fini =  elffile.get_section_by_name('.fini')


        mem =0;
        print(elffile['e_shnum'])
        for i in range(1,elffile['e_shnum']):
            print(elffile.get_section(i))

            section_numbering = elffile.get_section(i).header.sh_size
            mem =mem + section_numbering


        if not section_text:
            print('Is it rightA?')
            return
        if not section_fini:
            print('Is it right B?')
            return

        print(section_text)
        print(elffile.header)

        print("All section - :", hex(section_inter.header.sh_offset))
        print("All section -  size :", mem)
        print("section - text:",hex(section_text.header.sh_offset))
        print("section - text size :",hex(section_text.header.sh_size))
        print("section - fini:", hex(section_fini.header.sh_offset))
        print("section - fini size :", hex(section_fini.header.sh_size))


        functions = {'value': section_inter.header.sh_offset, 'size': mem}
        print(hex(section_text.header.sh_offset),section_text.header.sh_entsize,section_text.header.sh_size)
    #print(functions)
    return functions


import pandas as pd
import os

if __name__ == '__main__':
    #path_dir = 'C:/Users/82109/Desktop/answer/new_file/1'
    #path_dir = 'C:/Users/82109/Desktop/answer/core/exe/3'
    path_dir = 'F:/answer/core/exe/2'


    file_list = os.listdir(path_dir)
    print(file_list)
    for i in range(0, len(file_list)):
        list = []
        listA = []
        filename = path_dir+"/"+file_list[i]
        print(filename)
        print("최적화버전", path_dir[len(path_dir)-1:])
        functions_s =process_file_answer(filename)
        functions = process_file(filename)
        get_bytes(filename, functions)
        a=pd.DataFrame(list)
        #a.to_csv(filename+"_exe11.txt",index=False, header=None, sep="\t")
