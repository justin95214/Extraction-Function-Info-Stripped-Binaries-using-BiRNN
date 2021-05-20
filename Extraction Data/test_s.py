from elftools.elf.elffile import ELFFile
import sys

list = []


def get_bytes(filename, functions):
    with open(filename, 'rb') as f:
        f.seek(functions['value'])
        list1 = [0]
        count_0 = 0
        for b in f.read(functions['size']):
            list1.append(b)
            # if b > 0:
            #    list1.append(b)
            #    count_0 =count_0 + 1
        # print("0개수",count_0)

        list.append(list1)
        print('0x%x %d' % (functions['value'], functions['size']), list1)
        # ,['%02x' % b for b in f.read(functions['size'])])


def process_file(filename):
    print('Processing file:', filename)
    print("###############################################################")
    functions = {}
    with open(filename, 'rb') as f:
        p_vaddr = []
        p_offset = []
        p_filesz = []
        p_memsz = []
        elffile = ELFFile(f)

        for seg in elffile.iter_segments():
            if seg.header.p_type == 'PT_LOAD':
                p_vaddr.append(seg.header.p_vaddr)
                p_offset.append(seg.header.p_offset)
                p_memsz.append(seg.header.p_memsz)
                p_filesz.append(seg.header.p_filesz)
                print("p_offset", p_offset, "p_vaddr", p_vaddr, "p_memsz", p_memsz, "p_filesz", p_filesz)

        section_inter = elffile.get_section_by_name('.interp')
        section_text = elffile.get_section_by_name('.text')
        section_fini = elffile.get_section_by_name('.fini')

        mem = 0;
        print(elffile['e_shnum'])
        for i in range(1, elffile['e_shnum']):
            section_numbering = elffile.get_section(i).header.sh_size
            mem = mem + section_numbering

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
        print("section - text:", hex(section_text.header.sh_offset))
        print("section - text size :", hex(section_text.header.sh_size))
        print("section - fini:", hex(section_fini.header.sh_offset))
        print("section - fini size :", hex(section_fini.header.sh_size))

        functions = {'value': section_inter.header.sh_offset, 'size': mem}
        print(hex(section_text.header.sh_offset), section_text.header.sh_entsize, section_text.header.sh_size)
    # print(functions)
    return functions


import pandas as pd

if __name__ == '__main__':
    #dir0 = dir_name0[0]
    # Test_data = pd.read_csv(dir, index_col=0)
    #print(dir0)
    filename = sys.argv[1]
    functions = process_file(filename)
    get_bytes(filename, functions)
    a = pd.DataFrame(list)
    a.to_csv(filename + "_exe.csv", index=False, header=None, sep="\t")