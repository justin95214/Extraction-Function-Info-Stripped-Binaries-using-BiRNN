from elftools.elf.elffile import ELFFile
# pip install pyelftools

def get_bytes(filename, functions, r_size = 10):
    with open(filename, 'rb') as f:
        for func in functions.keys():
            f.seek(functions[func]['value'])
            print('%s 0x%x %d' % (func, functions[func]['value'], functions[func]['size']),
                 ['%02x' % b for b in f.read(r_size)])


def process_file(filename):
    print('Processing file:', filename)
    
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
                for i in range(len(p_vaddr)-1,-1,-1):
                    if (symb.entry.st_value >= p_vaddr[i]):
                        st_value = symb.entry.st_value - p_vaddr[i] + p_offset[i]
                #print(symb.name, st_value, symb.entry.st_size) #, symb.entry.st_shndx)
                functions[symb.name] = {'value':st_value, 'size':symb.entry.st_size}
    
    #print(functions)
    return functions
    

if __name__ == '__main__':
    filename = 'a.out'
    functions = process_file(filename)
    get_bytes(filename, functions, 10)