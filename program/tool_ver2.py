import sys
import os
import numpy as np
import cv2
import array
from PIL import Image
import pandas as pd
import numpy as np
from elftools.elf.elffile import ELFFile
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# input model sequence length
global file_name
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
abs = 'D:/Downloads/tool_verW/tool_verW/'

# MODEL_PATH = './model/Judge_Model.hdf5'
VERSION_MODEL_PATH = abs+'compilerCNN.h5'
BYTES_SAVE_PATH = abs+'bytes/'
IMAGE_SIZE = 160

def get_bytes(filename, functions):
    with open(filename, 'rb') as f:
        f.seek(functions['value'])
        list1 = []
        for b in f.read(functions['size']):
            list1.append(b)
        list.append(list1)
        # print('0x%x %d %d' % (functions['value'],functions['value'], functions['size']),list1)#,['%02x' % b for b in f.read(functions['size'])])
        array = [[0 for col in range(3)] for row in range((len(list1)))]
        print("바이너리 총길이(개수)", len(list1))
        for i in range(0, len(list1)):
            array[i][1] = list1[i]
            array[i][0] = functions['value']
            functions['value'] = functions['value'] + 1

        Dataframe = pd.DataFrame(array)
        # print(Dataframe)

        get_bytes_answer(filename, functions_s)
        print("Extracting Executable File.  Wait a moment please.")
        for i in range(0, len(list1)):
            for j in listA:
                if j == array[i][0]:
                    array[i][2] = 1

        Dataframe = pd.DataFrame(array, columns=['addr', 'bin', 'label'])

        # Dataframe.to_csv("C:/Users/82109/Desktop/answer/check/1_answer/"+filename[41:]+"_test.csv",index=False)
        # Dataframe.to_csv("C:/Users/82109/Desktop/answer/check/c_3_answer/" + filename[len(path_dir)+1:] + "_test.csv", index=False)
        print("csv파일 저장:", filename + "_test.csv")
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
                # print(seg.header)
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
            # print(elffile.get_section(i))

            section_numbering = elffile.get_section(i).header.sh_size
            mem = mem + section_numbering

        if not section_text:
            print('Is it rightA?')
            return
        if not section_fini:
            print('Is it right B?')
            return

        # print(section_text)
        # print(elffile.header)

        # print("All section - :", hex(section_inter.header.sh_offset))
        # print("All section -  size :", mem)
        # print("section - text:", hex(section_text.header.sh_offset))
        # print("section - text size :", hex(section_text.header.sh_size))
        # print("section - fini:", hex(section_fini.header.sh_offset))
        # print("section - fini size :", hex(section_fini.header.sh_size))

        functions = {'value': section_inter.header.sh_offset, 'size': mem}
        # print(hex(section_text.header.sh_offset), section_text.header.sh_entsize, section_text.header.sh_size)
    # print(functions)
    return functions

    import pandas as pd


def load_trained_model():
    # C:/Users/82109/Desktop/model/
    function_model = load_model(model_name)
    print('model load 완료')

    return function_model


def eval_load():
    # data load
    print(' 1. test data_load start')
    # print(filename)
    # 인자로 받아온 파일명
    dir0 = filename
    # print(dir0)
    # 실행파일을 csv로 바꾼 것
    file_n = dir0 + "_test.csv"
    # print(file_n)
    # csv파일 읽기 (여기부터 수정하면됨)
    data = pd.read_csv(file_n, index_col=0)
    data.reset_index(inplace=True, drop=True)
    print(' 2. input data load and reset_index finish')
    print("model_sequence_length :", model_sequence_length)
    # input sequence 로 나눔, 맨 뒤 나머지 빼기
    rest = len(data) % (model_sequence_length)
    print(' 3. rest data : ', rest)
    data_sub_rest = data[:len(data) - rest]
    print(' 4. subtracting the rest finish', data_sub_rest.shape)

    # one hot encoding
    data_onehot = pd.get_dummies(data_sub_rest['bin'])
    print(' 5. one hot encoding finish', data_onehot.shape)

    data_ndarray = data_onehot.to_numpy()  # data numpy 변환
    print(' 6. Transforming ndarray data finish', data_ndarray.shape)

    print(' 7. before reshape data shape', data_ndarray.shape)
    data_reshape = data_ndarray.reshape(-1, model_sequence_length, data_ndarray.shape[1])
    print(' 8. after reshape data shape', data_reshape.shape)
    print(' 9. final return data', data_reshape.shape)

    # label_testfile["text"] = filename +' load 완료'
    print(' 10. test file load 완료')
    # reshape 된 data return
    return data_reshape
    # eval_load해서 anaylsis()함수에서 test_data가 predict에 바로 넣을수있게


def anaylsis():
    print("anaylsis")
    global test_model
    test_model = load_trained_model()  # model load

    global test_data
    test_data = eval_load()

    dir0 = filename
    print(dir0)
    # 실행파일을 csv로 바꾼 것
    file_n = dir0 + "_test.csv"
    data0 = pd.read_csv(file_n, index_col=0)
    data0.reset_index(inplace=True, drop=True)
    check_list = []
    for i in data0['bin']:
        check_list.append(i)

    check = set(check_list)
    # print(check)

    print("Analyzing the binary... ")
    # print(test_data.shape)

    pr = test_model.predict(test_data)
    # print(type(pr))
    # print(pr.shape)
    # print(hex)
    pred = np.round(np.array(pr).flatten().tolist())

    dir0 = filename

    # print(dir0)
    file_name = dir0 + "_test.csv"
    # print("pred:",file_name)
    hex_data = pd.read_csv(file_name, index_col=0)

    answer_box = [[0 for col in range(3)] for row in range((len(pred)))]
    for i in range(0, len(pred)):
        answer_box[i][0] = data0['bin'][i]
        answer_box[i][1] = hex(int(data0['bin'][i]))
        answer_box[i][2] = pred[i]

    start_list = []
    count = 0
    for i in range(0, len(pred)):
        if answer_box[i][2] == 1:
            count = count + 1
            start_list.append(answer_box[i][1])

    for j in start_list:
        if j != answer_box[i][1]:
            start_list.append(answer_box[i][1])

    cols_name = ['binary', 'Hex', 'predict']
    pred_df = pd.DataFrame(answer_box, columns=cols_name)
    pred_df.to_csv(filename + "_pred.csv", sep=",")
    # 예측결과
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

    long = "saved directory - predicted executable file :" + file_name

    for i in range(0, len(long) + 5):
        str_list.append("=")

    str = ''.join(str_list)

    print("┏" + str + "┓")
    print("saved directory - predicted executable file :", file_name)
    print("┠" + str + "┨")
    print('accuracy_score', k_accuracy)
    print('recall_score', k_recall)
    print('precision_score', k_precision)
    print('f1_score', k_f1_score)
    print("┗" + str + "┛")


##### hwchoi's task #####
def bytes2png(f, width, save_dir):
    '''
    실제 .txt의 정보를 읽어들여서 이미지화 시키는 메소드
    :param f: filename
    :param width: image width
    :param save_dir: 저장 장소
    :return: image 저장 장소
    '''
    undecodedByte = 'FF'

    file = f

    """
        Construct image name and return if file already exists
    """
    image_name = f
    image_buf = image_name.split('/')[-1]
    image_buf = image_buf.split('.')[0]
    print('image buf : ', image_buf)
    image_name = './Images/' + image_buf + '.png'
    print('file : ', file)
    print('image name : ', image_name)

    # 이미지 저장 경로 설정
    # Images 폴더가 없으면 새로 생성
    if not os.path.exists(save_dir + '\\Images'):
        os.mkdir(save_dir + '\\Images')

    b_data = array.array('i')
    print(file)

    for line in open(file, 'r'):
        for byte in line.rstrip().split():
            # 각 라인마다 앞의 8개의 코드는 주소를 뜻하므로, 제외함.
            # 실제 데이터에서 가져올 때에는 주소 개념이 없음.

            # byte가 ??인 파트는 따로 처리하는 작업
            # 이 부분이 정확도에 크게 기여하는 곳이므로, 처리할 필요가 있음.
            if byte == '??':
                byte = undecodedByte

            # 간혹 byte 코드에 \\이라고 적힌 부분이 있어서 건너뛰어야 함.
            if byte.__contains__('\\'):
                continue

            b_data.append(int(byte, base=16))
            # 16진수 형태로 배열에 부착함

    i = 0
    while (True):
        i += 1
        if i * i <= len(b_data) and len(b_data) < (i + 1) * (i + 1):
            height = i + 1
            width = i + 1
            break

    if len(b_data) < (width * height):
        b_data += array.array('i', (0,) * (width * height - len(b_data)))
    image_buffer = np.fromiter(b_data, dtype=np.uint8).reshape((height, width))
    img = Image.fromarray(image_buffer, 'L')
    img.save(image_name)

    return image_name


# 파일을 읽어 데이터를 리스트로 반환
def read_file(file_path):
    '''
    한 실행파일에 대한 정보를 bytes 코드로 추출하는 메소드
    :param file_path:
    :return: 16진수화한 바이트코드
    '''
    try:
        data_list = []
        with open(file_path, mode="rb") as f:
            while True:
                buf = f.read(16)
                if not buf:
                    break
                else:
                    data_list.append(buf)

        return data_list

    except Exception as ex:
        print("[ERROR] : {0}".format(ex))
        raise Exception(ex) from ex


# data를 hex로 기록한 .txt파일 생성
def create_bytes_file(save_dir, file_name, data):
    '''
    read_file에서 추출한 bytes 코드를 .txt형태로 저장하는 메소드
    :param save_dir:
    :param file_name:
    :param data:
    :return:
    '''
    file_name = file_name.replace('/', '\\')
    file_path = save_dir + file_name.split('\\')[-1] + '.txt'
    try:
        with open(file_path, mode="wb") as f:
            for i in data:
                f.write(b' '.join(['{:02x}'.format(int(x)).upper().encode() for x in list(i)]))
                f.write(b'\r\n')

            print("[생성 완료] {}".format(file_name))

    except Exception as e:
        print(e)

    return file_path.strip()


def version_judge(version_model, path):
    # 실제로 판단하는 모듈
    hex_data = read_file(path)
    # File의 bytes값을 읽어들임.

    filename = path.split('/')[-1]
    bytes_file = create_bytes_file(BYTES_SAVE_PATH, filename, hex_data)
    # bytes값을 .txt 형태로 저장

    file_size = os.path.getsize(str(bytes_file))
    if (file_size < 10 * 1024):
        obj_img = bytes2png(bytes_file, 32, '.')
    elif (file_size < 30 * 1024):
        obj_img = bytes2png(bytes_file, 64, '.')
    elif (file_size < 60 * 1024):
        obj_img = bytes2png(bytes_file, 128, '.')
    elif (file_size < 100 * 1024):
        obj_img = bytes2png(bytes_file, 256, '.')
    elif (file_size < 200 * 1024):
        obj_img = bytes2png(bytes_file, 384, '.')
    elif (file_size < 500 * 1024):
        obj_img = bytes2png(bytes_file, 512, '.')
    elif (file_size < 1000 * 1024):
        obj_img = bytes2png(bytes_file, 768, '.')
    else:
        obj_img = bytes2png(bytes_file, 1024, '.')

    # .txt의 파일을 Image화 시키는 과정
    #obj_img = cv2.resize(cv2.imread(obj_img, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
    obj_img =plt.imread(obj_img).copy()
    obj_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    #obj_img = obj_img / 255
    #obj_img = np.column_stack([obj_img.flatten()])
    obj_img = np.reshape(obj_img, [IMAGE_SIZE, IMAGE_SIZE, 1])
    # 실제 데이터를 np.array에서 Image로 바꾸는 과정

    obj_img = np.expand_dims(obj_img, axis=0)

    y_prob = version_model.predict(obj_img)

    if y_prob.argmax() == 2:
        new_arr = y_prob
        new_arr[0][2] = 0
        y_prob[0][2] = 0
        if new_arr.argmax == 4:
            y_prob[0][2] /= 5

    y_classes = y_prob.argmax(axis=-1)

    result = y_classes[0]
    result = int(result)
    # 0 - 3, 1 - 4, 2 - 5, 3 - 6, 4 - 7, 5 - 8, 6 - 9로 대응
    result += 3

    return result


##########################################

if __name__ == '__main__':

    str0_list=[]
    long0 = " model :  gcc6_bin_core_s96_h144_o0123.h5"

    for i in range(0, len(long0) + 5):
        str0_list.append("=")

    str0 = ''.join(str0_list)


    print("start")
    version_model = load_model(VERSION_MODEL_PATH)
    print('Please wait for a second... Compiler version detection model loading...')

    filename = sys.argv[1]#input("directory (linux) : ")

    print("┏" + str0 + "┓")
    print(" filename:", filename)

    gcc_ver = version_judge(version_model, filename)
    print(" gcc_ver :", gcc_ver)

    if gcc_ver == 6 or gcc_ver ==  8:
        model_sequence_length = 96
        print(" model_sequence_length :", model_sequence_length)
        if gcc_ver == 6:
            print(" model : ", 'gcc6_bin_core_s96_h144_o0123.h5')
            model_name = abs+'gcc6_bin_core_s96_h144_o0123.h5'
        elif gcc_ver == 8:
            print(" model : ", 'gcc8_bin_core_s96_h144_o0123.h5')
            model_name = abs+'gcc8_bin_core_s96_h144_o0123.h5'
    elif gcc_ver == 7 or gcc_ver == 9:
        model_sequence_length = 192
        if gcc_ver == 7:
            print(" model : ", 'gcc7_bin_core_s192_h288_o0123.h5')
            model_name = abs+'gcc7_bin_core_s192_h288_o0123.h5'
        elif gcc_ver == 9:
            print(" model : ", 'gcc9_bin_core_s192_h288_o0123.h5')
            model_name =abs+ 'gcc9_bin_core_s192_h288_o0123.h5'
    else:
        print(' 지원하지 않는 컴파일러 버전')
    print("┗" + str0 + "┛")
    list = []
    listA = []
    str_list = []
    functions_s = process_file_answer(filename)
    functions = process_file(filename)
    get_bytes(filename, functions)
    a = pd.DataFrame(list)
    anaylsis()
    str = ''.join(str_list)

    print("┏" + str + "┓")
    print(" file name : ", filename)
    print(" compiler Ver : gcc", gcc_ver)
    print("┗" + str + "┛")

