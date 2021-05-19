import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings(action='ignore')

# 여러개 쳐도 나오게
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# C:\Users\server\Downloads/extract
# 파일읽기
# train Data
sequence = 192
unit = 288
#Train_data = pd.read_csv("C:/Users/USER/Desktop/extract/gcc6_3_.csv", index_col=0)
#Train_data = pd.read_csv("D:/path/concat_answer/3/answer_3.csv", index_col=0)
for m in range(0,4):
    path = "F:/except_binary" + "/bin_core_compare/"
    file =  'o'+str(m) + '_bincore9.csv'
    print("input:", input)
    print("unit:", unit)
    print("path:", path)
    print("file:", path + file)
    save_path = 'D:/bin_core_89/' + str(input) + '_' + str(m) + 'pred_model_' + file[:-4] + '.h5'
    print("save_path:", save_path)

    # 파일읽기
    Train_data = pd.read_csv(path + file, index_col=0)



    #Train_data = pd.read_csv("C:/Users/USER/Desktop/model/o_23/o3_test/o3binutils.csv", index_col=0)
    # Test Data
    #print(dir_name)
    #dir = dir_name[0]
    #Test_data = pd.read_csv(dir, index_col=0)
    #Test_data = pd.read_csv("C:/Users/USER/Desktop/extract/gcc6_3_.csv", index_col=0)
    #Test_data = pd.read_csv("C:/Users/USER/Desktop/model/o_23/o3_test/o3binutils.csv", index_col=0)
    # 형태 출력
    print(Train_data.shape)

    # reset_index (hex processing 하면서 값이 빠졌으니까 + n_gram 에서 index를 다루기 때문에)
    Train_data.reset_index(inplace=True, drop=True)

    print('reset_index 완료')
    print('input data shape')
    Train_data.head()

    print("Data")
    Train_data.tail()

    from sklearn.preprocessing import LabelEncoder

    encoder_train = LabelEncoder()
    encoder_train.fit(Train_data['bin'])
    Train_data_enc = encoder_train.transform(Train_data['bin'])


    # 데이터체크 - hex(16진수)가 256 label을 가져야 dummies 변환 가능

    # 16진수 256개 종류가 있어서 pd.get_dummies 사용 가능.

    # (3) get_dummies 변환

    # 훈련데이터 (gcc 최적화버전 0, 1, 2, 3 one hot encoding)
    print(Train_data_enc)


    Train_data_enc_onehot = pd.get_dummies(Train_data_enc)
    print(Train_data_enc_onehot)


    # 자르지않고 그냥 넣기 위한데이터 n_gram과 원핫한 결과는 같음
    # 변수를 따로 지정
    # test_gcc6_2_32_onehot  = gcc6_2_32_onehot
    print('원핫인코딩완료')

    print(Train_data_enc_onehot.shape)

    # (4) 데이터 체크 - 1, 0 비율 ==> 1이 함수의 갯수를 뜻함
    # 정답 데이터 1, 0 비율 확인  ==> 1이 함수의 갯수를 뜻함
    print(Train_data['label'].value_counts())

    Train_data_enc_onehot.tail(10)


    print("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\n")
    idx3 = Train_data_enc_onehot[Train_data['label'] == 1].index  # 407, 474 ...
    ls3 = list(idx3)

    # 최종 뽑을 행에 대한 index
    ls_idx3 = []
    left_idx3, right_idx3 = 0, sequence

    # 6gram
    for k in range(left_idx3, right_idx3):
        ls_idx3.extend(list(idx3 + k))
        # print(ls_idx3)# index 형이라서 가능

    # print(ls_idx3)

    # ls_idx3 = list(set(ls_idx3))
    print(len(ls_idx3))

    ls_idx3.sort()  # 인덱스 정렬

    # 1차 index 해당범위 초과한 것들 없애기
    ls_idx3 = list(filter(lambda x: x < len(Train_data_enc_onehot), ls_idx3))
    print(len(ls_idx3))
    # print(ls_idx3)

    # 2차 남은 index들 중 right_idx3 나눈 나머지 없애기
    sub = len(ls_idx3) % (right_idx3)
    print(sub)

    ls_idx3 = ls_idx3[:len(ls_idx3) - sub]
    print(len(ls_idx3))

    print('gcc6_2_32', len(ls_idx3))
    # print(ls_idx3)

    # loc 로 수정필요
    Train_data_enc_onehot_gram = Train_data_enc_onehot.loc[ls_idx3, :].copy()
    # print(test_gcc6_2_32_onehot )

    a = len(Train_data_enc_onehot) % sequence
    print(a)
    how = len(Train_data_enc_onehot) - a
    print("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\n")


    # 자르지않은 데이터, 따로지정한 데이터를 x/y를 나누기 위한 작업
    Train_data_enc_onehot_x = Train_data_enc_onehot.iloc[:how, ].to_numpy()
    Train_data_enc_onehot_y = Train_data['label'][:how, ].to_numpy()


    print("data")
    # print(x_gcc6_2_32_3.shape, y_gcc6_2_32_3.shape)

    # 자르지않은데이터 reshape
    # test_gcc6_2_32_onehot_x.shape[1]
    Train_data_enc_onehot_x = Train_data_enc_onehot_x.reshape(-1, right_idx3, Train_data_enc_onehot_x.shape[1])
    Train_data_enc_onehot_y = Train_data_enc_onehot_y.reshape(-1, right_idx3, 1)

    print("data")
    print(Train_data_enc_onehot_x.shape, Train_data_enc_onehot_y.shape)


    np.save('F:/except_binary/compare/' + file[:-4] + 'compare_X_data', Train_data_enc_onehot_x)  # x_save.npy
    np.save('F:/except_binary/compare/'+file[:-4]+'compare_Y_data', Train_data_enc_onehot_y) # y_save.npy