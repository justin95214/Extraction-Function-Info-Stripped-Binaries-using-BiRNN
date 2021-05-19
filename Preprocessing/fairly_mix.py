import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings(action='ignore')

# 여러개 쳐도 나오게

path = 'F:/except_binary/compare/'

bin9_0_x = np.load(path+'o0_bincore9compare_X_data.npy')
bin9_1_x = np.load(path+'o1_bincore9compare_X_data.npy')
bin9_2_x = np.load(path+'o2_bincore9compare_X_data.npy')
bin9_3_x = np.load(path+'o3_bincore9compare_X_data.npy')

bin9_0_y = np.load(path+'o0_bincore9compare_Y_data.npy')
bin9_1_y = np.load(path+'o1_bincore9compare_Y_data.npy')
bin9_2_y = np.load(path+'o2_bincore9compare_Y_data.npy')
bin9_3_y = np.load(path+'o3_bincore9compare_Y_data.npy')
print("0::",bin9_0_x.shape,bin9_0_y.shape)
print("1::",bin9_1_x.shape,bin9_1_y.shape)
print("2::",bin9_2_x.shape,bin9_2_y.shape)
print("3::",bin9_3_x.shape,bin9_3_y.shape)
bin9_x = np.concatenate([bin9_0_x,bin9_1_x,bin9_2_x,bin9_3_x])
bin9_y = np.concatenate([bin9_0_y,bin9_1_y,bin9_2_y,bin9_3_y])
list_length = []
list_length.append(bin9_0_y.shape[0])
list_length.append(bin9_1_y.shape[0])
list_length.append(bin9_2_y.shape[0])
list_length.append(bin9_3_y.shape[0])

list_length = sorted(list_length)
print(list_length)

print("before::",bin9_x.shape,bin9_y.shape)
bin9_x_remix = bin9_x
bin9_y_remix = bin9_y

right_idx = 192
print(list_length[0]*4)
for i in range(0,list_length[0]*4):
    if i%4 == 0 :
        bin9_x_remix[i] = bin9_0_x[int(i/4)].reshape(-1, right_idx, bin9_0_x.shape[2])
        bin9_y_remix[i] = bin9_0_y[int(i/4)].reshape(-1, right_idx, 1)
    elif i % 4 == 1:
        bin9_x_remix[i] = bin9_1_x[int(i/4)].reshape(-1, right_idx, bin9_1_x.shape[2])
        bin9_y_remix[i] = bin9_1_y[int(i/4)].reshape(-1, right_idx, 1)
    elif i % 4 == 2:
        bin9_x_remix[i] = bin9_2_x[int(i/4)].reshape(-1, right_idx, bin9_2_x.shape[2])
        bin9_y_remix[i] = bin9_2_y[int(i/4)].reshape(-1, right_idx, 1)
    elif i % 4 == 3:
        bin9_x_remix[i] = bin9_3_x[int(i/4)].reshape(-1, right_idx, bin9_3_x.shape[2])
        bin9_y_remix[i] = bin9_3_y[int(i/4)].reshape(-1, right_idx, 1)

bin9_x_remix = bin9_x_remix[:list_length[0]*4]
bin9_y_remix = bin9_y_remix[:list_length[0]*4]
print("after::",bin9_x_remix.shape,bin9_y_remix.shape)
s = 'F:/except_binary/gcc9o0_'
np.save(path+'comp_gcc8_x', bin9_x_remix)  # x_save.npy
np.save(path+'comp_gcc8_y', bin9_y_remix)  # y_save.npy
print("finish")