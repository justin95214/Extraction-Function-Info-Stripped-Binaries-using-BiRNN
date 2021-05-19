#!/usr/bin/env python
# coding: utf-8

# ## RNN 모델 N-Byte 방식 (함수정보 포함 vs 미포함 => 1:1 비율)
#
# ## (1) 데이터로드

# In[3]:
import warnings
warnings.filterwarnings(action='ignore')

# 여러개 쳐도 나오게
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# (1) 데이터로드
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings(action='ignore')

# 여러개 쳐도 나오게
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
path ="F:/except_binary/ "
file = 'o3_gcc8.csv'
# 파일읽기
bin8_0 = pd.read_csv(path + file, index_col=0)
print(bin8_0.shape)

# reset_index (hex processing 하면서 값이 빠졌으니까 + n_gram 에서 index를 다루기 때문에)
bin8_0.reset_index(inplace=True, drop=True)

print('reset_index 완료')
print('input data shape')
bin8_0.head()

# In[4]:


# (2-1) 데이터체크 1 - hex(16진수)가 256 label을 가져야 dummies 변환 가능
# 16진수 256개 종류가 있어서 pd.get_dummies 사용 가능.
print(len(bin8_0['bin'].unique()))

# (2-2) 데이터 체크 2 - 1, 0 비율 ==> 1이 함수의 갯수를 뜻함
# 정답 데이터 1, 0 비율 확인  ==> 1이 함수의 갯수를 뜻함
print(bin8_0['label'].value_counts())

# ## (3) N Byte씩 자르기

# In[5]:


idx_bin = bin8_0[bin8_0['label'] == 1].index  # 407, 474 ...
ls_bin = list(idx_bin)

# 최종 뽑을 행에 대한 index
ls_idx_bin = []

# n byte 자르기 방식
left_idx, right_idx = 0, 32  # 3개씩

# n byte 자르기
for k in range(left_idx, right_idx):
    ls_idx_bin.extend(list(idx_bin + k))  # index 형이라서 가능

# ls_idx = list(set(ls_idx))
ls_idx_bin.sort()  # 인덱스 정렬

# 1차 index 해당범위 초과한 것들 없애기
ls_idx_bin = list(filter(lambda x: x < len(bin8_0), ls_idx_bin))
print(len(ls_idx_bin))

# 2차 남은 index들 중 right_idx 나눈 나머지 없애기
sub_bin = len(ls_idx_bin) % (right_idx)
print('나머지', sub_bin)

ls_idx_bin = ls_idx_bin[:len(ls_idx_bin) - sub_bin]
print('최종 길이', len(ls_idx_bin))

print('bin8_0', len(ls_idx_bin))

# loc 로 수정필요
bin8_0_Ngram = bin8_0.loc[ls_idx_bin, :].copy()

# ## (4) false data 만들기

# In[6]:


# false data 만들기 - False 데이터 랜덤 생성

# 목표치
goal_bin = len(bin8_0_Ngram) / right_idx
count_bin = 0

print(goal_bin)

# 최종 데이터 Frame
d_bin = pd.DataFrame(columns=bin8_0.columns)

binutils_df = []
# goal 에 도달할 때까지
while True:
    if (count_bin == goal_bin):
        break
    # 진행상황 살펴보기 위함

    # 랜덤 N 바이트씩 뽑음
    # random index
    random_idx_bin = np.random.randint(len(bin8_0) - right_idx)

    if count_bin % 1000 == 0:
        print(count_bin, end=' ')
        print(random_idx_bin)

    df_bin = bin8_0[random_idx_bin: random_idx_bin + right_idx]

    # 뽑은 index의 N 바이트 중에 1이 없는 경우만
    if 1 not in df_bin['label'] and count_bin < goal_bin:
        binutils_df.append(df_bin)
        count_bin += 1

print('완료')
print(len(binutils_df))

# In[7]:


# True data와 False Data 같은지 체크
print(len(binutils_df))
print(bin8_0['label'].value_counts()[1])

# ## (5) False Data + True Data 합치기

# In[8]:


f_data = pd.concat(binutils_df)
final = pd.concat([f_data, bin8_0_Ngram])
final.shape

# ## (6) one hot encoding

# In[9]:


# 훈련데이터 (gcc 최적화버전 0, 1, 2, 3 one hot encoding)
bc8_0_onehot_Ngram = pd.get_dummies(final['bin'])
bc8_0_onehot_Ngram = pd.concat([final['label'], bc8_0_onehot_Ngram], axis=1)

print('원핫인코딩완료')
print(bc8_0_onehot_Ngram.shape)

# In[10]:


# 훈련 데이터, 훈련 라벨
x_bc8_0 = bc8_0_onehot_Ngram.iloc[:, 1:].to_numpy()
y_bc8_0 = bc8_0_onehot_Ngram['label'].to_numpy()
print(x_bc8_0.shape, x_bc8_0.shape)

x_bc8_0 = x_bc8_0.reshape(-1, right_idx, x_bc8_0.shape[1])
y_bc8_0 = y_bc8_0.reshape(-1, right_idx, 1)

print(x_bc8_0.shape, y_bc8_0.shape)

# In[11]:


# numpy 행, 열 섞기
p = np.random.permutation(x_bc8_0.shape[0])

x_bc8_0 = x_bc8_0[p]
y_bc8_0 = y_bc8_0[p]

print(x_bc8_0.shape, y_bc8_0.shape)

# ## (7) 모델

# In[12]:


# (10) 양방향 LSTM 모델링 작업
from tensorflow.keras import layers, models
# from tf.keras.models import Model, Sequential
# from tf.keras.layers import SimpleRNN, Input, Dense, LSTM
# from tf.keras.layers import Bidirectional, TimeDistributed

# 학습
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=3)  # 조기종료 콜백함수 정의

xInput = layers.Input(batch_shape=(None, right_idx, 256))
xBiLstm = layers.Bidirectional(layers.LSTM(48, return_sequences=True, stateful=False), merge_mode='concat')(xInput)
xOutput = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(
    xBiLstm)  # 각 스텝에서 cost가 전송되고, 오류가 다음 step으로 전송됨.

# ## (8) 학습 - 10 KFold

# In[13]:


# 교차검증 kfold
from sklearn.model_selection import KFold

# Accuracy, Precision, Recall, F1-Score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Confusion Matrix, ROC Curve
from sklearn.metrics import confusion_matrix, roc_auc_score

# 최종 평가지표들 평균용
accuracy, recall, precision, f1score, cm = [], [], [], [], []

# 11. 교차검증 kfold - k.split - 10회 / K-Fold 객체 생성
# kf = KFold(n_splits=10, shuffle=False, random_state=None) # KFold non shuffle 버전
kf = KFold(n_splits=10, shuffle=True, random_state=None)  # KFold non shuffle 버전

for train, validation in kf.split(x_bc8_0, y_bc8_0):
    model1 = models.Model(xInput, xOutput)
    model1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model1.summary()
    print('======Training stage======')
    model1.fit(x_bc8_0[train],
               y_bc8_0[train],
               epochs=10,
               batch_size=128,
               callbacks=[early_stopping])
    # k_accuracy = '%.4f' %(model1.evaluate(data_10000x[validation], data_10000y[validation])[1])
    model1.save('32_48_model_cv'+file[:-4]+'.h5')
    # 12. 교차검증결과 predict - 검증셋들
    # predict 값
    k_pr = model1.predict(x_bc8_0[validation])

    # 테스트 predict 결과들 비교 (평가지표 보기위함)
    pred = np.round(np.array(k_pr).flatten().tolist())
    y_test = np.array(y_bc8_0[validation]).flatten().tolist()

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
    # print('\nconfusion_matrix\n', k_cm)

    accuracy.append(k_accuracy)
    recall.append(k_recall)
    precision.append(k_precision)
    f1score.append(k_f1_score)
    # cm.append(k_cm)
#    print('roc_curve 면적', roc_auc_score(y_test, pred))

# 최종 결과지표
print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
print('\nK-fold cross validation Recall: {}'.format(recall))
print('\nK-fold cross validation Precision: {}'.format(precision))
print('\nK-fold cross validation F1-Score: {}'.format(f1score))
# print('\nK-fold cross validation ConfusionMatrix: {}'.format(cm))


# ## (9) 평가지표

# In[14]:


print('10-Fold Cross_validation. Accuracy :', np.mean(accuracy))
print('10-Fold Cross_validation. Recall :', np.mean(recall))
print('10-Fold Cross_validation. Precision :', np.mean(precision))
print('10-Fold Cross_validation. F1-Score :', np.mean(f1score))

# In[15]:


model1.save('gcc6_bin_core_s32_h48_o0.h5')
print('save 완료')

