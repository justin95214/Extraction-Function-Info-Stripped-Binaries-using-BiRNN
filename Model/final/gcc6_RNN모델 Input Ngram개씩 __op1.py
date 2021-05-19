#!/usr/bin/env python
# coding: utf-8

# In[1]:


# (1) 데이터로드
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

# 여러개 쳐도 나오게
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# 파일읽기
gcc6_1_32 = pd.read_csv("../data/binutils_gcc3~9_op0~4_csv/"+'gcc6'+"_1_32.csv", index_col=0)

# 형태 출력
print(gcc6_1_32.shape)

# reset_index (hex processing 하면서 값이 빠졌으니까 + n_gram 에서 index를 다루기 때문에)
gcc6_1_32.reset_index(inplace=True, drop=True)

print('reset_index 완료')
print('input data shape')
gcc6_1_32.head()


# In[14]:


gcc6_1_32.tail()


# In[19]:


# (2-1) 데이터체크 - hex(16진수)가 256 label을 가져야 dummies 변환 가능 

# 16진수 256개 종류가 있어서 pd.get_dummies 사용 가능.
print(len(gcc6_1_32['hex'].unique()))

# (3) get_dummies 변환 

# 훈련데이터 (gcc 최적화버전 0, 1, 2, 3 one hot encoding)
gcc6_1_32_onehot = pd.get_dummies(gcc6_1_32)


print('원핫인코딩완료')

print(gcc6_1_32_onehot.shape)

# (4) 데이터 체크 - 1, 0 비율 ==> 1이 함수의 갯수를 뜻함
# 정답 데이터 1, 0 비율 확인  ==> 1이 함수의 갯수를 뜻함
print(gcc6_1_32_onehot['bin'].value_counts())


# In[50]:


# (5-1) gcc3 6gram
######################## 
idx3 = gcc6_1_32_onehot[gcc6_1_32_onehot['bin']==1].index  # 407, 474 ...
ls3 = list(idx3)

# 최종 뽑을 행에 대한 index
ls_idx3 = [] 
left_idx3, right_idx3 = 0, 7 # 3개씩

# 6gram
for k in range(left_idx3, right_idx3):
    ls_idx3.extend(list(idx3 + k)) # index 형이라서 가능
print(len(ls_idx3))

#ls_idx3 = list(set(ls_idx3)) 
print(len(ls_idx3))

ls_idx3.sort() # 인덱스 정렬

# 1차 index 해당범위 초과한 것들 없애기
ls_idx3 = list(filter(lambda x: x<len(gcc6_1_32_onehot), ls_idx3))
print(len(ls_idx3))

# 2차 남은 index들 중 right_idx3 나눈 나머지 없애기
sub = len(ls_idx3)%(right_idx3)
print(sub)

ls_idx3 = ls_idx3[:len(ls_idx3)-sub]
print(len(ls_idx3))

print('gcc6_1_32', len(ls_idx3))

# loc 로 수정필요
gcc6_1_32_onehot_3gram = gcc6_1_32_onehot.loc[ls_idx3,:].copy()


# In[51]:


# 훈련 데이터, 훈련 라벨
x_gcc6_1_32_3 = gcc6_1_32_onehot_3gram.iloc[:,1:].to_numpy()
y_gcc6_1_32_3 = gcc6_1_32_onehot_3gram['bin'].to_numpy()
print(x_gcc6_1_32_3.shape, y_gcc6_1_32_3.shape)


# In[46]:


x_gcc6_1_32_3 = x_gcc6_1_32_3.reshape(-1, right_idx3, x_gcc6_1_32_3.shape[1])
y_gcc6_1_32_3 = y_gcc6_1_32_3.reshape(-1, right_idx3, 1)

print(x_gcc6_1_32_3.shape, y_gcc6_1_32_3.shape)


# ## 모델

# In[47]:


# (10) 양방향 LSTM 모델링 작업
from keras.models import Model, Sequential
from keras.layers import SimpleRNN, Input, Dense, LSTM
from keras.layers import Bidirectional, TimeDistributed

# 학습
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 3) # 조기종료 콜백함수 정의

xInput = Input(batch_shape=(None,right_idx3, 256)) 
xBiLstm = Bidirectional(LSTM(16, return_sequences=True), merge_mode = 'concat')(xInput)
xOutput = TimeDistributed(Dense(1, activation ='sigmoid'))(xBiLstm) # 각 스텝에서 cost가 전송되고, 오류가 다음 step으로 전송됨.

model1 = Model(xInput, xOutput)
model1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model1.summary()


# In[48]:


########## 3gram

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
kf = KFold(n_splits=10, shuffle=False, random_state=None) # KFold non shuffle 버전

for train, validation in kf.split(x_gcc6_1_32_3, y_gcc6_1_32_3):
    print('======Training stage======')
    model1.fit(x_gcc6_1_32_3[train],
               y_gcc6_1_32_3[train],
               epochs = 30,
               batch_size = 32)
    #k_accuracy = '%.4f' %(model1.evaluate(data_10000x[validation], data_10000y[validation])[1])

# 12. 교차검증결과 predict - 검증셋들
    # predict 값
    k_pr = model1.predict(x_gcc6_1_32_3[validation])
    
    # 테스트 predict 결과들 비교 (평가지표 보기위함)
    pred = np.round(np.array(k_pr).flatten().tolist())
    y_test = np.array(y_gcc6_1_32_3[validation]).flatten().tolist()
    
# 13. 평가지표들 출력
    ## 평가지표들
    k_accuracy = float(accuracy_score(y_test, pred))
    k_recall =  float(recall_score(y_test, pred))
    k_precision = float(precision_score(y_test, pred))
    k_f1_score = float(f1_score(y_test, pred))
    #k_cm = float(confusion_matrix(y_test, pred))
    
    print('accuracy_score', k_accuracy)
    print('recall_score', k_recall)
    print('precision_score', k_precision)
    print('f1_score', k_f1_score)
    #print('\nconfusion_matrix\n', k_cm)

    accuracy.append(k_accuracy)
    recall.append(k_recall)
    precision.append(k_precision)
    f1score.append(k_f1_score)
    #cm.append(k_cm)
#    print('roc_curve 면적', roc_auc_score(y_test, pred))

# 14. 최종 결과지표
print('\nK-fold cross validation Accuracy: {}'.format(accuracy))
print('\nK-fold cross validation Recall: {}'.format(recall))
print('\nK-fold cross validation Precision: {}'.format(precision))
print('\nK-fold cross validation F1-Score: {}'.format(f1score))
#print('\nK-fold cross validation ConfusionMatrix: {}'.format(cm))


# In[49]:


# 4gram 평가지표
print('10-Fold Cross_validation. Accuracy :', np.mean(accuracy))
print('10-Fold Cross_validation. Recall :', np.mean(recall))
print('10-Fold Cross_validation. Precision :', np.mean(precision))
print('10-Fold Cross_validation. F1-Score :', np.mean(f1score))

