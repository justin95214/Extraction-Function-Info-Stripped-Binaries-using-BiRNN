from numpy import array
from keras.layers import Dense, LSTM, Embedding
import pandas as pd
from keras.layers import Dense, Bidirectional, SimpleRNN, Flatten,Dropout
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import  sequence
import matplotlib.pyplot as plt
check_data = pd.read_csv("C:/Users/82109/PycharmProjects/untitled1/venv/check4_Q.csv", sep=',')
import  keras
import os
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore')


os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# 파일읽기
print("----------------------DATA check-----------------------------------------")
# Datafame변환
df = pd.DataFrame(check_data)

# bin / disbin = 엑셀의 columns 중 bin, disbin
bin = df["bin"]
endstart = df["label"]
df['bin'].value_counts().plot(kind='bar')
plt.show()
df['label'].value_counts().plot(kind='bar')
plt.show()
print("---------------------pre - processing-------------------------------------")
# label
e = LabelEncoder()
e.fit(bin)
bin = e.transform(bin)
print(type(bin))
print(bin)



print("---------------------One - hot Encoder-------------------------------------")
# One - hot Encoder
bin_onelist = to_categorical(bin)

print(bin_onelist)
print(bin_onelist.shape)
print("---------------------Test/Train __split-------------------------------------")
train_test_split = round(bin_onelist.shape[0]*0.8)
print(train_test_split)

X_train = bin_onelist[:train_test_split, :]
Y_train = endstart[:train_test_split]
X_test = bin_onelist[train_test_split:, :]
Y_test = endstart[train_test_split :]
# 1. 데이터




print('x shape : ', X_train.shape)  # (89만,256)
print('y shape : ', Y_train.shape)  # (89만,)
#  x    y
# 1~256 1/0
max_len = 30
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

# print('-------x reshape-----------')
#X_train = np.reshape(X_train, ( X_train.shape[0],  max_len,1))
#X_test = np.reshape(X_test, ( X_test.shape[0],  max_len,1))


print('x shape : ', X_train.shape)  # (89만,256)
print('y shape : ', Y_train.shape)  # (89만,)
# embedding(등장횟수 :1~256등까지, 다음 layer가 16으로 가기때문에 ,2차원)
# return sequences에 따라 True >>3차원 데이터를 넣어야됨
# true로 변경시에 Embedding input은 2차원이므로, 오류남
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
# 2. 모델 구성


model = Sequential()
model.add(Embedding(256, 16))
model.add(LSTM(16, input_shape = (max_len, 1)))
#model.add(Bidirectional(SimpleRNN(16,input_shape=(None,1), return_sequences=False)))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='relu'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


history = LossHistory() # 손실 이력 객체 생성
history.init()
class_weight ={0.: 1. ,1.: 7.}

hist = model.fit(X_train, Y_train, epochs=1, batch_size=32 ,validation_data=(X_train, Y_train),callbacks=[history]
                 , class_weight = class_weight
                 )


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
pred =model.predict(X_train)

print(pred.tolist())
print(Y_train)

scores = model.evaluate(X_train, Y_train)

print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pred = np.round(np.array(pred).flatten().tolist())

# 테스트 predict 정답들
Y_train = np.array(Y_train).flatten().tolist()

print('accuracy_score', float(accuracy_score(Y_train, pred)))
print('recall_score', float(recall_score(Y_train, pred)))
print('precision_score', float(precision_score(Y_train, pred)))
print('f1_score', f1_score(Y_train, pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print('\nconfusion_matrix\n', confusion_matrix(Y_train, pred))

# ROC Curve
from sklearn.metrics import roc_auc_score
roc_auc2 = roc_auc_score(Y_train, pred)
print('roc_curve 면적', roc_auc2)