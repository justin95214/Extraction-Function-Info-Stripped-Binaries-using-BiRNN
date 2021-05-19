from numpy import array
from keras.layers import Dense, LSTM, Embedding
import pandas as pd
from keras.layers import Dense, Bidirectional, SimpleRNN, Flatten
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing import  sequence
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend.tensorflow_backend as K
import tensorflow as tf
check_data = pd.read_csv("C:/Users/82109/PycharmProjects/untitled1/venv/gcc5_0_32a.csv", sep=',')


# 파일읽기
print("----------------------DATA check-----------------------------------------")
# Datafame변환
df = pd.DataFrame(check_data)

# bin / disbin = 엑셀의 columns 중 bin, disbin
X = df["bin"]
Y= df["label"]

sns.countplot(df.label)
plt.xlabel("label")
plt.title("start or not ")
plt.show()

print(X.shape)
print(Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

max_func = X.shape[0]
max_len = 256


sequences_tr = sequence.pad_sequences(X_train, maxlen=max_len)
sequences_te = sequence.pad_sequences(X_test, maxlen=max_len)
sequences_tr = to_categorical(sequences_tr)
sequences_te = to_categorical(sequences_te)

print(sequences_tr)
#print(type(sequences_hot.shape))
#print(sequences_hot.shape)
#sequences_hot = list(sequences_hot)
#print(sequences_hot)

#sequences_hot = pd.DataFrame(sequences_hot)
#sequences_hot.to_csv("seq.csv")
#print(sequences_hot)

model = Sequential()
model.add(Embedding(256, len(sequences_tr[0])))  # 사용된 단어 수 & input 하나 당 size
model.add(LSTM(len(sequences_tr[0])))
model.add(Dense(3, activation='softmax'))  # 카테고리 수

model.summary()
class_weight ={0.: 1. ,1.: 7.}

hist = model.fit(sequences_tr, Y_train, batch_size=128, epochs=10, validation_split=0.2, class_weight= class_weight )


