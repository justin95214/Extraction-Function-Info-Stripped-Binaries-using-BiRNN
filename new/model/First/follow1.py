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

check_data = pd.read_csv("C:/Users/82109/PycharmProjects/untitled1/venv/check4_Q.csv", sep=',')


# 파일읽기
print("----------------------DATA check-----------------------------------------")
# Datafame변환
df = pd.DataFrame(check_data)

# bin / disbin = 엑셀의 columns 중 bin, disbin
bin = df["bin"]
endstart = df["label"]

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

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print('x shape : ', X_train.shape)  # (89만,256)
print('y shape : ', Y_train.shape)  # (89만,)

model = Sequential()

model.add(LSTM(256, return_sequences= False, input_shape=(218, 1)))
model.add(Dense(16, activation='softmax'))
model.add(Dense(1, activation='relu'))

#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.summary()

class_weight ={0.: 1. ,1.: 9.}

model.fit(X_train, Y_train,
    validation_data=(X_test, Y_test),
    batch_size=32,
    epochs=2,
          class_weight= class_weight)

pred = model.predict(X_train)

print(pred.tolist())
import matplotlib.pyplot as plt

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(Y_train, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()