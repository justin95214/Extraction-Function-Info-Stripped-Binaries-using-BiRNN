import pandas as pd
import numpy as np

f = open('binutils-2.34-gcc6-O0-x86_strip.txt', 'r')
# 함수명 + 바이너리 처음 ~끝
line = f.read()
# line = line.replace("<* *>:","W")
print(line)

print("\n")
line = line.split()
print(line)

f2 = open('binutils-2.34-gcc6-O0-x86_stripW.txt', 'r')
# 함수명 텍스트
line2 = f2.read()
# line = line.replace("<* *>:","W")
print(line2)

print("\n")
line2 = line2.split()
print(line2)



array = [[0 for col in range(2)] for row in range((len(line)))]
for i in range(0, len(array)):
    array[i][0] = line[i]


array2 = [[0 for col in range(2)] for row in range((len(line2)))]
for i in range(0, len(array2)):
    array2[i][0] = line2[i]

print("\n")

print(len(line2))
print(len(line))

print("-------------------------------------------------")
k = 0
for i in range(0, len(line2)):
    k = k + 1
    print(k)
    for j in range(0, len(line)):
        if (array[j][0] == array2[i][0]):
            array[j][0] = np.nan
            array[j + 1][1] = 1


        if (array[j][0] == 0):
            array[j][0] = np.nan

print(array)
cols_name = ['strip', 'bin']

df = pd.DataFrame(array, columns=cols_name)

df = df.dropna(axis=0)

print(df)
df.to_csv("exe_strip_gcc6_0_32.csv")

print(type(line))

print('##### read 함수 #####')
# print(line)
f.close()