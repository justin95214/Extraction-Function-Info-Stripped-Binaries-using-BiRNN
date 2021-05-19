import pandas as pd
import numpy as np

f = open('g6.txt', 'r')
#함수명 + 바이너리 처음 ~끝
line = f.read()
#line = line.replace("<* *>:","W")
print(line)
print(len(line))
print("\n")
line = line.split()
#print(line)


f2 = open('g6w.txt', 'r')
#함수명 텍스트
line2 = f2.read()
#line = line.replace("<* *>:","W")
#print(line2)

print("\n")
line2 = line2.split()
#print(line2)
print(len(line2))

f3 = open('g6w_strip.txt', 'r')
#함수명 텍스트
line3 = f3.read()
#line = line.replace("<* *>:","W")
#print(line2)

print("\n")
line3 = line3.split()
#print(line2)
print(len(line3))


f4 = open('g6_strip.txt', 'r')
#함수명 텍스트
line4 = f4.read()
#line = line.replace("<* *>:","W")
#print(line2)

print("\n")
line4 = line4.split()
#print(line2)
print(len(line4))

array = [[0 for col in range(2)] for row in range((len(line)))]
for i in range(0, len(array)):
    array[i][0] = line[i]


array2 = [[0 for col in range(2)] for row in range((len(line2)))]
for i in range(0, len(array2)):
    array2[i][0] = line2[i]

print("\n")

array3 = [[0 for col in range(2)] for row in range((len(line3)))]
for i in range(0, len(array3)):
    array3[i][0] = line3[i]

print("\n")


array4 = [[0 for col in range(2)] for row in range((len(line4)))]
for i in range(0, len(array4)):
    array4[i][0] = line4[i]

print("\n")

print(len(line2))
print(len(line))

print("-------------------------------------------------")
k=0
for i in range(0,len(line2)) :

    for j in range(0, len(line)) :
        if (array[j][0] == array2[i][0]) :
            array[j][0] = np.nan
            array[j+1][1] = 1


        if (array[j][0] == 0) :
            array[j][0] = np.nan

for i in range(0,len(line3)) :
    for j in range(0, len(line)) :
        if (array[j][0] == array3[i][0]) :
            array[j][0] = np.nan
            array[j+1][1] = 0

array[0:897030]

I = int(len(line)/3)
print(array)
cols_name =['bin','label']
df = pd.DataFrame(array,columns=cols_name)

df4 = pd.DataFrame(array4,columns=cols_name)

df =df.dropna(axis=0)




print(df)
df.to_csv("exegcc6_3_32.csv")
df4.to_csv("exgcc6_strip.csv")

print(type(line))

print('##### read 함수 #####')
#print(line)
f.close()

df0 =pd.merge(df,df4)

