import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train[train['observation'] == 'PM2.5']
# print(train)
test = test[test['observation'] == 'PM2.5']
# 删除无关特征
train = train.drop(['Date', 'stations', 'observation'], axis=1)
test_x = test.iloc[:, 2:10]
test_y = test.iloc[:, 10]

train_x = []
train_y = []

for i in range(16):
    x = train.iloc[:, i:i + 8]
    # notice if we don't set columns name, it will have different columns name in each iteration
    x.columns = np.array(range(8))
    y = train.iloc[:, i + 8]
    y.columns = np.array(range(1))
    train_x.append(x)
    train_y.append(y)


train_x = pd.concat(train_x) # (3600, 9) Dataframe类型
train_y = pd.concat(train_y)

train_y = np.array(train_y, float)
test_y = np.array(test_y, float)

ss = StandardScaler()
ss.fit(train_x)
train_x = ss.transform(train_x)
ss.fit(test_x)
test_x = ss.transform(test_x)


def r2_score(y_true, y_predict):
    # 计算y_true和y_predict之间的MSE
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    # 计算y_true和y_predict之间的R Square
    print(MSE)



b = -120
w = np.full((8,),-4)
lr = 1
iteration = 1000


lr_b = 0
lr_w = np.full((8,),0)

for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(train_x)):
        b_grad = b_grad - 2.0*(train_y[n] - b - np.matmul(train_x[n], w))*1.0/3600
        w_grad = w_grad - 2.0*(train_y[n] - b - np.matmul(train_x[n], w))*train_x[n]/3600

    lr_b = lr_b+b_grad**2  ##
    lr_w = lr_w+np.square(w_grad)  ##


    b = b-lr/np.sqrt(lr_b)*b_grad
    w = w-lr/np.sqrt(lr_w)*w_grad
    if i >= 990:
        print('this is b', b)
        print('this is w', w)

print(b)
print('below is w')
print(w)

test_y_hat = b +np.matmul(test_x, w)
r2_score(test_y, test_y_hat)

train_y_hat = b+np.matmul(train_x,w)
r2_score(train_y, train_y_hat)
