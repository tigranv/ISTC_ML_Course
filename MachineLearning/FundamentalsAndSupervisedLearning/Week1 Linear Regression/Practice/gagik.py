import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

'''-----------------importing the data-----------------'''
data = np.load("clean_train.npy")

'''-----------------creating a DataFrame-----------------'''
# np.random.seed(10)
df=pd.DataFrame(list(data))
df=df.sample(frac=1)
df.set_index("username", inplace=True)
df["country_rank"].fillna(80, inplace=True)
# print('\n'.join(list(df)))

''' -----------------polynom creator-----------------'''
def pol_creator(df, col_names, deg):
    new_col_names = []
    for k in col_names:
        for i in range(2,deg+1):
            df[k+"_{}".format(i)] = df[k]**i
            new_col_names.append(k+"_{}".format(i))
    return new_col_names

''' -----------------logarithm creator-----------------'''
def log_creator(df, col_names):
    new_col_names = []
    for k in col_names:
        df[k+"_log"] = np.log(df[k]+1)
        new_col_names.append(k+"_log")
    return new_col_names

'''-----------------creating X and y numpy arrays-----------------'''
y=df.pp.values
df['country_freq'] = df.groupby('country')['country'].transform('count')
df["relative_position"]=df["country_rank"]*df['country_freq'].max()/df['country_freq']
pol_col = ["maximum_combo","total_score", "play_time", "hit_accuracy"]
log_col = ["follower_count", "relative_position"]
other_col = ["level"]
degree=2
new_pol_col = pol_creator(df, pol_col, degree)
new_log_col = log_creator(df, log_col)
X_col = pol_col+new_pol_col+log_col+new_log_col+other_col
X = np.array(df[X_col])
X = np.column_stack(([1 for _ in np.arange(len(X))], X))
print("columns are",X_col)

'''-----------------plotting variables for fun :)-----------------'''
# plt.xlim(0,2000)
# plt.scatter(np.array(df["relative_position"]),df.pp.values, alpha = 0.3)
# plt.show()

'''-----------------if needed standardizing X and y-----------------'''
# X = (X - np.mean(X)) / np.std(X)
# y = (y - np.mean(y)) / np.std(y)

'''-----------------functions for regression-----------------'''

def fit_linear_regression(X, Y):
    betta = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))
    return betta

def mean_square_loss(X, Y, b):
    mLoss=np.sqrt(np.sum((Y-np.dot(X,b))**2)/len(X))
    return mLoss

def fit_linear_regression_lmbd(X, Y,lmbd):
    betta = np.dot(np.linalg.inv(np.dot(X.T,X)+lmbd*np.identity(X.T.shape[0])),np.dot(X.T,Y))
    return betta


'''-----------------finding the best lyambda with cross-validation-----------------'''
k=5
folds = [X[i::k] for i in range(k)]
folds_y = [y[i::k] for i in range(k)]
lmbda=[]
mloss=[]

for lmbd in np.arange(0,50,0.05):
    loss = 0
    lmbda = np.append(lmbda, lmbd)
    for i in range(5):
        x = list(range(5))
        x.remove(i)
        traincv = np.vstack((folds[x[0]],folds[x[1]],folds[x[2]],folds[x[3]]))
        testcv = np.array(folds[i])
        train_ycv = np.hstack((folds_y[x[0]],folds_y[x[1]],folds_y[x[2]],folds_y[x[3]]))
        test_ycv = np.array(folds_y[i])
        paramcv = fit_linear_regression_lmbd(traincv, train_ycv,lmbd)
        predicted = np.dot(testcv, paramcv)
        loss += np.sqrt(sum((predicted - test_ycv)**2)/len(predicted))
    mloss = np.append(mloss, loss)
print('Mean loss is min [={}] when lyambda is {}'.format(np.mean(mloss/k),lmbda[np.argmin(mloss)]))
# plt.plot(lmbda,np.log(mloss))
# plt.axvline(x=lmbda[np.argmin(mloss)], color="magenta", alpha =0.3)
# plt.show

'''----------------- calculating parameters -----------------'''

coef = fit_linear_regression_lmbd(X,y, lmbda[np.argmin(mloss)])


'''-----------------if needed splitting the dataset into train and test sets and calculating losses -----------------'''

train = X[:int(len(X)*0.8)]
train_y=y[:int(len(y)*0.8)]
test = X[int(len(X)*0.8)+1:]
test_y = y[int(len(y)*0.8)+1:]

coef_train = fit_linear_regression_lmbd(train,train_y, lmbda[np.argmin(mloss)])
print("Parameters are:", list(zip(X_col, coef.flatten())))

mLoss_train=mean_square_loss(train,train_y,coef_train)
mLoss_test=mean_square_loss(test,test_y,coef_train)
print("Loss on train is:", mLoss_train)
print("Loss on test is:", mLoss_test)
print("Mean loss on train and test is:", (mLoss_train+mLoss_test)/2)

'''creating the test dictionary with predicted pp's'''
testdf = np.load("clean_test.npy")
dftest = pd.DataFrame(list(testdf))
dftest['country_freq'] = dftest.groupby('country')['country'].transform('count')
dftest["relative_position"]=dftest["country_rank"]*dftest['country_freq'].max()/dftest['country_freq']
username_test = dftest.username.values
pol_creator(dftest, pol_col, degree)
log_creator(dftest, log_col)
Xdf = np.array(dftest[X_col])
Xdf = np.column_stack(([1 for _ in np.arange(len(Xdf))], Xdf))
ydf=np.dot(Xdf,coef)
ydf1 = [np.mean(y) if np.isnan(x) else x for x in ydf]

'''saving to json'''
Gagik_Chakhoyan = dict(zip(username_test, ydf1))
# print(Gagik_Chakhoyan)
json.dump(Gagik_Chakhoyan, open("Gagik_Chakhoyan.json", "w"))

'''saving to .npy'''
np.save("Gagik_Chakhoyan", Gagik_Chakhoyan)

