import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import json

# Prepairing Data

train_data = np.load("clean_train.npy")
test_data = np.load("clean_test.npy")

train_df = pd.DataFrame(list(train_data))
test_df = pd.DataFrame(list(test_data))


use_list = ['country_rank','play_count', 'ranked_score','pp']
train_df = pd.DataFrame(train_df, columns=use_list)

use_list = ['username','country_rank','play_count', 'ranked_score']
test_df = pd.DataFrame(test_df, columns=use_list)


avg_rank = test_df['country_rank'].mean(axis=0)
train_df['country_rank'].replace(np.nan, avg_rank, inplace= True)
test_df['country_rank'].replace(np.nan, avg_rank, inplace= True)

for column in train_df.columns:
    if column != 'pp':
        mean = train_df[column].mean()
        std = train_df[column].std()
        train_df[column] = (train_df[column] - mean) / std
        test_df[column] = (test_df[column] - mean) / std
        
        
        
train_df = train_df[train_df['country_rank'] < 3]
train_df = train_df[train_df['play_count'] < 3]
train_df = train_df[train_df['ranked_score'] < 3]

labels = train_df.pp
l_mean = labels.mean()
l_std = labels.std()
labels = (labels - l_mean)/l_std

train_df.drop(['pp'], axis=1, inplace=True)
    
# processing data

X = np.array(train_df)
y = np.array(labels)
test_d = np.array(test_df)

def estimate_model(X, y,Lambda = 0):
    
    # X transpose
    Xtranspose = np.matrix.transpose(X)
    # Identity matrix (number of parameters is the dimension)
    Identity = np.identity(len(X[1,:]))
    
    # We don't add penalty to intercept
    Identity[0,0] = 0
    
    # Closed form solution is BetaHat = inv(X'X + Lambda*I)*X'y
    # Estimate model parameters (if Lambda = 0, we get standard square loss function result)
    BetaHat = np.dot(np.linalg.inv(np.add(np.dot(Xtranspose,X),Lambda*Identity)),np.dot(Xtranspose,y))

    return BetaHat

folds = [X[i::5] for i in range(5)]
folds_y = [y[i::5] for i in range(5)]


results = {}
for d in range(1, 10):
    loss = 0
    for i in range(5):
        train = np.array(folds[i])
        test = np.array(folds_y[i])
        
        x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.2)
        
        poly = PolynomialFeatures(d)
        X_train_poly = poly.fit_transform(x_train)
        X_test_poly = poly.fit_transform(x_test)
        
        betas = estimate_model(X_train_poly, y_train)
        predicted = [np.array(t).dot(betas) for t in X_test_poly]
        loss += sum([(predicted[index] - y_test[index])**2 for index in range(len(predicted))])/len(predicted)
    results[d] = loss
    print ("mean loss: {} , degree: {}".format(loss, d))
min_degree = min(results, key=results.get) 


results = {}
for l in np.arange(0, 1, 0.1):        
    mean_tmp = []
    loss = 0
    for i in range(5):
        train = np.array(folds[i])
        test = np.array(folds_y[i])

        x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.2)

        poly = PolynomialFeatures(min_degree)
        X_train_poly = poly.fit_transform(x_train)
        X_test_poly = poly.fit_transform(x_test)

        betas = estimate_model(X_train_poly, y_train, l)
        predicted = [np.array(t).dot(betas) for t in X_test_poly]
        loss += sum([(predicted[index] - y_test[index])**2 for index in range(len(predicted))])/len(predicted)
    results[l] = loss
    print ("lambda: {:.1f}, mean loss: {}".format(l, loss))
    
min_lambda = min(results, key=results.get) 

poly = PolynomialFeatures(min_degree)
X_poly = poly.fit_transform(X)
test_poly = poly.fit_transform(test_d[:,1:]) # without usernames
B = estimate_model(X_poly, y, min_lambda)
predict = test_poly.dot(B)
predict = predict * l_std + l_mean # recovering values
result = dict(zip(test_d[:,0], predict))

print("lambda: {}, degree: {}".format(min_lambda, min_degree))

with open("karen_ghandilyan.json", "w") as js_file:
        json.dump(result, js_file)
