import numpy as np
import pandas as pd
import json


def fit_linear_regression(X, Y):
    if X.shape[0] == Y.size and Y.ndim == 1 and X.ndim == 2:
        #print(X)
        xtx = np.dot(X.T, X)
        xtx = xtx.astype('float')
        #print(xtx.shape, xtx.dtype)
        q = np.dot(np.linalg.inv(xtx), X.T)
        #print(q)
        w = np.dot(q,Y)
        return w
    else:
        raise ValueError("Change the dimensions of input matrixes")
    pass

data = np.load("clean_train.npy")
frame = pd.DataFrame(list(data))
data = np.load("clean_test.npy")
test_frame = pd.DataFrame(list(data))
frame1 = pd.concat([frame, test_frame])
count = frame1.groupby('country').count()['A']
tuft = lambda x: count[x]
frame['country_count'] = frame['country'].apply(tuft)
frame['log_count/rank'] = np.log(frame['country_count']/frame['country_rank'])
frame['log_count/rank_sq'] = np.square(frame['log_count/rank'])
frame['log_count/rank_3'] = np.power(frame['log_count/rank'],3)
frame['log_log_count/rank'] = np.log(frame['log_count/rank']+0.3)
frame['log_repl'] = np.log(frame['replays_watched_by_others']+2)
frame['maximum_combo_log'] = np.log(frame['maximum_combo'])
frame_username = frame['username']
frame['A_log'] = np.log(frame['A']+2)
frame['S_log'] = np.log(frame['S']+2)
frame['SS_log'] = np.log(frame['SS']+2)
frame['follower_count_log'] = np.log(np.array(frame['follower_count'])+2)
frame['score/count_log'] = np.log(frame['total_score']/frame['play_count'])
frame['hits*combo/time'] = np.array(frame['total_hits']*frame['maximum_combo']/frame['play_time'])
frame['watch/count_log'] = np.log(frame['replays_watched_by_others']/frame['play_count']+0.0001)
frame['follower_count_log_log'] = np.log(frame['follower_count_log'])
Y = frame['pp']
frame = frame[['log_count/rank','follower_count_log','log_count/rank_3','follower_count_log_log','log_count/rank_sq','log_log_count/rank', 'level','maximum_combo_log','maximum_combo','play_count', 'log_repl', 'A_log', 'S_log', 'SS_log', 'is_active', 'is_supporter', 'total_score', 'score/count_log', 'total_hits', 'hits*combo/time', 'watch/count_log']]
#frame = pd.get_dummies(frame, columns=['country'])
#test = test[['country_rank', 'follower_count', 'hit_accuracy', 'age', 'level', 'maximum_combo','play_count']]
#frame1 = pd.concat([frame, test])
#frame1 = pd.get_dummies(frame1, columns=['country'])
#frame = frame1[:frame.shape[0]]
frame = frame.fillna(frame.mean())
X = np.array(frame)
X = np.hstack((np.ones((X.shape[0],1)),X))
X_test = X[4::5]
Y_test = Y[4::5]
X_cross = np.vstack(tuple([X[i::5] for i in range(4)]))
Y_cross = np.concatenate(tuple([Y[i::5] for i in range(4)]))
folds = [X_cross[i::4] for i in range(4)]
folds_y = [Y_cross[i::4] for i in range(4)]
loss = 0
for i in range(4):
    x = list(range(4))
    x.remove(i)
    train = np.array(np.vstack(tuple([folds[j] for j in x])))            
    test = np.array(folds[i])  
    train_y = np.array(np.hstack(tuple([folds_y[j] for j in x])))
    test_y = np.array(folds_y[i])
    #print(train_y.shape, train.shape)
    b = fit_linear_regression(train, train_y)
    #print(b)
    #print (a)
    a = np.sqrt(np.mean(np.square(np.dot(test,b)-test_y)))
    loss += a
loss = loss / 4
print(loss)

b = fit_linear_regression(X_cross, Y_cross)
print(np.std(np.dot(X_test, b) - Y_test))

test_frame = test_frame.fillna(test_frame.mean())
username = test_frame['username']
test_frame['country_count'] = test_frame['country'].apply(tuft)
test_frame['log_count/rank'] = np.log(test_frame['country_count']/test_frame['country_rank'])
test_frame['log_count/rank_sq'] = np.square(test_frame['log_count/rank'])
test_frame['log_count/rank_3'] = np.power(test_frame['log_count/rank'],3)
test_frame['log_log_count/rank'] = np.log(test_frame['log_count/rank']+0.3)
test_frame['log_repl'] = np.log(test_frame['replays_watched_by_others']+2)
test_frame['A_log'] = np.log(test_frame['A']+2)
test_frame['S_log'] = np.log(test_frame['S']+2)
test_frame['SS_log'] = np.log(test_frame['SS']+2)
test_frame['follower_count_log'] = np.log(np.array(test_frame['follower_count'])+2)
test_frame['maximum_combo_log'] = np.log(np.array(test_frame['maximum_combo'])+2)
test_frame['score/count_log'] = np.log(test_frame['total_score']/test_frame['play_count'])
test_frame['hits*combo/time'] = np.array(test_frame['total_hits']*test_frame['maximum_combo']/test_frame['play_time'])
test_frame['watch/count_log'] = np.log(test_frame['replays_watched_by_others']/test_frame['play_count']+0.0001)
test_frame['follower_count_log_log'] = np.log(test_frame['follower_count_log'])
test_frame = test_frame[['log_count/rank','follower_count_log','log_count/rank_3','follower_count_log_log','log_count/rank_sq','log_log_count/rank', 'level','maximum_combo_log','maximum_combo','play_count', 'log_repl', 'A_log', 'S_log', 'SS_log', 'is_active', 'is_supporter', 'total_score', 'score/count_log', 'total_hits', 'hits*combo/time', 'watch/count_log']]
#frame = pd.get_dummies(frame, columns=['country'])test_frame = test_frame.fillna(test_frame.mean())
test_frame = test_frame.fillna(test_frame.mean())
X_test = np.array(test_frame)
username = np.array(username)
X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test))
#username = np.array(frame_username)[4::5]
PP1_test = np.dot(X_test, b)
a = {}
for name, pp in zip(username, PP1_test):
    a[name] = pp
a
json.dump(a, open("edgar.json", 'w'))

