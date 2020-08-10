import sys
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.stats import rankdata

def average_score(row):
    score_list = ["fun-average","innovation-average","theme-average","graphics-average","audio-average","humor-average","mood-average"]
    total_score = 0
    for score in score_list:
        if row[score] != -1:
            total_score += row[score]
    return total_score/len(score_list)

def average_rank(row):
    rank_list = ["fun-rank","innovation-rank","theme-rank","graphics-rank","audio-rank","humor-rank","mood-rank"]
    total_rank = 0
    for rank in rank_list:
        if row[rank] != -1:
            total_rank += row[rank]
    return total_rank/len(rank_list)

def sum_rank(row):
    rank_list = ["fun-rank","innovation-rank","theme-rank","graphics-rank","audio-rank","humor-rank","mood-rank"]
    total_rank = 0
    for rank in rank_list:
        if row[rank] != -1:
            total_rank += row[rank]
    return total_rank

def sum_score(row):
    score_list = ["fun-average","innovation-average","theme-average","graphics-average","audio-average","humor-average","mood-average"]
    total_score = 0
    for score in score_list:
        if row[score] != -1:
            total_score += row[score]
    return total_score

def max_score(row, type_):
    score_list = ["fun-average","innovation-average","theme-average","graphics-average","audio-average","humor-average","mood-average"]
    min_score = float('inf')
    max_score = float('-inf')
    for score in score_list:
        if row[score] > max_score:
            max_score = row[score]
        if row[score] < min_score:
            min_score = row[score]
    if type_ == "min":
        return min_score
    return max_score

def max_rank(row, type_):
    rank_list = ["fun-rank","innovation-rank","theme-rank","graphics-rank","audio-rank","humor-rank","mood-rank"]
    min_rank = 9999999
    max_rank = -9999999
    for rank in rank_list:
        if row[rank] > max_rank:
            max_rank = row[rank]
        if row[rank] < min_rank:
            min_rank = row[rank]
    if type_ == "min":
        return min_rank
    return max_rank

def getTrainData():
    #uploaded = files.upload()
    train_loader = pd.read_csv("train.csv")
    #train_loader = train_loader[[b for b in list(train_loader.dtype.names) if b not in ["id","name", "slug","path","description","published","modified","ratings-given","link-tags","prev-games","num-authors","links"]]]
    train_loader = train_loader.drop(["id","name", "version","slug","path","description","published","modified","link-tags","prev-games","num-authors","links"], axis=1)
    train_loader.loc[(train_loader.category == 'jam'),'category']=1
    train_loader.loc[(train_loader.category == 'compo'),'category']=0

    #rank_list = ["fun-rank","innovation-rank","theme-rank","graphics-rank","audio-rank","humor-rank","mood-rank","fun-average","innovation-average","theme-average","graphics-average","audio-average","humor-average","mood-average"]
    
    #for rank in rank_list:
    #    train_loader[rank] = train_loader[rank] / train_loader.groupby('competition-num')[rank].transform('sum')

    for item in ["feedback-karma","ratings-given","ratings-received"]:
        train_loader[item] = train_loader[item]/ train_loader.groupby('competition-num')[item].transform('sum')
    #print(train_loader)
    #score_list = []
    #for score in score_list:
    #    train_loader[score] = train_loader[score] / train_loader.groupby('competition-num')[score].transform('sum')

    train_loader["avg_score"] = train_loader.apply(lambda row: average_score(row), axis=1)
    train_loader["avg_rank"] = train_loader.apply(lambda row: average_rank(row), axis=1)
    #train_loader["sum_score"] = train_loader.apply(lambda row: average_rank(row), axis=1)
    #train_loader["sum_rank"] = train_loader.apply(lambda row: average_rank(row), axis=1)
    train_loader["max_score"] = train_loader.apply(lambda row: max_score(row,"max"), axis=1)
    train_loader["min_score"] = train_loader.apply(lambda row: max_score(row,"min"), axis=1)
    train_loader["max_rank"] = train_loader.apply(lambda row: max_rank(row,"max"), axis=1)
    train_loader["min_rank"] = train_loader.apply(lambda row: max_rank(row,"min"), axis=1)

    #train_loader["feedback-karma"] = np.log(train_loader["feedback-karma"])
    #train_loader["ratings-given"] = np.log(train_loader["ratings-given"])
    #train_loader["ratings-received"] = np.log(train_loader["ratings-received"])
    print(train_loader)
    all_target = train_loader["label"]
    del train_loader["label"]
    all_data = train_loader.values
    return (all_data, all_target)

def getTestData():
    test = pd.read_csv("test.csv")
    test = test.drop(["name","version", "slug","path","description","published","modified","link-tags","prev-games","num-authors","links"], axis=1)
    test.loc[(test.category == 'jam'),'category']=1
    test.loc[(test.category == 'compo'),'category']=0
    for item in ["feedback-karma","ratings-given","ratings-received"]:
        test[item] = test[item]/ test.groupby('competition-num')[item].transform('sum')

    #rank_list = ["fun-rank","innovation-rank","theme-rank","graphics-rank","audio-rank","humor-rank","mood-rank"]
    #for rank in rank_list:
    #    test[rank] = test[rank] / test.groupby('competition-num')[rank].transform('sum')

    test["avg_score"] = test.apply(lambda row: average_score(row), axis=1)
    test["avg_rank"] = test.apply(lambda row: average_rank(row), axis=1)
    test["max_score"] = test.apply(lambda row: max_score(row,"max"), axis=1)
    test["min_score"] = test.apply(lambda row: max_score(row,"min"), axis=1)
    test["max_rank"] = test.apply(lambda row: max_rank(row,"max"), axis=1)
    test["min_rank"] = test.apply(lambda row: max_rank(row,"min"), axis=1)
    ids = test["id"]
    del test["id"]
    return (ids, test)

def rf_train(max_dep,random_st, ne,train_x,train_y):
    # 13,40 is the highest now
    clf = RandomForestClassifier(max_depth=max_dep,criterion= "entropy", random_state=random_st,n_estimators=ne)
    clf.fit(train_x, train_y)
    return clf

def svc_train(c,rs,gamma_, kernel_,train_x,train_y):
    # random_state = 30, C=108, gamma = scale, kernel = linear
    # random_state = 30, C=108, gamma = auto, kernel = linear
    clf = make_pipeline(StandardScaler(),SVC(random_state=rs, tol=1e-6,C=c,gamma = gamma_, kernel = kernel_))
    #clf.fit(train_x,train_y)
    return clf

def training_acc(clf,train_x, train_y):
    prediction = clf.predict(train_x)
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == train_y[i]:
            correct += 1
    print(correct)
    return correct/len(train_x)

def predict(clf, ids, test_x):
    pred = clf.predict(test_x)
    result = []
    for i in range(len(ids)):
       newrow = [ids[i],pred[i]]
       result.append(newrow)
    result = pd.DataFrame(result, columns = ["id","label"])
    result.to_csv("submit.csv",index = False)


if __name__ == "__main__":
    (train_x, train_y) = getTrainData()
    (ids, test_x) = getTestData()
    #train_x = preprocessing.normalize(train_x)
    print(train_x)
    #train_x = preprocessing.normalize(train_x)
    #test_x = preprocessing.normalize(test_x)
    max_dep = [10,11,12,13,14,15,16]
    random_st = [30,35,40,45,50]
    Cs = [8,18,108] # 18
    tol = [1e-5, 1e-6,1e-7]
    gamma = ["scale","auto"]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']

    #clf = GridSearchCV(estimator=svm.SVC(kernel='linear'), param_grid=dict(C=Cs), n_jobs=-1)
    rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=dict(max_depth = max_dep,random_state=random_st,n_estimators=[10,50,100] ), n_jobs=-1)
    #clf.fit(train_x, train_y)  
    rf.fit(train_x,train_y)
    '''
    cross_vali = 0
    best_fit = 0
    
    for dep in max_dep:
    
    for rs in random_st:
        for c in C:
        #for rs in random_st:
            for gam in gamma:
                for ker in kernel:
                    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
                    clf = svc_train(c,rs, gam, ker, train_x, train_y)
                    scores = cross_val_score(clf, train_x, train_y,cv=5)
                    print("scores for random_state = {}, C={}, gamma = {}, kernel = {}: {}".format(rs,c,gam,ker,scores))
                    print("mean score is: {}".format(scores.mean()))
                    if scores.mean() > cross_vali:
                        cross_vali = scores.mean()
                        best_fit = rs
    '''
    #print(clf.best_score_)                                  
    #print(clf.best_estimator_.C)
    print(rf.best_score_)                                  
    #print(rf.best_estimator_.max_depth)
    #print(rf.best_estimator_.random_state)
    #print(rf.best_estimator_.n_estimators)
    rf = rf_train(rf.best_estimator_.max_depth,rf.best_estimator_.random_state, rf.best_estimator_.n_estimators,train_x,train_y)
    #print(best_fit)
    #clf = rf_train(13,55, train_x, train_y)
    predict(rf, ids, test_x)
    
    #acc = []
    #for dep = max_dep:
    #    clf = rf_train(dep, train_x, train_y)
    #    acc.append(training_acc(clf, train_x, train_y))
    