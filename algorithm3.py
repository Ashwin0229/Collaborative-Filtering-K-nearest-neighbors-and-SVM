'''
Name   : Ashwin Sai C
Course : ML - CS6375-003
Title  : Mini Project 3
Term   : Fall 2023

'''

import numpy as np
import time
import sys
import math
import os
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from joblib import parallel_backend
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def read_file_data(filename):
    file_handle = open("netflix/"+filename+".txt","r")
    data        = file_handle.readlines()
    file_handle.close()

    print(len(data))
    
    user_vote = segregate_data(data)

    return user_vote

def segregate_data(data):
    user_id   = []
    movie_id  = []
    user_vote = {}

    for row in data:
        val = row.replace("\n","").split(",")        

        try:
            temp = user_vote[val[1]]
            temp[val[0]] = float(val[2])
            user_vote[val[1]] = temp
        except Exception as e:
            temp = {}
            temp[val[0]] = float(val[2])
            user_vote[val[1]] = temp 

    return user_vote

def segregate_movie(filename):
    user_id   = []
    movie_id  = []
    movie_user= {}

    file_handle = open("netflix/"+filename+".txt","r")
    data        = file_handle.readlines()
    file_handle.close()


    for row in data:
        val = row.replace("\n","").split(",")        

        try:
            temp = movie_user[val[0]]
            temp.append(val[1])
            movie_user[val[0]] = temp
        except Exception as e:
            temp = []
            temp.append(val[1])
            movie_user[val[0]] = temp 

    return movie_user

def read_test_data(filename):
    actual_vote = []
    user_id     = []
    movie_id    = []
    file_handle = open("netflix/"+filename+".txt","r")
    data        = file_handle.readlines()
    file_handle.close()

    for row in data:
        val = row.replace("\n","").split(",")
        movie_id.append(val[0])
        user_id.append(val[1])
        actual_vote.append(val[2])

    return [movie_id, user_id, actual_vote]

def correlation_calculation(a_ratings, i_ratings):
    common_items_mask = (a_ratings > 0) & (i_ratings > 0)
    if not common_items_mask.any():
        return 0

    common_a_ratings = a_ratings[common_items_mask]
    common_i_ratings = i_ratings[common_items_mask]
    mean_a           = np.mean(common_a_ratings)
    mean_i           = np.mean(common_i_ratings)
    numerator        = np.sum((common_a_ratings - mean_a) * (common_i_ratings - mean_i))
    denominator_a    = np.sqrt(np.sum((common_a_ratings - mean_a) ** 2))
    denominator_i    = np.sqrt(np.sum((common_i_ratings - mean_i) ** 2))
    correlation      = numerator / ((denominator_a * denominator_i)+sys.float_info.epsilon)

    return correlation

def predict_rating(active_user, item_to_predict, data, items_all,user_ratings,movie_user):
    user_items          = list(data.keys())
    items               = list(data[active_user].keys())    
    active_user_ratings = np.array(list(data[active_user].values()))
    active_user_mean    = np.mean(active_user_ratings)
    correlations        = np.array([correlation_calculation(active_user_ratings, np.array([data[user].get(item, 0) for item in items])) for user in movie_user[item_to_predict]])
    wa_i                = correlations[correlations > 0]
    non_zero_rows       = np.where(correlations > 0)[0]    
    user_ratings        = user_ratings[non_zero_rows,:]
    vij_vi              = user_ratings[:, list(items_all).index(item_to_predict)] - np.mean(user_ratings, axis=1)
    weighted_sums       = np.sum(wa_i * vij_vi , axis=0)
    weight_sums         = np.sum(np.abs(correlations[correlations > 0][:, np.newaxis]), axis=0)
    prediction          = active_user_mean + weighted_sums / (weight_sums*len(correlations[correlations>0]) + sys.float_info.epsilon)


    return prediction

def collaborative_filtering():

    data                              = read_file_data("TrainingRatings")
    [movie_id, user_id, actual_vote]  = read_test_data("TestingRatings")
    items_all = set()

    pred_list = []
    
    for user in data:
        movie_dict = data[user]
        for movie in movie_dict:
            items_all.add(movie)

    actual_vote  = [float(i) for i in actual_vote]
    user_items   = list(data.keys())
    user_ratings = np.array([[data[user].get(item, 0) for item in items_all] for user in user_items])
    movie_user   = segregate_movie("TrainingRatings")
    # start_time = time.time()
    # active_user_corr_dict = {}
    # for index,(movie,active_user) in enumerate(zip(movie_id,user_id)):
    #     print(index)
    #     items                 = list(data[active_user].keys())
    #     active_user_ratings   = np.array(list(data[active_user].values()))
    #     correlations          = np.array([correlation_calculation(active_user_ratings, np.array([data[user].get(item, 0) for item in items])) for user in movie_user[movie]])
    #     active_user_corr_dict[active_user] = correlations
    # end_time = time.time()

    # print("Time : ",end_time-start_time," s")
    # exit(0)

    with parallel_backend('threading', n_jobs=2):
        for user, movie in zip(user_id, movie_id):
            pred = predicted_rating = predict_rating(user, movie, data, items_all,user_ratings,movie_user)   
            if not math.isnan(pred[0]): 
                pred_list.append(min(pred[0],5.0))
            else:
                pred_list.append(0)
            print(movie, user, pred)
            end_time = time.time()


    print("Error Rate : ")
    MAE_value  = mean_absolute_error(actual_vote, pred_list)
    RMSE_value = np.sqrt(mean_squared_error(actual_vote, pred_list))

    print("MAE  : ", MAE_value)
    print("RMSE : ", RMSE_value)

def load_dataset():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,parser="auto")
    X = X / 255.
    # rescale the data, use the traditional train/test split
    # (60K: Train) and (10K: Test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return X_train, y_train, X_test, y_test

def load_parameters():

    paramaters_dict = {
                            "svm":
                                    {
                                        "1":
                                            {
                                               "C":10.0, "kernel":"rbf", "degree":3, "gamma":"scale", "coef0":0.0, "max_iter":100, "decision_function_shape":"ovr", "random_state":42 
                                            },
                                        "2":
                                            {
                                               "C":1.0, "kernel":"rbf", "degree":3, "gamma":"scale","coef0":0.0, "max_iter":1000, "decision_function_shape":"ovr", "random_state":42 
                                            },
                                        "3":
                                            {
                                               "C":1.0, "kernel":"poly", "degree":3, "gamma":"scale", "coef0":42.5, "max_iter":1000, "decision_function_shape":"ovr", "random_state":42 
                                            },
                                        "4":
                                            {
                                               "C":1.0, "kernel":"poly", "degree":4, "gamma":"auto", "coef0":1.2, "max_iter":100, "decision_function_shape":"ovr", "random_state":42 
                                            },
                                        "5":
                                            {
                                               "C":1.0, "kernel":"sigmoid", "degree":2, "gamma":"scale", "coef0":9.5, "max_iter":300, "decision_function_shape":"ovo", "random_state":42 
                                            },
                                        "6":
                                            {
                                               "C":0.25, "kernel":"linear", "degree":4, "gamma":"scale", "coef0":0.0, "max_iter":1000, "decision_function_shape":"ovr", "random_state":42 
                                            },
                                        "7":
                                            {
                                               "C":0.1, "kernel":"linear", "degree":4, "gamma":"auto", "coef0":0.0, "max_iter":500, "decision_function_shape":"ovr", "random_state":42 
                                            },
                                        "8":
                                            {
                                               "C":0.001, "kernel":"sigmoid", "degree":4, "gamma":0.001, "coef0":15.0, "max_iter":200, "decision_function_shape":"ovr", "random_state":42
                                            },
                                        "9":
                                            {
                                               "C":1000, "kernel":"rbf", "degree":4, "gamma":0.001, "coef0":0.0, "max_iter":100, "decision_function_shape":"ovr", "random_state":42 
                                            },
                                        "10":
                                            {
                                               "C":1000, "kernel":"poly", "degree":2, "gamma":0.001, "coef0":0.9, "max_iter":1000, "decision_function_shape":"ovo", "random_state":42 
                                            }
                                    },

                            "kn":
                                    {
                                        "1":
                                            {
                                                "n_neighbors":3, "algorithm":'auto', "weights":'uniform', "p":1
                                            },
                                        "2":
                                            {
                                                "n_neighbors":3, "algorithm":'auto', "weights":'distance', "p":2
                                            },
                                        "3":
                                            {
                                                "n_neighbors":3, "algorithm":'ball_tree', "weights":'uniform', "p":2
                                            },
                                        "4":
                                            {
                                                "n_neighbors":1, "algorithm":'ball_tree', "weights":'distance', "p":2
                                            },
                                        "5":
                                            {
                                                "n_neighbors":5, "algorithm":'kd_tree', "weights":'distance', "p":2
                                            },
                                        "6":
                                            {
                                                "n_neighbors":5, "algorithm":'brute', "weights":'uniform', "p":1
                                            },
                                        "7":
                                            {
                                                "n_neighbors":7, "algorithm":'ball_tree', "weights":'uniform', "p":1
                                            },
                                        "8":
                                            {
                                                "n_neighbors":7, "algorithm":'ball_tree', "weights":'uniform', "p":3
                                            },
                                        "9":
                                            {
                                                "n_neighbors":1, "algorithm":'auto', "weights":'uniform', "p":3
                                            },
                                        "10":
                                            {
                                                "n_neighbors":7, "algorithm":'brute', "weights":'uniform', "p":5
                                            },
                                        "11":
                                            {
                                                "n_neighbors":11, "algorithm":'auto', "weights":'uniform', "p":4
                                            }
                                    }     
                      }

    return paramaters_dict

def svm_classifier():

    X_train, y_train, X_test, y_test = load_dataset()
    y_test                           = y_test.values.tolist()
    y_train                          = [float(i) for i in y_train]
    result_list                      = []
    parameters                       = load_parameters()["svm"]
    header                           = ["C","kernel","degree","gamma","coef0","max_iter","decision_function_shape","random_state"]

    # parameters = {"8":
    #                 {
    #                    "C":0.001, "kernel":"sigmoid", "degree":4, "gamma":0.001, "coef0":15.0, "max_iter":200, "decision_function_shape":"ovr", "random_state":42 
    #                 }
    #              }
    # MAE           = make_scorer(mean_absolute_error, greater_is_better=False)
    # scoring_param = MAE

    # # #Uncomment for Tune sets
    # param_grid = {
    #               'C'                         : [0.1, 1, 10, 100, 1000],
    #               'kernel'                    : ['linear','poly','rbf','sigmoid'],
    #               'degree'                    : [2,3,4,5],
    #               'gamma'                     : [0.1,0.01, 1.0, 0.001],
    #               'coef0'                     : [0.0, 0.4, 0.9, 1.2, 9.5, ],
    #               'max_iter'                  : [10,100,1000]
    #              }
    # clf     = SVC()
    # _search = GridSearchCV(clf, param_grid, cv=2, scoring=scoring_param, n_jobs=os.cpu_count()-1)
    # _search.fit(X_train, y_train)
    # print(f"Best hyperparameters found by GridSearchCV: {_search.best_params_}")
    # parameters = _search.best_params_ 
    # results    = _search.cv_results_ 

    # print(parameters)
    # print(results) 

    for index in parameters.keys():
        print("Set ",index," :")        
        svm_classifier = SVC(C                       = parameters[index]['C'], 
                             kernel                  = parameters[index]['kernel'], 
                             degree                  = parameters[index]['degree'], 
                             gamma                   = parameters[index]['gamma'],
                             coef0                   = parameters[index]['coef0'],                         
                             max_iter                = parameters[index]['max_iter'],
                             decision_function_shape = parameters[index]['decision_function_shape'],
                             random_state            = parameters[index]['random_state']
                            )

        clf = make_pipeline(StandardScaler(), svm_classifier)
        clf.fit(X_train, y_train)

        pred_list = clf.predict(X_test)
        pred_list = [int(i) for i in pred_list]
        y_test    = [int(i) for i in y_test]

        MAE                 = mean_absolute_error(y_test, pred_list)
        RSME                = np.sqrt(mean_squared_error(y_test, pred_list))
        f1_score_val        = f1_score(y_test, pred_list,average="macro")
        precision_score_val = precision_score(y_test, pred_list,average="macro")
        recall_score_val    = recall_score(y_test, pred_list,average="macro")
        accuracy_score_val  = accuracy_score(y_test, pred_list)

        #print("Mean Absolute Error      : ",round(MAE,2))
        #print("Root Mean Squared Error  : ",round(RSME,2))
        print("F1-Score                  : ",round(f1_score_val,2))
        print("Precision                 : ",round(precision_score_val,2))
        print("Recall                    : ",round(recall_score_val,2))
        print("Accuracy                  : ",round((accuracy_score_val)*100,2))



        temp = []
        for val in parameters[index].values():
            temp.append(val)
        temp.append(round(f1_score_val,2))
        temp.append(round(precision_score_val,2))
        temp.append(round(recall_score_val,2))
        temp.append(round((accuracy_score_val)*100,2))

        result_list.append(temp)

    table_format_error_rate(result_list,"Support Vector Machines", header)

def kn_classifier():
    print("in KNC")
    X_train, y_train, X_test, y_test = load_dataset()
    y_test                           = y_test.values.tolist()
    y_train                          = [float(i) for i in y_train]
    parameters                       = load_parameters()["kn"]
    result_list                      = []
    header                           = ["n_neighbors","algorithm","weights","p"]

    # MAE  = make_scorer(mean_absolute_error, greater_is_better=False)
    # scoring_param = MAE

    # # #Uncomment for Tune sets
    # param_grid = {
    #               'n_neighbors'             : [1,3,5,7],
    #               'algorithm'               : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #                'weights'                : ['uniform', 'distance'],
    #               'p'                       : [1,2],
    #               'n_jobs'                  : [-1]
    #              }
    # clf = KNeighborsClassifier()
    # print("here")
    # _search = RandomizedSearchCV(clf, param_grid, n_iter=10, cv=2, scoring=scoring_param, n_jobs=os.cpu_count()-1)
    # _search.fit(X_train, y_train)
    # print(f"Best hyperparameters found by KNSearchCV: {_search.best_params_}")
    # parameters = _search.best_params_ 
    # results    = _search.cv_results_ 

    # print(parameters)
    # print(results)    

    for index in parameters.keys():
        print("Set ",index," :")
        neigh = KNeighborsClassifier(n_neighbors    = parameters[index]["n_neighbors"], 
                                     algorithm      = parameters[index]["algorithm"], 
                                     weights        = parameters[index]["weights"], 
                                     p              = parameters[index]["p"], 
                                     # n_jobs         = os.cpu_count()-1
                                     n_jobs         = 3
                                     )
        neigh.fit(X_train, y_train)

        pred_list = neigh.predict(X_test)
        pred_list = [int(i) for i in pred_list]
        y_test    = [int(i) for i in y_test]

        MAE  = mean_absolute_error(y_test, pred_list)
        RSME = np.sqrt(mean_squared_error(y_test, pred_list))
        f1_score_val        = f1_score(y_test, pred_list,average="macro")
        precision_score_val = precision_score(y_test, pred_list,average="macro")
        recall_score_val    = recall_score(y_test, pred_list,average="macro")
        accuracy_score_val  = accuracy_score(y_test, pred_list)

        #print("Mean Absolute Error      : ",round(MAE,2))
        #print("Root Mean Squared Error  : ",round(RSME,2))
        print("F1-Score                  : ",round(f1_score_val,2))
        print("Precision                 : ",round(precision_score_val,2))
        print("Recall                    : ",round(recall_score_val,2))
        print("Accuracy                  : ",round((accuracy_score_val)*100,2))

        # print("Mean Absolute Error : ",round(MAE,2))
        # print("Mean Squared Error  : ",round(RSME,2))

        temp = []
        for val in parameters[index].values():
            temp.append(val)
        temp.append(round(f1_score_val,2))
        temp.append(round(precision_score_val,2))
        temp.append(round(recall_score_val,2))
        temp.append(round((accuracy_score_val)*100,2))

        result_list.append(temp)

    table_format_error_rate(result_list,"K-Nearest Neighbours",header)

def table_format_error_rate(output_list,classifier_name, header):
    print("\n\n")
    header.append("F1-score")
    header.append("Precision")
    header.append("Recall")
    header.append("Accuracy")

    t = PrettyTable(header)
    
    for row in output_list:
        t.add_row(row)

    print(t)

if __name__ == "__main__":

    print("hi")

    #collaborative_filtering()
    # svm_classifier()
    # kn_classifier()