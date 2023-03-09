import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier


def prepare_data(df, test_size):
    #X = df.drop([0,'dist'], axis=1)
    X = df.iloc[:, :-1]
    y = df['dist']
    
    # Scaling data
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = test_size, random_state=10)
    return X, y, X_train, X_test, y_train, y_test


def svm_model(df, test_size, cv, plot=0):
    X, y, X_train, X_test, y_train, y_test = prepare_data(df, test_size)
    param_grid = [
        {'C':np.logspace(0,1,10),
         'gamma':np.logspace(0,1,10), 
         'kernel':['rbf']},
    ]

    optimal_params = GridSearchCV(SVC(), param_grid,cv=cv, verbose=0)
    optimal_params.fit(X_train, y_train)

    clf_svm = SVC(random_state=100, C=optimal_params.best_params_['C'], gamma=optimal_params.best_params_['gamma'])
    clf_svm.fit(X_train, y_train)

    if plot==1:
        y_pred = clf_svm.predict(X_test)
        c_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(c_matrix, display_labels=clf_svm.classes_)
        disp.plot(cmap=plt.cm.Blues, colorbar=False, xticks_rotation='vertical')
        plt.show()
    
    scores = cross_val_score(clf_svm, X_train, y_train, cv=cv)
    return scores


def cv_svm_moments_samples(distributions_dict, nr_sample_list, nr_moments_list):
    acc_mean = [] # list to keep track of acc
    acc_std = []

    for i in nr_sample_list:
        acc_mean_i = [] 
        acc_std_i = []
        for j in nr_moments_list:
            heavytail_samples = dm.get_samples(distributions_dict, i, sample_size)
            transformed_sampels = dict()
            for name, samples in heavytail_samples.items():
                transformed_sampels[name] = np.log1p(samples)

            moments_df = dem.get_moments_df(transformed_sampels, j)
            score = cm.svm_model(moments_df, test_size, cv)
            acc_mean_i.append(score.mean())
            acc_std_i.append(score.std())
        acc_mean.append(acc_mean_i)
        acc_std.append(acc_std_i)
    return acc_mean, acc_std