# optimize the number of features by iteration (dimentionality reduction)

import numpy as np
import pandas as pd
# import pymrmr
import mrmr
import sklearn_relief as relief
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import sklearn.preprocessing as prep
from sklearn.pipeline import make_pipeline
from sklearn import svm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgbm
from sklearn import decomposition
from sklearn.feature_selection import RFE

def data_normalization(classifier, data):
    if classifier == 'SVM':
        return prep.StandardScaler().fit_transform(data)
    else:
        return prep.MaxAbsScaler().fit_transform(data)

# feature_reduction = 'relief'
# classfier = "XGB"
# reduced_dimension = 50
process = [("SVM", 'pca'),
           ("SVM", 'mrmr'),
           ("SVM", 'relief')]
rep = 5
data_dir = rf'D:\DaMing\contusion_classification'
data_path = rf'{data_dir}\data.xlsx'
label_path = rf'{data_dir}\label.xlsx'
data = pd.read_excel(data_path)
label = pd.read_excel(label_path).squeeze().to_numpy()
max_feature_num = len(data.columns)
x_feature_names = data.columns

dim_range = np.arange(5, max_feature_num, 5)
train_acc = np.zeros((rep, dim_range.shape[0]))
val_acc = np.zeros((rep, dim_range.shape[0]))

for classifier, feature_reduction in process:
    for r in range(0, rep):
        for i in range(0, dim_range.shape[0]):
            reduced_dimension = dim_range[i]
            if feature_reduction == 'pca':
                scale_data = data_normalization(classifier, data)
                pca = decomposition.PCA(n_components=reduced_dimension)
                reduced_data = pca.fit_transform(scale_data)
            elif feature_reduction == 'mrmr':
                sf  = mrmr.mrmr_classif(X=data, y=label, K=reduced_dimension)
                reduced_data = data_normalization(classifier, data[sf])
            elif feature_reduction == 'rfe':
                reduced_data = data_normalization(classifier, data)
                if classifier=="SVM":
                    clf = svm.SVC(kernel='rbf', C=1, gamma='auto', probability=True)
                elif classifier=="XGB":
                    # importance_type='gain' should be given to generate feature_importances_ for further performance evaluation by RFE
                    clf = XGBClassifier(importance_type='gain')
                else:
                    clf = lgbm.LGBMClassifier(importance_type='gain')
                sf = RFE(estimator=clf, n_features_to_select=reduced_dimension)
                reduced_data = sf.fit_transform(reduced_data, label)
            elif feature_reduction == 'relief':
                scale_data = data_normalization(classifier, data)
                rl = relief.Relief(n_features=reduced_dimension, n_jobs=1) # Choose the best 3 features
                # r = relief.ReliefF(n_features=reduced_dimension, n_jobs=1)
                # r = relief.RReliefF(n_features=reduced_dimension, n_jobs=1)
                reduced_data = rl.fit_transform(scale_data, label)
            else:
                reduced_data = data_normalization(classifier, data)

            X_train, X_test, y_train, y_test = train_test_split(reduced_data, label, test_size=0.2)

            # 這邊有做正規化，沒有做結果會差很多
            if classifier=="SVM":
                clf = svm.SVC(kernel='rbf', C=1, gamma='auto', probability=True)
            elif classifier=="XGB":
                clf = XGBClassifier()
            else:
                clf = lgbm.LGBMClassifier()
                # clf = lgbm.LGBMClassifier(min_child_samples=40, num_leaves=127, max_depth=9, learning_rate=0.5, n_estimators=45)
            clf.fit(X_train,y_train)
            
            train_acc[r, i] = clf.score(X_train, y_train)
            val_acc[r, i] = clf.score(X_test, y_test)
        
        # print(rf'train: {train_acc[i]}')
        # print(rf'validation: {val_acc[i]}')
    np.save(rf'{data_dir}\{classifier}_{feature_reduction}_{rep}rep_train_acc', train_acc)
    np.save(rf'{data_dir}\{classifier}_{feature_reduction}_{rep}rep_val_acc', val_acc)