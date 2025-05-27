import optuna
import pandas as pd
import sklearn.preprocessing as prep
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import joblib
from sklearn import decomposition

data_path = rf'D:\DaMing\contusion_classification\data.xlsx'
label_path = rf'D:\DaMing\contusion_classification\label.xlsx'
data = pd.read_excel(data_path).to_numpy()
label = pd.read_excel(label_path).squeeze().to_numpy()
reduced_dimension = 40

scale_data = prep.MaxAbsScaler().fit_transform(data)
pca = decomposition.PCA(n_components=reduced_dimension)
reduced_data = pca.fit_transform(scale_data)
X_train, X_test, y_train, y_test = train_test_split(reduced_data, label, test_size=0.2, random_state=2)

def objective(trial):
    """
    A function to train a model using different hyperparamerters combinations provided by Optuna.
    """
    params = {
        'min_child_samples':trial.suggest_categorical('min_child_samples', [10, 15, 20, 25, 30, 35, 40, 45, 50]),
        'num_leaves': trial.suggest_categorical('num_leaves', [3, 7, 15, 31, 63, 127, 255]),
        'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 19]),
        'learning_rate': trial.suggest_categorical('learning_rate', [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]),
        'n_estimators': trial.suggest_categorical('n_estimators', [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    }
    clf = lgbm.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

search_space = {
    'min_child_samples':[10, 15, 20, 25, 30, 35, 40, 45, 50],
    'num_leaves': [3, 7, 15, 31, 63, 127, 255],
    'max_depth': [5, 7, 9, 11, 13, 15, 17, 19],
    'learning_rate': [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize')
study.optimize(objective, n_trials = 9*7*8*7*10)
joblib.dump(study, rf'D:\DaMing\contusion_classification\output.pkl')
print(study.best_trial)
print(study.best_params)
print(study.best_value)
