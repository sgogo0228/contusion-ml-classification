import shap
import numpy as np
import matplotlib.pyplot as plt

# show feature importance from feature reduction method
def plot_feature_importance(clf, x_feature_names):
    importance = clf.feature_importances_
    sort_importance_idx = np.argsort(importance)[::-1]
    sort_importance = np.sort(importance)[::-1]
    for i in range(0, 10):
        plt.barh(9-i, sort_importance[i])
    ax = plt.gca()
    plt.xlabel('Feature importance', fontsize=16, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('Feature', fontsize=16, fontweight='bold', fontfamily='Times New Roman')
    plt.xticks(fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    plt.yticks(np.arange(0, 10))
    ax.set_yticklabels(x_feature_names[sort_importance_idx[0:10]][::-1], fontsize=12, fontweight='bold', fontfamily='Times New Roman')

# execute SHAP
def plot_shap_value(clf, X_test, x_feature_names):
    class_names = ['normal', 'destruction', 'repair', 'remodeling']
    explainer = shap.KernelExplainer(model=clf.predict_proba, data=X_test, link='logit')
    shap_values = explainer.shap_values(X=X_test)
    # This line can only run on jupyter and show visual picture of shapley value
    shap.summary_plot(shap_values[0], X_test, max_display=10, class_names= class_names, feature_names = x_feature_names)

