# muscle-contusion-ml-classification
A machine learning pipeline for classifying recovery stages of muscle contusion using SVM, XGBoost, and LightGBM models, combined with various feature selection methods (PCA, RFE, mRMR, Relief). Subsequently, methods assessing feature importances attempt to explain the results.

---
## Features

- 3 ML classifiers: SVM, XGBoost, LGBM
- 4 feature reduction methods: PCA, RFE, mRMR, Relief
- Accuracy evaluation and confusion matrix
- SHAP explainability for model interpretation
- Modular code structure for easy expansion

---
## Modules

- src/train_and_save_results.py:\
Train multiple classification models with various feature selection techniques and save results
- src/plot_results.py:\
Visualize model performances from saved .npy result files
- src/importance_visualization.py:\
Model interpretability using SHAP values and feature importance plots

---
## Example Outputs
**Only partial examples are provided due to conflict of interests.**
- examples/svm_acc.jpg: classification accuracy with various feature reduction methods and feature dimentions (from plot_results.py)
- examples/relief_importance.jpg: importance values and ranking of top 10 features (from importance_visualization.py)
- examples/shap_value.jpg: result of SHAP method (from importance_visualization.py)

---
## File Structure
```
muscle-contusion-ml-classification
├── README.md
├── requirements.txt
├── src
│   ├── train_and_save_results.py
│   ├── plot_results.py
│   └── shap_visualization.py
├── data
│   ├── data.xlsx
│   └── label.xlsx
└── examples
    ├── svm_acc.jpg
    ├── relief_importance.jpg
    └── shap_value.jpg
```

---
## Requirements
```bash
pip install -r requirements.txt
```

---
### Installation of packages mrmr and shap
Due to PyPI limitations, the following packages may require manual installation or alternative handling
```bash
pip install mrmr
pip install shap
```
If installation fails, please don't use mrmr as a feature reduction method and/or comment out the shap block in the code, and proceed with other model evaluation sections.

---
## Usage

```bash
python src/train_and_save_results.py
python src/plot_results.py
python src/shap_visualization.py
```

---
## Notes

- Replace data.xlsx and label.xlsx with your data or simulated examples
