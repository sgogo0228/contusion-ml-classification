# muscle-contusion-ml-classification
A machine learning pipeline for classifying recovery stages of muscle contusion using SVM, XGBoost, and LightGBM models, combined with various feature selection methods (PCA, RFE, mRMR, Relief).

## Features

- 3 ML classifiers: SVM, XGBoost, LGBM
- 4 feature reduction methods: PCA, RFE, mRMR, Relief
- Accuracy evaluation and confusion matrix
- SHAP explainability for model interpretation
- Modular code structure for easy expansion

## File Structure

muscle-contusion-ml-classification/
├── README.md
├── requirements.txt
├── src/
│   └── contusion_classification_ML.py
├── data/
│   ├── data.xlsx  ←（如不能公開可用 dummy）
│   └── label.xlsx
├── examples/
│   └── shap_summary.png

## Requirements

## Usage

```bash
python src/contusion_classification_ML.py
```

## Notes

- Replace data.xlsx and label.xlsx with your data or simulated examples
- SHAP summary plots are generated at the end of execution
