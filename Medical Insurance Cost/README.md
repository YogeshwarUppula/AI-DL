# Medical Insurance Cost Prediction Using Deep Learning Techniques

A regression project that predicts individual medical insurance charges using **PyTorch**. The notebook compares a PyTorch Linear Regression baseline with two Artificial Neural Network (ANN) models trained using SGD and Adam optimizers.

## Project objectives

- Load and inspect the Medical Cost Personal Dataset.
- Clean duplicate records and preprocess numerical and categorical features.
- Train and evaluate three PyTorch regression models.
- Compare model performance using MAE, MSE, RMSE, and R².
- Use the trained model workflow to predict insurance charges for a new input.

## Models implemented

1. **Linear Regression (PyTorch)** — baseline model trained with SGD.
2. **ANN with SGD** — fully connected network with 64 and 32 hidden units, ReLU activations, and SGD optimizer.
3. **ANN with Adam** — the same ANN architecture trained with the Adam optimizer.

## Dataset

The dataset file `insurance.csv` is included in this project folder. It is the Medical Cost Personal Dataset, available from:

[Medical Cost Personal Dataset – Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

The notebook uses `insurance.csv` as its input dataset. Keep the CSV in the same project folder as the notebook when running locally.

Dataset columns used:
- Inputs: `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- Target: `charges`

The notebook reports 1,338 original rows and 7 columns. After duplicate removal, 1,337 rows remain. The train/test split is 80:20 (1,069 training rows and 268 test rows). Preprocessing produces 11 model input features.

## Results

The following are the recorded results from the executed project notebook:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression (PyTorch) | 4,133.62 | 35,670,502.40 | 5,972.48 | 0.8059 |
| ANN with SGD | **2,499.37** | **19,199,533.17** | **4,381.73** | **0.8955** |
| ANN with Adam | 3,266.49 | 27,251,220.08 | 5,220.27 | 0.8517 |

**Best-performing model in this run:** ANN with SGD, with the lowest RMSE and highest R² score among the three models.

> Results can vary if the notebook is re-run with different software versions, random seeds, or training conditions.

## Repository contents

```text
Medical-Insurance-Cost-Prediction/
├── DL_PROJECT.ipynb
├── README.md
├── requirements.txt
├── insurance.csv
├── PROJECT_INFO.txt
└── .gitignore
```

## Run the project in Google Colab

1. Open `DL_PROJECT.ipynb` in Google Colab.
2. Run the notebook cells in order.
3. When the upload prompt appears, select the downloaded `insurance.csv`.
4. Run the remaining cells to preprocess the data, train the models, evaluate them, and view the plots.

## Requirements

The notebook uses Python with:
- pandas
- numpy
- scikit-learn
- torch
- matplotlib
- seaborn

Google Colab already provides most of these packages. For a local environment, install them with:

```bash
pip install -r requirements.txt
```

## Evaluation metrics

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

## Notes

- The dataset is included as `insurance.csv`.
- The notebook is the main executable project artifact.
- This is an academic course project for Artificial Intelligence and Deep Learning.
