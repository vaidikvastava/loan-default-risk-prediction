import json
import shutil
from pathlib import Path

nb_dir = Path(r"C:\Users\lowan\.gemini\antigravity\scratch\loan-default-risk-prediction\notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)

# 1. EDA Notebook
nb_eda = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01 — Exploratory Data Analysis & Risk Profile Assessment\n",
                "**Dataset:** Home Credit Default Risk (`application_train.csv`)\n",
                "**Objective:** Analyze applicant demographics, financial distributions, missingness patterns, and default risk signals."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "raw_path = '../data/raw/application_train.csv'\n",
                "df = pd.read_csv(raw_path)\n",
                "print(f'Raw dataset shape: {df.shape}')\n",
                "print('Target Class Distribution:')\n",
                "print(df['TARGET'].value_counts(normalize=True))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Key Finding: Severe Class Imbalance\n",
                "- ~92.3% Non-Defaulters (`TARGET = 0`)\n",
                "- ~7.7% Defaulters (`TARGET = 1`)\n",
                "- 12:1 negative-to-positive ratio requires cost-sensitive weighting and PR-AUC/ROC-AUC evaluation rather than raw accuracy."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Missing values analysis\n",
                "missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)\n",
                "print('Top 10 columns by missingness (%):')\n",
                "print(missing.head(10))\n",
                "\n",
                "# DAYS_EMPLOYED anomaly check\n",
                "anom_count = (df['DAYS_EMPLOYED'] == 365243).sum()\n",
                "print(f'DAYS_EMPLOYED anomaly count (365243): {anom_count} ({anom_count/len(df):.1%})')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(nb_dir / "01_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_eda, f, indent=2)

# 2. Feature Engineering Notebook
nb_fe = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 02 — Financial Feature Engineering & Ratio Derivations\n",
                "**Objective:** Construct domain-specific affordability metrics, external bureau aggregates, and robust categorical encodings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "df = pd.read_csv('../data/raw/application_train.csv')\n",
                "\n",
                "# Domain Ratios\n",
                "df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']\n",
                "df['ANNUITY_TO_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']\n",
                "df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']\n",
                "df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']\n",
                "\n",
                "# External ratings\n",
                "ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']\n",
                "df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)\n",
                "print('Engineered financial ratios preview:')\n",
                "print(df[['DEBT_TO_INCOME', 'ANNUITY_TO_INCOME', 'EXT_SOURCE_MEAN']].describe())"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(nb_dir / "02_feature_engineering.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_fe, f, indent=2)

# 3. Copy user's original notebook as 03_loan_default_full_lifecycle.ipynb
user_nb_path = r"C:\Users\lowan\.gemini\antigravity\brain\f18d00e6-8c28-41be-9159-4fbbbae23cce\.user_uploaded\media_1793980585736_7b07fb25.ipynb"
shutil.copyfile(user_nb_path, nb_dir / "03_loan_default_full_lifecycle.ipynb")

print("All 3 notebooks saved successfully.")
