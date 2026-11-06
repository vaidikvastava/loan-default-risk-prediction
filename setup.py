from setuptools import setup, find_packages

setup(
    name="loan-default-risk-prediction",
    version="1.0.0",
    author="CreditIQ Machine Learning Team",
    description="End-to-End Loan Default Prediction and Credit Underwriting Intelligence System",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "full": [
            "imbalanced-learn>=0.11.0",
            "seaborn>=0.12.0",
            "streamlit>=1.28.0",
            "pymysql>=1.1.0",
            "mysql-connector-python>=8.0.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "shap>=0.42.0",
        ],
        "test": [
            "pytest>=7.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-loan-model=src.train:run_training_pipeline",
            "score-loan=src.predict:main",
        ],
    },
)
