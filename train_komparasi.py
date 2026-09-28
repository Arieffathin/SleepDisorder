"""
Skrip Pelatihan Model Komparasi (Logistic Regression & Random Forest)
Sebagai pembanding untuk XGBoost Model (F1-Score: 0.9319)

Spesifikasi:
1. Preprocessing Robust: Drop Person ID, Gender, Occupation.
2. Logistic Regression (Baseline) + 5-Fold Cross Validation.
3. Random Forest (Tuning via GridSearchCV) + 5-Fold Cross Validation.
4. Output Evaluasi: Classification Report, Accuracy, Precision, Recall, F1-Score (Console & training_results_komparasi.pkl).
5. Model Saving: lr_model_robust.pkl dan rf_model_robust.pkl.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Set stdout encoding for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

from preprocessing import load_data


def prepare_robust_dataset(df: pd.DataFrame, target_col: str = 'Sleep Disorder') -> pd.DataFrame:
    """
    Menyiapkan dataset robust tanpa fitur demografi (Gender, Occupation, Person ID)
    agar persis sama dengan pipeline model XGBoost.
    """
    print("=" * 70)
    print("DATASET PREPARATION (Removing Demographic Bias)")
    print("=" * 70)

    df_clean = df.copy()
    # Handle NaN/None pada target
    df_clean[target_col] = df_clean[target_col].fillna('Normal')
    df_clean[target_col] = df_clean[target_col].replace('None', 'Normal')

    print(f"\n[OK] Total baris: {df_clean.shape[0]}")
    print(f"[OK] Distribusi Target:\n{df_clean[target_col].value_counts()}")

    # Fitur demografi yang di-drop
    DEMOGRAPHIC_FEATURES = ['Gender', 'Occupation', 'Person ID']
    print(f"\n[WARNING] DROPPING DEMOGRAPHIC FEATURES:")
    for feature in DEMOGRAPHIC_FEATURES:
        if feature in df_clean.columns:
            print(f"  - {feature}")
            df_clean = df_clean.drop(columns=[feature])

    print(f"\n[OK] Fitur yang digunakan ({df_clean.shape[1] - 1} fitur + 1 target):")
    remaining_features = [col for col in df_clean.columns if col != target_col]
    for feature in remaining_features:
        print(f"  - {feature}")

    return df_clean


def preprocess_robust_pipeline(
    df: pd.DataFrame,
    target_col: str = 'Sleep Disorder',
    test_size: float = 0.2,
    apply_smote_flag: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocessing pipeline: Split -> Impute -> Encode -> Scale -> Target Encode -> SMOTE
    """
    print("\n" + "=" * 70)
    print("ROBUST PREPROCESSING PIPELINE")
    print("=" * 70)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"\n[1/5] Data Split (80% Train, 20% Test):")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Handling missing values jika ada
    imputer_dict = {}
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            if X_train[col].dtype in ['int64', 'float64']:
                fill_val = X_train[col].median()
            else:
                fill_val = X_train[col].mode()[0]
            imputer_dict[col] = fill_val
            X_train[col].fillna(fill_val, inplace=True)
            X_test[col].fillna(fill_val, inplace=True)

    # Encode Fitur Kategorikal (BMI Category, Blood Pressure)
    encoders = {}
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le
    print(f"  [OK] Fitur Kategorikal Di-encode: {categorical_cols}")

    # Standarisasi (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  [OK] Standarisasi: {X_train_scaled.shape[1]} fitur")

    # Encode Target
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)
    print(f"  [OK] Kelas Target: {target_encoder.classes_}")

    # SMOTE Oversampling
    if apply_smote_flag:
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train_encoded = smote.fit_resample(X_train_scaled, y_train_encoded)
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        print(f"  [OK] SMOTE Diterapkan pada Training Set: {X_train_scaled.shape[0]} sampel")
        print(f"  [OK] Keseimbangan Kelas Train: {dict(zip(target_encoder.classes_, counts))}")

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_encoded,
        'y_test': y_test_encoded,
        'scaler': scaler,
        'encoders': encoders,
        'target_encoder': target_encoder,
        'imputer_dict': imputer_dict,
        'feature_names': X_train.columns.tolist()
    }


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[LogisticRegression, float]:
    """
    Melatih model Logistic Regression (Baseline) dengan 5-Fold Cross Validation.
    """
    print("\n" + "=" * 70)
    print("1. TRAINING LOGISTIC REGRESSION (BASELINE)")
    print("=" * 70)

    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)

    # 5-Fold Cross Validation
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='f1_weighted')
    mean_cv_f1 = float(np.mean(cv_scores))

    print(f"[OK] 5-Fold CV F1-Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"[OK] Rata-rata CV Weighted F1-Score: {mean_cv_f1:.4f}")

    # Fit pada seluruh data train
    lr_model.fit(X_train, y_train)

    return lr_model, mean_cv_f1


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    """
    Melatih model Random Forest dengan Hyperparameter Tuning via GridSearchCV.
    """
    print("\n" + "=" * 70)
    print("2. TRAINING RANDOM FOREST (HYPERPARAMETER TUNING)")
    print("=" * 70)

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 5, 8, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    base_rf = RandomForestClassifier(random_state=random_state)

    print(f"Running GridSearchCV (CV={cv}, scoring='f1_weighted')...")
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)

    print(f"\n[OK] Parameter Terbaik Random Forest:")
    for param, val in best_params.items():
        print(f"  - {param}: {val}")
    print(f"[OK] Best CV Weighted F1-Score: {best_cv_score:.4f}")

    return best_rf, best_params, best_cv_score


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_encoder: LabelEncoder,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluasi model pada Test Set dan cetak metrik lengkap.
    """
    print("\n" + "-" * 70)
    print(f"EVALUASI MODEL: {model_name}")
    print("-" * 70)

    y_pred = model.predict(X_test)
    class_names = target_encoder.classes_

    acc = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average='weighted'))
    f1_macro = float(f1_score(y_test, y_pred, average='macro'))
    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=class_names, digits=4, output_dict=True)

    print(f"\n[REPORT] CLASSIFICATION REPORT ({model_name}):")
    print(report_str)

    print(f"[CONFUSION MATRIX] ({model_name}):")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    print(f"\n[SUMMARY] Metrik {model_name}:")
    print(f"  - Accuracy          : {acc:.4f}")
    print(f"  - Weighted F1-Score : {f1_weighted:.4f}")
    print(f"  - Macro F1-Score    : {f1_macro:.4f}")

    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'classification_report_str': report_str,
        'classification_report_dict': report_dict
    }


def main():
    print("=" * 70)
    print("SKRIP KOMPARASI MODEL UNTUK SKRIPSI")
    print("Logistic Regression (Baseline) vs Random Forest (Tuned) vs XGBoost")
    print("=" * 70)

    # 1. Load Dataset & Robust Preparation
    dataset_path = os.path.join('Dataset', 'Sleep_health_and_lifestyle_dataset.csv')
    df_raw = load_data(dataset_path)
    df_robust = prepare_robust_dataset(df_raw)

    # 2. Robust Preprocessing (Drop Demographics + SMOTE + Scaler)
    prep = preprocess_robust_pipeline(df_robust, apply_smote_flag=True)

    # 3. Melatih Logistic Regression (Baseline)
    lr_model, lr_cv_f1 = train_logistic_regression(
        prep['X_train'], prep['y_train'], cv=5
    )

    # 4. Melatih Random Forest (Hyperparameter Tuning)
    rf_model, rf_best_params, rf_cv_f1 = train_random_forest(
        prep['X_train'], prep['y_train'], cv=5
    )

    # 5. Evaluasi Kedua Model pada Data Test
    lr_eval = evaluate_model(
        lr_model, prep['X_test'], prep['y_test'], prep['target_encoder'], "Logistic Regression"
    )

    rf_eval = evaluate_model(
        rf_model, prep['X_test'], prep['y_test'], prep['target_encoder'], "Random Forest"
    )

    # 6. Simpan Model ke File .pkl
    with open('lr_model_robust.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("\n[OK] Model Logistic Regression disimpan ke: lr_model_robust.pkl")

    with open('rf_model_robust.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("[OK] Model Random Forest disimpan ke: rf_model_robust.pkl")

    # 7. Benchmark XGBoost Reference
    xgb_f1_reference = 0.9319  # Benchmark XGBoost Robust F1-Score

    # 8. Tabel Ringkasan Perbandingan
    print("\n" + "=" * 75)
    print("TABEL PERBANDINGAN PERFORMA MODEL (DATA TEST)")
    print("=" * 75)
    header = f"{'Model':<25} | {'CV F1-Score':<12} | {'Test Accuracy':<14} | {'Test F1-Score':<14} | {'Selisih vs XGB'}"
    print(header)
    print("-" * 75)

    lr_diff = lr_eval['f1_weighted'] - xgb_f1_reference
    rf_diff = rf_eval['f1_weighted'] - xgb_f1_reference

    print(f"{'Logistic Regression':<25} | {lr_cv_f1:<12.4f} | {lr_eval['accuracy']:<14.4f} | {lr_eval['f1_weighted']:<14.4f} | {lr_diff:+.4f}")
    print(f"{'Random Forest (Tuned)':<25} | {rf_cv_f1:<12.4f} | {rf_eval['accuracy']:<14.4f} | {rf_eval['f1_weighted']:<14.4f} | {rf_diff:+.4f}")
    print(f"{'XGBoost (Reference)':<25} | {'N/A':<12} | {'N/A':<14} | {xgb_f1_reference:<14.4f} | {'Baseline'}")
    print("=" * 75)

    # 9. Simpan Hasil Komparasi ke File training_results_komparasi.pkl
    komparasi_results = {
        'logistic_regression': {
            'model_file': 'lr_model_robust.pkl',
            'cv_f1_score': lr_cv_f1,
            'test_accuracy': lr_eval['accuracy'],
            'test_f1_weighted': lr_eval['f1_weighted'],
            'test_f1_macro': lr_eval['f1_macro'],
            'confusion_matrix': lr_eval['confusion_matrix'],
            'classification_report_str': lr_eval['classification_report_str'],
            'classification_report_dict': lr_eval['classification_report_dict']
        },
        'random_forest': {
            'model_file': 'rf_model_robust.pkl',
            'best_params': rf_best_params,
            'cv_f1_score': rf_cv_f1,
            'test_accuracy': rf_eval['accuracy'],
            'test_f1_weighted': rf_eval['f1_weighted'],
            'test_f1_macro': rf_eval['f1_macro'],
            'confusion_matrix': rf_eval['confusion_matrix'],
            'classification_report_str': rf_eval['classification_report_str'],
            'classification_report_dict': rf_eval['classification_report_dict']
        },
        'xgboost_reference': {
            'test_f1_weighted': xgb_f1_reference
        }
    }

    with open('training_results_komparasi.pkl', 'wb') as f:
        pickle.dump(komparasi_results, f)
    print("[OK] Hasil komparasi lengkap disimpan ke: training_results_komparasi.pkl")
    print("\nProses selesai dengan sukses!")


if __name__ == '__main__':
    main()
