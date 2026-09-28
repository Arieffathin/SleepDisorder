"""
Robust Model Training - Without Demographic Bias
DROP Gender & Occupation untuk fokus pada clinical metrics
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from preprocessing import load_data


def prepare_robust_dataset(df: pd.DataFrame, target_col: str = 'Sleep Disorder') -> pd.DataFrame:
    """
    Prepare dataset dengan DROP demographic features untuk mengurangi bias
    
    Args:
        df: Raw dataset
        target_col: Target column name
    
    Returns:
        Cleaned dataframe tanpa demographic features
    """
    print("=" * 70)
    print("DATASET PREPARATION (Removing Demographic Bias)")
    print("=" * 70)
    
    # Handle missing target (NaN = Normal)
    df_clean = df.copy()
    df_clean[target_col] = df_clean[target_col].fillna('Normal')
    df_clean[target_col] = df_clean[target_col].replace('None', 'Normal')
    
    print(f"\n✓ Total rows: {df_clean.shape[0]}")
    print(f"✓ Target distribution:\n{df_clean[target_col].value_counts()}")
    
    # DROP demographic features yang bisa menyebabkan bias
    DEMOGRAPHIC_FEATURES = ['Gender', 'Occupation', 'Person ID']
    
    print(f"\n⚠️  DROPPING DEMOGRAPHIC FEATURES to reduce bias:")
    for feature in DEMOGRAPHIC_FEATURES:
        if feature in df_clean.columns:
            print(f"  - {feature}")
            df_clean = df_clean.drop(columns=[feature])
    
    print(f"\n✓ Remaining features ({df_clean.shape[1]} columns):")
    remaining_features = [col for col in df_clean.columns if col != target_col]
    for feature in remaining_features:
        print(f"  - {feature}")
    
    return df_clean


def preprocess_robust_pipeline(
    df: pd.DataFrame,
    target_col: str = 'Sleep Disorder',
    test_size: float = 0.2,
    apply_smote: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Preprocessing pipeline untuk robust model (tanpa demographic features)
    """
    print("\n" + "=" * 70)
    print("ROBUST PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # 1. Split data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"\n[1/5] Data split:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Train classes: {y_train.value_counts().to_dict()}")
    
    # 2. Handle missing values (should be none, but just in case)
    print("\n[2/5] Handling missing values...")
    imputer_dict = {}
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            if X_train[col].dtype in ['int64', 'float64']:
                fill_value = X_train[col].median()
            else:
                fill_value = X_train[col].mode()[0]
            imputer_dict[col] = fill_value
            X_train[col].fillna(fill_value, inplace=True)
            X_test[col].fillna(fill_value, inplace=True)
    print(f"  ✓ Missing values handled: {list(imputer_dict.keys()) if imputer_dict else 'None'}")
    
    # 3. Encode categorical features (BMI Category, Blood Pressure)
    print("\n[3/5] Encoding categorical features...")
    encoders = {}
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le
    print(f"  ✓ Encoded: {categorical_cols}")
    
    # 4. Scale features
    print("\n[4/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  ✓ Scaled: {X_train_scaled.shape[1]} features")
    
    # 5. Encode target
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)
    print(f"  ✓ Target classes: {target_encoder.classes_}")
    
    # 6. SMOTE
    if apply_smote:
        print("\n[5/5] Applying SMOTE...")
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train_encoded = smote.fit_resample(X_train_scaled, y_train_encoded)
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        print(f"  ✓ After SMOTE: {X_train_scaled.shape[0]} samples")
        print(f"  ✓ Class balance: {dict(zip(unique, counts))}")
    
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


def train_robust_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 50,
    cv: int = 5,
    random_state: int = 42
):
    """Train XGBoost dengan hyperparameter tuning"""
    print("\n" + "=" * 70)
    print("TRAINING ROBUST XGBOOST MODEL")
    print("=" * 70)
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        random_state=random_state
    )
    
    # Hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'n_estimators': [50, 100, 150, 200, 300],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.5]
    }
    
    print(f"\n[1/2] Running RandomizedSearchCV...")
    print(f"  Iterations: {n_iter}")
    print(f"  CV Folds: {cv}")
    print(f"  Scoring: f1_weighted")
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='f1_weighted',
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    
    print(f"\n[2/2] Best hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\n✓ Best CV F1-Score: {random_search.best_score_:.4f}")
    
    return best_model, random_search.best_params_, random_search.best_score_


def evaluate_robust_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_encoder,
    save_plot: bool = True
):
    """Evaluate model dengan fokus pada clinical sensitivity"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (Robust Model)")
    print("=" * 70)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Class names
    class_names = target_encoder.classes_
    
    # Classification report
    print("\n📊 CLASSIFICATION REPORT:")
    print("=" * 70)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📋 CONFUSION MATRIX:")
    print("=" * 70)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Per-class metrics
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"\n📈 F1-Score (weighted): {f1_weighted:.4f}")
    
    # False Negatives Analysis
    print("\n⚠️  FALSE NEGATIVES ANALYSIS (Clinical Focus):")
    print("=" * 70)
    for i, class_name in enumerate(class_names):
        fn_count = cm[i, :].sum() - cm[i, i]
        total_actual = cm[i, :].sum()
        fn_rate = (fn_count / total_actual) * 100 if total_actual > 0 else 0
        
        print(f"\n  {class_name}:")
        print(f"    Total actual: {total_actual}")
        print(f"    Correctly predicted: {cm[i, i]}")
        print(f"    False Negatives: {fn_count} ({fn_rate:.2f}%)")
    
    # Plot confusion matrix
    if save_plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'}, linewidths=0.5)
        plt.title('Confusion Matrix - Robust Model (No Demographic Bias)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix_robust.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved: confusion_matrix_robust.png")
        plt.close()
    
    return {
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': report
    }


def test_bias_scenarios(model, scaler, encoders, target_encoder, feature_names):
    """Test model dengan scenarios untuk detect bias"""
    print("\n" + "=" * 70)
    print("BIAS DETECTION TEST")
    print("=" * 70)
    
    # Scenario: Extreme poor sleep (should predict disorder, not Normal)
    print("\n🧪 TEST SCENARIO: Extreme Poor Sleep Profile")
    print("-" * 70)
    
    test_case = {
        'Age': 27,
        'Sleep Duration': 4.0,
        'Quality of Sleep': 1,
        'Physical Activity Level': 0,
        'Stress Level': 10,
        'BMI Category': 'Overweight',
        'Blood Pressure': '140/95',
        'Heart Rate': 95,
        'Daily Steps': 1000
    }
    
    print("Input:")
    for k, v in test_case.items():
        print(f"  {k}: {v}")
    
    # Create DataFrame
    test_df = pd.DataFrame([test_case])
    
    # Encode categorical
    for col, encoder in encoders.items():
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str).apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
    
    # Scale
    X_test = scaler.transform(test_df)
    
    # Predict
    prediction = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    pred_name = target_encoder.inverse_transform(prediction)[0]
    confidence = np.max(pred_proba) * 100
    
    print(f"\nPrediction: {pred_name}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"All probabilities:")
    for class_name, prob in zip(target_encoder.classes_, pred_proba[0]):
        print(f"  {class_name}: {prob*100:.2f}%")
    
    if pred_name == 'Normal' and confidence > 80:
        print("\n❌ BIAS DETECTED: Model still predicts 'Normal' with high confidence")
        print("   despite extreme poor sleep indicators!")
    else:
        print(f"\n✅ GOOD: Model correctly identified risk (predicted {pred_name})")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROBUST MODEL TRAINING (Demographic Bias Reduction)")
    print("=" * 70)
    
    # 1. Load and prepare dataset
    print("\n[STAGE 1] Loading dataset...")
    df_raw = load_data('Dataset/Sleep_health_and_lifestyle_dataset.csv')
    df_robust = prepare_robust_dataset(df_raw)
    
    # 2. Preprocessing
    print("\n[STAGE 2] Preprocessing...")
    preprocess_results = preprocess_robust_pipeline(df_robust, apply_smote=True)
    
    # 3. Training
    print("\n[STAGE 3] Training XGBoost...")
    best_model, best_params, best_score = train_robust_xgboost(
        preprocess_results['X_train'],
        preprocess_results['y_train'],
        n_iter=50,
        cv=5
    )
    
    # 4. Evaluation
    print("\n[STAGE 4] Evaluation...")
    eval_results = evaluate_robust_model(
        best_model,
        preprocess_results['X_test'],
        preprocess_results['y_test'],
        preprocess_results['target_encoder']
    )
    
    # 5. Bias detection test
    print("\n[STAGE 5] Bias detection...")
    test_bias_scenarios(
        best_model,
        preprocess_results['scaler'],
        preprocess_results['encoders'],
        preprocess_results['target_encoder'],
        preprocess_results['feature_names']
    )
    
    # 6. Save robust model and preprocessor
    print("\n" + "=" * 70)
    print("SAVING ROBUST MODEL")
    print("=" * 70)
    
    # Save model
    with open('xgb_model_robust.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("✓ Model saved: xgb_model_robust.pkl")
    
    # Save preprocessor
    preprocessor_robust = {
        'scaler': preprocess_results['scaler'],
        'encoders': preprocess_results['encoders'],
        'target_encoder': preprocess_results['target_encoder'],
        'imputer_dict': preprocess_results['imputer_dict'],
        'feature_names': preprocess_results['feature_names']
    }
    with open('preprocessor_robust.pkl', 'wb') as f:
        pickle.dump(preprocessor_robust, f)
    print("✓ Preprocessor saved: preprocessor_robust.pkl")
    
    # Save training results
    training_results = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'test_f1_score': eval_results['f1_weighted'],
        'confusion_matrix': eval_results['confusion_matrix'],
        'classification_report': eval_results['classification_report']
    }
    with open('training_results_robust.pkl', 'wb') as f:
        pickle.dump(training_results, f)
    print("✓ Training results saved: training_results_robust.pkl")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"✓ CV F1-Score: {best_score:.4f}")
    print(f"✓ Test F1-Score: {eval_results['f1_weighted']:.4f}")
    print(f"\n✓ Demographic features removed: Gender, Occupation")
    print(f"✓ Focus on clinical metrics for unbiased predictions")
    print("\nFiles generated:")
    print("  - xgb_model_robust.pkl")
    print("  - preprocessor_robust.pkl")
    print("  - training_results_robust.pkl")
    print("  - confusion_matrix_robust.png")
