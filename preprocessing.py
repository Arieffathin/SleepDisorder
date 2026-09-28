"""
Modul Preprocessing untuk Sleep Disorder Prediction
Anti-Data Leakage: Split -> Fit pada Train -> Transform pada Train & Test
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset dari CSV"""
    df = pd.read_csv(filepath)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = 'Sleep Disorder',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split untuk menjaga proporsi kelas target
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def handle_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = 'median'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Handling missing values: Fit pada train, transform pada train & test
    
    Args:
        strategy: 'mean', 'median', atau 'mode'
    
    Returns:
        X_train_filled, X_test_filled, imputer_dict
    """
    imputer_dict = {}
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()
    
    for col in X_train.columns:
        if X_train[col].isnull().sum() > 0:
            if strategy == 'median':
                fill_value = X_train[col].median()
            elif strategy == 'mean':
                fill_value = X_train[col].mean()
            elif strategy == 'mode':
                fill_value = X_train[col].mode()[0]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            imputer_dict[col] = fill_value
            X_train_filled[col].fillna(fill_value, inplace=True)
            X_test_filled[col].fillna(fill_value, inplace=True)
    
    return X_train_filled, X_test_filled, imputer_dict


def encode_categorical(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label Encoding untuk kategorikal: Fit pada train, transform pada train & test
    
    Returns:
        X_train_encoded, X_test_encoded, encoders_dict
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    encoders_dict = {}
    
    if categorical_cols is None:
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
        
        # Handle unseen categories di test set
        X_test_encoded[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        
        encoders_dict[col] = le
    
    return X_train_encoded, X_test_encoded, encoders_dict


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    StandardScaler: Fit pada train, transform pada train & test
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def encode_target(
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode target variable: Fit pada train, transform pada train & test
    
    Returns:
        y_train_encoded, y_test_encoded, target_encoder
    """
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)
    
    return y_train_encoded, y_test_encoded, target_encoder


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE untuk handle imbalance: Hanya pada training set
    
    Returns:
        X_train_resampled, y_train_resampled
    """
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled


def save_preprocessor(
    scaler: StandardScaler,
    encoders: Dict[str, LabelEncoder],
    target_encoder: LabelEncoder,
    imputer_dict: Dict[str, Any],
    filepath: str = 'preprocessor.pkl'
) -> None:
    """Save semua komponen preprocessing ke file .pkl"""
    preprocessor = {
        'scaler': scaler,
        'encoders': encoders,
        'target_encoder': target_encoder,
        'imputer_dict': imputer_dict
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    print(f"Preprocessor saved to {filepath}")


def load_preprocessor(filepath: str = 'preprocessor.pkl') -> Dict[str, Any]:
    """Load preprocessor dari file .pkl"""
    with open(filepath, 'rb') as f:
        preprocessor = pickle.load(f)
    
    return preprocessor


def preprocess_pipeline(
    df: pd.DataFrame,
    target_col: str = 'Sleep Disorder',
    test_size: float = 0.2,
    apply_smote_flag: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Pipeline lengkap preprocessing dengan anti-data leakage
    
    Returns:
        Dictionary berisi:
        - X_train, X_test, y_train, y_test (scaled & encoded)
        - scaler, encoders, target_encoder, imputer_dict
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE (Anti-Data Leakage)")
    print("=" * 50)
    
    # 0. Handle target: NaN/None = 'Normal' (3 kelas: Normal, Insomnia, Sleep Apnea)
    print("\n[0/6] Handling target values...")
    df_clean = df.copy()
    df_clean[target_col] = df_clean[target_col].fillna('Normal')
    df_clean[target_col] = df_clean[target_col].replace('None', 'Normal')
    print(f"Total rows: {df_clean.shape[0]}")
    print(f"Target distribution: {df_clean[target_col].value_counts().to_dict()}")
    
    # 1. Split data (STRATIFIED)
    print("\n[1/6] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df_clean, target_col, test_size, random_state)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution (Train): {y_train.value_counts().to_dict()}")
    
    # 2. Handle missing values (FIT pada train)
    print("\n[2/6] Handling missing values...")
    X_train, X_test, imputer_dict = handle_missing_values(X_train, X_test)
    print(f"Missing values handled: {list(imputer_dict.keys())}")
    
    # 3. Encode categorical (FIT pada train)
    print("\n[3/6] Encoding categorical features...")
    X_train, X_test, encoders = encode_categorical(X_train, X_test)
    print(f"Encoded columns: {list(encoders.keys())}")
    
    # 4. Scale features (FIT pada train)
    print("\n[4/6] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"Features scaled: {X_train_scaled.shape[1]} features")
    
    # 5. Encode target (FIT pada train)
    print("\n[5/6] Encoding target variable...")
    y_train_encoded, y_test_encoded, target_encoder = encode_target(y_train, y_test)
    print(f"Target classes: {target_encoder.classes_}")
    
    # 6. SMOTE (HANYA pada train)
    if apply_smote_flag:
        print("\n[6/6] Applying SMOTE on training set...")
        print(f"Before SMOTE: {X_train_scaled.shape}")
        X_train_scaled, y_train_encoded = apply_smote(X_train_scaled, y_train_encoded, random_state)
        print(f"After SMOTE: {X_train_scaled.shape}")
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        print(f"Class balance: {dict(zip(unique, counts))}")
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED")
    print("=" * 50)
    
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


if __name__ == "__main__":
    # Contoh penggunaan
    df = load_data('Dataset/Sleep_health_and_lifestyle_dataset.csv')
    print(f"Dataset loaded: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nTarget distribution:\n{df['Sleep Disorder'].value_counts()}")
    
    # Jalankan preprocessing pipeline
    results = preprocess_pipeline(df, target_col='Sleep Disorder', apply_smote_flag=True)
    
    # Save preprocessor
    save_preprocessor(
        scaler=results['scaler'],
        encoders=results['encoders'],
        target_encoder=results['target_encoder'],
        imputer_dict=results['imputer_dict'],
        filepath='preprocessor.pkl'
    )
