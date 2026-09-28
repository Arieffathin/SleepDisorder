"""
Flask Web Application untuk Sleep Disorder Prediction
Endpoint: /, /predict
WITH INPUT VALIDATION untuk mencegah OOD predictions
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import re
from preprocessing import load_preprocessor

app = Flask(__name__)

def load_model(filepath: str):
    """Load model pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Display name mapping for available models
MODEL_NAMES = {
    'xgb': 'XGBoost',
    'rf': 'Random Forest',
    'lr': 'Logistic Regression'
}

# Load kesemua 3 model dan preprocessor saat startup
print("Loading ROBUST models (XGBoost, Random Forest, Logistic Regression) and preprocessor...")
MODELS = {
    'xgb': load_model('xgb_model_robust.pkl'),
    'rf': load_model('rf_model_robust.pkl'),
    'lr': load_model('lr_model_robust.pkl')
}
PREPROCESSOR = load_preprocessor('preprocessor_robust.pkl')
TARGET_ENCODER = PREPROCESSOR['target_encoder']
print(f"[OK] Robust models loaded: {list(MODELS.keys())}. Target classes: {TARGET_ENCODER.classes_}")
print(f"[OK] Features used: {PREPROCESSOR['feature_names']}")

# Input Validation Ranges (based on training dataset analysis)
VALIDATION_RULES = {
    'Age': (27, 59),
    'Sleep Duration': (4.0, 10.0),
    'Quality of Sleep': (1, 10),
    'Physical Activity Level': (0, 180),
    'Stress Level': (1, 10),
    'Heart Rate': (40, 120),
    'Daily Steps': (0, 20000),
}

# Valid categorical values (updated for robust model - no Gender/Occupation)
VALID_CATEGORIES = {
    'BMI Category': ['Normal', 'Normal Weight', 'Overweight', 'Obese']
}


def validate_input(form_data: dict) -> tuple:
    """
    Validasi input KETAT untuk mencegah OOD (Out-of-Distribution) predictions
    
    Args:
        form_data: Dictionary dari form/API input
    
    Returns:
        (is_valid: bool, error_message: str)
    """
    errors = []
    
    # Map API field names to display names
    field_mapping = {
        'age': 'Age',
        'sleep_duration': 'Sleep Duration',
        'quality_of_sleep': 'Quality of Sleep',
        'physical_activity': 'Physical Activity Level',
        'stress_level': 'Stress Level',
        'heart_rate': 'Heart Rate',
        'daily_steps': 'Daily Steps'
    }
    
    # 1. STRICT Numeric Validation
    for field_name, display_name in field_mapping.items():
        if field_name not in form_data:
            errors.append(f"{display_name} is required")
            continue
        
        try:
            value = float(form_data[field_name])
            
            if display_name not in VALIDATION_RULES:
                continue
            
            min_val, max_val = VALIDATION_RULES[display_name]
            
            if value < min_val or value > max_val:
                errors.append(
                    f"{display_name} must be between {min_val} and {max_val}. "
                    f"Your value: {value}"
                )
        except (ValueError, TypeError):
            errors.append(f"{display_name} must be a valid number")
    
    # 2. STRICT Categorical Validation (updated for robust model - no Gender)
    if 'bmi_category' in form_data:
        if form_data['bmi_category'] not in VALID_CATEGORIES['BMI Category']:
            errors.append(
                f"BMI Category must be one of: {', '.join(VALID_CATEGORIES['BMI Category'])}. "
                f"Your value: '{form_data['bmi_category']}'"
            )
    else:
        errors.append("BMI Category is required")
    
    # 3. STRICT Blood Pressure Validation
    if 'blood_pressure' not in form_data:
        errors.append("Blood Pressure is required")
    else:
        bp = str(form_data['blood_pressure'])
        bp_pattern = re.match(r'^(\d{2,3})/(\d{2,3})$', bp)
        
        if not bp_pattern:
            errors.append(
                f"Blood Pressure must be in format XXX/YY (e.g., 120/80). "
                f"Your value: '{bp}'"
            )
        else:
            systolic = int(bp_pattern.group(1))
            diastolic = int(bp_pattern.group(2))
            
            if not (90 <= systolic <= 180):
                errors.append(
                    f"Systolic pressure must be between 90-180. "
                    f"Your value: {systolic}"
                )
            if not (60 <= diastolic <= 120):
                errors.append(
                    f"Diastolic pressure must be between 60-120. "
                    f"Your value: {diastolic}"
                )
    
    # 4. ADDITIONAL: Extreme Combination Warning (non-blocking, just log)
    if not errors:  # Only if no errors so far
        try:
            sleep_duration = float(form_data.get('sleep_duration', 7))
            quality = int(form_data.get('quality_of_sleep', 7))
            stress = int(form_data.get('stress_level', 5))
            
            # Case: Very short sleep + very low quality + high stress
            # Changed to WARNING instead of ERROR - let model predict but warn user
            if sleep_duration < 5 and quality <= 3 and stress >= 7:
                pass  # Don't block, model should handle it
                
        except (ValueError, KeyError, TypeError):
            pass  # Already caught in numeric validation
    
    if errors:
        # Format error messages
        error_text = " | ".join(errors)
        return False, error_text
    
    return True, ""


def analyze_health_insights(form_data: dict) -> dict:
    """
    Analisis kebiasaan baik dan area yang perlu diperbaiki dari input pengguna
    
    Args:
        form_data: Dictionary dari form input
    
    Returns:
        Dictionary dengan 'good_habits' dan 'needs_improvement' lists
    """
    good_habits = []
    needs_improvement = []
    
    try:
        # 1. Sleep Duration Analysis
        sleep_duration = float(form_data.get('sleep_duration', 7))
        if 7 <= sleep_duration <= 9:
            good_habits.append(f"Durasi tidur optimal ({sleep_duration} jam/hari)")
        elif sleep_duration < 6:
            needs_improvement.append(f"Durasi tidur terlalu pendek ({sleep_duration} jam) - Target: 7-9 jam")
        elif sleep_duration > 9:
            needs_improvement.append(f"Durasi tidur terlalu panjang ({sleep_duration} jam) - Target: 7-9 jam")
        else:  # 6-7 or 9-10
            good_habits.append(f"Durasi tidur cukup baik ({sleep_duration} jam/hari)")
        
        # 2. Sleep Quality Analysis
        quality = int(form_data.get('quality_of_sleep', 7))
        if quality >= 7:
            good_habits.append(f"Kualitas tidur baik ({quality}/10)")
        elif quality <= 4:
            needs_improvement.append(f"Kualitas tidur rendah ({quality}/10) - Target: ≥ 7/10")
        else:  # 5-6
            needs_improvement.append(f"Kualitas tidur perlu ditingkatkan ({quality}/10) - Target: ≥ 7/10")
        
        # 3. Stress Level Analysis
        stress = int(form_data.get('stress_level', 5))
        if stress <= 4:
            good_habits.append(f"Tingkat stres terkendali ({stress}/10)")
        elif stress >= 7:
            needs_improvement.append(f"Tingkat stres tinggi ({stress}/10) - Target: ≤ 4/10")
        else:  # 5-6
            needs_improvement.append(f"Tingkat stres cukup tinggi ({stress}/10) - Pertimbangkan teknik relaksasi")
        
        # 4. Physical Activity Analysis
        activity = int(form_data.get('physical_activity', 50))
        if activity >= 60:
            good_habits.append(f"Aktivitas fisik mencukupi ({activity} menit/hari)")
        elif activity < 30:
            needs_improvement.append(f"Aktivitas fisik kurang ({activity} menit) - Target: ≥ 60 menit/hari")
        else:  # 30-59
            needs_improvement.append(f"Aktivitas fisik cukup ({activity} menit) - Tingkatkan ke 60+ menit/hari")
        
        # 5. BMI Category Analysis
        bmi = form_data.get('bmi_category', 'Normal')
        if bmi in ['Normal', 'Normal Weight']:
            good_habits.append("Kategori BMI normal")
        elif bmi == 'Overweight':
            needs_improvement.append("BMI kategori Overweight - Pertimbangkan program penurunan berat badan")
        elif bmi == 'Obese':
            needs_improvement.append("BMI kategori Obese - Konsultasi dengan ahli gizi disarankan")
        
        # 6. Blood Pressure Analysis
        bp = str(form_data.get('blood_pressure', '120/80'))
        bp_match = re.match(r'^(\d{2,3})/(\d{2,3})$', bp)
        if bp_match:
            systolic = int(bp_match.group(1))
            diastolic = int(bp_match.group(2))
            
            if systolic < 120 and diastolic < 80:
                good_habits.append(f"Tekanan darah normal ({bp})")
            elif systolic >= 140 or diastolic >= 90:
                needs_improvement.append(f"Tekanan darah tinggi ({bp}) - Target: < 120/80")
            elif systolic >= 130 or diastolic >= 85:
                needs_improvement.append(f"Tekanan darah elevated ({bp}) - Pantau dan jaga pola makan")
            else:  # 120-129
                good_habits.append(f"Tekanan darah cukup baik ({bp})")
        
        # 7. Heart Rate Analysis
        heart_rate = int(form_data.get('heart_rate', 70))
        if 60 <= heart_rate <= 75:
            good_habits.append(f"Detak jantung istirahat optimal ({heart_rate} bpm)")
        elif heart_rate > 85:
            needs_improvement.append(f"Detak jantung tinggi ({heart_rate} bpm) - Target: 60-75 bpm")
        elif heart_rate < 60:
            good_habits.append(f"Detak jantung rendah ({heart_rate} bpm) - Indikator kebugaran baik")
        else:  # 76-85
            good_habits.append(f"Detak jantung cukup baik ({heart_rate} bpm)")
        
        # 8. Daily Steps Analysis
        steps = int(form_data.get('daily_steps', 5000))
        if steps >= 10000:
            good_habits.append(f"Langkah harian sangat baik ({steps:,} langkah)")
        elif steps >= 7000:
            good_habits.append(f"Langkah harian baik ({steps:,} langkah)")
        elif steps < 5000:
            needs_improvement.append(f"Langkah harian kurang ({steps:,} langkah) - Target: ≥ 10,000 langkah")
        else:  # 5000-6999
            needs_improvement.append(f"Langkah harian cukup ({steps:,} langkah) - Tingkatkan ke 10,000 langkah")
    
    except Exception as e:
        # Jika terjadi error parsing, skip insights
        pass
    
    return {
        'good_habits': good_habits,
        'needs_improvement': needs_improvement
    }


def preprocess_input_data(form_data: dict) -> np.ndarray:
    """
    Transform form data menjadi format siap prediksi (ROBUST MODEL - no demographics)
    
    Args:
        form_data: Dictionary dari form input (semua string)
    
    Returns:
        Preprocessed numpy array
    """
    # Konversi form data ke format yang sesuai dengan robust model
    # PENTING: Tidak ada Gender dan Occupation
    data = {
        'Age': int(form_data['age']),
        'Sleep Duration': float(form_data['sleep_duration']),
        'Quality of Sleep': int(form_data['quality_of_sleep']),
        'Physical Activity Level': int(form_data['physical_activity']),
        'Stress Level': int(form_data['stress_level']),
        'BMI Category': form_data['bmi_category'],
        'Blood Pressure': form_data['blood_pressure'],
        'Heart Rate': int(form_data['heart_rate']),
        'Daily Steps': int(form_data['daily_steps'])
    }
    
    # Konversi ke DataFrame (array 2D)
    df = pd.DataFrame([data])
    
    # Apply preprocessing
    # 1. Handle missing values
    imputer_dict = PREPROCESSOR['imputer_dict']
    for col, fill_value in imputer_dict.items():
        if col in df.columns:
            df[col].fillna(fill_value, inplace=True)
    
    # 2. Encode categorical
    encoders = PREPROCESSOR['encoders']
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )
    
    # 3. Scale
    scaler = PREPROCESSOR['scaler']
    X_scaled = scaler.transform(df)
    
    return X_scaled


@app.route('/')
def index():
    """Homepage dengan form input"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint prediksi dengan SELEKSI MODEL, INPUT VALIDATION dan CLINICAL OVERRIDE
    Menerima form data, validasi, preprocessing, dan return hasil prediksi
    """
    try:
        # Ambil data dari form
        form_data = request.form.to_dict()
        
        # Ambil parameter pilihan model dari input pengguna
        model_choice = form_data.get('model_choice', 'xgb')
        if model_choice not in MODELS:
            model_choice = 'xgb'
            
        selected_model = MODEL_NAMES.get(model_choice, 'XGBoost')
        active_model = MODELS[model_choice]
        
        # VALIDATE INPUT (OOD prevention)
        is_valid, error_message = validate_input(form_data)
        
        if not is_valid:
            return render_template(
                'index.html',
                error=f"❌ Validasi Input Gagal:\n\n{error_message}",
                input_data=form_data,
                selected_model=selected_model
            )
        
        # Extract clinical metrics for override logic
        sleep_duration = float(form_data.get('sleep_duration', 7))
        quality = int(form_data.get('quality_of_sleep', 7))
        stress = int(form_data.get('stress_level', 5))
        
        # Check for extreme combination (warning)
        warning = None
        if sleep_duration < 5 and quality <= 3 and stress >= 7:
            warning = "⚠️ WARNING: Input shows extreme health indicators. Strongly recommend immediate medical consultation."
        
        # Analyze health insights
        health_insights = analyze_health_insights(form_data)
        
        # Preprocess input
        X_processed = preprocess_input_data(form_data)
        
        # Predict menggunakan model yang dipilih
        prediction = active_model.predict(X_processed)
        pred_id = prediction[0]
        pred_name = TARGET_ENCODER.inverse_transform([pred_id])[0]
        
        # Get prediction probabilities
        pred_proba = active_model.predict_proba(X_processed)
        confidence = float(np.max(pred_proba)) * 100
        
        # ========================================
        # CLINICAL EXPERT SYSTEM OVERRIDE
        # ========================================
        is_overridden = False
        override_reason = None
        
        # Rule 5: Sleep Apnea Override (Overweight/Obese + Hypertension)
        bmi_category = form_data.get('bmi_category', '')
        blood_pressure = str(form_data.get('blood_pressure', ''))
        try:
            sys_bp, dia_bp = map(int, blood_pressure.split('/'))
        except (ValueError, AttributeError):
            sys_bp, dia_bp = 0, 0
            
        if bmi_category in ['Overweight', 'Obese'] and (sys_bp >= 140 or dia_bp >= 90):
            is_overridden = True
            pred_name = 'Sleep Apnea'
            confidence = 99.0
            override_reason = 'Indikator klinis kuat untuk risiko Sleep Apnea (Obesitas + Hipertensi).'
            warning = f"🚨 CLINICAL ALERT: {override_reason}"
        # Rule: Override 'Normal' prediction if clinical metrics are severe
        elif pred_name == 'Normal':
            override_triggered = False
            reasons = []
            
            # Rule 1: Severe sleep deprivation
            if sleep_duration <= 5:
                override_triggered = True
                reasons.append(f"critically low sleep duration ({sleep_duration}h ≤ 5h)")
            
            # Rule 2: Very high stress
            if stress >= 8:
                override_triggered = True
                reasons.append(f"very high stress level ({stress} ≥ 8/10)")
            
            # Rule 3: Very poor sleep quality
            if quality <= 3:
                override_triggered = True
                reasons.append(f"very poor sleep quality ({quality} ≤ 3/10)")
            
            # Rule 4: Combination of moderate issues
            if sleep_duration <= 6 and quality <= 4 and stress >= 7:
                override_triggered = True
                reasons.append("combination of poor sleep, low quality, and high stress")
            
            # Apply override if any rule triggered
            if override_triggered:
                is_overridden = True
                pred_name = 'High Risk - Insomnia'
                confidence = 99.0  # Expert system confidence
                override_reason = "Clinical Expert System Override: " + ", ".join(reasons)
                warning = f"🚨 CLINICAL ALERT: Model predicted 'Normal' but clinical metrics indicate HIGH RISK. {override_reason}"
        
        # Return hasil ke template dengan variabel selected_model
        return render_template(
            'index.html',
            prediction=pred_name,
            confidence=f"{confidence:.2f}",
            input_data=form_data,
            warning=warning,
            is_overridden=is_overridden,
            override_reason=override_reason,
            health_insights=health_insights,
            selected_model=selected_model
        )
    
    except Exception as e:
        return render_template(
            'index.html',
            error=f"❌ Error: {str(e)}",
            input_data=form_data if 'form_data' in locals() else None,
            selected_model=selected_model if 'selected_model' in locals() else 'XGBoost'
        )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint untuk prediksi (JSON input/output)
    WITH STRICT INPUT VALIDATION and CLINICAL OVERRIDE
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
            
        # Ambil parameter pilihan model dari JSON data
        model_choice = data.get('model_choice', 'xgb')
        if model_choice not in MODELS:
            model_choice = 'xgb'
            
        selected_model = MODEL_NAMES.get(model_choice, 'XGBoost')
        active_model = MODELS[model_choice]
        
        # VALIDATE INPUT (OOD prevention) - STRICT
        is_valid, error_message = validate_input(data)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_message
            }), 400
        
        # Extract clinical metrics for override logic
        sleep_duration = float(data.get('sleep_duration', 7))
        quality = int(data.get('quality_of_sleep', 7))
        stress = int(data.get('stress_level', 5))
        
        # Check for extreme combination (warning)
        warning = None
        if sleep_duration < 5 and quality <= 3 and stress >= 7:
            warning = "Extreme health indicators detected. Strongly recommend immediate medical consultation."
        
        # Preprocess
        X_processed = preprocess_input_data(data)
        
        # Predict menggunakan model yang dipilih
        prediction = active_model.predict(X_processed)
        pred_id = int(prediction[0])
        pred_name = TARGET_ENCODER.inverse_transform([pred_id])[0]
        
        # Probabilities
        pred_proba = active_model.predict_proba(X_processed)
        confidence = float(np.max(pred_proba)) * 100
        
        # ========================================
        # CLINICAL EXPERT SYSTEM OVERRIDE
        # ========================================
        is_overridden = False
        override_reason = None
        
        # Rule 5: Sleep Apnea Override (Overweight/Obese + Hypertension)
        bmi_category = data.get('bmi_category', '')
        blood_pressure = str(data.get('blood_pressure', ''))
        try:
            sys_bp, dia_bp = map(int, blood_pressure.split('/'))
        except (ValueError, AttributeError):
            sys_bp, dia_bp = 0, 0
            
        if bmi_category in ['Overweight', 'Obese'] and (sys_bp >= 140 or dia_bp >= 90):
            is_overridden = True
            pred_name = 'Sleep Apnea'
            confidence = 99.0
            override_reason = 'Indikator klinis kuat untuk risiko Sleep Apnea (Obesitas + Hipertensi).'
            warning = override_reason
        # Rule: Override 'Normal' prediction if clinical metrics are severe
        elif pred_name == 'Normal':
            override_triggered = False
            reasons = []
            
            # Rule 1: Severe sleep deprivation
            if sleep_duration <= 5:
                override_triggered = True
                reasons.append(f"critically low sleep duration ({sleep_duration}h)")
            
            # Rule 2: Very high stress
            if stress >= 8:
                override_triggered = True
                reasons.append(f"very high stress level ({stress}/10)")
            
            # Rule 3: Very poor sleep quality
            if quality <= 3:
                override_triggered = True
                reasons.append(f"very poor sleep quality ({quality}/10)")
            
            # Rule 4: Combination of moderate issues
            if sleep_duration <= 6 and quality <= 4 and stress >= 7:
                override_triggered = True
                reasons.append("combination of poor metrics")
            
            # Apply override
            if override_triggered:
                is_overridden = True
                pred_name = 'High Risk - Insomnia'
                confidence = 99.0
                override_reason = "Clinical Expert System Override: " + ", ".join(reasons)
                warning = override_reason
        
        # Return JSON
        response = {
            'success': True,
            'prediction': pred_name,
            'prediction_id': pred_id,
            'confidence': round(confidence, 2),
            'selected_model': selected_model,
            'model_choice': model_choice,
            'all_probabilities': {
                class_name: round(float(prob) * 100, 2)
                for class_name, prob in zip(TARGET_ENCODER.classes_, pred_proba[0])
            },
            'is_overridden': is_overridden
        }
        
        if warning:
            response['warning'] = warning
        
        if override_reason:
            response['override_reason'] = override_reason
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FLASK APP STARTING")
    print("=" * 70)
    print("Access the web interface at: http://127.0.0.1:5000")
    print("API endpoint: http://127.0.0.1:5000/api/predict (POST JSON)")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
