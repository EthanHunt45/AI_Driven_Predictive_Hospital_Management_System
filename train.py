import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score,
                             confusion_matrix)
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Gereksiz uyarıları kapatalım
warnings.filterwarnings('ignore')

# Çıktı klasörü kontrolü
if not os.path.exists('outputs'):
    os.makedirs('outputs')

print("--- 1. DATA LOADING & PREPARATION ---")

# Tüm veri setlerini yüklüyoruz
try:
    patients_df = pd.read_csv('hospital_data/patients.csv')
    services_df = pd.read_csv('hospital_data/services_weekly.csv')
    staff_schedule_df = pd.read_csv('hospital_data/staff_schedule.csv')
    print("All datasets (Patients, Services, Staff) loaded successfully.")
except FileNotFoundError:
    print(
        "ERROR: CSV files not found. Please ensure 'patients.csv', 'services_weekly.csv', and 'staff_schedule.csv' are in the same directory.")
    exit()

# Tarih ve LOS hesaplamaları
patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])
patients_df['LOS'] = (patients_df['departure_date'] - patients_df['arrival_date']).dt.days
patients_df['arrival_month'] = patients_df['arrival_date'].dt.month

# --- 2. EXPLORATORY DATA ANALYSIS (EDA) ---
print("Generating EDA plots...")

# Grafik 1: Yaş Dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(patients_df['age'], bins=20, kde=True, color='skyblue')
plt.title('Patient Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('outputs/1_eda_age_distribution.png')
plt.close()

# Grafik 2: Servis Dağılımı
plt.figure(figsize=(10, 6))
sns.countplot(data=patients_df, x='service', palette='viridis')
plt.title('Patient Count by Service')
plt.xlabel('Service')
plt.ylabel('Patient Count')
plt.savefig('outputs/2_eda_service_distribution.png')
plt.close()

# Grafik 3: Personel Dağılımı (YENİ)
# Hangi serviste toplam kaç vardiya (shift) yapılmış?
plt.figure(figsize=(10, 6))
staff_counts = staff_schedule_df[staff_schedule_df['present'] == 1].groupby('service')['staff_id'].count().reset_index()
sns.barplot(data=staff_counts, x='service', y='staff_id', palette='magma')
plt.title('Total Staff Shifts by Service')
plt.xlabel('Service')
plt.ylabel('Total Shifts (Present)')
plt.savefig('outputs/3_eda_staff_distribution.png')
plt.close()

# --- 3. TASK A: REGRESSION (Predicting Length of Stay) ---
print("\n--- Task A: Regression Models (LOS Prediction) ---")

patients_encoded = pd.get_dummies(patients_df, columns=['service'], drop_first=True)
features_reg = ['age', 'arrival_month'] + [col for col in patients_encoded.columns if 'service_' in col]
X = patients_encoded[features_reg]
y = patients_encoded['LOS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
}

print(f"{'Model':<25} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
print("-" * 65)

for name, model in reg_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name:<25} | {mae:<10.2f} | {rmse:<10.2f} | {r2:<10.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'{name}: Actual vs Predicted LOS')
    plt.xlabel('Actual LOS (Days)')
    plt.ylabel('Predicted LOS (Days)')
    plt.savefig(f'outputs/4_reg_scatter_{name.replace(" ", "_").lower()}.png')
    plt.close()

# --- 4. TASK B: CLASSIFICATION (Predicting High Risk > 7 Days) ---
print("\n--- Task B: Classification Models (High Risk Prediction) ---")

y_clf = (patients_encoded['LOS'] > 7).astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print(f"{'Model':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
print("-" * 50)

for name, model in clf_models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict(X_test_c)

    acc = accuracy_score(y_test_c, y_pred)
    f1 = f1_score(y_test_c, y_pred)

    print(f"{name:<25} | {acc:<10.2f} | {f1:<10.2f}")

    cm = confusion_matrix(y_test_c, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'High Risk'],
                yticklabels=['Normal', 'High Risk'])
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'outputs/5_clf_cm_{name.replace(" ", "_").lower()}.png')
    plt.close()

# --- 5. TASK C: TIME SERIES (Demand Forecasting) ---
print("\n--- Task C: Demand Forecasting (SARIMA) ---")

ts_data = services_df[services_df['service'] == 'emergency'].sort_values('week').reset_index(drop=True)
train_size = int(len(ts_data) * 0.8)
train_ts, test_ts = ts_data.iloc[:train_size], ts_data.iloc[train_size:]

try:
    model_sarima = SARIMAX(train_ts['patients_request'], order=(1, 1, 1))
    model_fit = model_sarima.fit(disp=False)
    sarima_pred_full = model_fit.predict(start=0, end=len(ts_data) - 1)

    test_pred_only = sarima_pred_full.iloc[len(train_ts):]
    mae_ts = mean_absolute_error(test_ts['patients_request'], test_pred_only)
    print(f"SARIMA Model MAE (Test Set Only): {mae_ts:.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(ts_data['week'], ts_data['patients_request'], label='Actual Data', color='green', linewidth=2, alpha=0.7)
    plt.plot(ts_data['week'], sarima_pred_full, label='SARIMA Prediction', color='red', linestyle='--', linewidth=2)
    plt.axvline(x=train_ts['week'].iloc[-1], color='black', linestyle=':', label='Train/Test Split')
    plt.title('Emergency Service Weekly Demand Forecast (Full Dataset)')
    plt.xlabel('Week')
    plt.ylabel('Patient Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/6_ts_forecast_full_year.png')
    plt.close()
except Exception as e:
    print(f"Error in Task C: {e}")

# --- 6. TASK D: STAFF FORECASTING (YENİ BÖLÜM) ---
print("\n--- Task D: Staff Schedule Forecasting (Random Forest) ---")

# Veri Hazırlığı:
# 1. Sadece 'Emergency' servisindeki 'Doctor'ları filtreliyoruz (Örnek senaryo)
# 2. Haftalık toplam çalışan doktor sayısını buluyoruz.
staff_ts = staff_schedule_df[
    (staff_schedule_df['service'] == 'emergency') &
    (staff_schedule_df['role'] == 'doctor')
    ].groupby('week')['present'].sum().reset_index()

# Basit bir regresyon için özellikler (Features) üretiyoruz
# Model: Gelecek haftaki personel sayısını, sadece 'Hafta Numarası'na bakarak tahmin etmeye çalışacak.
# (Daha karmaşık modellerde 'Lag' (geçmiş veriler) kullanılabilir)
X_staff = staff_ts[['week']]
y_staff = staff_ts['present']

# Eğitim ve Test ayrımı
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_staff, y_staff, test_size=0.2, shuffle=False)

# Random Forest Modeli
rf_staff = RandomForestRegressor(n_estimators=100, random_state=42)
rf_staff.fit(X_train_s, y_train_s)

# Tahminler (Tüm veri seti üzerinde görselleştirme için)
y_pred_staff_full = rf_staff.predict(X_staff)

# Metrikler (Sadece Test seti üzerinde)
y_pred_test_s = rf_staff.predict(X_test_s)
mae_staff = mean_absolute_error(y_test_s, y_pred_test_s)

print(f"Target: Weekly Doctor Count in Emergency")
print(f"Model: Random Forest Regressor")
print(f"MAE (Mean Absolute Error): {mae_staff:.2f} shifts")

# Grafik Çizimi
plt.figure(figsize=(14, 7))
# Gerçek Veri
plt.plot(staff_ts['week'], staff_ts['present'], label='Actual Staff Count', color='purple', linewidth=2, marker='o',
         markersize=4)
# Tahmin
plt.plot(staff_ts['week'], y_pred_staff_full, label='Predicted Staff Count (RF)', color='orange', linestyle='--',
         linewidth=2)
# Ayrım Çizgisi
plt.axvline(x=X_train_s['week'].iloc[-1], color='black', linestyle=':', label='Train/Test Split')

plt.title('Staff Capacity Forecast: Emergency Doctors (Weekly)')
plt.xlabel('Week')
plt.ylabel('Number of Doctors Present')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/7_staff_forecast.png')
plt.close()

print("\n" + "=" * 40)
print("ALL TASKS COMPLETED SUCCESSFULLY.")
print("Check 'outputs/' folder for 7 generated graphs.")
print("=" * 40)