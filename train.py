import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#Uyarıları görmemek için yaptım.
warnings.filterwarnings('ignore')

# Klasör yapıları
if not os.path.exists('outputs'): os.makedirs('outputs')
if not os.path.exists('models'): os.makedirs('models')

print("--- 1. VERİ YÜKLEME VE HAZIRLIK ---")
try:
    patients_df = pd.read_csv('patients_augmented.csv')
    services_df = pd.read_csv('hospital_data/services_weekly.csv')
    staff_schedule_df = pd.read_csv('hospital_data/staff_schedule.csv')
    print("Veriler başarıyla yüklendi.")
except FileNotFoundError:
    print("HATA: 'patients_augmented.csv' bulunamadı. Önce generate_data.py çalıştırın.")
    exit()

# Tarih dönüşümleri
patients_df['arrival_date'] = pd.to_datetime(patients_df['arrival_date'])
patients_df['departure_date'] = pd.to_datetime(patients_df['departure_date'])
# stay_length zaten augmented veride var ama garanti olsun diye float yaptım
patients_df['stay_length'] = patients_df['stay_length'].astype(float)

# --- 2. GELİŞMİŞ EDA (YENİ ÖZELLİKLER İÇİN) ---
print("Gelişmiş EDA Grafikleri çiziliyor...")

# Grafik 1: Tanı (Diagnosis) bazlı Yatış Süresi (Boxplot) - Yeni özelliğin etkisi
plt.figure(figsize=(12, 6))
sns.boxplot(data=patients_df, x='service', y='stay_length', hue='diagnosis')
plt.title('Length of Stay Distribution by Diagnosis & Service')
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('outputs/1_eda_diagnosis_los.png')
plt.close()

# Grafik 2: Şiddet Puanı (Severity) vs Risk Durumu - Yeni özelliğin etkisi
plt.figure(figsize=(10, 6))
sns.kdeplot(data=patients_df[patients_df['stay_length'] <= 7], x='severity_score', shade=True, label='Normal Stay',
            color='green')
sns.kdeplot(data=patients_df[patients_df['stay_length'] > 7], x='severity_score', shade=True,
            label='Long Stay (High Risk)', color='red')
plt.title('Impact of Severity Score on Length of Stay Risk')
plt.legend()
plt.savefig('outputs/2_eda_severity_risk.png')
plt.close()

# --- 3. TASK A: REGRESSION (LOS Prediction) ---
print("\n--- Task A: Regression (Linear, RF, XGBoost) ---")

# Feature Engineering
categorical_cols = ['service', 'gender', 'diagnosis']
df_encoded = pd.get_dummies(patients_df, columns=categorical_cols, drop_first=True)

ignore_cols = ['patient_id', 'name', 'arrival_date', 'departure_date', 'stay_length', 'satisfaction']
feature_cols = [c for c in df_encoded.columns if c not in ignore_cols]

X = df_encoded[feature_cols]
y = df_encoded['stay_length']
y_log = np.log1p(y)  # Log Transformation

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Modeller: Linear, RF ve şimdi XGBoost
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                                          random_state=42)
}

best_reg_model = None
best_r2 = -float('inf')

print(f"{'Model':<25} | {'MAE':<10} | {'R2 Score':<10}")
print("-" * 50)

for name, model in reg_models.items():
    model.fit(X_train, y_train_log)
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_test_actual = np.expm1(y_test_log)

    mae = mean_absolute_error(y_test_actual, preds)
    r2 = r2_score(y_test_actual, preds)

    print(f"{name:<25} | {mae:<10.2f} | {r2:<10.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_reg_model = model

# --- SHAP ANALİZİ (EXPLAINABILITY)  ---
print("\n--- SHAP Explainability Analysis ---")
# XGBoost veya RF için SHAP değerlerini hesapladım
explainer = shap.TreeExplainer(best_reg_model)
# Hesaplama hızı için test setinden 100 örnek alıyoruz
shap_values = explainer.shap_values(X_test.iloc[:100])

plt.figure()
shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
plt.title(f"SHAP Summary for {type(best_reg_model).__name__}")
plt.savefig('outputs/3_shap_summary_regression.png', bbox_inches='tight')
plt.close()
print("SHAP grafiği kaydedildi (outputs/3_shap_summary_regression.png).")

# --- 4. TASK B: CLASSIFICATION ---
print("\n--- Task B: Classification (High Risk > 7 Days) ---")
y_clf = (patients_df['stay_length'] > 7).astype(int)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# XGBoost Classifier ekliyoruz
clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

best_clf_model = None
best_f1 = 0

print(f"{'Model':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
print("-" * 50)

for name, model in clf_models.items():
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)
    acc = accuracy_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds)

    print(f"{name:<25} | {acc:<10.2f} | {f1:<10.2f}")

    if f1 > best_f1:
        best_f1 = f1
        best_clf_model = model

# --- 5. TASK C: TIME SERIES (SARIMA vs LSTM)  ---
print("\n--- Task C: Time Series Forecasting (SARIMA vs LSTM) ---")

ts_data = services_df[services_df['service'] == 'emergency'].sort_values('week').reset_index(drop=True)
data_values = ts_data['patients_request'].values.astype(float)

# Veriyi Hazırla
train_size = int(len(data_values) * 0.8)
train_data, test_data = data_values[:train_size], data_values[train_size:]

# 1. SARIMA MODELİ
try:
    sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.predict(start=len(train_data), end=len(data_values) - 1)
    sarima_mae = mean_absolute_error(test_data, sarima_pred)
    print(f"SARIMA MAE: {sarima_mae:.2f}")
except:
    print("SARIMA hatası.")
    sarima_pred = np.zeros(len(test_data))

# 2. LSTM MODELİ (Derin Öğrenme)
# LSTM veriyi 0-1 arasına sıkıştırmayı sever (Scaling)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_values.reshape(-1, 1))


# LSTM için veri hazırlama fonksiyonu (Sliding Window)
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 3  # Geçmiş 3 haftaya bakarak gelecek haftayı tahmin et
X_lstm, y_lstm = create_dataset(scaled_data, look_back)

# Train/Test Split (LSTM için)
train_size_lstm = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

# LSTM [samples, time steps, features] formatı ister
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], 1, X_test_lstm.shape[1]))

# Model Mimarisi
model_lstm = Sequential()
model_lstm.add(LSTM(50, input_shape=(1, look_back)))  # 50 Nöronlu LSTM katmanı
model_lstm.add(Dropout(0.2))  # Overfitting engelleme
model_lstm.add(Dense(1))  # Çıktı katmanı
model_lstm.compile(loss='mean_squared_error', optimizer='adam')

# Modeli Eğit (Verbose=0 ile çıktı kirliliğini önle)
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=1, verbose=0)

# Tahmin
lstm_pred_scaled = model_lstm.predict(X_test_lstm)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)  # Skalayı geri çevir
y_test_actual_lstm = scaler.inverse_transform([y_test_lstm])

lstm_mae = mean_absolute_error(y_test_actual_lstm[0], lstm_pred[:, 0])
print(f"LSTM Model MAE: {lstm_mae:.2f}")

# Karşılaştırma Grafiği
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(data_values)), data_values, label='Actual Data', color='gray', alpha=0.5)
# Test seti indeksleri
test_indices = np.arange(len(train_data), len(data_values))
plt.plot(test_indices, sarima_pred, label=f'SARIMA (MAE: {sarima_mae:.1f})', color='red', linestyle='--')
# LSTM indeksleri (Look back nedeniyle biraz daha kısa olabilir, uyduruyoruz)
lstm_indices = np.arange(len(data_values) - len(lstm_pred), len(data_values))
plt.plot(lstm_indices, lstm_pred, label=f'LSTM (MAE: {lstm_mae:.1f})', color='blue', linestyle='-.')

plt.title('Time Series Benchmarking: SARIMA vs LSTM ')
plt.legend()
plt.savefig('outputs/6_timeseries_benchmark.png')
plt.close()

# --- 6. MODELLERİ KAYDETME ---
print("\n--- Modeller Kaydediliyor ---")
# En iyi Regresyon modelini kaydet
joblib.dump(best_reg_model, 'models/los_regression_model.pkl')
# En iyi Sınıflandırma modelini kaydet
joblib.dump(best_clf_model, 'models/risk_classification_model.pkl')
# Sütunları kaydet
joblib.dump(feature_cols, 'models/feature_columns.pkl')

print("BÜTÜN İŞLEMLER (XGBoost, LSTM, SHAP dahil) TAMAMLANDI.")
print("Lütfen 'outputs/' klasöründeki yeni grafikleri inceleyin.")