import pandas as pd
import numpy as np
import random

# 1. Mevcut veriyi yükle
try:
    df = pd.read_csv('hospital_data/patients.csv')
    print("Orijinal veri yüklendi.")
except FileNotFoundError:
    print("HATA: 'patients.csv' dosyası bulunamadı.")
    exit()

# Tarih formatlarını düzelt
df['arrival_date'] = pd.to_datetime(df['arrival_date'])

# Orijinal dosyada gender yok rastgele ekledim.
if 'gender' not in df.columns:
    np.random.seed(42)
    df['gender'] = np.random.choice(['Male', 'Female'], size=len(df))
    print("Eksik 'gender' sütunu rastgele verilerle oluşturuldu.")

# 2. Tıbbi Mantık Haritası
diagnosis_map = {
    'surgery': [('Appendectomy', 2, 4), ('Hip Replacement', 5, 10), ('Hernia Repair', 1, 3), ('Heart Bypass', 7, 14)],
    'general_medicine': [('Flu', 2, 5), ('Pneumonia', 4, 8), ('Diabetes Crisis', 3, 7), ('Hypertension', 1, 3)],
    'emergency': [('Trauma', 1, 5), ('Food Poisoning', 1, 2), ('Cardiac Arrest', 5, 12), ('Minor Injury', 0, 1)],
    'ICU': [('Sepsis', 10, 20), ('Respiratory Failure', 7, 15), ('Post-Op Critical', 5, 10), ('Stroke', 10, 25)]
}

new_diagnoses = []
new_severities = []
new_stay_lengths = []

np.random.seed(42)

for index, row in df.iterrows():
    service = row['service']
    age = row['age']

    # Tanı Atama
    if service in diagnosis_map:
        diag_info = random.choice(diagnosis_map[service])
        diagnosis, min_d, max_d = diag_info
    else:
        diagnosis, min_d, max_d = 'Other', 2, 5

    # Şiddet Puanı (Severity)
    base_sev = np.random.randint(1, 7)
    if age > 60: base_sev += np.random.randint(1, 4)
    severity = min(base_sev, 10)

    # LOS Hesaplama
    base_los = np.random.randint(min_d, max_d + 1)
    sev_impact = (severity - 3) * 0.8 if severity > 3 else 0
    age_impact = 0.05 * age if age > 50 else 0
    noise = np.random.normal(0, 0.5)

    final_los = base_los + sev_impact + age_impact + noise
    final_los = max(1, round(final_los))

    new_diagnoses.append(diagnosis)
    new_severities.append(severity)
    new_stay_lengths.append(final_los)

# Yeni sütunları ekleim.
df['diagnosis'] = new_diagnoses
df['severity_score'] = new_severities
df['stay_length'] = new_stay_lengths
df['departure_date'] = df['arrival_date'] + pd.to_timedelta(df['stay_length'], unit='D')

# Kaydet
df.to_csv('patients_augmented.csv', index=False)
print("BAŞARILI: 'patients_augmented.csv' (Gender sütunu ile) oluşturuldu.")