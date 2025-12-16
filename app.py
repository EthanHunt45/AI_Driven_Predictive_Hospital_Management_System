import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

# Sayfa Ayarları
st.set_page_config(page_title="AI Hospital Management", layout="wide", page_icon="🏥")

# --- MODELLERİ YÜKLEME ---
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load('models/los_regression_model.pkl')
        clf_model = joblib.load('models/risk_classification_model.pkl')
        try:
            staff_model = joblib.load('models/staff_forecast_model.pkl')
        except:
            staff_model = None
        feature_cols = joblib.load('models/feature_columns.pkl')
        return reg_model, clf_model, staff_model, feature_cols
    except FileNotFoundError:
        st.error("Modeller bulunamadı! Lütfen önce train.py dosyasını çalıştırın.")
        return None, None, None, None


reg_model, clf_model, staff_model, feature_cols = load_models()

# --- KENAR ÇUBUĞU (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
st.sidebar.title("Hospital Management System")
st.sidebar.info("AI-Driven Hospital Management System")
avg_daily_cost = st.sidebar.number_input("Average Daily Cost per Patient ($)", value=1500, step=100)

# --- ANA EKRAN ---
st.title("🏥 AI-Driven Hospital Decision Support System")
st.markdown("Predictive Analytics, Financial Estimation & Scenario Simulation")

# Sekmeler
tab1, tab2, tab3, tab4, tab5= st.tabs([
    "📝 Single Prediction",
    "📂 Batch Prediction (Excel)",
    "📈 Simulation & What-If",
    "📊 Analytics Dashboard",
    "👨‍⚕️ Staff Management",
])

# --- TAB 1: TEKLİ HASTA TAHMİNİ ---
with tab1:
    st.header("Patient Admission & Cost Estimation")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        service = st.selectbox("Department", ["surgery", "general_medicine", "emergency", "ICU"])
        if service == 'surgery':
            diagnosis_opts = ['Appendectomy', 'Hip Replacement', 'Hernia Repair', 'Heart Bypass']
        elif service == 'general_medicine':
            diagnosis_opts = ['Flu', 'Pneumonia', 'Diabetes Crisis', 'Hypertension']
        elif service == 'emergency':
            diagnosis_opts = ['Trauma', 'Food Poisoning', 'Cardiac Arrest', 'Minor Injury']
        else:
            diagnosis_opts = ['Sepsis', 'Respiratory Failure', 'Post-Op Critical', 'Stroke']
        diagnosis = st.selectbox("Diagnosis", diagnosis_opts)

    with col3:
        severity = st.slider("Severity Score (1-10)", 1, 10, 5, help="1: Mild, 10: Critical")

        # Aylar
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
            "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month_name = st.selectbox("Arrival Month", list(month_map.keys()))
        arrival_month = month_map[selected_month_name]  # Model için sayıya çeviriyoruz
        # -------------------------------

    if st.button("Predict LOS, Risk & Cost", type="primary"):
        if reg_model is not None:
            # Input Hazırlama
            input_data = pd.DataFrame({
                'age': [age], 'severity_score': [severity], 'arrival_month': [arrival_month],
                'service': [service], 'gender': [gender], 'diagnosis': [diagnosis]
            })

            # One-Hot Encoding
            input_encoded = pd.get_dummies(input_data, columns=['service', 'gender', 'diagnosis'])
            input_final = pd.DataFrame(0, index=[0], columns=feature_cols)
            common = list(set(input_encoded.columns) & set(feature_cols))
            input_final[common] = input_encoded[common]

            # Tahminler
            pred_log = reg_model.predict(input_final)[0]
            pred_days = np.expm1(pred_log)
            pred_risk = clf_model.predict(input_final)[0]
            pred_prob = clf_model.predict_proba(input_final)[0][1]

            # Maliyet Hesabı
            estimated_cost = pred_days * avg_daily_cost

            # Sonuç Gösterimi
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Predicted LOS", f"{pred_days:.1f} Days")
            with c2:
                risk_color = "inverse" if pred_risk == 0 else "normal"
                st.metric("Risk Probability", f"{pred_prob:.1%}", delta="High Risk" if pred_risk == 1 else "Normal",
                          delta_color=risk_color)
            with c3:
                st.metric("Estimated Cost", f"${estimated_cost:,.2f}")

            if pred_risk == 1:
                st.error("⚠️ ALERT: High resource usage expected. Consider prioritizing bed allocation.")
            else:
                st.success("✅ Routine admission recommended.")

# --- TAB 2: TOPLU TAHMİN (BATCH) ---
with tab2:
    st.header("Upload Patient Data (CSV/Excel)")
    st.write("Upload a CSV file with columns: age, gender, service, diagnosis, severity_score")

    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview:", batch_df.head())

            if st.button("Run Batch Prediction"):
                results = []
                progress_bar = st.progress(0)

                for i, row in batch_df.iterrows():
                    input_data = pd.DataFrame([row])
                    if 'arrival_month' not in input_data.columns:
                        input_data['arrival_month'] = 1
                    input_encoded = pd.get_dummies(input_data, columns=['service', 'gender', 'diagnosis'])
                    input_final = pd.DataFrame(0, index=[0], columns=feature_cols)
                    common = list(set(input_encoded.columns) & set(feature_cols))
                    input_final[common] = input_encoded[common]
                    input_final = input_final.astype(float)

                    d_log = reg_model.predict(input_final)[0]
                    days = np.expm1(d_log)
                    risk_prob = clf_model.predict_proba(input_final)[0][1]

                    results.append({
                        'Patient_ID': i,
                        'Predicted_LOS': round(days, 2),
                        'High_Risk_Prob': round(risk_prob, 4),
                        'Est_Cost': round(days * avg_daily_cost, 2)
                    })
                    progress_bar.progress((i + 1) / len(batch_df))

                res_df = pd.DataFrame(results)
                st.success("Batch Prediction Complete!")
                st.dataframe(res_df)
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# --- TAB 3: SENARYO VE SİMÜLASYON (WHAT-IF) ---
with tab3:
    st.header("⚡ What-If Analysis: Resource Planning")
    col_sim1, col_sim2 = st.columns([1, 2])

    with col_sim1:
        st.subheader("Simulation Parameters")
        demand_increase = st.slider("📈 Increase in Patient Demand (%)", 0, 100, 20)
        staff_efficiency = st.slider("⚡ Staff Efficiency (Patients/Dr)", 5, 20, 10)
        base_patients = 1000

    with col_sim2:
        st.subheader("Projected Impact")
        new_demand = base_patients * (1 + demand_increase / 100)
        required_doctors = new_demand / staff_efficiency

        fig_sim, ax_sim = plt.subplots(figsize=(8, 4))
        categories = ['Current Demand', 'Simulated Demand']
        values = [base_patients, new_demand]
        colors = ['gray', 'red']
        ax_sim.bar(categories, values, color=colors)
        ax_sim.set_ylabel("Weekly Patients")
        ax_sim.set_title(f"Demand Surge Simulation (+{demand_increase}%)")
        st.pyplot(fig_sim)

        st.metric("Projected Weekly Patients", int(new_demand), delta=f"{int(new_demand - base_patients)}")
        st.metric("Required Doctors", int(required_doctors), delta="Calculated based on efficiency")

# --- TAB 4: ANALİTİK PANELİ ---
with tab4:
    st.header("Global Analytics")
    try:
        df = pd.read_csv('patients_augmented.csv')
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Severity Distribution")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['severity_score'], kde=True, ax=ax1, color="purple")
            st.pyplot(fig1)
        with c2:
            st.subheader("LOS by Diagnosis")
            fig2, ax2 = plt.subplots()
            top_diag = df['diagnosis'].value_counts().nlargest(5).index
            sns.boxplot(data=df[df['diagnosis'].isin(top_diag)], x='diagnosis', y='stay_length', ax=ax2)
            plt.xticks(rotation=45)
            st.pyplot(fig2)
    except:
        st.warning("Data not found.")

# --- TAB 5: PERSONEL YÖNETİMİ ---
with tab5:
    st.header("👨‍⚕️ Staff Management System")

    staff_file = 'hospital_data/staff.csv'

    try:
        staff_df = pd.read_csv(staff_file)
    except FileNotFoundError:
        staff_df = pd.DataFrame(columns=['staff_id', 'staff_name', 'role', 'service'])

    # Üst Kısım: Personel Listesi
    st.subheader("Current Staff List")
    st.dataframe(staff_df, use_container_width=True)

    st.divider()

    # --- EKLEME/SİLME ve SCHEDULER ---
    col_mgmt, col_schedule = st.columns([1, 1])

    with col_mgmt:
        st.subheader("🛠️ Manage Staff")

        # Sekmeli Ekle/Sil
        sub_tab_add, sub_tab_remove = st.tabs(["Add Staff", "Remove Staff"])

        with sub_tab_add:
            with st.form("add_staff_form"):
                new_name = st.text_input("Full Name")
                new_role = st.selectbox("Role", ["doctor", "nurse", "nursing_assistant", "admin"])
                new_service = st.selectbox("Department", ["emergency", "surgery", "general_medicine", "ICU"])
                submitted_add = st.form_submit_button("Add Staff")

                if submitted_add and new_name:
                    new_id = f"STF-{str(uuid.uuid4())[:8]}"
                    new_row = pd.DataFrame(
                        [{"staff_id": new_id, "staff_name": new_name, "role": new_role, "service": new_service}])
                    staff_df = pd.concat([staff_df, new_row], ignore_index=True)
                    staff_df.to_csv(staff_file, index=False)
                    st.success(f"Added: {new_name}")
                    st.rerun()

        with sub_tab_remove:
            if not staff_df.empty:
                staff_options = staff_df.apply(lambda x: f"{x['staff_name']} ({x['staff_id']})", axis=1)
                selected_staff = st.selectbox("Select Staff", staff_options)
                if st.button("Remove Selected"):
                    staff_id_to_remove = selected_staff.split("(")[-1].replace(")", "")
                    staff_df = staff_df[staff_df['staff_id'] != staff_id_to_remove]
                    staff_df.to_csv(staff_file, index=False)
                    st.success("Removed.")
                    st.rerun()

    with col_schedule:
        st.subheader("📅 Weekly Schedule View")
        try:
            schedule_df = pd.read_csv('hospital_data/staff_schedule.csv')
            # Haftaları al
            weeks = sorted(schedule_df['week'].unique())

            # Hafta Seçimi Slider
            selected_week = st.select_slider("Select Week to View", options=weeks, value=weeks[0])

            # O haftaya ait veriyi filtrele
            weekly_view = schedule_df[schedule_df['week'] == selected_week]

            # Göster
            st.write(f"**Week {selected_week} Roster**")
            st.dataframe(weekly_view, use_container_width=True, hide_index=True)

        except FileNotFoundError:
            st.warning("staff_schedule.csv not found.")
        except Exception as e:
            st.error(f"Error loading schedule: {e}")
