import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="Dubai Property Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size:2.4rem !important; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size:1.1rem !important; color: #4B5563; margin-bottom: 2rem; }
    .card-valuation { background-color: #F8FAFC; padding: 25px; border-radius: 12px; border-left: 6px solid #10B981; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #F1F5F9; border-radius: 4px 4px 0 0; padding: 10px 20px; font-weight: 600;}
    .stTabs [aria-selected="true"] { background-color: #1E3A8A; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# 2. Load Assets
@st.cache_resource
def load_analytics_assets():
    model = joblib.load('dubai_xgb_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    area_price_map = joblib.load('area_price_map.pkl')
    return model, model_columns, area_price_map

try:
    model, model_columns, area_price_map = load_analytics_assets()
    area_prefix = "area_name_en_"
    neighborhoods = sorted([col.replace(area_prefix, "") for col in model_columns if col.startswith(area_prefix)])
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# 3. Geo Map
geo_map = {
    "Downtown Dubai": [25.1972, 55.2744],
    "Dubai Marina": [25.0805, 55.1403],
    "Jumeirah Village Circle (JVC)": [25.0601, 55.2058],
    "Business Bay": [25.1843, 55.2723],
    "Palm Jumeirah": [25.1124, 55.1390],
    "Dubai South": [24.8967, 55.1500],
    "Dubai Hills Estate": [25.1097, 55.2635]
}

# 4. Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluent/100/000000/real-estate.png", width=80)
    st.markdown("### **System Architecture**")
    st.info("Built on an XGBoost Regression framework analyzing Dubai Land Department (DLD) transactions.")
    st.markdown("---")
    st.markdown("### **Model Hyperparameters**")
    st.markdown("- **Estimators:** 300 Trees")
    st.markdown("- **Max Depth:** 8")
    st.markdown("- **Learning Rate:** 0.05")
    st.markdown("- **Training Method:** Histogram")

# 5. Header
st.markdown('<div class="main-header">🏙️ Dubai Property Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-driven secondary market appraisal and financial forecasting framework</div>', unsafe_allow_html=True)

# 6. Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Valuation Engine", "💰 Financial Forecaster", "📍 Location Intelligence"])

# ----------------- TAB 1 -----------------
with tab1:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📋 Structural Parameters")
        property_type = st.segmented_control("Structure Type", ["Apartment", "Villa"], default="Apartment")
        procedure_area = st.slider("Total Built-up Area (Sqft)", min_value=300, max_value=15000, value=1200, step=50)
        rooms_en = st.number_input("Number of Bedrooms (0 = Studio)", min_value=0, max_value=10, value=2, step=1)
        has_parking = st.toggle("Includes Dedicated Parking Space", value=True)

    with col2:
        st.subheader("📍 Geographic Context")
        default_index = neighborhoods.index("Business Bay") if "Business Bay" in neighborhoods else 0
        selected_area = st.selectbox("Target Neighborhood Cluster", neighborhoods, index=default_index)

        # NEW: Transaction year input
        transaction_year = st.slider("Transaction Year", min_value=2010, max_value=2025, value=2024, step=1)
        transaction_month = st.slider("Transaction Month", min_value=1, max_value=12, value=6, step=1)

        st.markdown("---")
        st.markdown("### 📊 Input Matrix Summary")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Unit Density", f"{(procedure_area / (rooms_en + 1)):,.1f} sqft/room")
        m_col2.metric("Configuration", f"{rooms_en} Bed {property_type}")

    st.markdown("---")

    if st.button("🚀 Execute Market Appraisal", use_container_width=True):
        with st.spinner("Processing machine learning vectors..."):

            # Build input row
            input_data = pd.DataFrame(0, index=[0], columns=model_columns)

            # Core features
            input_data['rooms_en'] = rooms_en
            input_data['has_parking'] = 1 if has_parking else 0
            input_data['size_per_room'] = procedure_area / (rooms_en + 1)

            # NEW: Log area (if model was trained with log area)
            if 'procedure_area_log' in model_columns:
                input_data['procedure_area_log'] = np.log1p(procedure_area)
            elif 'procedure_area' in model_columns:
                input_data['procedure_area'] = procedure_area

            # NEW: Time features
            if 'transaction_year' in model_columns:
                input_data['transaction_year'] = transaction_year
            if 'transaction_month' in model_columns:
                input_data['transaction_month'] = transaction_month

            # NEW: Area mean price
            if 'area_mean_price' in model_columns:
                input_data['area_mean_price'] = area_price_map.get(selected_area, np.mean(list(area_price_map.values())))

            # Property type
            if property_type == "Villa" and 'property_type_en_Villa' in model_columns:
                input_data['property_type_en_Villa'] = 1

            # Area dummy
            target_area_col = area_prefix + selected_area
            if target_area_col in model_columns:
                input_data[target_area_col] = 1

            # Predict
            predicted_price = model.predict(input_data)[0]
            st.session_state['predicted_price'] = predicted_price

            # Display
            st.markdown(f"""
            <div class="card-valuation">
                <h4 style='margin-top:0; color:#4B5563;'>🎯 AI Predicted Market Value</h4>
                <h1 style='color:#1E3A8A; margin: 10px 0;'>AED {predicted_price:,.2f}</h1>
                <p style='margin:0; color:#64748B; font-weight:500;'>
                    Implied Rate: AED {predicted_price / procedure_area:,.2f} per Sqft
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.success("Appraisal complete! Check the **Financial Forecaster** tab for mortgage calculations.")

# ----------------- TAB 2 -----------------
with tab2:
    st.subheader("🏦 Mortgage & Investment Calculator")

    if 'predicted_price' in st.session_state:
        pred_price = st.session_state['predicted_price']
        st.info(f"Using AI Predicted Property Value: **AED {pred_price:,.2f}**")

        f_col1, f_col2 = st.columns([1, 1])

        with f_col1:
            down_payment_pct = st.slider("Down Payment (%)", min_value=15, max_value=80, value=20, step=1)
            interest_rate = st.slider("Annual Interest Rate (%)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
            loan_tenure = st.selectbox("Loan Tenure (Years)", [5, 10, 15, 20, 25, 30], index=4)

        with f_col2:
            down_payment_amount = pred_price * (down_payment_pct / 100)
            principal = pred_price - down_payment_amount
            monthly_rate = (interest_rate / 100) / 12
            n_payments = loan_tenure * 12

            if monthly_rate > 0:
                emi = principal * monthly_rate * ((1 + monthly_rate)**n_payments) / (((1 + monthly_rate)**n_payments) - 1)
            else:
                emi = principal / n_payments

            st.markdown("### Financial Breakdown")
            st.metric("Initial Down Payment Required", f"AED {down_payment_amount:,.2f}")
            st.metric("Total Loan Amount (Principal)", f"AED {principal:,.2f}")
            st.markdown("---")
            st.metric("Estimated Monthly EMI", f"AED {emi:,.2f}", delta="Monthly Outflow", delta_color="inverse")
    else:
        st.warning("⚠️ Please execute a valuation in the 'Valuation Engine' tab first.")

# ----------------- TAB 3 -----------------
with tab3:
    st.subheader("📍 Geospatial Market Mapping")

    current_area = selected_area

    if current_area in geo_map:
        coords = geo_map[current_area]
        st.success(f"Mapping coordinates for: **{current_area}**")
    else:
        coords = [25.2048, 55.2708]
        st.info(f"Showing general Dubai area. Exact coordinates for '{current_area}' not mapped.")

    map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
    st.map(map_data, zoom=11, use_container_width=True)