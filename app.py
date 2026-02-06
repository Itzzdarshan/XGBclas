import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# --- 1. LUXURY THEME & BACKGROUND ---
st.set_page_config(page_title="MilkVision Pro", page_icon="🥛", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.85)), 
                          url("https://images.unsplash.com/photo-1528498033373-3c6c08e93d79?q=80&w=1920");
        background-attachment: fixed;
        background-size: cover;
        color: #ffffff;
    }
    
    .main-title {
        text-align: center;
        font-size: 60px;
        font-weight: 900;
        background: linear-gradient(90deg, #bf953f, #fcf6ba, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(212, 175, 55, 0.3);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(10px);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #d4af37, #fcf6ba) !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 50px !important;
        border: none !important;
        width: 100%;
        padding: 10px;
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD DATA ---
@st.cache_resource
def load_milk_pack():
    return joblib.load("milk_model.pkl")

data = load_milk_pack()
model, encoder, metrics, importances = data['model'], data['encoder'], data['metrics'], data['importances']

# --- 3. HEADER ---
st.markdown("<h1 class='main-title'>MILK QUALITY ANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#d4af37; letter-spacing:3px;'>AI-POWERED CLASSIFICATION ENGINE</p>", unsafe_allow_html=True)

# Metric Row
col_a, col_b, col_c = st.columns(3)
col_a.metric("XGBoost Accuracy", f"{metrics['accuracy']*100:.2f}%")
col_b.metric("Model Task", "Classification")
col_c.metric("Parameters", "Optimized (Task 5)")

st.markdown("<br>", unsafe_allow_html=True)

# --- 4. INPUTS ---
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.write("🧪 **Chemical Profile**")
    ph = st.slider("pH Level", 3.0, 9.5, 6.6)
    temp = st.slider("Temperature (°C)", 34, 90, 40)

with c2:
    st.write("👅 **Sensory Profile**")
    taste = st.radio("Taste Quality", ["Bad (0)", "Good (1)"], horizontal=True)
    odor = st.radio("Odor Quality", ["Bad (0)", "Good (1)"], horizontal=True)
    fat = st.radio("Fat Content", ["Low (0)", "Optimal (1)"], horizontal=True)

with c3:
    st.write("👁️ **Visual Profile**")
    turb = st.radio("Turbidity", ["Low (0)", "High (1)"], horizontal=True)
    colour = st.number_input("Colour Value", 240, 255, 254)

st.markdown("</div>", unsafe_allow_html=True)

# --- 5. PREDICTION ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("💎 ANALYZE MILK QUALITY"):
    # Convert radio labels to 0/1
    t_val = 1 if "Good" in taste else 0
    o_val = 1 if "Good" in odor else 0
    f_val = 1 if "Optimal" in fat else 0
    tr_val = 1 if "High" in turb else 0
    
    features = np.array([[ph, temp, t_val, o_val, f_val, tr_val, colour]])
    pred_idx = model.predict(features)[0]
    grade = encoder.inverse_transform([pred_idx])[0].upper()
    
    st.markdown("---")
    res_l, res_r = st.columns(2)
    
    with res_l:
        st.write("### 🏆 Prediction Result")
        color_map = {"HIGH": "#d4af37", "MEDIUM": "#f9d71c", "LOW": "#ff4b4b"}
        st.markdown(f"""
            <div style='background: {color_map.get(grade, "#333")}; padding: 30px; border-radius: 20px; text-align: center;'>
                <h1 style='color: black; margin: 0;'>GRADE: {grade}</h1>
            </div>
        """, unsafe_allow_html=True)
        st.balloons()

    with res_r:
        st.write("### 📊 Decision Driver (Task 4.3)")
        fig = go.Figure(go.Bar(
            x=importances.values, y=importances.index,
            orientation='h', marker=dict(color='#d4af37')
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': 'white'}, height=300)
        st.plotly_chart(fig, use_container_width=True)