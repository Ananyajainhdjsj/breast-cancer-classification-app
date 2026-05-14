import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="Breast Cancer Classification System",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elegant, professional design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    :root {
        --primary: #1e3a8a;
        --primary-light: #3b82f6;
        --primary-dark: #1e40af;
        --accent: #0891b2;
        --success: #059669;
        --danger: #dc2626;
        --warning: #f59e0b;
        --bg-light: #f8fafc;
        --border: #e2e8f0;
        --text-dark: #ffffff;
        --text-light: #ffffff;
    }
    
    body {
        background-color: #f4f7fb;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #000000;
    }
    
.main {
    background:
        radial-gradient(circle at 15% 20%, rgba(59,130,246,0.22), transparent 25%),
        radial-gradient(circle at 85% 15%, rgba(14,165,233,0.18), transparent 22%),
        radial-gradient(circle at 70% 80%, rgba(99,102,241,0.18), transparent 24%),
        linear-gradient(145deg, #020617 0%, #0f172a 40%, #111827 100%);
    
    min-height: 100vh;
    color: white;
}
    /* Page Header */
.page-header {
    position: relative;

    background:
        linear-gradient(
            135deg,
            rgba(15,23,42,0.95),
            rgba(30,64,175,0.88),
            rgba(14,165,233,0.82)
        );

    border-radius: 30px;

    padding: 4rem 3rem;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 25px 60px rgba(0,0,0,0.45),
        inset 0 1px 1px rgba(255,255,255,0.08);
}
    .page-header::before {
    content: "";

    position: absolute;

    width: 400px;
    height: 400px;

    background: rgba(255,255,255,0.08);

    border-radius: 50%;

    top: -200px;
    right: -100px;

    filter: blur(10px);
}

    .page-header::after {
        content: "";
        position: absolute;
        right: -4rem;
        bottom: -5rem;
        width: 16rem;
        height: 16rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
        margin-bottom: 0;
        max-width: 58ch;
    }
    
    /* Card Styling */
.card {
    background: rgba(15, 23, 42, 0.62);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 2rem;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.35),
        inset 0 1px 1px rgba(255,255,255,0.08);

    margin-bottom: 1.5rem;

    transition: all 0.35s ease;

    color: #f8fafc;
}
.card:hover {
    transform: translateY(-6px) scale(1.01);

    border-color: rgba(59,130,246,0.35);

    box-shadow:
        0 20px 50px rgba(59,130,246,0.18),
        0 10px 40px rgba(0,0,0,0.45);
}
    
    .card-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
    }
    
    /* Button Styling */
  .stButton > button {
    background:
        linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        );

    color: white !important;

    border: none;

    border-radius: 18px;

    padding: 0.9rem 1.4rem;

    font-weight: 600;

    box-shadow:
        0 10px 25px rgba(37,99,235,0.35);

    transition: all 0.3s ease;
}
    
   .stButton > button:hover {
    transform: translateY(-3px);

    box-shadow:
        0 18px 40px rgba(37,99,235,0.4);

    filter: brightness(1.05);
}
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Sample Button Special Styling */
    .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%) !important;
    }
    
    /* Input & Slider */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    .stSlider > div[data-baseweb="slider"] {
    padding-top: 1rem;
}
.stSlider [role="slider"] {
    background: white !important;

    border: 4px solid #38bdf8 !important;

    width: 22px !important;
    height: 22px !important;

    box-shadow:
        0 0 20px rgba(56,189,248,0.8);
}

    .stSlider > div > div > div > input {
        border-radius: 8px;
        border: 2px solid var(--primary-light);
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        color: var(--text-dark);
        height: 40px;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    .stSlider > div > div > div > input:hover {
        border-color: #0ea5e9;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.25);
    }
    
    .stSlider > div > div > span {
        color: #1e40af;
        font-weight: 500;
    }
    
    /* Slider Track */
    .stSlider > div > div > div:nth-child(1) {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.2), rgba(14, 165, 233, 0.3));
        border-radius: 999px;
        height: 6px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.96);
        border-radius: 18px;
        padding: 1.5rem;
        border: 1px solid rgba(226, 232, 240, 0.9);
        text-align: center;
        transition: all 0.3s ease;
        color: #000000;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    
    .metric-card:hover {
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.08);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #000000;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #000000;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Alert Boxes */
.alert-success {
    background: linear-gradient(
        135deg,
        rgba(6, 78, 59, 0.95),
        rgba(5, 150, 105, 0.85)
    );

    color: white;

    border-radius: 20px;
    padding: 1.5rem;

    border: 1px solid rgba(16,185,129,0.25);

    box-shadow:
        0 10px 30px rgba(16,185,129,0.2);
}
.alert-danger {
    background: linear-gradient(
        135deg,
        rgba(127, 29, 29, 0.95),
        rgba(220, 38, 38, 0.85)
    );

    border: 1px solid rgba(248,113,113,0.25);

    color: white;

    padding: 1.5rem;
    border-radius: 20px;

    box-shadow:
        0 10px 30px rgba(220,38,38,0.25);

    backdrop-filter: blur(10px);
}
    .alert-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid var(--warning);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.08);
    }
    
    .alert-info {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border-left: 4px solid var(--primary-light);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        color: #000000;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.08);
    }

    .alert-info * {
        color: #000000;
    }
    
    .alert-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        color: #000000;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f4f9 0%, #d9e8f5 50%, #e8f1fa 100%);
        color: #000000;
    }

    [data-testid="stSidebar"] * {
        color: #000000;
    }
    
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #000000;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
    }
    
    /* Radio/Select */
    .stRadio [role="radiogroup"] {
        gap: 1rem;
    }
    
    .stRadio > label {
        background: white;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        cursor: pointer;
        transition: all 0.2s;
        color: #000000;
    }
    
    .stRadio > label:hover {
        border-color: rgba(59, 130, 246, 0.55);
        background: var(--bg-light);
    }

    .stRadio > label[data-selected="true"] {
        border-color: var(--primary-light);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.12);
    }

    .stRadio label span {
        color: #000000;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 2rem 0;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    
    /* Progress */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--success) 0%, var(--primary-light) 100%);
        border-radius: 4px;
    }
    
    /* Headings */
    h1 { color: var(--text-dark); margin-top: 0; }
    h2 { color: #000000; margin-top: 1.5rem; }
    h3 { color: var(--text-dark); }

    h1, h2, h3 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    /* Links */
    a {
        color: #000000;
        text-decoration: none;
        transition: color 0.2s;
    }
    
    a:hover {
        color: #000000;
    }
    
    /* Column gap */
    [data-testid="column"] {
        gap: 1rem;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes smoothFade {
        from {
            opacity: 0.8;
        }
        to {
            opacity: 1;
        }
    }
    
    .card {
        animation: slideIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    /* Smooth gradient transitions */
    * {
        transition: background 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD PIPELINE & DATA
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("breast_cancer_pipeline.pkl")

pipeline = load_model()

data = load_breast_cancer()
feature_names = data.feature_names
feature_means = data.data.mean(axis=0)

feature_info = {
    "mean radius": "Average distance from center to perimeter points of the tumor.",
    "mean texture": "Measures variation in gray-scale intensity.",
    "mean perimeter": "Average perimeter size of the tumor.",
    "mean area": "Average area occupied by the tumor.",
    "mean smoothness": "Measures local variation in radius lengths.",
    "mean compactness": "Describes how compact the tumor shape is.",
    "mean concavity": "Measures inward curves in tumor contour.",
    "mean concave points": "Number of concave portions in contour.",
    "mean symmetry": "Measures tumor symmetry.",
    "mean fractal dimension": "Indicates complexity of tumor boundary."
}

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Healthcare AI System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Prediction", "Analytics", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    with st.container():
        st.markdown("""
        <div class="alert-info">
            <div class="alert-title">Project Details</div>
            <p style="margin: 0; font-size: 0.9rem;">
                Benchmarked <strong>14 ML models</strong> across <strong>3 preprocessing pipelines</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Developed with Streamlit & Scikit-learn")

# ============================================================
# SAMPLE DATA
# ============================================================
benign_sample = [
    12.0, 14.0, 78.0, 450.0, 0.09, 0.08, 0.03, 0.02, 0.18, 0.06,
    0.25, 1.2, 1.8, 20.0, 0.006, 0.02, 0.02, 0.01, 0.02, 0.003,
    13.0, 18.0, 85.0, 550.0, 0.12, 0.18, 0.12, 0.05, 0.25, 0.07
]

malignant_sample = [
    22.0, 30.0, 150.0, 1500.0, 0.12, 0.25, 0.30, 0.15, 0.25, 0.08,
    1.2, 2.5, 8.0, 150.0, 0.01, 0.06, 0.08, 0.03, 0.04, 0.007,
    28.0, 40.0, 190.0, 2500.0, 0.16, 0.50, 0.60, 0.30, 0.40, 0.12
]

if "sample_values" not in st.session_state:
    st.session_state.sample_values = feature_means.tolist()

# ============================================================
# PAGE: PREDICTION
# ============================================================
if page == "Prediction":
    
    # Header
    st.markdown("""
    <div class="page-header">
        <div class="header-title">Breast Cancer Classification</div>
        <div class="header-subtitle">AI-Powered Diagnostic Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Info Box
    st.markdown("""
    <div class="alert-info">
        <div class="alert-title">Important Disclaimer</div>
        <p style="margin: 0; font-size: 0.9rem;">
            This tool is for educational and research purposes only. It does not replace professional medical diagnosis. Always consult with qualified healthcare professionals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Demo Section
    st.markdown("### Quick Demo Cases")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Load Benign Sample", use_container_width=True):

            st.session_state.sample_values = benign_sample

            for idx, val in enumerate(benign_sample):
                st.session_state[f"slider_{idx}"] = val

            st.rerun()

    with col2:
        if st.button("Load Malignant Sample", use_container_width=True):

            st.session_state.sample_values = malignant_sample

            for idx, val in enumerate(malignant_sample):
                st.session_state[f"slider_{idx}"] = val

            st.rerun()

    with col3:
        if st.button("Reset to Defaults", use_container_width=True):

            default_values = feature_means.tolist()

            st.session_state.sample_values = default_values

            for idx, val in enumerate(default_values):
                st.session_state[f"slider_{idx}"] = val

            st.rerun()
    
    # Input Section
    st.markdown("---")
    st.markdown("### Patient Data Input")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #eff6ff 0%, #e0f2fe 100%); border-left: 4px solid #0ea5e9; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <p style="color: #0c4a6e; font-size: 1.05rem; font-weight: 500; margin: 0; line-height: 1.6;">
            Use the sliders below to enter tumor measurements. Hover over feature names for detailed information.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    inputs = []
    
    categories = {
        "Tumor Size Features": list(range(0, 4)),
        "Shape and Texture Features": list(range(4, 10)),
        "Measurement Error Features": list(range(10, 20)),
        "Worst Case Features": list(range(20, 30))
    }
    
    for category, indices in categories.items():
        with st.container():
            st.markdown(f"#### {category}")
            
            cols = st.columns(2)
            
            for idx_num, idx in enumerate(indices):
                feature = feature_names[idx]
                mean_val = float(feature_means[idx])
                max_val = float(np.max(data.data[:, idx]) * 1.2)
                
                with cols[idx_num % 2]:
                    value = st.slider(
                        label=feature.replace("mean ", "").title(),
                        min_value=0.0,
                        max_value=max_val,
                        value=float(st.session_state.sample_values[idx]),
                        step=max_val / 100,
                        help=feature_info.get(feature, "Medical feature used in prediction."),
                        key=f"slider_{idx}"
                    )
                    inputs.append(value)
            
            st.markdown("---")
    
    # Prediction Button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("Analyze Tumor", use_container_width=True)
    
    # Perform Prediction
    if predict_button:
        input_array = np.array([inputs])
        
        prediction = pipeline.predict(input_array)[0]
        probability = pipeline.predict_proba(input_array)[0]
        
        benign_prob = probability[0] * 100
        malignant_prob = probability[1] * 100
        confidence = max(probability) * 100
        
        # Results Section
        st.markdown("---")
        st.markdown("### Prediction Results")
        
        # Main Result Card
        if prediction == 1:
            st.markdown("""
            <div class="alert-danger">
                <div class="alert-title">Malignant Tumor Detected</div>
                <p style="margin: 0; font-size: 0.95rem;">
                    High probability of malignant tumor. Immediate medical consultation is strongly recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(malignant_prob) / 100, text=f"Malignant Confidence: {malignant_prob:.1f}%")
        else:
            st.markdown("""
            <div class="alert-success">
                <div class="alert-title">Benign Tumor Detected</div>
                <p style="margin: 0; font-size: 0.95rem;">
                    Lower probability of malignant tumor, but follow-up with healthcare provider is recommended.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(benign_prob) / 100, text=f"Benign Confidence: {benign_prob:.1f}%")
        
        # Metrics
        st.markdown("#### Confidence Breakdown")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Confidence</div>
                <div class="metric-value">{confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Benign Probability</div>
                <div class="metric-value">{benign_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Malignant Probability</div>
                <div class="metric-value">{malignant_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Comparison Chart
        st.markdown("#### Probability Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Benign', 'Malignant'],
                y=[benign_prob, malignant_prob],
                marker=dict(
                    color=['#10b981', '#ef4444'],
                    line=dict(color='white', width=3),
                    opacity=0.85
                ),
                text=[f'{benign_prob:.1f}%', f'{malignant_prob:.1f}%'],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>',
                textfont=dict(size=14, color='#1e40af', weight=600)
            )
        ])
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(240, 249, 255, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Segoe UI, sans-serif', size=13, color='#1e40af'),
            margin=dict(t=20, b=20, l=20, r=20),
            height=400,
            xaxis=dict(showgrid=False, title='', tickfont=dict(size=13, color='#1e40af')),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e0f2fe', title='Probability (%)', tickfont=dict(size=12, color='#1e40af')),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: ANALYTICS
# ============================================================
elif page == "Analytics":
    
    st.markdown("""
    <div class="page-header">
        <div class="header-title">Model Benchmark Analytics</div>
        <div class="header-subtitle">Performance Comparison of 14 ML Models</div>
    </div>
    """, unsafe_allow_html=True)
    
    benchmark_df = pd.DataFrame({
        "Model": [
            "AdaBoost",
            "QDA",
            "Gradient Boosting",
            "SVM (RBF)",
            "Logistic Regression",
            "Neural Network",
            "Bagging",
            "SVM (Linear)",
            "Decision Tree",
            "Extra Trees",
            "Random Forest",
            "Naive Bayes",
            "KNN",
            "Ridge Classifier"
        ],
        "Accuracy": [
            0.9656, 0.9656, 0.9563, 0.9563, 0.9531,
            0.9531, 0.9531, 0.9531, 0.9500, 0.9469,
            0.9438, 0.9406, 0.9375, 0.9343
        ]
    }).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    
    # Top Models
    st.markdown("### Top Performing Models")
    
    top_5 = benchmark_df.head(5)
    
    top_col1, top_col2, top_col3 = st.columns(3)
    
    medals = ["1st", "2nd", "3rd", "4th", "5th"]
    colors = ["#059669", "#0891b2", "#f59e0b", "#6b7280", "#6b7280"]
    
    for idx, (col, (_, row)) in enumerate(zip([top_col1, top_col2, top_col3, top_col1, top_col2], top_5.iterrows())):
        if idx < 3:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {colors[idx]};">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{medals[idx]}</div>
                    <div class="metric-label">{row['Model']}</div>
                    <div class="metric-value">{row['Accuracy']*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Full Comparison
    st.markdown("---")
    st.markdown("### Detailed Model Comparison")
    
    # Create interactive chart
    fig = px.bar(
        benchmark_df,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison",
        labels={"Accuracy": "Accuracy Score"},
        color="Accuracy",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(240, 249, 255, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, sans-serif', size=11, color='#1e40af'),
        xaxis_tickangle=-45,
        height=500,
        margin=dict(b=100),
        hovermode='x unified',
        title={'font': {'size': 16, 'color': '#1e40af'}}
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.2%}<extra></extra>',
        marker=dict(line=dict(color='white', width=1))
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("### Key Insights")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best Accuracy</div>
            <div class="metric-value">{benchmark_df['Accuracy'].max()*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Accuracy</div>
            <div class="metric-value">{benchmark_df['Accuracy'].mean()*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Models Benchmarked</div>
            <div class="metric-value">{len(benchmark_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy Range</div>
            <div class="metric-value">{(benchmark_df['Accuracy'].max() - benchmark_df['Accuracy'].min())*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Table
    st.markdown("---")
    st.markdown("### Complete Model Rankings")
    
    display_df = benchmark_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Rank"] = range(1, len(display_df) + 1)
    display_df = display_df[["Rank", "Model", "Accuracy"]]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================
# PAGE: ABOUT
# ============================================================
else:

    st.markdown("""
    <div class="page-header">
        <div class="header-title">About This Project</div>
        <div class="header-subtitle">
            Research-Oriented Machine Learning Framework for Breast Cancer Classification
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # INTRODUCTION
    # =========================================================
    st.markdown("""
    <div class="card">

    <div class="card-header">Project Overview</div>

    <p>
    This project presents a research-oriented end-to-end machine learning system
    for breast cancer classification using comparative benchmarking of multiple
    preprocessing pipelines and classification algorithms.
    </p>

    <p>
    The system was developed to investigate how different feature engineering,
    dimensionality reduction, and ensemble learning strategies influence
    predictive performance in healthcare-focused binary classification tasks.
    </p>

    <p>
    The implementation combines concepts inspired by breast cancer prediction
    research literature, comparative machine learning studies, and applied
    healthcare AI workflows.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # DATASET + PIPELINES
    # =========================================================
    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("""
        <div class="card">

        <div class="card-header">Dataset & Feature Space</div>

        <ul style="padding-left: 1.5rem; color: #ffffff;">

        <li><strong>Dataset:</strong> Wisconsin Breast Cancer Dataset</li>

        <li><strong>Total Samples:</strong> 1601(569 Wisconsin records) patient records</li>

        <li><strong>Diagnostic Features:</strong> 30 real-valued tumor attributes</li>

        <li><strong>Classification Objective:</strong> Benign vs Malignant prediction</li>

        <li><strong>Data Representation:</strong> Digitized FNA (Fine Needle Aspirate) measurements</li>

        </ul>

        <p style="color: #ffffff;">
        Feature groups include:
        </p>

        <ul style="padding-left: 1.5rem; color: #ffffff;">

        <li>Radius-based geometric measurements</li>
        <li>Texture and smoothness descriptors</li>
        <li>Concavity and compactness indicators</li>
        <li>Fractal dimension characteristics</li>
        <li>Worst-case and error-based measurements</li>

        </ul>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

        <div class="card-header">Preprocessing Pipelines</div>

        <p>
        Three independent preprocessing pipelines were designed and benchmarked
        to evaluate the impact of feature engineering strategies on model performance.
        </p>

        <ol style="padding-left: 1.5rem; color: #ffffff;">

        <li>
        <strong>Pipeline 1: Full Feature Space</strong><br>
        StandardScaler normalization applied to all 30 diagnostic features.
        </li>

        <br>

        <li>
        <strong>Pipeline 2: Statistical Feature Selection</strong><br>
        StandardScaler + SelectKBest feature filtering using ANOVA F-score ranking.
        </li>

        <br>

        <li>
        <strong>Pipeline 3: PCA Dimensionality Reduction</strong><br>
        StandardScaler + Principal Component Analysis reducing feature space to 11 orthogonal components.
        </li>

        </ol>

        <p>
        The PCA-based pipeline demonstrated superior generalization performance
        while reducing dimensional complexity and feature redundancy.
        </p>

        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # MODELS SECTION
    # =========================================================
    st.markdown("---")

    st.markdown("""
    <div class="card">

    <div class="card-header">Machine Learning Models Evaluated</div>

    <p>
    A total of 14 machine learning algorithms were benchmarked across all preprocessing pipelines.
    The evaluation framework included linear, probabilistic, distance-based,
    tree-based, and ensemble learning approaches.
    </p>

    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">

    <div>

    <strong style="color: #ffffff;">Linear & Statistical Models</strong>

    <ul style="padding-left: 1.5rem; color: #ffffff;">
    <li>Logistic Regression</li>
    <li>Linear Discriminant Analysis (LDA)</li>
    <li>Quadratic Discriminant Analysis (QDA)</li>
    <li>Naive Bayes</li>
    </ul>

    </div>

    <div>

    <strong style="color: #ffffff;">Distance & Margin-Based Models</strong>

    <ul style="padding-left: 1.5rem; color: #ffffff;">
    <li>K-Nearest Neighbors</li>
    <li>Support Vector Machine (Linear)</li>
    <li>Support Vector Machine (RBF)</li>
    </ul>

    </div>

    <div>

    <strong style="color: #ffffff;">Tree-Based Models</strong>

    <ul style="padding-left: 1.5rem; color: #ffffff;">
    <li>Decision Tree</li>
    <li>Random Forest</li>
    <li>Extra Trees</li>
    </ul>

    </div>

    <div>

    <strong style="color: #ffffff;">Ensemble Learning Models</strong>

    <ul style="padding-left: 1.5rem; color: #ffffff;">
    <li>AdaBoost</li>
    <li>Gradient Boosting</li>
    <li>Bagging Classifier</li>
    <li>Neural Network (MLP)</li>
    </ul>

    </div>

    </div>

    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # PERFORMANCE SECTION
    # =========================================================
    st.markdown("---")

    perf_col1, perf_col2 = st.columns(2, gap="large")

    with perf_col1:

        st.markdown("""
        <div class="card">

        <div class="card-header">Best Performing Pipeline</div>

        <p style="color: #ffffff;">

        <strong>Final Deployed Model:</strong> AdaBoost Classifier<br><br>

        <strong>Accuracy:</strong> 96.56%<br>
        <strong>Precision:</strong> 97.39%<br>
        <strong>Recall:</strong> 93.33%<br>
        <strong>F1-Score:</strong> 95.32%<br>
        <strong>ROC-AUC:</strong> 0.9964<br><br>

        <strong>Pipeline:</strong> StandardScaler + PCA (11 Components)

        </p>

        <p style="color: #ffffff;">
        Ensemble boosting methods consistently demonstrated superior
        classification robustness and higher discriminative capability
        compared to standalone classifiers.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with perf_col2:

        st.markdown("""
        <div class="card">

        <div class="card-header">Evaluation Methodology</div>

        <p style="color: #ffffff;">
        Models were evaluated using stratified train-test splitting
        to preserve class distribution across target categories.
        </p>

        <ul style="padding-left: 1.5rem; color: #ffffff;">

        <li>Train-Test Split: 80/20</li>
        <li>Random State Control for Reproducibility</li>
        <li>Stratified Sampling</li>
        <li>Multi-metric Evaluation Framework</li>

        </ul>

        <p style="color: #ffffff;">
        Performance evaluation included:
        </p>

        <ul style="padding-left: 1.5rem; color: #ffffff;">

        <li>Accuracy</li>
        <li>Precision</li>
        <li>Recall</li>
        <li>F1-Score</li>
        <li>ROC-AUC Analysis</li>

        </ul>

        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # DEPLOYMENT SECTION
    # =========================================================
    st.markdown("---")

    st.markdown("""
    <div class="card">

    <div class="card-header">Deployment Architecture</div>

    <p style="color: #ffffff;">
    The final production pipeline integrates preprocessing,
    dimensionality reduction, and classification into a unified
    Scikit-learn deployment pipeline serialized using Joblib.
    </p>

    <p style="color: #ffffff;">
    Real-time predictions are served through an interactive Streamlit
    web application featuring:
    </p>

    <ul style="padding-left: 1.5rem; color: #ffffff;">

    <li>Interactive patient feature simulation</li>
    <li>Real-time probabilistic inference</li>
    <li>Dynamic confidence scoring</li>
    <li>Comparative analytics dashboard</li>
    <li>Research-oriented model documentation</li>

    </ul>

    <p style="color: #ffffff;">
    The deployment architecture ensures preprocessing consistency
    between training and inference environments through serialized ML pipelines.
    </p>

    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # TECHNOLOGY STACK
    # =========================================================
    st.markdown("---")
    st.markdown("### Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown("""
        <div class="card">

        <div style="font-weight: 600; color: #ffffff; margin-bottom: 1rem;">
        Core Libraries
        </div>

        <ul style="padding-left: 1.5rem; color: #ffffff;">
        <li>Scikit-learn</li>
        <li>Pandas</li>
        <li>NumPy</li>
        <li>Joblib</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    with tech2:

        st.markdown("""
        <div class="card">

        <div style="font-weight: 600; color: #ffffff; margin-bottom: 1rem;">
        Frontend & Visualization
        </div>

        <ul style="padding-left: 1.5rem; color: #ffffff;">
        <li>Streamlit</li>
        <li>Plotly</li>
        <li>Custom CSS</li>
        <li>HTML Components</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    with tech3:

        st.markdown("""
        <div class="card">

        <div style="font-weight: 600; color: #ffffff; margin-bottom: 1rem;">
        ML Concepts
        </div>

        <ul style="padding-left: 1.5rem; color: #ffffff;">
        <li>Feature Engineering</li>
        <li>PCA Optimization</li>
        <li>Ensemble Learning</li>
        <li>Comparative Benchmarking</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    # =========================================================
    # FOOTER
    # =========================================================
    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #ffffff;">

    <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
    <strong>Breast Cancer Classification System</strong>
    </p>

    <p style="font-size: 0.95rem;">
    Developed by Ananya Jain
    </p>

    <p style="font-size: 0.85rem;">
    Machine Learning • Healthcare AI • Comparative Benchmarking • Streamlit Deployment
    </p>

    <p style="margin-top: 1rem; font-size: 0.8rem;">
    This system is intended for educational and research purposes only
    and should not be considered a substitute for professional medical diagnosis.
    </p>

    </div>
    """, unsafe_allow_html=True)