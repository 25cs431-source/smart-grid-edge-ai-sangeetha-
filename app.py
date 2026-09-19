from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Grid Edge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL ANIMATED THEME
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(14,165,233,0.15), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(59,130,246,0.14), transparent 30%),
            radial-gradient(circle at 50% 90%, rgba(16,185,129,0.08), transparent 32%),
            linear-gradient(135deg, #030712 0%, #071525 50%, #06101c 100%);
        color: #e5eefb;
        min-height: 100vh;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(3,10,22,0.98),
                rgba(7,20,38,0.98)
            );
        border-right: 1px solid rgba(96,165,250,0.18);
    }

    [data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }


    /* ================= TITLE ================= */

    .main-title {
        font-size: clamp(2.1rem, 4vw, 3.4rem);
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -1.5px;
        margin-bottom: 8px;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #93c5fd,
                #67e8f9,
                #ffffff
            );

        background-size: 250% auto;

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        animation: titleAnimation 7s linear infinite;
    }

    .subtitle {
        color: #94a9c2;
        font-size: 1rem;
        margin-bottom: 18px;
    }


    /* ================= STATUS BADGES ================= */

    .status-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 20px;
        animation: fadeUp 0.8s ease;
    }

    .status-badge {
        padding: 8px 14px;
        border-radius: 999px;

        background: rgba(14,165,233,0.08);
        border: 1px solid rgba(14,165,233,0.25);

        color: #7dd3fc;
        font-size: 0.8rem;
        font-weight: 600;

        box-shadow: 0 0 18px rgba(14,165,233,0.08);
    }


    /* ================= NOTICE ================= */

    .notice {
        padding: 17px 20px;
        border-radius: 17px;

        background:
            linear-gradient(
                135deg,
                rgba(245,158,11,0.13),
                rgba(15,23,42,0.65)
            );

        border: 1px solid rgba(245,158,11,0.28);

        box-shadow:
            0 15px 40px rgba(0,0,0,0.2);

        margin-bottom: 25px;

        animation: fadeUp 0.9s ease;
    }


    /* ================= METRIC CARDS ================= */

    [data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(15,31,53,0.88),
                rgba(7,19,34,0.72)
            );

        border: 1px solid rgba(96,165,250,0.20);

        padding: 18px;

        border-radius: 18px;

        box-shadow:
            0 12px 40px rgba(0,0,0,0.22);

        transition:
            transform 0.3s ease,
            border-color 0.3s ease,
            box-shadow 0.3s ease;

        animation: fadeUp 0.6s ease;
    }

    [data-testid="stMetric"]:hover {

        transform: translateY(-6px);

        border-color:
            rgba(103,232,249,0.55);

        box-shadow:
            0 20px 55px rgba(14,165,233,0.14);
    }

    [data-testid="stMetricLabel"] {
        color: #91a8c2 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800;
    }


    /* ================= BUTTON ================= */

    .stButton > button {

        border-radius: 13px;

        border:
            1px solid rgba(103,232,249,0.35);

        background:
            linear-gradient(
                135deg,
                #0ea5e9,
                #2563eb
            );

        color: white;

        font-weight: 750;

        min-height: 47px;

        box-shadow:
            0 8px 28px rgba(37,99,235,0.28);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            filter 0.2s ease;
    }

    .stButton > button:hover {

        transform: translateY(-3px);

        filter: brightness(1.08);

        box-shadow:
            0 14px 38px rgba(14,165,233,0.35);
    }


    /* ================= RISK CARDS ================= */

    .risk-high,
    .risk-moderate,
    .risk-low {

        padding: 22px;

        border-radius: 19px;

        backdrop-filter: blur(18px);

        box-shadow:
            0 15px 45px rgba(0,0,0,0.25);

        animation:
            riskPulse 2.8s ease-in-out infinite;
    }

    .risk-high {

        background:
            linear-gradient(
                135deg,
                rgba(127,29,29,0.48),
                rgba(30,10,15,0.80)
            );

        border:
            1px solid rgba(248,113,113,0.42);

        border-left:
            6px solid #ef4444;
    }

    .risk-moderate {

        background:
            linear-gradient(
                135deg,
                rgba(124,45,18,0.45),
                rgba(30,18,8,0.80)
            );

        border:
            1px solid rgba(251,146,60,0.40);

        border-left:
            6px solid #f97316;
    }

    .risk-low {

        background:
            linear-gradient(
                135deg,
                rgba(6,78,59,0.43),
                rgba(5,30,27,0.80)
            );

        border:
            1px solid rgba(52,211,153,0.38);

        border-left:
            6px solid #10b981;
    }


    /* ================= WORKFLOW ================= */

    .flow {

        text-align: center;

        padding: 18px 10px;

        min-height: 95px;

        border-radius: 17px;

        background:
            rgba(15,32,54,0.70);

        border:
            1px solid rgba(96,165,250,0.20);

        box-shadow:
            0 10px 30px rgba(0,0,0,0.18);

        transition:
            transform 0.3s ease,
            border-color 0.3s ease;

        animation:
            floating 4s ease-in-out infinite;
    }

    .flow:hover {

        transform:
            translateY(-7px)
            scale(1.02);

        border-color:
            rgba(103,232,249,0.50);
    }


    /* ================= TABS ================= */

    div[data-testid="stTabs"] button {

        color: #9fb3c9;

        transition:
            all 0.25s ease;
    }

    div[data-testid="stTabs"] button:hover {

        color: #e0f2fe;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {

        color: #67e8f9;
    }


    /* ================= EXPANDER ================= */

    .stExpander {

        background:
            rgba(10,24,42,0.55);

        border:
            1px solid rgba(96,165,250,0.16);

        border-radius: 16px;
    }


    /* ================= ANIMATIONS ================= */

    @keyframes titleAnimation {

        0% {
            background-position: 0% 50%;
        }

        50% {
            background-position: 100% 50%;
        }

        100% {
            background-position: 0% 50%;
        }
    }


    @keyframes fadeUp {

        from {
            opacity: 0;
            transform: translateY(16px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }


    @keyframes floating {

        0%, 100% {
            transform: translateY(0);
        }

        50% {
            transform: translateY(-4px);
        }
    }


    @keyframes riskPulse {

        0%, 100% {
            box-shadow:
                0 15px 45px rgba(0,0,0,0.22);
        }

        50% {
            box-shadow:
                0 18px 55px rgba(14,165,233,0.12);
        }
    }


    /* ================= SCROLLBAR ================= */

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #030712;
    }

    ::-webkit-scrollbar-thumb {

        background:
            linear-gradient(
                #0ea5e9,
                #2563eb
            );

        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATHS = [
    BASE_DIR / "data" / "synthetic_smart_grid_demand_dataset.csv",
    BASE_DIR / "synthetic_smart_grid_demand_dataset.csv",
]


# ============================================================
# DATASET
# ============================================================

def find_dataset():

    for path in DATA_PATHS:

        if path.exists():
            return path

    return None


@st.cache_data
def load_data(path):

    return pd.read_csv(path)


def clean_data(df):

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    aliases = {

        "Day Type": "Day_Type",

        "Temperature C":
            "Temperature_C",

        "Previous Demand MW":
            "Previous_Demand_MW",

        "Demand MW":
            "Demand_MW",

        "Demand":
            "Demand_MW",
    }

    df.rename(
        columns=aliases,
        inplace=True
    )

    required_columns = [

        "Day",
        "Day_Type",
        "Hour",
        "Temperature_C",
        "Previous_Demand_MW",
        "Demand_MW",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    numeric_columns = [

        "Hour",
        "Temperature_C",
        "Previous_Demand_MW",
        "Demand_MW",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["Day_Type"] = (
        df["Day_Type"]
        .astype(str)
        .str.strip()
    )

    df.dropna(
        subset=numeric_columns + ["Day_Type"],
        inplace=True
    )

    df = df[
        (df["Hour"] >= 0)
        &
        (df["Hour"] <= 23)
    ]

    return df.reset_index(drop=True)


# ============================================================
# LOAD DATA
# ============================================================

dataset_path = find_dataset()


st.markdown(
    """
    <div class="main-title">
        ⚡ Smart Grid Demand Forecasting
    </div>

    <div class="subtitle">
        Edge AI • Demand Prediction • Risk Assessment • Early Warning
    </div>

    <div class="status-row">

        <span class="status-badge">
            🟢 EDGE ONLINE
        </span>

        <span class="status-badge">
            ⚡ LOCAL INFERENCE
        </span>

        <span class="status-badge">
            🤖 AI MONITORING
        </span>

        <span class="status-badge">
            📊 LIVE SIMULATION
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="notice">

    <b>⚠️ Academic Prototype</b><br>

    This application uses a
    <b>synthetic smart-grid dataset</b>
    for educational demonstration.

    It is not connected to a real electrical grid
    and does not control electrical equipment.

    </div>
    """,
    unsafe_allow_html=True
)


if dataset_path is None:

    st.error(
        "Dataset not found. Please place "
        "synthetic_smart_grid_demand_dataset.csv "
        "inside the data folder."
    )

    st.stop()


try:

    raw_data = load_data(
        str(dataset_path)
    )

    df = clean_data(raw_data)

except Exception as error:

    st.error("Dataset loading failed.")

    st.exception(error)

    st.stop()


# ============================================================
# MODEL
# ============================================================

FEATURES = [

    "Hour",
    "Temperature_C",
    "Previous_Demand_MW",
    "Day_Type",
]

TARGET = "Demand_MW"


@st.cache_resource
def train_model(data):

    X = data[FEATURES]

    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


    numeric_features = [

        "Hour",
        "Temperature_C",
        "Previous_Demand_MW",
    ]

    categorical_features = [
        "Day_Type"
    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                "passthrough",
                numeric_features
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )


    model = RandomForestRegressor(

        n_estimators=250,

        random_state=42,

        n_jobs=-1
    )


    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    return pipeline, {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "train": len(X_train),

        "test": len(X_test)
    }


with st.spinner(
    "🤖 Training Edge AI model locally..."
):

    model, metrics = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Prediction Control"
    )

    st.caption(
        "Enter current grid conditions"
        " and forecast future demand."
    )


    hours = st.slider(

        "🕐 Hour of Day",

        min_value=0,

        max_value=23,

        value=18
    )


    day_types = sorted(
        df["Day_Type"].unique()
    )


    day_type = st.selectbox(

        "📅 Day Type",

        day_types
    )


    temperature = st.number_input(

        "🌡️ Temperature (°C)",

        min_value=-20.0,

        max_value=60.0,

        value=float(
            df["Temperature_C"].median()
        ),

        step=0.5
    )


    previous_demand = st.number_input(

        "⚡ Current / Previous Demand (MW)",

        min_value=0.0,

        max_value=5000.0,

        value=float(
            df["Previous_Demand_MW"].mean()
        ),

        step=1.0
    )


    st.markdown("---")


    st.markdown(
        "## 🏗️ Grid Capacity"
    )


    grid_capacity = st.number_input(

        "Grid Capacity (MW)",

        min_value=1.0,

        max_value=10000.0,

        value=1200.0,

        step=10.0
    )


    st.markdown("---")


    st.markdown(
        "## 🚨 Risk Thresholds"
    )


    low_percentage = st.slider(

        "Low → Moderate (%)",

        50,

        95,

        80
    )


    high_percentage = st.slider(

        "Moderate → High (%)",

        low_percentage,

        130,

        100
    )


    predict_button = st.button(

        "⚡ PREDICT DEMAND",

        type="primary",

        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

input_data = pd.DataFrame(

    [{
        "Hour": hours,

        "Temperature_C":
            temperature,

        "Previous_Demand_MW":
            previous_demand,

        "Day_Type":
            day_type
    }]
)


if (
    predict_button
    or
    "prediction" not in st.session_state
):

    prediction = float(
        model.predict(
            input_data
        )[0]
    )

    st.session_state.prediction = prediction


else:

    prediction = st.session_state.prediction


# ============================================================
# RISK CALCULATION
# ============================================================

ratio = (
    prediction
    /
    grid_capacity
)


if ratio < low_percentage / 100:

    risk = "LOW"

elif ratio <= high_percentage / 100:

    risk = "MODERATE"

else:

    risk = "HIGH"


headroom = (
    grid_capacity
    -
    prediction
)


overload = max(
    0,
    prediction - grid_capacity
)


# ============================================================
# EDGE STATUS
# ============================================================

st.markdown(
    "### 🖥️ Edge AI System Status"
)


status1, status2, status3 = st.columns(3)


with status1:

    st.success(
        "🟢 **SYSTEM ONLINE**"
    )


with status2:

    st.info(
        "⚡ **LOCAL INFERENCE**"
    )


with status3:

    st.success(
        "🤖 **MODEL READY**"
    )


st.caption(
    "Edge AI is simulated by running "
    "the trained machine-learning model "
    "locally on this computer."
)


# ============================================================
# GRID OVERVIEW
# ============================================================

st.markdown(
    "### 📊 Grid Overview"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(

    "⚡ Current Demand",

    f"{previous_demand:.1f} MW"
)


c2.metric(

    "🌡️ Temperature",

    f"{temperature:.1f} °C"
)


c3.metric(

    "🏗️ Grid Capacity",

    f"{grid_capacity:.1f} MW"
)


c4.metric(

    "🔮 Predicted Demand",

    f"{prediction:.1f} MW",

    delta=f"{prediction - previous_demand:.1f} MW"
)


# ============================================================
# RISK DISPLAY
# ============================================================

st.markdown(
    "### 🚨 AI Demand Risk Assessment"
)


if risk == "HIGH":

    st.markdown(

        f"""
        <div class="risk-high">

        <h2>🔴 HIGH LOAD WARNING</h2>

        <b>Predicted demand exceeds the configured capacity.</b>

        <br><br>

        Predicted Demand:
        <b>{prediction:.1f} MW</b>

        <br>

        Grid Capacity:
        <b>{grid_capacity:.1f} MW</b>

        <br>

        Overload:
        <b>{overload:.1f} MW</b>

        <br>

        Load Ratio:
        <b>{ratio * 100:.1f}%</b>

        <br>

        Risk Level:
        <b>HIGH</b>

        </div>
        """,

        unsafe_allow_html=True
    )


elif risk == "MODERATE":

    st.markdown(

        f"""
        <div class="risk-moderate">

        <h2>🟠 MODERATE RISK</h2>

        <b>
        Predicted demand is approaching
        the configured capacity.
        </b>

        <br><br>

        Predicted Demand:
        <b>{prediction:.1f} MW</b>

        <br>

        Grid Capacity:
        <b>{grid_capacity:.1f} MW</b>

        <br>

        Remaining Headroom:
        <b>{headroom:.1f} MW</b>

        <br>

        Load Ratio:
        <b>{ratio * 100:.1f}%</b>

        </div>
        """,

        unsafe_allow_html=True
    )


else:

    st.markdown(

        f"""
        <div class="risk-low">

        <h2>🟢 LOW RISK</h2>

        <b>
        Predicted demand is within
        the configured capacity.
        </b>

        <br><br>

        Predicted Demand:
        <b>{prediction:.1f} MW</b>

        <br>

        Grid Capacity:
        <b>{grid_capacity:.1f} MW</b>

        <br>

        Available Headroom:
        <b>{headroom:.1f} MW</b>

        <br>

        Load Ratio:
        <b>{ratio * 100:.1f}%</b>

        </div>
        """,

        unsafe_allow_html=True
    )


# ============================================================
# RECOMMENDED ACTION
# ============================================================

st.markdown(
    "### 🛠️ Recommended Preventive Action"
)


if risk == "HIGH":

    actions = [

        "Monitor the simulated grid load immediately.",

        "Reduce or shift non-critical loads.",

        "Redistribute flexible loads where appropriate.",

        "Review high-demand areas and hours."
    ]


elif risk == "MODERATE":

    actions = [

        "Monitor the grid load closely.",

        "Consider reducing optional loads.",

        "Keep flexible-load redistribution options ready."
    ]


else:

    actions = [

        "Continue routine monitoring.",

        "Predicted demand is within configured capacity.",

        "No immediate demonstration action is required."
    ]


for action in actions:

    st.write(
        "🔹 " + action
    )


st.caption(
    "These are demonstration suggestions only. "
    "The application does not automatically "
    "control electrical equipment."
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    "### 🤖 AI Model Performance"
)


p1, p2, p3, p4 = st.columns(4)


p1.metric(
    "MAE",
    f"{metrics['MAE']:.2f} MW"
)


p2.metric(
    "RMSE",
    f"{metrics['RMSE']:.2f} MW"
)


p3.metric(
    "R² Score",
    f"{metrics['R2']:.3f}"
)


p4.metric(
    "Training Records",
    f"{metrics['train']}"
)


st.caption(
    f"Random Forest Regressor • "
    f"Training: {metrics['train']} records • "
    f"Testing: {metrics['test']} records"
)


# ============================================================
# DEMAND GRAPH
# ============================================================

st.markdown(
    "### 📈 Demand Analytics"
)


tab1, tab2 = st.tabs(

    [
        "📊 Historical Demand",
        "🔮 Prediction"
    ]
)


# ============================================================
# HISTORICAL GRAPH
# ============================================================

with tab1:

    graph_data = df.copy()

    graph_data[
        "Time Index"
    ] = np.arange(
        len(graph_data)
    )


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=graph_data[
                "Time Index"
            ],

            y=graph_data[
                "Demand_MW"
            ],

            mode="lines",

            name="Historical Demand",

            line=dict(
                width=2
            ),

            fill="tozeroy"
        )
    )


    fig.add_hline(

        y=grid_capacity,

        line_dash="dash",

        annotation_text=
        "Grid Capacity"
    )


    fig.update_layout(

        height=450,

        paper_bgcolor=
        "rgba(0,0,0,0)",

        plot_bgcolor=
        "rgba(5,15,28,0.55)",

        font=dict(
            color="#cbd5e1"
        ),

        xaxis_title=
        "Dataset Time Index",

        yaxis_title=
        "Demand (MW)",

        hovermode=
        "x unified"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PREDICTION GRAPH
# ============================================================

with tab2:

    recent = df.tail(24).copy()

    recent[
        "Sequence"
    ] = np.arange(
        len(recent)
    )


    fig2 = go.Figure()


    fig2.add_trace(

        go.Scatter(

            x=recent[
                "Sequence"
            ],

            y=recent[
                "Demand_MW"
            ],

            mode="lines+markers",

            name="Historical"
        )
    )


    fig2.add_trace(

        go.Scatter(

            x=[
                len(recent)
            ],

            y=[
                prediction
            ],

            mode="markers",

            name="AI Prediction",

            marker=dict(

                size=18,

                symbol="star"
            )
        )
    )


    fig2.add_hline(

        y=grid_capacity,

        line_dash="dash",

        annotation_text=
        "Grid Capacity"
    )


    fig2.update_layout(

        height=450,

        paper_bgcolor=
        "rgba(0,0,0,0)",

        plot_bgcolor=
        "rgba(5,15,28,0.55)",

        font=dict(
            color="#cbd5e1"
        ),

        xaxis_title=
        "Recent Time Sequence",

        yaxis_title=
        "Demand (MW)"
    )


    st.plotly_chart(

        fig2,

        use_container_width=True
    )


# ============================================================
# DATASET
# ============================================================

with st.expander(
    "📁 Dataset Information"
):

    d1, d2, d3 = st.columns(3)


    d1.metric(
        "Records",
        len(df)
    )


    d2.metric(
        "Minimum Demand",
        f"{df['Demand_MW'].min():.1f} MW"
    )


    d3.metric(
        "Maximum Demand",
        f"{df['Demand_MW'].max():.1f} MW"
    )


    st.dataframe(

        df.head(10),

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# SYSTEM WORKFLOW
# ============================================================

st.markdown(
    "### 🧭 System Workflow"
)


flow_columns = st.columns(5)


workflow = [

    (
        "⚡",
        "Grid Data",
        "Synthetic readings"
    ),

    (
        "🖥️",
        "Edge Processing",
        "Local preprocessing"
    ),

    (
        "🤖",
        "AI Forecast",
        "Random Forest"
    ),

    (
        "🚨",
        "Risk Assessment",
        "Capacity comparison"
    ),

    (
        "🔔",
        "Early Warning",
        "Preventive action"
    )
]


for column, item in zip(
    flow_columns,
    workflow
):

    icon, title, description = item


    with column:

        st.markdown(

            f"""
            <div class="flow">

            <div style="
                font-size:2rem;
                margin-bottom:6px;
            ">
                {icon}
            </div>

            <b>{title}</b>

            <br>

            <small>
                {description}
            </small>

            </div>
            """,

            unsafe_allow_html=True
        )


# ============================================================
# PROJECT EXPLANATION
# ============================================================

st.markdown("---")

st.markdown(
    "### 🧠 Project Explanation"
)


with st.expander(
    "🔍 Understand how the project works"
):

    st.markdown(

        """
        **1. Problem**

        Electricity demand changes throughout the day.
        Unexpected demand increases can create capacity
        challenges.

        **2. Solution**

        This prototype uses historical synthetic demand
        records to train a machine-learning regression
        model and forecast future demand.

        **3. Edge AI**

        The trained model performs inference locally
        on the computer running the Streamlit application.

        **4. Machine Learning Model**

        A Random Forest Regressor uses:

        - Hour
        - Temperature
        - Previous Demand
        - Day Type

        to predict Demand in MW.

        **5. Risk Assessment**

        The predicted demand is compared with the
        configured grid capacity.

        - Below 80% → LOW
        - 80% to 100% → MODERATE
        - Above 100% → HIGH

        **6. Important Limitation**

        This is an academic prototype using synthetic data.
        It is not connected to a real electrical grid and
        does not automatically control electrical equipment.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:45px;
        padding:20px;
        color:#64748b;
        border-top:1px solid rgba(148,163,184,0.12);
    ">

        ⚡ <b>Smart Grid Edge AI</b>

        <br>

        <small>
        Demand Forecasting • Edge AI • Risk Assessment
        </small>

        <br><br>

        <small>
        Student Mini-Project • Synthetic Data •
        Educational Demonstration Only
        </small>

    </div>
    """,
    unsafe_allow_html=True
)