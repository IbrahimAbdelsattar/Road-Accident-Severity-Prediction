import streamlit as st
import pandas as pd
import joblib

# =========================
# Load saved model bundle
# =========================
@st.cache_resource
def load_model():
    bundle = joblib.load("severity_xgb_bundle.pkl")  # تأكد إن الاسم نفس اسم ملفك
    return bundle

bundle = load_model()
model = bundle["model"]
le_y = bundle["label_encoder"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]
bool_cols = bundle["bool_cols"]

st.title("🚗 Accident Severity Prediction")
st.write("Predict accident severity (1–4) using your trained XGBoost model.")

st.sidebar.header("Input Features")

user_input = {}

# =========================
# 1️⃣ Numeric Inputs
# =========================
st.subheader("Numeric Features")

# لو حابب تحط default قريب من القيم الشائعة
default_numeric = {
    "Start_Lat": 37.808498,
    "Start_Lng": -122.366852,
    "Distance(mi)": 0.0,
    "Temperature(F)": 64.0,
    "Humidity(%)": 93.0,
    "Pressure(in)": 29.86,
    "Visibility(mi)": 10.0,
}

for col in numeric_cols:
    user_input[col] = st.number_input(
        col,
        value=float(default_numeric.get(col, 0.0))
    )

# =========================
# 2️⃣ Categorical Inputs
# =========================
st.subheader("Categorical Features")

# ----- Source -----
source_options = ["Source1", "Source2", "Source3"]
user_input["Source"] = st.selectbox("Source", source_options)

# ----- City -----
user_input["City"] = st.text_input("City", value="Miami")

# ----- County -----
user_input["County"] = st.text_input("County", value="Los Angeles")

# ----- State -----
state_options = [
    "CA","FL","TX","SC","NY","NC","VA","PA","MN","OR","AZ","GA","IL","TN","MI","LA",
    "NJ","MD","OH","WA","AL","UT","CO","OK","MO","CT","IN","MA","WI","KY","NE","MT",
    "IA","AR","NV","KS","DC","RI","MS","DE","WV","ID","NM","NH","WY","ND","ME","VT","SD"
]
user_input["State"] = st.selectbox("State", state_options)

# ----- Country -----
user_input["Country"] = st.selectbox("Country", ["US"])

# ----- Timezone -----
tz_options = ["US/Eastern", "US/Pacific", "US/Central", "US/Mountain"]
user_input["Timezone"] = st.selectbox("Timezone", tz_options)

# ----- Airport_Code -----
user_input["Airport_Code"] = st.text_input("Airport_Code", value="KCQT")

# ----- Wind_Direction -----
wind_options = [
    "CALM","S","SSW","W","WNW","NW","Calm","SW","WSW","SSE","NNW","N","SE",
    "E","ESE","NE","ENE","NNE","VAR","South","West","North","Variable","East"
]
user_input["Wind_Direction"] = st.selectbox("Wind_Direction", wind_options)

# ----- Weather_Condition -----
weather_common = [
    "Fair",
    "Mostly Cloudy",
    "Cloudy",
    "Clear",
    "Partly Cloudy",
]
user_input["Weather_Condition"] = st.selectbox(
    "Weather_Condition",
    weather_common + ["Other"]
)
if user_input["Weather_Condition"] == "Other":
    # free text لو عايز تكتب حالة جو مختلفة
    user_input["Weather_Condition"] = st.text_input("Other Weather_Condition", value="Fair")

# ----- Twilight / Sun -----
user_input["Sunrise_Sunset"] = st.selectbox("Sunrise_Sunset", ["Day", "Night"])
user_input["Civil_Twilight"] = st.selectbox("Civil_Twilight", ["Day", "Night"])
user_input["Nautical_Twilight"] = st.selectbox("Nautical_Twilight", ["Day", "Night"])
user_input["Astronomical_Twilight"] = st.selectbox("Astronomical_Twilight", ["Day", "Night"])

# =========================
# 3️⃣ Boolean Inputs
# =========================
st.subheader("Boolean Features")

for col in bool_cols:
    # checkbox أسهل من selectbox
    user_input[col] = st.checkbox(col, value=False)

# =========================
# Build DataFrame from user input
# =========================
input_df = pd.DataFrame([user_input])

st.markdown("### Current Input")
st.write(input_df)

# =========================
# Prediction
# =========================
if st.button("Predict Severity"):
    # الموديل جوه الـ pipeline هيتكفّل بالـ encoding والـ preprocessing
    pred_encoded = model.predict(input_df)[0]               # 0,1,2,3
    pred_label = le_y.inverse_transform([pred_encoded])[0]  # يرجّع 1,2,3,4

    st.success(f"Predicted Severity: **{int(pred_label)}**")
