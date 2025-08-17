import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle, os

st.set_page_config(page_title="Telecom Churn Analysis", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

data = load_data()

st.title("📊 Telecom Churn Analysis & Prediction")

# ---------- Show Data ----------
st.subheader("Dataset Preview")
st.write(data.head())

# ---------- Basic EDA ----------
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Churn", data=data, ax=ax)
st.pyplot(fig)

st.subheader("Churn by Contract Type")
fig, ax = plt.subplots()
sns.countplot(x="Contract", hue="Churn", data=data, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------- Data Preprocessing ----------
df = data.copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)
df = df.drop(["customerID"], axis=1)

# Encode categorical
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# ---------- Train or Load Model ----------
model_path = "models/churn_model.pkl"
if not os.path.exists(model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pickle.dump(model, open(model_path, "wb"))
else:
    model = pickle.load(open(model_path, "rb"))
with open("churn_model.pkl", "rb") as file:
    model = pickle.load(file)
# ---------- Model Accuracy ----------
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
st.subheader("Model Accuracy")
st.write(f"✅ Logistic Regression Accuracy: {acc:.2f}")

# ---------- Prediction Form ----------
st.subheader("Try Customer Churn Prediction")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("MonthlyCharges", 0, 200, 50)
    total = st.number_input("TotalCharges", 0, 10000, 1000)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# Make input row like training data
if st.button("Predict Churn"):
    new_df = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
        "Contract_One year": [1 if contract=="One year" else 0],
        "Contract_Two year": [1 if contract=="Two year" else 0],
        "InternetService_Fiber optic": [1 if internet=="Fiber optic" else 0],
        "InternetService_No": [1 if internet=="No" else 0],
        "gender_Male": [1 if gender=="Male" else 0]
    })

    # Add missing cols
    for col in X.columns:
        if col not in new_df.columns:
            new_df[col] = 0
    new_df = new_df[X.columns]

    pred = model.predict(new_df)[0]
    st.success("🔴 Customer is likely to CHURN" if pred==1 else "🟢 Customer will STAY")




