# 📊 Telecom Churn Analysis & Prediction

This project analyzes customer churn for a telecom company using the Kaggle *Telco Customer Churn Dataset*.  
It includes *exploratory data analysis (EDA), **visualizations, a **logistic regression model, and a **Streamlit web app* for interactive churn prediction.

---

## 🚀 Features
- *Data Exploration*
  - Preview dataset, clean missing values, encode categorical features
- *Visualization*
  - Churn distribution plots
  - Churn by contract type
- *Machine Learning*
  - Logistic Regression model for churn prediction
  - Model is auto-trained and saved in /models
- *Interactive Web App*
  - Built with Streamlit
  - Input customer details and predict churn in real time
- *Deployment Ready*
  - Easy to deploy on Streamlit Cloud

---

## 📂 Project Structure
telecom-churn-analysis/ │── app.py                  # Main Streamlit App │── requirements.txt        # Python dependencies │── README.md               # Project Documentation │── LICENSE                 # Open-source license (MIT) │── data/ │     └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset │── models/ │     └── churn_model.pkl   # Saved Logistic Regression model (auto-created)

---

## 📊 Dataset
- *Source:* [Kaggle – Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)  
- *Rows:* 7043  
- *Columns:* 21  
- *Target Variable:* Churn (Yes/No)

*Sample Columns*
- gender, SeniorCitizen, Partner, Dependents
- tenure, MonthlyCharges, TotalCharges
- Contract, PaymentMethod, InternetService
- Churn

---

## ▶ Run Locally
Clone this repository and run the app:

```bash
git clone <your-repo-link>
cd telecom-churn-analysis
pip install -r requirements.txt
streamlit run app.py 

---

🌐 Deployment (Streamlit Cloud)

1. Push this project to GitHub


2. Go to Streamlit Cloud


3. Connect your repo


4. Choose app.py as the entry file


5. Deploy 🚀




---

🛠 Tech Stack

Python – Core programming

Pandas, NumPy – Data preprocessing

Matplotlib, Seaborn – Data visualization

Scikit-learn – Machine learning (Logistic Regression)

Streamlit – Interactive web app



---

📈 Example Output

EDA Charts

Churn distribution

Churn vs Contract type


Prediction

Input: Tenure, Charges, Contract type, Internet type, Gender

Output: Churn / Not Churn




---

📜 License

This project is licensed under the MIT License – free to use and modify.
See the LICENSE file for details.


---

👤 Author

Rajkumar Suryavanshi
📧 Email: krajsuryaaa@gmail.copm
💼 LinkedIn: http://www.linkedin.com/in/rajkumar-suryavanshi-963703254
📂 GitHub: https://github.com/Suryavanshii