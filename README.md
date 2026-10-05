# Customer Churn Prediction

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company or continue using its services.

The project analyzes customer information such as tenure, contract type, internet service, monthly charges, and other service-related features to identify customers who are at risk of churning.

## 🎯 Objective

The main objective of this project is to help companies identify customers who are likely to leave their services so that they can take suitable actions to improve customer retention.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 🤖 Machine Learning

The project includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Encoding
- Model Training
- Model Evaluation
- Hyperparameter Tuning
- Probability Threshold Tuning
- Customer Churn Prediction

The trained model is saved using `Joblib` and integrated into a Streamlit web application.

## 📊 Dataset

The project uses a Telecom Customer Churn dataset containing information such as:

- Customer ID
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract
- Monthly Charges
- Total Charges
- Payment Method
- Churn

## 📈 Model Performance

The model achieved approximately **80% test accuracy**.

The project also evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🌐 Streamlit Web Application

A Streamlit-based web application was developed to allow users to enter customer details and predict their churn probability.

The application provides:

- Customer churn prediction
- Churn probability
- Risk level
- Data visualizations
- Customer-related analytics

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
```

### 2. Open the project folder

```bash
cd customer-churn-prediction
```

### 3. Install required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
├── dataset.csv
└── other project files
```

## 🔮 Future Improvements

- Improve model performance
- Add more machine learning algorithms
- Add customer segmentation
- Improve the Streamlit dashboard
- Deploy the application online
- Add real-time customer data integration

## 👨‍💻 Author

**Prince Raikwar**

Artificial Intelligence & Machine Learning Student

---

⭐ If you find this project useful, feel free to star the repository.