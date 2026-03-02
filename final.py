import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Page Config
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Survival Classification")
st.markdown("Upload Titanic dataset, train models, and predict survival probability using 4 inputs.")

# Sidebar
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Titanic CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Remove unwanted columns
    df = df.drop(['SibSp', 'Parch', 'Fare', 'Ticket'], axis=1, errors='ignore')

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    if 'Survived' not in df.columns:
        st.error("Dataset must contain 'Survived' column")
    else:

        # ======================
        # PREPROCESSING
        # ======================
        def preprocess_data(df):
            df_processed = df.copy()

            if 'Age' in df_processed.columns:
                df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())

            if 'Embarked' in df_processed.columns:
                df_processed['Embarked'] = df_processed['Embarked'].fillna(
                    df_processed['Embarked'].mode()[0]
                )

            df_processed = df_processed.drop(
                ['Name', 'Cabin', 'PassengerId'],
                axis=1,
                errors='ignore'
            )

            if 'Sex' in df_processed.columns:
                df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})

            if 'Embarked' in df_processed.columns:
                df_processed = pd.get_dummies(
                    df_processed,
                    columns=['Embarked'],
                    drop_first=True
                )

            return df_processed

        df_processed = preprocess_data(df)

        X = df_processed.drop('Survived', axis=1)
        y = df_processed['Survived']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        st.subheader("⚙️ Model Training & Comparison")

        if st.button("🚀 Train Models"):

            results = []

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

                accuracy = accuracy_score(y_test, preds)
                precision = precision_score(y_test, preds)
                recall = recall_score(y_test, preds)

                results.append({
                    "Model": name,
                    "Accuracy (%)": round(accuracy * 100, 2),
                    "Precision (%)": round(precision * 100, 2),
                    "Recall (%)": round(recall * 100, 2)
                })

            results_df = pd.DataFrame(results).sort_values(
                by="Accuracy (%)", ascending=False
            )

            st.dataframe(results_df, use_container_width=True)
            st.bar_chart(results_df.set_index("Model")["Accuracy (%)"])

            best_model_name = results_df.iloc[0]["Model"]
            best_model = models[best_model_name]
            best_model.fit(X_train_scaled, y_train)

            # Save in session_state
            st.session_state["best_model"] = best_model
            st.session_state["scaler"] = scaler
            st.session_state["columns"] = X.columns
            st.session_state["trained"] = True

            st.success(f"🏆 Best Model: {best_model_name}")

        # ======================
        # CUSTOM 4 INPUT PREDICTION
        # ======================

        if "trained" in st.session_state:

            st.subheader("🎯 Predict Survival (4 Inputs)")

            col1, col2 = st.columns(2)

            with col1:
                pclass = st.selectbox("Passenger Class", [1, 2, 3])
                sex = st.selectbox("Sex", ["male", "female"])

            with col2:
                age = st.number_input("Age", min_value=0, max_value=100, value=25)
                embarked = st.selectbox("Embarked", ["C", "Q", "S"])

            if st.button("🔮 Predict Survival"):

                # Encode
                sex_val = 0 if sex == "male" else 1
                embarked_C = 1 if embarked == "C" else 0
                embarked_Q = 1 if embarked == "Q" else 0

                input_data = pd.DataFrame(
                    [[pclass, sex_val, age, embarked_C, embarked_Q]],
                    columns=st.session_state["columns"]
                )

                input_scaled = st.session_state["scaler"].transform(input_data)

                model = st.session_state["best_model"]

                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]

                if prediction == 1:
                    st.success("🎉 Passenger Survived")
                else:
                    st.error("❌ Passenger Did Not Survive")

                st.info(f"📊 Survival Probability: {probability*100:.2f}%")
                st.progress(int(probability * 100))

else:
    st.info("👈 Please upload Titanic dataset to begin.")