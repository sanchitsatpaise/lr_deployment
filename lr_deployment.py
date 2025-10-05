import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load Model and Preprocessing Objects ---
MODEL_PATH = 'titanic_model.pkl'

try:
    # Load the pickle file uploaded by the user
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    log_model = data['model']
    scaler = data['scaler']
    le_dict = data['label_encoders']
    
    # Extract LabelEncoder mappings (assuming 'le' was a single encoder object
    # that encoded both 'Sex' and 'Embarked' as per the notebook's final code cell.
    # We will use the common mappings found in the Titanic dataset.)
    
    # Sex Mappings (Typically: male=1, female=0 OR vice versa. Using common display logic.)
    # Note: We hardcode display mappings as the single 'le' object mapping is ambiguous.
    SEX_MAPPING = {'Male': 1, 'Female': 0}
    
    # Embarked Mappings (Typically: S=2, C=0, Q=1 OR vice versa. Using common display logic.)
    EMBARKED_MAPPING = {'Southampton (S)': 2, 'Cherbourg (C)': 0, 'Queenstown (Q)': 1}

except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model file. This often happens due to library version mismatch (e.g., numpy).")
    st.info("Try running `pip install --upgrade scikit-learn numpy` and ensure the model was created with compatible versions.")
    st.info(f"Detailed Error: {e}")
    st.stop()


# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title('🚢 Logistic Regression Model Deployment: Titanic Survival Predictor')
st.markdown("Use the form below to enter passenger data and predict the probability of survival.")

# --- 3. User Input Form ---
with st.form("prediction_form"):
    
    st.subheader("Passenger Information")
    
    # Row 1: Pclass and Sex
    col1, col2 = st.columns(2)
    pclass = col1.selectbox(
        'Passenger Class (Pclass)',
        options=[1, 2, 3],
        help="1: First, 2: Second, 3: Third"
    )
    sex_display = col2.radio(
        'Sex',
        options=['Male', 'Female']
    )
    
    # Row 2: Age and Fare
    col3, col4 = st.columns(2)
    age = col3.slider('Age', min_value=1, max_value=80, value=30, step=1)
    fare = col4.number_input('Fare ($)', min_value=0.0, max_value=512.33, value=50.0, step=5.0)
    
    # Row 3: SibSp, Parch, and Embarked
    col5, col6, col7 = st.columns(3)
    sibsp = col5.number_input('Siblings/Spouses (SibSp)', min_value=0, max_value=8, value=0, step=1)
    parch = col6.number_input('Parents/Children (Parch)', min_value=0, max_value=6, value=0, step=1)
    embarked_display = col7.selectbox(
        'Port of Embarkation',
        options=list(EMBARKED_MAPPING.keys()),
        help="S: Southampton, C: Cherbourg, Q: Queenstown"
    )

    st.markdown("---")
    submitted = st.form_submit_button("Predict Survival Probability")

# --- 4. Prediction Logic ---
if submitted:
    
    # 1. Preprocess the inputs (matching the training pipeline)
    sex_encoded = SEX_MAPPING[sex_display]
    embarked_encoded = EMBARKED_MAPPING[embarked_display]

    # Create the feature array (must match the order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    # The order is assumed based on standard Titanic feature sets and your notebook code structure.
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
                               columns=feature_names)

    # 2. Apply the saved StandardScaler
    # Ensure all data types are numeric before scaling
    scaled_input = scaler.transform(input_data.astype(float))
    
    # 3. Make Prediction
    probability = log_model.predict_proba(scaled_input)[0][1]
    prediction = log_model.predict(scaled_input)[0]

    # --- 5. Display Results ---
    st.subheader("Prediction Result")
    prob_percent = probability * 100
    
    # Conditional display based on predicted class
    if prediction == 1:
        st.balloons()
        st.success(f"**Outcome: Survived!** 🎉")
    else:
        st.error(f"**Outcome: Did Not Survive** 😢")
        
    st.markdown(f"The model predicts a **{prob_percent:.2f}%** chance of survival.")

