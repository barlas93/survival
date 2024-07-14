pip install -r requirements.txt

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

# Load data and preprocess
data = pd.read_csv('survival.csv')
data = data.drop(columns=['Name']).dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

X = data[['Age', 'Sex', 'BMI', 'Diagnosis', 'Location', 'Resection', 'Infection', 'CT', 'RT', 'Revision']]
y = data['SURV1']

# Random Oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Function to predict and display result
def predict(age, sex, bmi, diagnosis, location, resection, infection, ct, rt, revision):
    # Encode categorical input
    sex_encoded = label_encoder.transform([sex])[0]
    diagnosis_encoded = label_encoder.transform([diagnosis])[0]

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_encoded],
        'BMI': [bmi],
        'Diagnosis': [diagnosis_encoded],
        'Location': [location],
        'Resection': [resection],
        'Infection': [infection],
        'CT': [ct],
        'RT': [rt],
        'Revision': [revision]
    })

    # Make prediction
    y_pred = rf_model.predict(input_data)
    y_proba = rf_model.predict_proba(input_data)[:, 1]

    # Map predicted values to labels
    labels = {0: "Failure", 1: "Survival"}
    prediction = labels[y_pred[0]]

    return prediction, y_proba[0]

# Streamlit UI
def main():
    st.title('Implant Survival Prediction Model (12 months)')
    st.markdown('This app predicts survival based on input data.')

    # Input form
    age_options = ['<18', '18-40', '41-65', '>65']
    age = st.selectbox('Age', age_options)

    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex', sex_options)

    bmi_options = ['<18.5', '18.5-24.9', '25-29.9', '30-39.9', '>40']
    bmi = st.selectbox('BMI', bmi_options)

    diagnosis_options = ['Primary', 'Metastatic']
    diagnosis = st.selectbox('Diagnosis', diagnosis_options)

    location_options = ['Upper extremity', 'Lower extremity']
    location = st.selectbox('Location', location_options)

    resection = st.slider('Resection (mm)', 0, 500, 200)

    infection = st.checkbox('History of infection')

    ct = st.checkbox('Chemotherapy')

    rt = st.checkbox('Radiation therapy')

    revision = st.slider('Number of surgeries', 1, 10, 1)

    if st.button('Predict'):
        # Process input and get prediction
        age_value = 0
        if age == '18-40':
            age_value = 1
        elif age == '41-65':
            age_value = 2
        elif age == '>65':
            age_value = 3

        bmi_value = 0
        if bmi == '18.5-24.9':
            bmi_value = 1
        elif bmi == '25-29.9':
            bmi_value = 2
        elif bmi == '30-39.9':
            bmi_value = 3
        elif bmi == '>40':
            bmi_value = 4

        prediction, probability = predict(age_value, sex, bmi_value, diagnosis, location, resection, infection, ct, rt, revision)

        # Display prediction result
        st.markdown('## Prediction Result')
        st.markdown(f'**Prediction:** {prediction}')
        st.markdown(f'**Probability of Survival:** {probability:.2f}')

if __name__ == '__main__':
    main()

