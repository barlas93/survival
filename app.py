import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv('survival.csv')
data = data.drop(columns=['Name'])
data = data.dropna()

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

X = data[['Age', 'Sex', 'BMI', 'Diagnosis', 'Location', 'Resection', 'Revision']]
y = data['SURV2']

oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X, y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X_resampled, y_resampled)

def main():
    st.markdown("<h1 style='text-align: center;'>Megaprosthesis Survival Prediction 12 months</h1>", unsafe_allow_html=True)
    
    st.sidebar.header('Input Features')

    age_options = ['<18', '18-40', '41-65', '>65']
    age = st.sidebar.selectbox('Age', age_options)

    bmi_options = ['<18.5', '18.5-24.9', '25-29.9', '30-39.9', '>40']
    bmi = st.sidebar.selectbox('BMI', bmi_options)

    sex_options = ['Male', 'Female']
    sex = st.sidebar.selectbox('Gender', sex_options)

    diagnosis_options = ['Primary', 'Metastatic', 'Non-Oncologic']
    diagnosis = st.sidebar.selectbox('Diagnosis', diagnosis_options)

    location_options = ['Proximal Femur Replacement', 'Distal Femur Replacement', 'Proximal Tibia Replacement', 'Proximal Humerus Replacement']
    location = st.sidebar.selectbox('Location', location_options)

    resection_options = ['<120', '121-199', '200-299', '>300']
    resection = st.sidebar.selectbox('Resection (mm)', resection_options)

    revision_options = ['1', '2', '3', '>3']
    revision = st.sidebar.selectbox('Number of surgeries', revision_options)

    age_mapping = {'<18': 1, '18-40': 2, '41-65': 3, '>65': 4}
    bmi_mapping = {'<18.5': 1, '18.5-24.9': 2, '25-29.9': 3, '30-39.9': 4, '>40': 5}
    sex_mapping = {'Male': 1, 'Female': 0}
    diagnosis_mapping = {'Non-Oncologic':2, 'Primary': 1, 'Metastatic': 0}
    location_mapping = {'Proximal Femur Replacement': 1, 'Distal Femur Replacement': 2, 'Proximal Tibia Replacement': 3, 'Proximal Humerus Replacement': 4}
    resection_mapping = {'<120': 1, '121-199': 2, '200-299': 3, '>300': 4}
    revision_mapping = {'1': 1, '2': 2, '3': 3, '>3': 4}

    input_data = pd.DataFrame({
        'Age': [age_mapping[age]],
        'Sex': [sex_mapping[sex]],
        'BMI': [bmi_mapping[bmi]],
        'Diagnosis': [diagnosis_mapping[diagnosis]],
        'Location': [location_mapping[location]],
        'Resection': [resection_mapping[resection]],
        'Revision': [revision_mapping[revision]]
    })

    y_pred = rf_model.predict(input_data)
    y_proba = rf_model.predict_proba(input_data)[:, 1]

    labels = {0: "failure", 1: "survival"}
    prediction_label = labels[y_pred[0]]

    prob_html = f"""
    <div style="text-align: center; margin-top: 20px;">
            <span style="font-size: 28px">Probability of survival: {y_proba[0]}</span>
    </div>
    """
    st.markdown(prob_html, unsafe_allow_html=True)
    
    if y_pred == 0:
        result_html = f"""
        <div style="text-align: center;">
            <span style="font-size: 28px">The predicted outcome is </span><span style="font-size: 28px; color: red;">{prediction_label}</span>
        </div>
        """
    else:
        result_html = f"""
        <div style="text-align: center;">
            <span style="font-size: 28px">The predicted outcome is </span><span style="font-size: 28px; color: green;">{prediction_label}</span>
        </div>
        """

    st.markdown(result_html, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
