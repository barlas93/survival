import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('survival.csv')
data = data.drop(columns=['Name'])
data = data.dropna()

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

X = data[['Age', 'Sex', 'BMI', 'Diagnosis', 'Location', 'Resection', 'Infection', 'CT', 'RT', 'Revision']]
y = data['SURV1']

oversample = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = oversample.fit_resample(X, y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X_resampled, y_resampled)

def main():
    st.title('Survival Prediction App')

    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, step=0.1)
    diagnosis = st.selectbox('Diagnosis', ['Type A', 'Type B', 'Type C'])
    location = st.selectbox('Location', ['Location A', 'Location B', 'Location C'])
    resection = st.checkbox('Resection')
    infection = st.checkbox('Infection')
    ct = st.checkbox('CT')
    rt = st.checkbox('RT')
    revision = st.checkbox('Revision')

    if st.button('Predict'):
        sex_encoded = label_encoder.transform([sex])[0]
        diagnosis_encoded = label_encoder.transform([diagnosis])[0]

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

        y_pred = rf_model.predict(input_data)
        y_proba = rf_model.predict_proba(input_data)[:, 1]

        st.write('Prediction:', y_pred[0])
        st.write('Probability:', y_proba[0])

if __name__ == '__main__':
    main()
