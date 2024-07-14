pip install -r requirements.txt

from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

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

label_encoder_sex = LabelEncoder()
label_encoder_diagnosis = LabelEncoder()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    bmi = request.form['bmi']
    diagnosis = request.form['diagnosis']
    location = request.form['location']
    resection = request.form['resection']
    infection = request.form['infection']
    ct = request.form['ct']
    rt = request.form['rt']
    revision = request.form['revision']
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'BMI': [bmi],
        'Diagnosis': [diagnosis],
        'Location': [location],
        'Resection': [resection],
        'Infection': [infection],
        'CT': [ct],
        'RT': [rt],
        'Revision': [revision]
    })

    y_pred = rf_model.predict(input_data)
    y_proba = rf_model.predict_proba(input_data)[:, 1]

    labels = {0: "Failure", 1: "Survival"}
    y_pred_labels = [labels[pred] for pred in y_pred]

    return render_template('result.html', prediction=y_pred_labels[0], probability=y_proba[0])

if __name__ == '__main__':
    app.run(debug=True)
