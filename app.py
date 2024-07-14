from flask import Flask, render_template, request
import pickle
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

# Random Oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = oversample.fit_resample(X, y)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X, y)

# Load label encoders
label_encoder_sex = LabelEncoder()
label_encoder_diagnosis = LabelEncoder()

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    diagnosis = int(request.form['diagnosis'])
    location = int(request.form['location'])
    resection = int(request.form['resection'])
    infection = int(request.form['infection'])
    ct = int(request.form['ct'])
    rt = int(request.form['rt'])
    revision = int(request.form['revision'])
    
    # Create input DataFrame
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

    # Make prediction
    y_pred = rf_model.predict(input_data)
    y_proba = rf_model.predict_proba(input_data)[:, 1]

    return render_template('result.html', prediction=y_pred[0], probability=y_proba[0])

if __name__ == '__main__':
    app.run(debug=True)
