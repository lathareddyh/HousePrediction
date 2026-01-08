import pickle 
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')

#initialize flask app
app = Flask(__name__)
data = pd.read_csv("C:\Latha\LearnBay\Projects\Linear-Regression\Deploy\cleaned_data.csv")
pipe = pickle.load(open("C:\Latha\LearnBay\Projects\Linear-Regression\Deploy\house_prediction.pkl", "rb"))

# Logic
@app.route('/') #Decorator
def index():
    locations = sorted(data['location'].unique())
    # locations = ""
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    total_sqft = float(request.form.get('total_sqft'))

    # location	bhk	total_sqft	bath
    input = pd.DataFrame([[location, bhk, total_sqft, bath]], columns=['location', 'bhk', 'total_sqft', 'bath'])
    prediction = np.round(pipe.predict(input)[0], 2)
    return str(prediction)
#Run the application
# app.run(host='0.0.0.0', port=7080, debug=True)
if __name__ == "__main__":
    app.run(debug=True)