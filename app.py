from flask import Flask, render_template, request, jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask("car_model")

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET'])

def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])

def predict():
    # fuelType_Diesel=0
    temp_array = list()
    if request.method == 'POST':

        model_C_Class = request.form['model_ C Class']
        if model_C_Class == 'model_ C Class':
            temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ A Class':
            temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ B Class':
            temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ CL Class':
            temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ CLA Class':
            temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ CLC Class':
            temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ CLK Class':
            temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ CLS Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ E Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ G Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ GL Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ GLA Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ GLB Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ GLC Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ GLE Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ GLS Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ M Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ R Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ S Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif model_C_Class == 'model_ SL Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif model_C_Class == 'model_ SLK Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif model_C_Class == 'model_ V Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif model_C_Class == 'model_ X Class':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif model_C_Class == 'model_180':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif model_C_Class == 'model_200':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        else:
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        
        transmission_Manual=request.form['transmission_Manual']
        if(transmission_Manual=='Manual'):
            transmission_Manual=1
            transmission_Automatic=0
        else:
            transmission_Manual=0
            transmission_Automatic=1

        fuelType_Petrol=request.form['fuelType_Petrol']
        if(fuelType_Petrol=='Petrol'):
            fuelType_Petrol=1
            fuelType_Diesel=0
        else:
            fuelType_Petrol=0
            fuelType_Diesel=1

        age = int(request.form['Age'])
        mileage=int(request.form['mileage'])
        tax=int(request.form['tax'])
        mpg=float(request.form['mpg'])
        engineSize=int(request.form['engineSize'])
        temp_array = temp_array + [transmission_Manual,transmission_Automatic,fuelType_Petrol,fuelType_Diesel,age, mileage, tax, mpg, engineSize]
        temp_array = np.array([temp_array])
        prediction=model.predict(temp_array)
        output=round(prediction[0],2)

        if output<0:
            return render_template('index.html',prediction_text="❌ Sorry you cannot sell this car. 🙁")

        else:
            return render_template('index.html',prediction_text="✅ You Can Sell the Car at {} Pounds 🤑👍".format(output))

    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
