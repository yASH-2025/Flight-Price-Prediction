from flask import Flask,request,render_template
from datetime import date
import pickle
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

model = joblib.load('flight_pred.pkl')

app = Flask(__name__)
@app.route('/')
# @cross_origin()
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
# @cross_origin()
def predict():
    if request.method=='POST':
        dep_time = request.form['Dep_Time']
        arrival_time=request.form['Arrival_Time']

        Journey_day = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").day
        Journey_month = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").month

        Departure_hour = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").hour
        Departure_min = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").minute 


        if Departure_hour>=6 and Departure_hour<=12:
            d6_12=1
            da6=0
            db6=0
        
        elif Departure_hour>=18 :
            d6_12=0
            da6=1
            db6=0

        elif Departure_hour<6:
            d6_12=0
            da6=0
            db6=1
        
        else:
            d6_12=0
            da6=0
            db6=0

        today_day=date.today().day
        Days_left=Journey_day-today_day

        Arrival_hour =  pd.to_datetime(arrival_time,format="%Y-%m-%dT%H:%M").hour
        Arrival_min =  pd.to_datetime(arrival_time,format="%Y-%m-%dT%H:%M").minute

        if Arrival_hour>=6 and Arrival_hour<=12:
            a6_12=1
            aa6=0
            ab6=0
        
        elif Arrival_hour>=18 :
            a6_12=0
            aa6=1
            ab6=0

        elif Arrival_hour<6:
            a6_12=0
            aa6=0
            ab6=1
        
        else:
            a6_12=0
            aa6=0
            ab6=0

        Total_stops= int(request.form['stops'])
        Class=int(request.form['Class'])

        dur_hour = abs(Arrival_hour-Departure_hour)
        dur_min = abs(Arrival_min-Departure_min)

        Source=request.form["Source"]

        if (Source == 'Delhi'):
            Source_Delhi = 1
            Source_Kolkata = 0
            Source_Mumbai = 0
            Source_Chennai = 0
            Source_Banglore =0
            Source_Hyderabad =0

        elif (Source == 'Kolkata'):
            Source_Delhi = 0
            Source_Kolkata = 1
            Source_Mumbai = 0
            Source_Chennai = 0
            Source_Banglore =0
            Source_Hyderabad =0

        elif (Source == 'Mumbai'):
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 1
            Source_Chennai = 0
            Source_Banglore =0
            Source_Hyderabad =0

        elif (Source == 'Chennai'):
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 0
            Source_Chennai = 1
            Source_Banglore =0
            Source_Hyderabad =0

        elif (Source == 'Hydrabad'):
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 0
            Source_Chennai = 0
            Source_Banglore =0
            Source_Hyderabad =0

        elif (Source == 'Bangalore'):
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 0
            Source_Chennai = 0
            Source_Banglore =1
            Source_Hyderabad =0
        
        else:
            Source_Delhi = 0
            Source_Kolkata = 0
            Source_Mumbai = 0
            Source_Chennai = 0
            Source_Banglore =0
            Source_Hyderabad =1


        # Destination_Bangalore	Destination_Chennai	Destination_Delhi	Destination_Hyderabad	Destination_Kolkata	Destination_Mumbai
        Destination = request.form["Destination"]

        if (Destination == 'Bangalore'):
            Destination_Bangalore = 1
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_Mumbai = 0
            Destination_Chennai = 0
        
        elif (Destination == 'Delhi'):
            Destination_Bangalore = 0
            Destination_Delhi = 1
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_Mumbai = 0
            Destination_Chennai = 0

        elif (Destination == 'Hyderabad'):
            Destination_Bangalore = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 1
            Destination_Kolkata = 0
            Destination_Mumbai = 0
            Destination_Chennai = 0

        elif (Destination == 'Kolkata'):
            Destination_Bangalore = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 1
            Destination_Mumbai = 0
            Destination_Chennai = 0

        elif (Destination =='Mumbai'):
            Destination_Bangalore = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_Mumbai = 1
            Destination_Chennai = 0
        
        else:
            Destination_Bangalore = 0
            Destination_Delhi = 0
            Destination_Hyderabad = 0
            Destination_Kolkata = 0
            Destination_Mumbai = 0
            Destination_Chennai = 1

        inp = pd.DataFrame({
            'Journey_day': [Journey_day],
            'Class': [Class],
            'Total_stops': [Total_stops],
            'Duration_in_hours': [dur_hour],
            'Days_left': [Days_left],
            'Journey_month': [Journey_month],
            '6 AM - 12 PM': [a6_12],
            'After 6 PM': [aa6],
            'Before 6 AM': [ab6],
            'Departure_6 AM - 12 PM': [d6_12],
            'Departure_After 6 PM': [da6],
            'Departure_Before 6 AM': [db6],
            'Source_Bangalore': [Source_Banglore],
            'Source_Chennai': [Source_Chennai],
            'Source_Delhi': [Source_Delhi],
            'Source_Hyderabad': [Source_Hyderabad],
            'Source_Kolkata': [Source_Kolkata],
            'Source_Mumbai': [Source_Mumbai],
            'Destination_Bangalore': [Destination_Bangalore],
            'Destination_Chennai': [Destination_Chennai],
            'Destination_Delhi': [Destination_Delhi],
            'Destination_Hyderabad': [Destination_Hyderabad],
            'Destination_Kolkata': [Destination_Kolkata],
            'Destination_Mumbai': [Destination_Mumbai]
        })

        data = xgb.DMatrix(inp)

        output = model.predict(data)

        output = round(float(output[0]), 2)
        return render_template('home.html', predictions='You will have to Pay approx Rs. {}'.format(output))


if __name__ == '__main__':
	app.run(debug=True)