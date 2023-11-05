from flask import Flask,request,render_template
from datetime import date
import subprocess
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
import knnclass
import knnstop
from sklearn.impute import KNNImputer


model = joblib.load('C:/Users/sangh/Downloads/Comding/BTP2/Prediction/x/flight_xgb.pkl')


app = Flask(__name__)
@app.route('/')
# @cross_origin()
def home():
	return render_template('home.html')

def classify_hour(hour):
    if 6 <= hour <= 12:
        return 1, 0, 0
    elif hour >= 18:
        return 0, 1, 0
    elif hour <6:
        return 0, 0, 1
    else:
        return 0, 0, 0


@app.route('/predict',methods=['GET','POST'])
# @cross_origin()


def predict():
    if request.method=='POST':
        dep_time = request.form['Dep_Time']
        arrival_time=request.form['Arrival_Time']

        Journey_date = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").day
        Journey_month = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").month

        Journey_day =pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").day_of_week
        Departure_hour = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").hour
        Departure_min = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").minute 

        today_day=date.today().day
        Days_left=Journey_date-today_day
        Arrival_hour =  pd.to_datetime(arrival_time,format="%Y-%m-%dT%H:%M").hour
        Arrival_min =  pd.to_datetime(arrival_time,format="%Y-%m-%dT%H:%M").minute

        d6_12, da6, db6 = classify_hour(Departure_hour)
        a6_12, aa6, ab6 = classify_hour(Arrival_hour)

        Total_stops= int(request.form['stops'])
        Class=int(request.form['Class'])


        dur_hour = abs(Arrival_hour-Departure_hour)
        dur_min = abs(Arrival_min-Departure_min)

        Source = request.form["Source"]
        source_mapping = {
            'Delhi': (1, 0, 0, 0, 0, 0),
            'Kolkata': (0, 1, 0, 0, 0, 0),
            'Mumbai': (0, 0, 1, 0, 0, 0),
            'Chennai': (0, 0, 0, 1, 0, 0),
            'Hyderabad': (0, 0, 0, 0, 1, 0),
            'Bangalore': (0, 0, 0, 0, 0, 1),
            'Ahmedabad':(0, 0, 0, 0, 0, 0)
        }

        source_values = source_mapping.get(Source)

        Source_Delhi, Source_Kolkata, Source_Mumbai, Source_Chennai, Source_Hyderabad, Source_Banglore = source_values

        Destination = request.form["Destination"]
        destination_mapping = {
            'Bangalore': (1, 0, 0, 0, 0, 0),
            'Delhi': (0, 1, 0, 0, 0, 0),
            'Hyderabad': (0, 0, 1, 0, 0, 0),
            'Kolkata': (0, 0, 0, 1, 0, 0),
            'Mumbai': (0, 0, 0, 0, 1, 0),
            'Chennai':(0, 0, 0, 0, 0, 1),
            'Ahmedabad':(0, 0, 0, 0, 0, 0)
        }

        destination_values = destination_mapping.get(Destination)

        Destination_Bangalore, Destination_Delhi, Destination_Hyderabad, Destination_Kolkata, Destination_Mumbai, Destination_Chennai = destination_values
        Class_early=Class
        Stops_early=Total_stops

        if Class==-1 and Total_stops!=-1:
            inp1=knnclass.knnclass1({
            'Journey_day': Journey_day,
            'Class': np.nan,
            'Total_stops': Total_stops,
            'Duration_in_hours': dur_hour,
            'Days_left': Days_left,
            'Journey_date': Journey_date,
            'Journey_month': Journey_month,
            '6 AM - 12 PM': a6_12,
            'After 6 PM': aa6,
            'Before 6 AM': ab6,
            'Departure_6 AM - 12 PM': d6_12,
            'Departure_After 6 PM': da6,
            'Departure_Before 6 AM': db6,
            'Source_Bangalore': Source_Banglore,
            'Source_Chennai': Source_Chennai,
            'Source_Delhi': Source_Delhi,
            'Source_Hyderabad': Source_Hyderabad,
            'Source_Kolkata': Source_Kolkata,
            'Source_Mumbai': Source_Mumbai,
            'Destination_Bangalore': Destination_Bangalore,
            'Destination_Chennai': Destination_Chennai,
            'Destination_Delhi': Destination_Delhi,
            'Destination_Hyderabad': Destination_Hyderabad,
            'Destination_Kolkata': Destination_Kolkata,
            'Destination_Mumbai': Destination_Mumbai
            })
            if inp1['Class']==0:
                 Class_pred='Economy'
            elif inp1['Class']==1:
                 Class_pred='Business'
            elif inp1['Class']==2:
                 Class_pred='Premium Economy'
            elif inp1['Class']==3:
                 Class_pred='First'
            
            inp = pd.DataFrame({key: [value] for key, value in inp1.items()})
        elif Total_stops==-1 and Class!=-1:
            inp1=knnstop.knnstop1({
            'Journey_day': Journey_day,
            'Class': Class,
            'Total_stops': np.nan,
            'Duration_in_hours': dur_hour,
            'Days_left': Days_left,
            'Journey_date': Journey_date,
            'Journey_month': Journey_month,
            '6 AM - 12 PM': a6_12,
            'After 6 PM': aa6,
            'Before 6 AM': ab6,
            'Departure_6 AM - 12 PM': d6_12,
            'Departure_After 6 PM': da6,
            'Departure_Before 6 AM': db6,
            'Source_Bangalore': Source_Banglore,
            'Source_Chennai': Source_Chennai,
            'Source_Delhi': Source_Delhi,
            'Source_Hyderabad': Source_Hyderabad,
            'Source_Kolkata': Source_Kolkata,
            'Source_Mumbai': Source_Mumbai,
            'Destination_Bangalore': Destination_Bangalore,
            'Destination_Chennai': Destination_Chennai,
            'Destination_Delhi': Destination_Delhi,
            'Destination_Hyderabad': Destination_Hyderabad,
            'Destination_Kolkata': Destination_Kolkata,
            'Destination_Mumbai': Destination_Mumbai
            })
            Stops_pred=inp1['Total_stops']
            inp = pd.DataFrame({key: [value] for key, value in inp1.items()})
        else:
            inp = pd.DataFrame({
                'Journey_day': [Journey_day],
                'Class': [Class],
                'Total_stops': [Total_stops],
                'Duration_in_hours': [dur_hour],
                'Days_left': [Days_left],
                'Journey_date': [Journey_date],
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
        
        data = xgb.DMatrix(inp,enable_categorical=True)

        output = model.predict(data)

        output1 = round(float(output[0]),2)
        if Class_early==-1 and Stops_early==-1:
             return render_template('home.html',predictions='Can Only leave One value as NULL')
        if Class_early==-1:
             return render_template('home.html',predictions='You will have to Pay approx Rs. {}'.format(output1),class1='Class Predicted : {}'.format(Class_pred))
        elif Stops_early==-1:
             return render_template('home.html',predictions='You will have to Pay approx Rs. {}'.format(output1),stops1='Number of Stops Predicted : {}'.format(Stops_pred.astype(int)))
        return render_template('home.html',predictions='You will have to Pay approx Rs. {}'.format(output1))


if __name__ == '__main__':
	app.run(debug=True)
