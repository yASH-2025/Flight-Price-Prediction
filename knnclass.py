import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import math

df = pd.read_csv('Encoded.csv')
def knnclass1(data):
        user_input = pd.Series(data)
        k=7
        imputer = KNNImputer(n_neighbors=k)
        imputer.fit(df[['Journey_day',
                    'Class',
                    'Total_stops',
                    'Duration_in_hours',
                    'Days_left',
                    'Journey_date',
                    'Journey_month',
                    '6 AM - 12 PM',
                    'After 6 PM',
                    'Before 6 AM',
                    'Departure_6 AM - 12 PM',
                    'Departure_After 6 PM',
                    'Departure_Before 6 AM',
                    'Source_Bangalore',
                    'Source_Chennai',
                    'Source_Delhi',
                    'Source_Hyderabad',
                    'Source_Kolkata',
                    'Source_Mumbai',
                    'Destination_Bangalore',
                    'Destination_Chennai',
                    'Destination_Delhi',
                    'Destination_Hyderabad',
                    'Destination_Kolkata',
                    'Destination_Mumbai']])

        imputed_user_input = imputer.transform(user_input.to_frame().T)
        imputed_user_input[0][1]=math.floor(imputed_user_input[0][1]*4)
        
        imputed_user_input_series = pd.Series(imputed_user_input[0], index=user_input.index)

        print("Imputed User Input:")
        print(imputed_user_input_series)
        return imputed_user_input_series
