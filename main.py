import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import sklearn.metrics as sm
from flask import Flask, request, render_template
  
# Flask constructor
app = Flask(__name__)   
  
# A decorator used to tell the application
# which URL is associated function
@app.route('/')
def gfg():
   return render_template("index.html")

@app.route('/send', methods =["GET", "POST"])
def success():
   if request.method == 'POST':
        area = float(request.form.get("inputArea"))
        state = request.form.get("inputState")
        district = request.form.get("inputDistrict")
        season = request.form.get("inputSeason")
        year = 2018
        
        scaler = StandardScaler()
        # Predicting rainfall using KNN regression
        df = pd.read_csv(r'selected_rainfall.csv')
        data = df.loc[(df['State_Name'] == state)]
        x_train = data[['Crop_Year']]
        y_train = data[[season]]
        x_test = [[year]]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        rainfall_model = KNeighborsRegressor(n_neighbors=8, weights='distance')
        rainfall_model.fit(x_train, y_train)
        y_test = rainfall_model.predict(x_test)
        rainfall = y_test[0]

        # Predicting max temperature using KNN regression
        df = pd.read_csv(r'maxtemp.csv')
        data = df.loc[(df['State_Name'] == state)]
        x_train = data[['Crop_Year']]
        y_train = data[[season]]
        x_test = [[year]]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        maxtemp_model = KNeighborsRegressor(n_neighbors=8, weights='distance')
        maxtemp_model.fit(x_train, y_train)
        y_test = maxtemp_model.predict(x_test)
        maxtemp = y_test[0]

        # Predicting min temperature using KNN regression
        df = pd.read_csv(r'mintemp.csv')
        data = df.loc[(df['State_Name'] == state)]
        x_train = data[['Crop_Year']]
        y_train = data[[season]]
        x_test = [[year]]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        mintemp_model = KNeighborsRegressor(n_neighbors=8, weights='distance')
        mintemp_model.fit(x_train, y_train)
        y_test = mintemp_model.predict(x_test)
        mintemp = y_test[0]

        # Predicting relative humidity using KNN regression
        df = pd.read_csv(r'humidity.csv')
        data = df.loc[(df['State_Name'] == state)]
        x_train = data[['Crop_Year']]
        y_train = data[[season]]
        x_test = [[year]]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        hum_model = KNeighborsRegressor(n_neighbors=8, weights='distance')
        hum_model.fit(x_train, y_train)
        y_test = hum_model.predict(x_test)
        humidity = y_test[0]

        # Predicting solar radiation using KNN regression
        df = pd.read_csv(r'solar.csv')
        data = df.loc[(df['State_Name'] == state)]
        x_train = data[['Crop_Year']]
        y_train = data[[season]]
        x_test = [[year]]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        solar_model = KNeighborsRegressor(n_neighbors=8, weights='distance')
        solar_model.fit(x_train, y_train)
        y_test = solar_model.predict(x_test)
        solar = y_test[0]

        # Predicting wind speed using KNN regression
        df = pd.read_csv(r'wind.csv')
        data = df.loc[(df['State_Name'] == state)]
        x_train = data[['Crop_Year']]
        y_train = data[[season]]
        x_test = [[year]]
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        wind_model = KNeighborsRegressor(n_neighbors=8, weights='distance')
        wind_model.fit(x_train, y_train)
        y_test = wind_model.predict(x_test)
        wind_speed = y_test[0]

        # Creating feature array for the new prediction
        x_predict = [[year, rainfall, maxtemp, mintemp, wind_speed, humidity, solar]]

        # Creating a dictionary to hold the values
        production = {}

        # Retrieving production data from csv file
        df = pd.read_csv(r'trial1.csv')
        df1 = df.loc[(df['State_Name'] == state) & (df['District_Name'] == district) & (df['Season'] == season)]
        crops = df1['Crop'].unique()

        for crop in crops:
            data = df1.loc[(df1['Crop'] == crop)]

            if data.size < 35:
                continue

            x = data[['Crop_Year', 'Rainfall', 'Max_temp', 'Min_temp', 'Wind_Speed', 'Relative_humidity', 'Solar_Radiation']]
            y = data[['P/A']]
            scaler.fit(x)
            x = scaler.transform(x)

            # Train Test Split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

            # Predicting using RandomForestRegressor
            model1 = RandomForestRegressor(n_estimators=200, max_depth=10)
            model1.fit(x_train, y_train.values.ravel())
            y_pred1 = model1.predict(x_test)
            err1 = round(sm.mean_absolute_error(y_test, y_pred1), 6)
            model1.fit(x, y.values.ravel())
            y_new1 = model1.predict(x_predict)
            production[crop] = (err1, y_new1[0])

            # Predicting using GradientBoostingRegressor
            model2 = GradientBoostingRegressor(learning_rate=0.05, n_estimators=200)
            model2.fit(x_train, y_train.values.ravel())
            y_pred2 = model2.predict(x_test)
            err2 = round(sm.mean_absolute_error(y_test, y_pred2), 6)
            t = production[crop]
            if err2 < t[0]:
                model2.fit(x, y.values.ravel())
                y_new2 = model2.predict(x_predict)
                production[crop] = (err2, y_new2[0])

            # Predicting using SVR
            model3 = SVR(kernel='rbf')
            model3.fit(x_train, y_train.values.ravel())
            y_pred3 = model3.predict(x_test)
            err3 = round(sm.mean_absolute_error(y_test, y_pred3), 6)
            t = production[crop]
            if err3 < t[0]:
                model3.fit(x, y.values.ravel())
                y_new3 = model3.predict(x_predict)
                production[crop] = (err3, y_new3[0])

        # Retrieving price data
        prices = pd.read_csv('crop_prices.csv')

        # Retrieving cultivation cost data
        cultivation_cost = pd.read_csv('cultivation_cost.csv')

        predicted_crop = 'NIL'
        max_profit = -10000000.00
        # Iterating through the dictionary to find the best crop
        for crop, details in production.items():
            total_production = details[1]*area
            total_selling_price = 10*total_production*(prices[(prices['Commodity'] == crop)][str(year)].values[0])
            total_cost_price = area*(cultivation_cost[(cultivation_cost['State_Name'] == state) & (cultivation_cost['Crop'] == crop)][str(year)].values[0])
            total_profit = total_selling_price - total_cost_price
            if total_profit > max_profit:
                predicted_crop = crop
                max_profit = total_profit
        return render_template('final.html', result=predicted_crop, profit=max_profit)
    
if __name__=='__main__':
    app.run()
