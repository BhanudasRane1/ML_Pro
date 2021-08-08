
from flask import Flask, render_template, url_for,request
from sklearn.linear_model import LinearRegression
import joblib
import pandas
import numpy as np


app = Flask(__name__)





@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        data = np.array(data).reshape(-1,1)
        db = pandas.read_csv("marks.csv")
        X = db["hours"]
        X = X.values
        X = X.reshape(4,1)
        y = db["marks"]
        mind = LinearRegression()
        mind.fit(X,y)
        my_prediction = mind.predict(data)
    return render_template('result.html',prediction = my_prediction)



@app.route('/salary',methods=['POST'])
def salary():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        data = np.array(data).reshape(-1,1)
        db = pandas.read_csv("salary.csv")
        x = db["YearsExperience"]
        x=x.values
        x=x.reshape(-1,1)
        y = db["Salary"]
        mind = LinearRegression()
        mind.fit(x,y)
        my_prediction = mind.predict(data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000, debug=True)





