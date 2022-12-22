from flask import Flask, render_template, jsonify, request
from utils import Predictor
import numpy as np


app = Flask(__name__)

@app.route('/')
def hello_flask():
    print("Welcome to Laptop Price Predictor ")
    return render_template("home.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method=="GET":
        Company=request.args.get("Company")
        TypeName=request.args.get("TypeName")
        Ram=eval(request.args.get("Ram"))
        Weight=eval(request.args.get("Weight"))
        Touchscreen=eval(request.args.get("Touchscreen"))
        IPS=eval(request.args.get("IPS"))
        ppi=eval(request.args.get("ppi"))
        Cpu_brand=request.args.get("Cpu_brand")
        HDD=eval(request.args.get("HDD"))
        SSD=(request.args.get("SSD"))
        Gpu_brand=request.args.get("Gpu_brand")
        os=request.args.get("os")

        pre=Predictor(Company,TypeName,Ram,Weight,Touchscreen,IPS,ppi,Cpu_brand,HDD,SSD,Gpu_brand,os)
        predict=pre.get_pred_value()

        return render_template("home.html", prediction = np.around(predict,2))




if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5050,debug=True)  