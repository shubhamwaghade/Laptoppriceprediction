import pickle
import pandas as pd
import numpy as np
import sklearn
import warnings
import json
warnings.filterwarnings('ignore')

class Predictor():

    def __init__(self,Company,TypeName,Ram,Weight,Touchscreen,IPS,ppi,Cpu_brand,HDD,SSD,Gpu_brand,os):

        self.Company="Company_"+Company
        self.TypeName="TypeName_"+TypeName
        self.Ram=Ram
        self.Weight=Weight
        self.Touchscreen=Touchscreen
        self.IPS=IPS
        self.ppi=ppi
        self.Cpu_brand="Cpu_brand_"+Cpu_brand
        self.HDD=HDD
        self.SSD=SSD
        self.Gpu_brand="Gpu_brand_"+Gpu_brand
        self.os="os_"+os

    def load_saved_data(self):    
        with open(r"linear_reg_model.pkl","rb") as f:
            self.model = pickle.load(f)

        with open(r"project_data.json",'r') as r:
            self.project_data = json.load(r)   

    def get_pred_value(self):   
        self.load_saved_data()
        Company_index=self.project_data["Columns"].index(self.Company)
        TypeName_index=self.project_data["Columns"].index(self.TypeName)
        Cpu_brand_index=self.project_data["Columns"].index(self.Cpu_brand)
        Gpu_brand_index=self.project_data["Columns"].index(self.Gpu_brand)
        os_index=self.project_data["Columns"].index(self.os)

        col_count=len(self.project_data["Columns"])
        array=np.zeros(col_count)

        array[Company_index] =1
        array[TypeName_index]=1
        array[0]=self.Ram
        array[1]=self.Weight
        array[2]=self.Touchscreen
        array[3]=self.IPS
        array[4]=self.ppi
        array[Cpu_brand_index]=1
        array[5]=self.HDD
        array[Gpu_brand_index]=1
        array[os_index]=1

        predicted_value=np.around(self.model.predict([array])[0],2)
        return predicted_value

if __name__=="__main__":
    obj=Predictor()
    obj

        
