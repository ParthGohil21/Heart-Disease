import streamlit as st
import joblib 
import numpy as np
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from lightgbm import LGBMClassifier


st.title('Heart Disease Checker')

st.sidebar.info(f'''
         This app predicts wheather You are likely to have a **heart realated disease** or not.  
         This model is trained on the data of over **50K patients** 🩺💓''') 


st.info('**To know your heart disease status fill up the below information**')




Weight=st.number_input('Weight')

Height=st.number_input('Height in cm',min_value=0.1)

Sex=st.selectbox('Sex',['Male','Female'])

GenHealth=st.selectbox('How is your overall health?',['Very good', 'Good', 'Excellent', 'Fair', 'Poor'])

AgeCategory=st.selectbox('Age Category',['18-24','25-29', '30-34','35-39','40-44','45-49','50-54',
                                    '55-59','60-64','65-69','70-74','75-79', '80 or older'])

SleepTime=st.selectbox('On average, how many hours of sleep do you get in a 24-hour period?',
              [ 1.0,  2.0,  3.0,  4.0,  5.0, 6.0,  7.0, 8.0,  9.0,  10.0,  11.0,
              12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
              23.0, 24.0])

DiffWalking=st.selectbox('Do you have serious difficulty walking or climbing stairs?',['Yes','No'])

Smoking=st.selectbox('have you smoked 100 or more ciggarates in your lifetime?',['Yes','No'])

AlcoholDrinking=st.selectbox('Do you drink Alcohol regularly?',['Yes','No'])

Stroke=st.selectbox('Did you ever had Stroke?',['Yes','No'])

PhysicalHealth=st.number_input('how many days during the past 30 days, you had bad physical health?',step=1.0,min_value=0.0,max_value=30.0)

Diabetic=st.selectbox('Do you have Diabetes?',['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])

PhysicalActivity=st.selectbox('Have you done any physical activity or exercise in last 30 days?',['Yes','No'])

Asthma=st.selectbox('Do you have Asthma?',['Yes','No'])            


BMI=(Weight/(Height**2))*10000
BMI=round(BMI,2)


def load_pipeline():
   pipeline=joblib.load(r'C:\Users\parth\OneDrive\Desktop\heart disease project\lgb\lgb_model.pkl')
   return pipeline


def make_df():

   df=pd.DataFrame({'Sex':Sex, 
                  'GenHealth':GenHealth,
                  'AgeCategory':AgeCategory,
                  'SleepTime':SleepTime,
                  'Smoking':Smoking,
                  'BMI':BMI,
                  'AlcoholDrinking':AlcoholDrinking,
                  'Stroke':Stroke,
                  'PhysicalHealth':PhysicalHealth,
                  'DiffWalking':DiffWalking,
                  'Diabetic':Diabetic,
                  'PhysicalActivity':PhysicalActivity,
                  'Asthma':Asthma},
                  index=[0]
                  )  
   return df 


Submit=st.button('Submit')


def make_classification():
   df=make_df()
   pipeline=load_pipeline()
   prediction=pipeline.predict(df)
   prediction=prediction[0]

   if prediction==1:
      return st.error('You are more likely to have heart disease ☹️!! - Dr. Light GBM')

   elif prediction==0:
      return st.success('You are not likely to have heart disease 🥳!! - Dr. Light GBM')



def output():
   if Submit:
      st.write('Results:')
      make_classification()
   pass 


output()
