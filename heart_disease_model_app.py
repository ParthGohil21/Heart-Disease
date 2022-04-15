import streamlit as st
import streamlit.components.v1 as components
import shap
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.title('Heart Disease Checker')

st.sidebar.info(f'''
         This app predicts wheather You are likely to have a **heart realated disease** or not.  
         This model is trained on the data of over **500K patients** 🩺💓''')

st.sidebar.image('header.png')

st.info('**To know your heart disease status fill up the below information**')

Weight = st.number_input('Weight in KG', min_value=0, step=10)

Height = st.number_input('Height in CM', min_value=1, step=10)

Sex = st.selectbox('Sex', ['Male', 'Female'])

GenHealth = st.selectbox(f'''How is your overall health(Your satisfaction level)?
                           (Excellent if no chronic condition or health complaint at all)''',
                         ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])

AgeCategory = st.selectbox('Age Category', ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
                                            '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])

SleepTime = st.slider('On average, how many hours of sleep do you get in a 24-hour period?', 0, 24, 1)

DiffWalking = st.selectbox('Do you have serious difficulty walking or climbing stairs?', ['No', 'Yes'])

Smoking = st.selectbox('have you smoked 100 or more ciggarates in your lifetime?', ['No', 'Yes'])

AlcoholDrinking = st.selectbox('Do you drink Alcohol regularly?', ['No', 'Yes'])

Stroke = st.selectbox('Did you ever had Stroke?', ['No', 'Yes'])

PhysicalHealth = st.slider(
    'how many days during the past 30 days, you had bad physical health? (which includes physical illness and injury)',
    step=1.0, min_value=0.0, max_value=30.0)

Diabetic = st.selectbox('Do you have Diabetes?', ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])

PhysicalActivity = st.selectbox('Have you done any physical activity or exercise in last 30 days?', ['Yes', 'No'])

Asthma = st.selectbox('Do you have Asthma?', ['No', 'Yes'])

BMI = (Weight / (Height ** 2)) * 10000
BMI = round(BMI, 2)


def load_pipeline():
    pipeline = joblib.load('lgb_pipe.pkl')
    return pipeline


def make_df():
    df = pd.DataFrame({'Sex': Sex,
                       'GenHealth': GenHealth,
                       'AgeCategory': AgeCategory,
                       'SleepTime': SleepTime,
                       'Smoking': Smoking,
                       'BMI': BMI,
                       'AlcoholDrinking': AlcoholDrinking,
                       'Stroke': Stroke,
                       'PhysicalHealth': PhysicalHealth,
                       'DiffWalking': DiffWalking,
                       'Diabetic': Diabetic,
                       'PhysicalActivity': PhysicalActivity,
                       'Asthma': Asthma},
                      index=[0]
                      )
    return df


Submit = st.button('Submit')


def make_classification():
    df = make_df()
    pipeline = load_pipeline()
    prediction = pipeline.predict(df)
    prediction = prediction[0]

    if prediction == 1:
        return st.error('You are more likely to have heart disease ☹️!! - Dr. Light GBM')

    elif prediction == 0:
        return st.success('You are not likely to have heart disease 🥳!! - Dr. Light GBM')


#                               changes begin from here


#  additional shap functions
def streamlit_shap(plot, height=None):
    """
    This function allows streamlit to render
    shap plots
    :param plot: shap plot to be rendered
    :param height: height of plot
    :return: None
    """
    shap_html = f'<head>{shap.getjs()}</head><body>{plot.html()}</body>'
    components.html(shap_html, height=height)


def shap_force_plot(mask_csv):
    """
    This function derives shap values and visualises
    shap plots
    :param mask_csv: this should be a portion of your training set
    :return: shap force plot
    """
    #  loading model pipeline
    pipeline = load_pipeline()
    #  loading mask
    mask = pd.read_csv(mask_csv)
    #  creating explainer. arguments are model and transformed dataset so make sure
    #  you index accordingly if you come across an exception
    explainer = shap.TreeExplainer(pipeline[1], pipeline[0].transform(mask))
    #  Calculating shap values
    shap_vals = explainer.shap_values(pipeline[0].transform(make_df()))
    return shap.force_plot(explainer.expected_value, shap_vals, make_df())


def output():
    if Submit:
        st.write('Results:')
        make_classification()
        shap_force_plot('mask.csv')  # please pass a portion of your training set as argument 'mask_csv'
    pass


#  make sure you test and debug this locally before deployment.
#  if you come across any problems commit changes to the shapPlots branch and I'll check it out.


output()
