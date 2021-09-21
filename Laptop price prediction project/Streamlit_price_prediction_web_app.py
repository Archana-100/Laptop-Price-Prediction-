# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:12:39 2021
pip install xgboost==0.90
@author: archa
"""

import streamlit as st
import pickle 
import numpy as np
from astyles import *

def ui_headr():
    try :
        #st.set_page_config(page_title="Prediction", page_icon=None, layout='centered', initial_sidebar_state='auto',)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown( main_body_style ,unsafe_allow_html=True,)
        st.markdown(hide_streamlit_style, unsafe_allow_html=True,) 
        st.markdown(sidebar_style,unsafe_allow_html=True,)
        st.markdown(label_txt_style, unsafe_allow_html=True,)
        st.markdown(button_radio_style, unsafe_allow_html=True,)
        st.markdown(btn_style, unsafe_allow_html=True,)
    except Exception as e :
        st.error(str(e))
        
ui_headr()

#imort our model and dataframe
def load_df():
    lap_df = pickle.load(open("laptop_df.pkl",'rb'))
    return lap_df

def load_model():
    model= pickle.load(open("xgb_model.pkl",'rb'))
    return model 

def get_data_from_ui():
    st.title("Predict Laptop Price")
    laptop_df = load_df()
    company= st.selectbox('Company Brand Name',laptop_df['Company'].unique())

    typename = st.selectbox('Type of laptop',laptop_df['TypeName'].unique())

    ram = st.selectbox('Ram(In GB)',[2,4,8,16,24,32,64])

    weight = st.number_input("Enter weight of the laptop")

    touchscreen = st.selectbox('Touchscreen',["No","Yes"])

    ips = st.selectbox('IPS',["No","Yes"])

    screen_size = st.number_input('Enter Screen Size(In Inches')

    screen_resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

    cpu = st.selectbox('CPU',laptop_df['Cpu Brand'].unique())

    hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

    ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

    gpu = st.selectbox('GPU',laptop_df['Gpu brand'].unique())

    os = st.selectbox('OS',laptop_df['OS'].unique())


    if st.button('Predict Laptop Price'):
        #query
        ppi= None

        if touchscreen == 'Yes':
           touchscreen = 1
        else:
           touchscreen = 0

        if ips== 'Yes':
           ips = 1
        else:
           ips = 0

        x_res = int(screen_resolution.split("x")[0])
        y_res = int(screen_resolution.split("x")[1])

        ppi= (((x_res**2)+(y_res**2))**0.5)/screen_size
        query = np.array([company,typename,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

        query= query.reshape(1,12)
        model = load_model()
        st.title("The predicted price :- " + str(int(np.exp(model.predict(query)[0]))))
        
get_data_from_ui()

    