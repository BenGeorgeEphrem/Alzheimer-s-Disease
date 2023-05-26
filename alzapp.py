import pandas as pd
import streamlit as st
import numpy as np
from pycaret.classification import load_model,predict_model
rf_model = load_model('best')

st.set_page_config(page_title="Alzheimer's Disease Prediction System", layout = 'wide', initial_sidebar_state = 'auto')
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("Alzheimer's Disease Prediction System")
tx = """"This prediction app is created based on the Alzheimer's disease dataset
https://www.kaggle.com/datasets/brsdincer/alzheimer-features

The ML model used for the prediction is Catboost Classifier"""
st.sidebar.info(tx)
gender = st.radio("Gender of the person",('M','F'))
age = st.number_input("Age of the person",60,100,step=1)
mmse = st.number_input("Mini-Mental State Examination Score(0 to 30)",0,30,step=1,
                       help ="MMSE is a 30-point questionnaire that has been proven to be accurate and reliable in detecting dementia.https://oxfordmedicaleducation.com/wp-content/uploads/2016/10/MMSE-printable-mini-mental-state-examination.pdf")
cdr = st.slider('Clinical Dementia Rating (0 - 2)',0.0,2.0,step=0.5,
                help = "measure dementia severity{0= non-demented; 0.5 = very mild dementia; 1,1.5 = mild dementia; 2 = moderate dementia}")
etiv = st.number_input("estimated total intracranial volume (1106–2004) mm3",1106,2004,1106,
                       help = "eTIV variable estimates intracranial brain volume")
nwbv = st.number_input("Normalized whole brain volume (0.64–0.90) mg",0.64,0.9,step=0.01,
                       help="nWBV variable measures the volume of the whole brain")
asf = st.number_input("Atlas scaling factor (0.88–1.56)",0.88,1.56,step=0.01,
                      help="scaling factor that allows for comparison of the estimated total intracranial volume (eTIV) based on differences in human anatomy")
dic = {'M/F':gender, 'Age':age, 'MMSE':mmse, 'CDR':cdr, 
       'eTIV':etiv, 'nWBV':nwbv, 'ASF':asf}
df = pd.DataFrame([dic])
if st.button('Predict'):
  res = predict_model(rf_model,df)
  if res['Label'][0] == 'Demented':
    st.write('Demented')
    st.image('demented.jpg')
  else:
    st.write('Non- Demented')
    st.image('nondemented.jpg')
