import streamlit as st
import json 
import requests
import pickle

st.title("Credit Score Rating")

st.write("Select the inputs from the number_input below 👇")
returnOnAssets = st.number_input('returnOnAssets',-1.,1.,value=0.047044,)
netProfitMargin = st.number_input('netProfitMargin',-101.85,198.52,value=0.068535)
debtRatio = st.number_input('debtRatio',0.00,1.93,value=0.649932)
returnOnCapitalEmployed = st.number_input('returnOnCapitalEmployed',-87162.16,2.44,value=0.058498)
operatingProfitMargin = st.number_input('operatingProfitMargin',-124.34,410.18,value=0.105987)
quickRatio = st.number_input('quickRatio',-1.89,916.66,value=1.042165)
operatingCashFlowPerShare = st.number_input('operatingCashFlowPerShare',-11950.49,6439270.41,value=3.806772)
operatingCashFlowSalesRatio = st.number_input('operatingCashFlowSalesRatio',-4.46,688.53,value=0.136786)
returnOnEquity = st.number_input('returnOnEquity',-63.81,141350.21,value=0.162230)
pretaxProfitMargin = st.number_input('pretaxProfitMargin',-124.34,309.69,value=0.073767)

inputs = {'returnOnAssets':returnOnAssets,'netProfitMargin':netProfitMargin,
          'debtRatio':debtRatio, 'returnOnCapitalEmployed':returnOnCapitalEmployed,
          'operatingProfitMargin':operatingProfitMargin, 'quickRatio':quickRatio,
          'operatingCashFlowPerShare':operatingCashFlowPerShare, 'operatingCashFlowSalesRatio':operatingCashFlowSalesRatio,
          'returnOnEquity':returnOnEquity,'pretaxProfitMargin': pretaxProfitMargin}

loaded_model = pickle.load(open("front_end/rf.sav", 'rb'))
data_in = [[inputs['returnOnAssets'], inputs['netProfitMargin'], inputs['debtRatio'],inputs['returnOnCapitalEmployed'],inputs['operatingProfitMargin'],inputs['quickRatio'], inputs['operatingCashFlowPerShare'], inputs['operatingCashFlowSalesRatio'], inputs['returnOnEquity'],inputs['pretaxProfitMargin']]]

prediction = loaded_model.predict(data_in)

if st.button("Predict"):
    # res = requests.post(url = "http://127.0.0.1:8000/predict", data = json.dumps(inputs))
    prediction = loaded_model.predict(data_in)
    probability = loaded_model.predict_proba(data_in).max()
    st.subheader(f"Response from API 🛫")
    st.markdown(f"#### **Prediction: {prediction[0]}**")
    st.markdown(f"#### **Probability: {probability.max()}**")
    
    st.write("AAA: 10,   AA:9,   A:8, BBB: 7,  BB:6,B:5,   CCC:4,   CC:3,   C:2,  D:1")
