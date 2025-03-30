# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("An artificial intelligence platform to predict the early suicide risk after discharge for suicide patients")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Overeating = st.sidebar.selectbox("Over eating", ("No", "Yes"))
Vegetable = st.sidebar.selectbox("Love eating vegetable", ("No", "Yes"))
Sedentarytime = st.sidebar.selectbox("Sedentary time (hours)", ("﹤1", "≧1 and ﹤3", "≧3 and ﹤6", "≧6"))
Exercisefrequency = st.sidebar.selectbox("Exercise frequency", ("0", "1-2", "3-4", ">5"))
Suicidetime = st.sidebar.selectbox("Suicide frequency", ("1", "2", "3", "≧4"))
Previousdepression= st.sidebar.selectbox("Previous diagnosis of depression", ("Yes1", "No2"))
Previouspsychiologicaldisorder= st.sidebar.selectbox("Previous diagnosis of psychiological disorders", ("Yes1", "No2"))
BECK20 = st.sidebar.slider("BECK 20 score", 0, 19)
Antithyroglobulin = st.sidebar.slider("Anti-thyroglobulin (IU/ml)", 1.0, 40.0)

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_roundweb.pkl")
    x = pd.DataFrame([[Overeating, Vegetable, Sedentarytime, Exercisefrequency, Suicidetime, Previousdepression, Previouspsychiologicaldisorder, BECK20, Antithyroglobulin]],
                     columns=["Overeating", "Vegetable", "Sedentarytime", "Exercisefrequency", "Suicidetime", "Previousdepression", "Previouspsychiologicaldisorder", "BECK20", "Antithyroglobulin"])

    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["﹤1", "≧1 and ﹤3", "≧3 and ﹤6", "≧6"], [1, 2, 3, 4])
    x = x.replace(["0", "1-2", "3-4", ">5"], [1, 2, 3, 4])
    x = x.replace(["1", "2", "3", "≧4"], [1, 2, 3, 4])
    x = x.replace(["Yes1", "No2"], [1, 2])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Early suicide risk after discharge: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.415:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")

st.subheader('About the model')
st.markdown('The complimentary online calculator utilizes an advanced machine learning algorithm and has demonstrated exceptional performance during validation. The risk of patient suicide within three months post-discharge can be predicted through this platform, providing important reasons for the high-risk status based on the risk assessment report. Based on this, it can serve as a reference for suicide management in clinical patients after discharge.')