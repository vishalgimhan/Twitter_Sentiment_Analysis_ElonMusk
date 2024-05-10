import streamlit as st
import pickle
import time

st.title("Twitter Sentiment Analysis")

#load model
#model = pickle.load(open('TwitterSentiment-RF.pkl', 'rb'))
from joblib import load
model = load('TwitterSentiment-RF.joblib')
tweet = st.text_input("Enter the tweet")

submit = st.button('Predict')
if submit:
    with st.spinner('Wait for it...'):
        time.sleep(5)
        start = time.time()
        prediction = model.predict([tweet])
        end = time.time()
        st.write('Prediction time: {}'.format(end-start, 2), 'seconds')
        st.write('Predicted Sentiment: ', prediction[0])


#streamlit run app.py