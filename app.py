import streamlit as st
#import joblib

#model_tfidf = joblib.load('tfidf_model.bkl')
#model_classification = joblib.load('tag_classification.bkl')
#tag_label = joblib.load('multilabel_binarizer.bkl')

# Header of tag Prediction
html_temp="""<div style="background-color:#F5F5F5"> <h1 style="color:#31333F;text-align:center;"> Customer Satisfaction Prediction </h1></div>"""

st.text_input("Enter your text here", key="text")

# You can access the value at any point with:
text = st.session_state.text

text = cleanText(text)
input_vector = model_tfidf.transform(text)        
res = model_classification.predict(input_vector)
res = tag_label.inverse_transform(res)

st.success(f'The tag list is {res}')
