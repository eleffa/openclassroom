import streamlit as st
import joblib

model_tfidf = joblib.load('tfidf_model.bkl')
model_classification = joblib.load('tag_classification.bkl')
indice = joblib.load('list_indice.bkl')
title = joblib.load('list_title.bkl')
docs = joblib.load('list_doc.bkl')
tag_label = joblib.load('multilabel_binarizer.bkl')
tf_idf_test = joblib.load('tfidf_test.bkl')
true_tag = joblib.load('true_test.bkl')

# Header of tag Prediction
html_temp="""<div style="background-color:#F5F5F5"> <h1 style="color:#31333F;text-align:center;">Tag Prediction </h1></div>"""

x = st.slider('x')

num =x
ind = indice[num]
title = title[ind]
tags = true_tag[ind]
input_vector = tf_idf_test[num]
res = model_classification.predict(input_vector)
res = tag_label.inverse_transform(res)

st.success(f'Document Title is {title}')
st.success(f'Document Tags are {tags}')
st.success(f'The predict s tags list is {res}')
