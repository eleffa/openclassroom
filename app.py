import streamlit as st
import joblib
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords    
from nltk.stem.porter import PorterStemmer 
import pickle


vectorizer = joblib.load('tf_model.bkl')
xgb = joblib.load('tag_classification.bkl')
#multilabel_binarizer = joblib.load('multilabel_binarizer.pyc')
file = open('multilabel_binarizer.pyc', 'rb')
multilabel_binarizer = pickle.load(file)
file.close()


#defining the funtion that will be used to create the dictionnary
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()      
    #
    # 4. Stem all the words
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]                       
    #
    # 5. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 6. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 7. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 

# Header of tag Prediction
html_temp="""<div style="background-color:#F5F5F5"> <h1 style="color:#31333F;text-align:center;">Tag Prediction </h1></div>"""

text_input = st.text_input( "Entrer votre texte ici 👇" )

if text_input:
  #st.write("You entered: ", text_input)
  review_cleaned = review_to_words(text_input)
  #vectorize the cleaned text
  review_vectorized = vectorizer.transform([review_cleaned]).toarray()
  #predict tags
  y_pred = xgb.predict(review_vectorized)

  #get the tags in text form
  tags_pred = multilabel_binarizer.inverse_transform(y_pred)

  st.success(f'The predict s tags list is {tags_pred}')
