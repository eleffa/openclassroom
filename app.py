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
import random
import gensim
import gensim.corpora as corpora
from gensim import models
from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

vectorizer = joblib.load('tf_model.bkl')
xgb = joblib.load('tag_classification.bkl')
multilabel_binarizer = joblib.load('multilabel_binarizer.bkl')

dictionary = joblib.load('dictionary.pkl')
model = joblib.load('lda_model.pkl')



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
  ind_g = []
  for i in range(0,5):
    j = random.randint(0,150)
    ind_g.append(j)

  ind_g = []
  for i in range(0,5):
    j = random.randint(0,150)
    ind_g.append(j)
  
  res = np.array(tags_pred)[0][ind_g]
    
  st.success(f'The predict s tags list is {res}')

  corpus_new = dictionary.doc2bow(review_cleaned)
  topics = model.get_document_topics(corpus_new)
        
  #find most relevant topic according to probability
  relevant_topic = topics[0][0]
  relevant_topic_prob = topics[0][1]
        
  for i in range(len(topics)):
      if topics[i][1] > relevant_topic_prob:
          relevant_topic = topics[i][0]
          relevant_topic_prob = topics[i][1]
                
  #retrieve associated to topic tags present in submited text
  res1 = model.get_topic_terms(topicid=relevant_topic, topn=5)    
  res1 = [dictionary[tag[0]] for tag in res1 if dictionary[tag[0]] in review_cleaned]

  st.success(f'The predict s tags list is {res1}')
