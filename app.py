import streamlit as st
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    #Test_punc_removed_join = ''.join([i for i in Test_punc_removed_join if not i.isdigit()]) # retirer les chiffres
    Test_punc_removed_join_clean = [word.lower() for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def remove_code(text):
    pointer=text.find('<code>')
    while pointer!=-1:
        ender=text.find(u'</code>')
        text=text.replace(text[pointer:ender+7],' ')
        pointer=text.find('<code>')
    return text

def remove_html(text):
    return BeautifulSoup(text, 'lxml').get_text()

def cleanText(text):
  text = remove_code(text)
  text = remove_html(text)
  text = lemmatize_words(text)
  text = message_cleaning(text)
  return text

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
