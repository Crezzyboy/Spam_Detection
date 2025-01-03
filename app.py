import streamlit as st
# text preproceing
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import re

def text_pre(text):
    text=text.lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    text= re.sub(r'\d+(\.\d+)?', '', text)
    text=word_tokenize(text)
    stop_words=set(stopwords.words("english"))
    words=[word for word in text if word not in stop_words]
    lem=WordNetLemmatizer()
    words=[lem.lemmatize(word) for word in words]
    x=" ".join(words)
    return x

st.title("sms_detecion model")
text=st.text_input("Enter your sms")

with open("model.pkl","rb") as file:
    model=pickle.load(file)

with open("vecterizer.pkl","rb") as file:
    vectorizer=pickle.load(file)
if st.button("predict"):
    text=text_pre(text)
    text=vectorizer.transform([text])
    pred=model.predict(text)[0]
    if pred==1:
        st.write("This is a spam message")
    else:
        st.write("This is a notspam message")