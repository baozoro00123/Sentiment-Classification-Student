import streamlit as st
import joblib
import string
import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from PIL import Image
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import speech_recognition as sr


model_ug_cbow = KeyedVectors.load('w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('w2v_model_ug_sg.word2vec')

# Load Mô Hình 
Svm_TF_IDF = joblib.load('svm_tfidf.pkl')
Svm_BoW = joblib.load('svm_BoW.pkl')
print(1)
# Load Vector
vec_TF_IDF = joblib.load('tfidf_vec.pkl')
vec_BoW = joblib.load('BoW_vec.pkl')

# Load Vector SVD
svd_TF_IDF = joblib.load('svd_tfidf.pkl')
svd_BoW = joblib.load('svd_BoW.pkl')

# tokenizer
tokenizer = joblib.load('tokenizer.joblib')

# CNN- LSTM
LSTM_CNN = load_model('model_lstm_cnn_33.h5')

print("Load Thành Công")
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenize = word_tokenize(text)
    filename = 'stopwords.csv'
    data = pd.read_csv(filename, names=['word'])
    list_stopwords = data['word']
    myarray = np.asarray(list_stopwords)
    words = [word for word in tokenize if word.lower() not in myarray]
    sentence = ' '.join(words)
    return [sentence]

# doc = input("Nhập text dự đoán: ")

def transform_doc (doc):
    doc = preprocess_text(doc)
    # vec tf
    vec_doc_tf = vec_TF_IDF.transform(doc)
    vec_doc_tf= svd_TF_IDF.transform(vec_doc_tf)
    # vector bow
    vec_doc_bow = vec_BoW.transform(doc)
    vec_doc_bow= svd_BoW.transform(vec_doc_bow)
    return vec_doc_tf, vec_doc_bow 

def predict_tf_bow( vec_doc_tf, vec_doc_bow):
    y_pred_tf = Svm_TF_IDF.predict(vec_doc_tf)
    y_pred_bow = Svm_BoW.predict(vec_doc_bow)
    return y_pred_tf, y_pred_bow

# embeddings
embeddings_index = {}
for w in list(model_ug_cbow.wv.index_to_key):
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])
num_words = 10000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
        
def w2v (doc):
    doc = preprocess_text(text)
    doc_sequence = tokenizer.texts_to_sequences([doc])
    doc_padded = pad_sequences(doc_sequence, maxlen=150)
    predictions = (LSTM_CNN.predict(doc_padded)> 0.5).astype("float32")
    a = np.argmax(predictions)
    return a

import base64

def autoplay_audio(file_path: str, volume: float):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true" volume="{volume}" onended="this.currentTime=0; this.play()">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


file_path = "Một Ngày Chẳng Nắng.mp3"
volume = 0.5
autoplay_audio(file_path, volume)

st.title("Sentiment Classification Student App")
st.write('\n')

image = Image.open('sentiment-analysis.png')
show = st.image(image, use_column_width=True)

option = st.radio(
    "Lựa chọn thành phần Input",
    ("Text", "Audio")
)
if option == "Text":
    with st.form(key='emotion_clf_form'):
        text = st.text_area("Nhập văn bản cần dự đoán tại đây: ")
        submit = st.form_submit_button(label='Dự đoán cảm xúc')
        
        if submit:
            st.success('Văn bản vừa nhập: ')
            st.write(text)
            st.success("Prediction")
            
            vec_tf, vec_bow = transform_doc(text)
            vec_tf, vec_bow = predict_tf_bow(vec_tf, vec_bow)
            
            vec_w2v = w2v (text)
            
            st.spinner('Classifying ...')
            if vec_tf[0] and  vec_bow[0] and vec_w2v  == 1:
                st.write("SVM + BoW dự đoán: Tích cực 🤗")
                st.write("SVM + TF-IDF dự đoán: Tích cực 🤗")
                
                st.balloons()
                
            else:
                st.write("SVM + BoW dự đoán: Tiêu cực 😔")
                st.write("TF-IDF dự đoán: Tiêu cực 😔")

else:
    with st.form(key='emotion_clf_form'):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Hãy nói gì đó...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            # st.write("Bạn vừa nói: ",  text)
        except sr.UnknownValueError:
            st.write("Không thể nhận dạng giọng nói!")
        except sr.RequestError as e:
            st.write("Lỗi kết nối tới Google Speech Recognition service; {0}".format(e))
        
        submit = st.form_submit_button(label='Dự đoán cảm xúc')
        
        if submit:
            st.success("Prediction")
            
            vec_tf, vec_bow = transform_doc(text)
            vec_tf, vec_bow = predict_tf_bow(vec_tf, vec_bow)
            
            vec_w2v = w2v (text)
            
            st.spinner('Classifying ...')
            if vec_tf[0] and  vec_bow[0] and vec_w2v  == 1:
                st.write("SVM + BoW dự đoán: Tích cực 🤗")
                st.write("SVM + TF-IDF dự đoán: Tích cực 🤗")
                
                st.balloons()
                
            else:
                st.write("SVM + BoW dự đoán: Tiêu cực 😔")
                st.write("TF-IDF dự đoán: Tiêu cực 😔")
        

#python -m streamlit run DA4.py