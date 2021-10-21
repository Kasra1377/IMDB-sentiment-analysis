from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import pickle
import os
import re

app =Flask(__name__ , template_folder="templates")

MODEL_PATH="models/cnn-model.h5"
cnnmodel = load_model(MODEL_PATH, compile=False)

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000 
EMBEDDING_DIM = 100

data = pd.read_csv("data/cleaned_data.csv")
data = data.drop("Unnamed: 0" , axis=1)

X = data.drop("sentiment", axis = 1)
y = data["sentiment"]

X_train , X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
                                                     test_size=0.2, random_state=1)

X_train = X_train["review"].to_list()
X_test = X_test["review"].to_list()
y_train = y_train.to_list()
y_test = y_test.to_list()

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index

trainvalid_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


@app.route("/home/" , methods=["GET" , "POST"])
def home():
    result = ""
    if request.method == "POST":

        text = request.form["review"]
        ps = PorterStemmer()
        CLEAN = re.compile("<.*?>")    # to remove everything between "<>"
        result = re.sub(CLEAN, " ", text)
        result = re.sub("[^a-zA-Z]" , " " , result)
        result = result.lower()
        result = result.split()   # to break sentences into words
        word = [ps.stem(word) for word in result if word not in stopwords.words("english")]
        result = " ".join(word)

        tokens = tokenizer.texts_to_sequences([result])
        sent = pad_sequences(tokens, maxlen=MAX_SEQUENCE_LENGTH)
        pred = cnnmodel.predict(np.array(sent))

        result = np.argmax(pred)
        if result == 1:
            result = "\U0001F603"
        else:
            result = "\U0001F641"

    return render_template("template.html" , result=result)    

if __name__=="__main__":
    app.run(debug=True)