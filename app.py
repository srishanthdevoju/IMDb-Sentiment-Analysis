import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

MAXLEN = 200

# Load Models
model_rnn = load_model("simple_rnn_model.h5")
model_lstm = load_model("lstm_model.h5")
model_gru = load_model("gru_model.h5")

# IMDb Vocabulary
word_index = imdb.get_word_index()

MAX_FEATURES = 10000

word_index = imdb.get_word_index()

def encode_review(text):

    words = text.lower().split()

    encoded = [1]

    for word in words:

        idx = word_index.get(word)

        if idx is not None and idx < MAX_FEATURES:
            encoded.append(idx + 3)
        else:
            encoded.append(2)

    return encoded

def preprocess(text):

    seq = encode_review(text)

    seq = [min(x, 9999) for x in seq]

    return pad_sequences(
        [seq],
        maxlen=MAXLEN,
        padding='pre',
        truncating='pre'
    )

def predict(model, review):

    processed = preprocess(review)

    st.write("Max token:", processed.max())
    score = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if score >= 0.5
        else "Negative"
    )

    confidence = (
        score
        if score >= 0.5
        else 1-score
    )

    return sentiment, confidence, score

# UI

st.title("🎬 Movie Review Sentiment Analysis System")

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

review = st.text_area(
    "Enter your movie review here..."
)

selected_model = st.radio(
    "Select Model",
    ["SimpleRNN","LSTM","GRU"]
)

if st.button("Analyze Review"):

    models = {
        "SimpleRNN": model_rnn,
        "LSTM": model_lstm,
        "GRU": model_gru
    }

    sentiment, confidence, score = predict(
        models[selected_model],
        review
    )

    st.success(
        f"Sentiment: {sentiment}"
    )

    st.info(
        f"Confidence: {confidence*100:.2f}%"
    )

    st.subheader(
        "Probability Distribution"
    )

    chart_df = pd.DataFrame({
        "Probability":[
            score,
            1-score
        ]
    },
    index=[
        "Positive",
        "Negative"
    ])

    st.bar_chart(chart_df)

    st.subheader(
        "Compare All Models"
    )

    results = []

    for name, model in models.items():

        pred, conf, _ = predict(
            model,
            review
        )

        results.append(
            [name,pred,
             round(conf*100,2)]
        )

    st.dataframe(
        pd.DataFrame(
            results,
            columns=[
                "Model",
                "Prediction",
                "Confidence (%)"
            ]
        )
    )