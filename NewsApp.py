import streamlit as st
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re

# Load the models from pickle files
LR = load('LR.pkl')
DT = load('DT.pkl')
GB = load('GB.pkl')
RF = load('RF.pkl')
Xgb = load('Xgb.pkl')
vectorizer = load('vectorizer.pkl')

# Define the function to output label
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Define the manual_testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)  # Corrected variable name
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    pred_Xgb = Xgb.predict(new_xv_test)  # Corrected variable name
    return {
        "LR": output_label(pred_LR[0]),
        "DT": output_label(pred_DT[0]),
        "GB": output_label(pred_GB[0]),
        "RF": output_label(pred_RF[0]),
        "XGb": output_label(pred_Xgb[0])
    }

# Streamlit app
def main():
    st.title("Fake News Detection")
    st.write("Enter the news text below:")

    # Input text area for user input
    news_text = st.text_area("Input News Text:", "")

    # Button to trigger prediction
    if st.button("Predict"):
        # Perform manual testing
        predictions = manual_testing(news_text)
        
        # Display predictions
        st.write("Predictions:")
        st.write("- LR:", predictions["LR"])
        st.write("- DT:", predictions["DT"])
        st.write("- GB:", predictions["GB"])
        st.write("- RF:", predictions["RF"])
        st.write("- XGb:", predictions["XGb"])

if __name__ == "__main__":
    main()
