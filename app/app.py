import gradio as gr
import pickle
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import liwc
import nltk
import os

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('wordnet')

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# --- 1. Load All Models and Preprocessing Objects ---
try:
    with open('trained_logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('preprocessing_objects.pkl', 'rb') as preprocessor_file:
        preprocessing_objects = pickle.load(preprocessor_file)

    with open('mentor_data.pkl', 'rb') as mentor_data_file:
        recovery_dict = pickle.load(mentor_data_file)
        
    tfidf_vectorizer = preprocessing_objects['tfidf_vectorizer']
    word_vectorizer = preprocessing_objects['word_vectorizer']
    imputer = preprocessing_objects['imputer']
    scaler = preprocessing_objects['scaler']
    # class_labels is not needed since the model predicts string labels
    
    # Load drug list for mentor logic only once at startup
    drug_names_df = pd.read_csv("unique_drug_names.csv")
    drug_list = drug_names_df['DrugName'].tolist()

except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing required file: {e}. Please ensure all model and preprocessing files are uploaded.")

# Initialize static tools
sid = SentimentIntensityAnalyzer()
try:
    liwc_analyzer, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')
except FileNotFoundError:
    raise FileNotFoundError("The LIWC dictionary file 'LIWC2007_English100131.dic' is required but was not found.")
lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---
def clean_and_lemmatize(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    words = re.findall(r'\w+', text)
    return [lemmatizer.lemmatize(word) for word in words]

def find_mentioned_drugs(post_text, drugs):
    mentioned = []
    if not isinstance(post_text, str):
        return mentioned
    post_text = post_text.lower()
    for drug in drugs:
        if re.search(r'\b{}\b'.format(re.escape(drug.lower())), post_text):
            mentioned.append(drug)
    return mentioned

# --- 2. Prediction Function ---
def classify_stage_and_recommend(input_text):
    if not input_text.strip():
        return "Please enter some text to classify.", ""
        
    try:
        words = clean_and_lemmatize(input_text)
        
        liwc_counts = Counter(cat for word in words for cat in liwc_analyzer(word) if cat in category_names)
        liwc_features = np.array([liwc_counts[cat] for cat in category_names]).reshape(1, -1)
        
        sentiment_scores = sid.polarity_scores(input_text)
        sentiment_features = np.array([sentiment_scores['neg'], sentiment_scores['pos']]).reshape(1, -1)
        
        tfidf_features = tfidf_vectorizer.transform([input_text]).toarray()
        word_features = word_vectorizer.transform([input_text]).toarray()

        combined_features = np.concatenate((liwc_features, sentiment_features, tfidf_features, word_features), axis=1)
        imputed_features = imputer.transform(combined_features)
        scaled_features = scaler.transform(imputed_features)

        # Make a prediction - model returns string labels directly
        prediction_result = model.predict(scaled_features)
        
        # Extract the predicted stage (it's already a string label)
        predicted_stage = prediction_result[0]

        # --- Mentor Recommendation Logic ---
        mentor_recommendation = ""
        if predicted_stage == 'Addicted':
            user_drugs = set(find_mentioned_drugs(input_text, drug_list))
            if user_drugs:
                common_drugs_count = {
                    mentor: len(user_drugs.intersection(set(mentor_drugs)))
                    for mentor, mentor_drugs in recovery_dict.items()
                }
                potential_mentors = {mentor: count for mentor, count in common_drugs_count.items() if count > 0}
                ranked_mentors = sorted(potential_mentors.items(), key=lambda item: item[1], reverse=True)
                top_3_mentors = [mentor for mentor, _ in ranked_mentors[:3]]

                if top_3_mentors:
                    mentor_recommendation = "You may find support from these mentors: " + ", ".join(top_3_mentors)
                else:
                    mentor_recommendation = "No suitable mentors found for the drugs mentioned."
            else:
                mentor_recommendation = "Please mention specific drugs for mentor recommendations."
        else:
            mentor_recommendation = "Mentor recommendations are provided for users in the 'Addicted' stage."

        return f"The text indicates the person is in the **{predicted_stage}** stage.", mentor_recommendation

    except Exception as e:
        return "An error occurred during classification.", f"Error: {e}"

# --- 3. Gradio Interface ---
iface = gr.Interface(
    fn=classify_stage_and_recommend,
    inputs=gr.Textbox(lines=5, label="Enter a Reddit post or text related to drug use"),
    outputs=[
        gr.Textbox(label="Predicted Stage"),
        gr.Textbox(label="Mentor Recommendation")
    ],
    title="Drug Addiction Stage Classifier and Mentor Recommender",
    description="This application classifies text into five stages of addiction and provides mentor recommendations for users in the 'Addicted' stage.",
    examples=[
        ["I am still trying to process last night's events. We had made the decision to experiment with a cocaine for the first time, and it was certainly a wild ride. We each had a small dose and hung out, putting on some music to pass the time. After about an hour and a half, we suddenly realized one of our friends wasn't in the room with us anymore. A quick search revealed that he was in his bedroom, standing perfectly still and fixated on a corner of the ceiling. It was clear he was in an entirely different world, and it was quite a shock to see him like that. For a first experience, it was certainly unforgettable.", "The text indicates the person is in the **Addicted** stage.", "You may find support from these mentors: NAME_768, NAME_916, NAME_1084"],
        ["Today marks one year of sobriety! So grateful for my support system.", "The text indicates the person is in the **A-Recovery** stage.", "Mentor recommendations are provided for users in the 'Addicted' stage."],
        ["I’ve been clean for three months and it’s a daily struggle, but I’m doing it.", "The text indicates the person is in the **M-Recovery** stage.", "Mentor recommendations are provided for users in the 'Addicted' stage."],
        ["Feeling proud of myself for reaching 100 days without using.", "The text indicates the person is in the **A-Recovery** stage.", "Mentor recommendations are provided for users in the 'Addicted' stage."]]
)

if __name__ == "__main__":
    iface.launch()