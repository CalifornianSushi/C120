import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# words to be ignored/omitted while framing the dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

# load the model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def preprocess_user_input(user_input):
    bag_of_words = []

    # tokenize the user_input
    words = nltk.word_tokenize(user_input)

    # convert the user input into its root words: stemming
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]

    # Remove duplicacy and sort the user_input
    words = sorted(list(set(words)))

    # Input data encoding: Create BOW for user_input
    for w in words:
        for word in get_stem_words(w):
            bag_of_words.append(1) if word in words else bag_of_words.append(0)

    return np.array([bag_of_words])

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)

    # extract the class from the predicted_class_label
    predicted_class = classes[predicted_class_label]

    # now we have the predicted tag, select a random response
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            bot_response = random.choice(intent['responses'])
            return bot_response

print("Hi, I am Stella. How can I help you?")

while True:
    # take input from the user
    user_input = input('Type your message here: ')

    response = bot_response(user_input)
    print("Bot Response: ", response)
