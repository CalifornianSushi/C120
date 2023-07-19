import nltk
nltk.download('punkt')
nltk.download('wordnet')

# To stem words
from nltk.stem import PorterStemmer

# Create an instance of class PorterStemmer
stemmer = PorterStemmer()

# Importing json lib
import json
import pickle
import numpy as np

# List of unique root words in the data
words = []

# List of unique tags in the data
classes = []

# List of the pair of (['words', 'of', 'the', 'sentence'], 'tags')
pattern_word_tags_list = []

# Words to be ignored while creating the Dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# Open the JSON file and load data from it.
with open('intents.json') as train_data_file:
    data = json.load(train_data_file)

# Function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)      
    return stem_words

# Function to create the corpus for the chatbot
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    for intent in data['intents']:
        # Add all patterns and tags to a list
        for pattern in intent['patterns']:
            # Tokenize the pattern          
            pattern_words = nltk.word_tokenize(pattern)
            # Add the tokenized words to the words list
            words.extend(pattern_words)      
            # Add the 'tokenized word list' along with the 'tag' to pattern_word_tags_list
            pattern_word_tags_list.append((pattern_words, intent['tag']))
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    stem_words = get_stem_words(words, ignore_words) 

    # Remove duplicate words from stem_words
    stem_words = set(stem_words)

    # Sort the stem_words list and classes list
    stem_words = sorted(list(stem_words))
    classes = sorted(classes)

    # Print stem_words
    print('stem_words list:', stem_words)

    return stem_words, classes, pattern_word_tags_list

# Training Dataset: 
# Input Text ----> as Bag of Words 
# Tags ---------> as Label

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        # Example: word_tags = (['hi', 'there'], 'greetings')
        pattern_words = word_tags[0]  # ['Hi', 'There]
        bag_of_words = []

        # Stemming pattern words before creating Bag of Words
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        # Input data encoding 
        for word in stem_words:
            if word in stemmed_pattern_word:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)

        bag.append(bag_of_words)

    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        # Start with a list of 0s 
        labels_encoding = [0] * len(classes)  
        # Example: word_tags = (['hi', 'there'], 'greetings')
        tag = word_tags[1]   # 'greetings'
        tag_index = classes.index(tag)
        # Labels Encoding
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)

    return np.array(labels)

def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Convert Stem words and Classes to Python pickel file format
    pickle.dump(stem_words, open('words.pkl', 'wb'))
    pickle.dump(tag_classes, open('classes.pkl', 'wb'))

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

bow_data, label_data = preprocess_train_data()

# After completing the code, remove comments from print statements
print("First BOW encoding:", bow_data[0])
print("First Label encoding:", label_data[0])


