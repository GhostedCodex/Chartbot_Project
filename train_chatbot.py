import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the intents JSON file
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Initialize lists to hold data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Looping throught the intents JSON file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in our corpus
        documents.append((word_list, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower())
         for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Print stats to see what we have
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save words and classes to pickle files
pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
# Creating an empty array for our output
output_empty = [0] * len(classes)

# Loop through each document
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]

    # Create our bag of words array with 1, if word match found in current pattern
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# Create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data created")

# Building and Training the model

# Create model - 3 layers.
# First layer 128 neurons, second layer 64 neurons
# 3rd output layer contains number of neurons equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile mode. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6,
          momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# Fit the model to the training data
# 200 epochs, batch size of 5
hist = model.fit(np.array(train_x), np.array(
    train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('model/chatbot_model.h5', hist)

print("Model created and saved to model/ folder")
