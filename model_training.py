# model_training.py
import random # for shuffling data
import json # for reading intents file
import pickle # allows you to save and retrieve complex data structures easily
import numpy as np # for numerical operations
import tensorflow as tf # for building the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt') # Used for tokenizing text into words or sentences
nltk.download('wordnet') # Used for lemmatizing words

class ChatbotTrainer:
    def __init__(self, intents_file='intents.json'): 
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open(intents_file).read())
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']
        self.model = None
    
    def preprocess_data(self):
        # Extract words and classes from intents
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Lemmatize and sort words
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        
        # Save processed data
        pickle.dump(self.words, open('words.pkl', 'wb'))     # Save words
        pickle.dump(self.classes, open('classes.pkl', 'wb')) # Save classes
        
        return self.words, self.classes, self.documents
    
    def prepare_training_data(self): 
        training = []
        output_empty = [0] * len(self.classes)
        
        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append(bag + output_row)
        
        # Shuffle and convert to numpy array
        random.shuffle(training)
        training = np.array(training)
        
        # Split features and labels
        train_x = training[:, :len(self.words)]
        train_y = training[:, len(self.words):]
        
        return train_x, train_y
    
    def build_model(self, input_shape):
        # Create a proper model architecture
        model = Sequential()
        
        # Input layer
        model.add(Dense(128, input_shape=(input_shape,), activation='relu')) # First layer
        model.add(Dropout(0.5)) # Dropout layer to prevent overfitting
        
        # Second dense layer
        model.add(Dense(64, activation='relu')) # Second layer
        model.add(Dropout(0.3)) # Dropout layer
        
        # Output layer
        model.add(Dense(len(self.classes), activation='softmax')) # Output layer with softmax activation
        return model
    
    def build_lstm_model(self, input_shape):
        # Create a model with LSTM (if you really want to use LSTM)
        model = Sequential()
        
        # Reshape input for LSTM (add time dimension)
        model.add(Reshape((1, input_shape), input_shape=(input_shape,)))
        
        # LSTM layer(Long Short-Term Memory Layer)
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.5))
        
        # Dense layer
        model.add(Dense(64, activation='relu')) # Second layer
        model.add(Dropout(0.3))
        
        # Output layer
        model.add(Dense(len(self.classes), activation='softmax'))
        
        return model
    
    def train_model(self, epochs=200, batch_size=8, learning_rate=0.001, use_lstm=False):
        # Preprocess and prepare data
        self.preprocess_data()
        train_x, train_y = self.prepare_training_data()
        
        # Build model based on preference
        if use_lstm:
            self.model = self.build_lstm_model(len(train_x[0]))
        else:
            self.model = self.build_model(len(train_x[0]))
        
        # Use Adam optimizer for better convergence
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # Add model checkpointing to save the best model
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        # Train model
        history = self.model.fit(
            np.array(train_x),
            np.array(train_y),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[checkpoint]
        )
        
        # Save the final model
        self.model.save('chatbot_model.h5')
        
        # Save the words and classes again to ensure they match the model
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))
        
        return history


# Example intents file creation if it doesn't exist
def create_sample_intents_if_needed():
    try:
        with open('intents.json', 'r') as f:
            # File exists, no need to create
            pass
    except FileNotFoundError:
        # Create a sample intents file
        sample_intents = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": [
                        "Hi", "Hello", "Hey", "How are you", "What's up"
                    ],
                    "responses": [
                        "Hello!", "Hey there!", "Hi, how can I help you today?"
                    ]
                },
                {
                    "tag": "goodbye",
                    "patterns": [
                        "Bye", "See you later", "Goodbye", "I'm leaving"
                    ],
                    "responses": [
                        "Goodbye!", "See you soon!", "Take care!"
                    ]
                },
                {
                    "tag": "thanks",
                    "patterns": [
                        "Thank you", "Thanks", "That's helpful"
                    ],
                    "responses": [
                        "You're welcome!", "Happy to help!", "Anytime!"
                    ]
                }
            ]
        }
        
        with open('intents.json', 'w') as f:
            json.dump(sample_intents, f, indent=4)
        print("Created sample intents.json file")


# Main execution block
if __name__ == "__main__":
    # Create sample intents file if needed
    create_sample_intents_if_needed()
    
    # Create and train the model
    trainer = ChatbotTrainer(intents_file='intents.json')
    
    # Choose whether to use LSTM or simple Dense layers
    use_lstm = False  # Set to True if you want to use LSTM
    
    # Train the model
    history = trainer.train_model(epochs=100, batch_size=8, use_lstm=use_lstm)
    print("Model training complete!")
    print("Files created: words.pkl, classes.pkl, chatbot_model.h5, best_model.h5")
    
    # Check if files were created
    import os
    for file in ['words.pkl', 'classes.pkl', 'chatbot_model.h5', 'best_model.h5']:
        if os.path.exists(file):
            print(f"✓ {file} was created successfully")
        else:
            print(f"✗ {file} was NOT created")