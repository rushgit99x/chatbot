import os
import sys
import json
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
import nltk
from nltk.stem import WordNetLemmatizer
from feedback_collector import FeedbackCollector

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class ModelRetrainer:
    def __init__(self, intents_file='intents.json', model_path='chatbot_model.h5', 
                 feedback_file='user_feedback.json', min_feedback_count=5):
        self.intents_file = intents_file
        self.model_path = model_path
        self.feedback_file = feedback_file
        self.min_feedback_count = min_feedback_count
        self.feedback_collector = FeedbackCollector(feedback_file)
        self.lemmatizer = WordNetLemmatizer()
        
        # Load existing data
        try:
            self.words = pickle.load(open('words.pkl', 'rb'))
            self.classes = pickle.load(open('classes.pkl', 'rb'))
            self.intents = json.loads(open(intents_file).read())
            self.model = tf.keras.models.load_model(model_path)
            print("Loaded existing model and data successfully")
        except Exception as e:
            print(f"Error loading model data: {e}")
            print("Please run model_training.py first to create initial model files.")
            sys.exit(1)
    
    def update_intents_from_feedback(self, ignore_min_feedback=False):
        """Update intents based on collected feedback, including positive feedback as new patterns and responses"""
        unprocessed_feedback = self.feedback_collector.get_unprocessed_feedback()
        
        if not ignore_min_feedback and len(unprocessed_feedback) < self.min_feedback_count:
            print(f"Not enough new feedback to retrain model. Found {len(unprocessed_feedback)}, need {self.min_feedback_count}")
            return False
        
        print(f"Processing {len(unprocessed_feedback)} new feedback entries")
        
        # Group feedback by tag to update response weights and collect positive feedback
        feedback_by_tag = {}
        positive_feedback = []
        for item in unprocessed_feedback:
            try:
                tag = item["tag"]
                score = item.get("feedback_score")
                response = item["bot_response"]
                
                if score is None:
                    print(f"Skipping feedback entry with missing feedback_score: {item}")
                    continue
                
                if tag not in feedback_by_tag:
                    feedback_by_tag[tag] = []
                
                feedback_by_tag[tag].append({
                    "response": response,
                    "score": score
                })
                
                # Collect positive feedback for adding to intents
                if score == 1:
                    positive_feedback.append(item)
            except KeyError as e:
                print(f"Skipping invalid feedback entry missing key {e}: {item}")
                continue
        
        # Update weights and add positive feedback to intents
        modified = False
        for intent in self.intents["intents"]:
            tag = intent["tag"]
            
            # Update response weights based on feedback
            if tag in feedback_by_tag:
                # Initialize response weights if not present
                if "response_weights" not in intent:
                    intent["response_weights"] = [1.0] * len(intent["responses"])
                
                # Update weights based on feedback
                for feedback_item in feedback_by_tag[tag]:
                    response = feedback_item["response"]
                    score = feedback_item["score"]
                    
                    # Find response index
                    try:
                        response_idx = intent["responses"].index(response)
                        
                        # Update weight
                        current_weight = intent["response_weights"][response_idx]
                        learning_rate = 0.1
                        
                        if score > 0:
                            intent["response_weights"][response_idx] = min(5.0, current_weight + learning_rate)
                        elif score < 0:
                            intent["response_weights"][response_idx] = max(0.1, current_weight - learning_rate)
                        
                        modified = True
                    except ValueError:
                        # Response might have been changed or removed
                        print(f"Response '{response}' not found in intent '{tag}', skipping weight update")
                        continue
            
            # Add positive feedback to patterns and responses
            for feedback_item in positive_feedback:
                if feedback_item["tag"] == tag:
                    user_message = feedback_item["user_message"]
                    bot_response = feedback_item["bot_response"]
                    
                    # Check for duplicate pattern (case-insensitive)
                    existing_patterns = [p.lower() for p in intent["patterns"]]
                    if user_message.lower() not in existing_patterns:
                        intent["patterns"].append(user_message)
                        modified = True
                        print(f"Added pattern '{user_message}' to intent '{tag}'")
                    else:
                        print(f"Skipped duplicate pattern '{user_message}' for intent '{tag}'")
                    
                    # Add bot_response to responses if not already present
                    if bot_response not in intent["responses"]:
                        intent["responses"].append(bot_response)
                        if "response_weights" not in intent:
                            intent["response_weights"] = [1.0] * len(intent["responses"])
                        else:
                            intent["response_weights"].append(1.0)  # Initialize weight for new response
                        modified = True
                        print(f"Added response '{bot_response}' to intent '{tag}'")
        
        # Save updated intents
        if modified:
            with open(self.intents_file, 'w') as f:
                json.dump(self.intents, f, indent=4)
            print("Updated intents with new feedback weights and positive feedback")
        
        # Mark feedback as processed
        self.feedback_collector.mark_as_processed()
        
        return modified
    
    def rebuild_model(self):
        """Rebuild and retrain the model with updated data"""
        from model_training import ChatbotTrainer
        
        # Create trainer with updated intents
        trainer = ChatbotTrainer(intents_file=self.intents_file)
        
        # Save current model as backup
        backup_path = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        try:
            self.model.save(backup_path)
            print(f"Created backup of current model: {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create model backup: {e}")
        
        # Train model with fewer epochs for faster retraining
        history = trainer.train_model(epochs=50, batch_size=8, use_lstm=False)
        
        # Clear the feedback file after successful retraining
        self.feedback_collector.clear_feedback()
        
        print("Model retraining complete!")
        return True
    
    def retrain_if_needed(self):
        """Main method to check and retrain model if necessary"""
        print(f"Checking for new feedback at {datetime.now().isoformat()}")
        
        # Update intents with feedback
        modified = self.update_intents_from_feedback()
        
        # If intents were modified, retrain model
        if modified:
            print("Retraining model with updated feedback data")
            self.rebuild_model()
            return True
        else:
            print("No retraining needed")
            return False