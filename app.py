from flask import Flask, render_template, request, jsonify, session
import uuid
import json
import pickle
import numpy as np
import datetime
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import nltk
from feedback_collector import FeedbackCollector
from model_retrainer import ModelRetrainer

# Make sure NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')

# RLChatbot class definition
class RLChatbot:
    def __init__(self, model_path='chatbot_model.h5'):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('intents.json').read())
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = tf.keras.models.load_model(model_path)
        
        # User feedback tracking for reinforcement learning
        self.conversation_history = {}  # Using dictionary to track by user_id
        self.feedback_history = []
        self.learning_rate = 0.01
        
        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector()
        
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for word in sentence_words:
            for i, w in enumerate(self.words):
                if w == word:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({
                'intent': self.classes[r[0]],
                'probability': str(r[1])
            })
        return return_list
    
    def get_response(self, user_id, user_message, intents_list):
        if not intents_list:
            response = "I'm not sure I understand. Could you rephrase that?"
            tag = "unknown"
        else:
            tag = intents_list[0]['intent']
            list_of_intents = self.intents['intents']
            
            for i in list_of_intents:
                if i['tag'] == tag:
                    # For reinforcement learning, we select responses weighted by their previous success
                    responses = i['responses']
                    # Check if we have response weights, if not initialize
                    if 'response_weights' not in i:
                        i['response_weights'] = [1.0] * len(responses)
                    
                    # Select response based on weights
                    weights = np.array(i['response_weights'])
                    weights = weights / np.sum(weights)  # Normalize
                    response_idx = np.random.choice(len(responses), p=weights)
                    response = responses[response_idx]
                    break
            else:
                response = "I'm not sure I understand. Could you rephrase that?"
                tag = "unknown"
        
        # Store this interaction for later reinforcement
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
            
        self.conversation_history[user_id].append({
            'user_message': user_message,
            'bot_response': response,
            'tag': tag
        })
        
        return response, tag
    
    def provide_feedback(self, user_id, feedback_score):
        """
        User provides feedback on the last response
        Score: 1 (positive), 0 (neutral), -1 (negative)
        """
        if user_id not in self.conversation_history or not self.conversation_history[user_id]:
            return False
        
        last_interaction = self.conversation_history[user_id][-1]
        tag = last_interaction['tag']
        user_message = last_interaction['user_message']
        bot_response = last_interaction['bot_response']
        
        # Save feedback to the central collector
        self.feedback_collector.save_feedback(
            user_id=user_id, 
            user_message=user_message, 
            bot_response=bot_response,
            tag=tag, 
            feedback_score=feedback_score
        )
        
        # Also update in-memory weights for immediate use
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                # Find the response in the list
                try:
                    response_idx = intent['responses'].index(bot_response)
                    
                    # Initialize weights if needed
                    if 'response_weights' not in intent:
                        intent['response_weights'] = [1.0] * len(intent['responses'])
                    
                    # Update weights based on feedback
                    current_weight = intent['response_weights'][response_idx]
                    if feedback_score > 0:
                        intent['response_weights'][response_idx] = min(5.0, current_weight + self.learning_rate)
                    elif feedback_score < 0:
                        intent['response_weights'][response_idx] = max(0.1, current_weight - self.learning_rate)
                    
                    # Save updated intents for immediate use
                    with open('intents.json', 'w') as f:
                        json.dump(self.intents, f, indent=4)
                    
                    return True
                except ValueError:
                    # Response might have been changed or removed
                    return False
        
        return False

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "chatbot_reinforcement_learning_key"

# Simple admin password (replace with proper authentication in production)
ADMIN_PASSWORD = "admin123"  # Change this to a secure password

# Check if required files exist, if not prompt user to run training
def check_required_files():
    import os
    required_files = ['intents.json', 'words.pkl', 'classes.pkl', 'chatbot_model.h5']
    missing_files = [file for file in required_files if not os.path.exists(file)]
    return missing_files

# Initialize chatbot
try:
    chatbot = RLChatbot()
    chatbot_ready = True
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    chatbot_ready = False

@app.route("/")
def home():
    # Generate a session ID if not exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Check if required files exist
    missing_files = check_required_files()
    if missing_files:
        return render_template("error.html", message=f"Missing required files: {', '.join(missing_files)}. Please run model_training.py first.")
    
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Check if chatbot is ready
    if not chatbot_ready:
        return jsonify({
            "response": "The chatbot is not initialized properly. Please make sure the model files exist.",
            "tag": "error",
            "confidence": 0
        })
    
    data = request.get_json()
    user_message = data.get("message")
    user_id = session.get('user_id', str(uuid.uuid4()))
    
    try:
        intents_list = chatbot.predict_class(user_message)
        response, tag = chatbot.get_response(user_id, user_message, intents_list)
        
        return jsonify({
            "response": response,
            "tag": tag,
            "confidence": float(intents_list[0]["probability"]) if intents_list else 0
        })
    except Exception as e:
        return jsonify({
            "response": f"An error occurred: {str(e)}",
            "tag": "error",
            "confidence": 0
        })

@app.route("/feedback", methods=["POST"])
def feedback():
    if not chatbot_ready:
        return jsonify({
            "success": False,
            "message": "The chatbot is not initialized properly."
        })
    
    data = request.get_json()
    feedback_score = data.get("score", 0)  # 1 (positive), 0 (neutral), -1 (negative)
    user_id = session.get('user_id', str(uuid.uuid4()))
    
    try:
        success = chatbot.provide_feedback(user_id, feedback_score)
        
        return jsonify({
            "success": success,
            "message": "Thank you for your feedback!" if success else "Could not process feedback"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"An error occurred: {str(e)}"
        })

@app.route("/status")
def status():
    missing_files = check_required_files()
    
    return jsonify({
        "status": "ready" if not missing_files and chatbot_ready else "not_ready",
        "missing_files": missing_files,
        "chatbot_initialized": chatbot_ready
    })

@app.route("/admin", methods=["GET", "POST"])
def admin():
    # Load feedback data for display
    try:
        with open('user_feedback.json', 'r') as f:
            feedback_data = json.load(f)
    except Exception as e:
        feedback_data = {"feedback": []}
        print(f"Error loading feedback data: {e}")

    if request.method == "POST":
        password = request.form.get("password")
        if password == ADMIN_PASSWORD:
            session['admin_authenticated'] = True
            return render_template("admin.html", feedback_data=feedback_data)
        else:
            return render_template("admin_login.html", error="Invalid password")
    else:
        if session.get('admin_authenticated'):
            return render_template("admin.html", feedback_data=feedback_data)
        return render_template("admin_login.html")

@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    if not session.get('admin_authenticated'):
        return jsonify({"success": False, "message": "Unauthorized access"}), 401
    
    try:
        retrainer = ModelRetrainer()
        # Force retraining by ignoring minimum feedback count
        modified = retrainer.update_intents_from_feedback(ignore_min_feedback=True)
        if modified:
            retrainer.rebuild_model()
            return jsonify({"success": True, "message": "Model retraining completed"})
        else:
            return jsonify({"success": False, "message": "No new feedback to process"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error during retraining: {str(e)}"})

@app.route("/admin/add_feedback", methods=["POST"])
def add_feedback():
    if not session.get('admin_authenticated'):
        return jsonify({"success": False, "message": "Unauthorized access"}), 401
    
    try:
        data = request.get_json()
        user_message = data.get("user_message")
        bot_response = data.get("bot_response")
        tag = data.get("tag")
        feedback_score = data.get("feedback_score")

        # Validate input
        if not all([user_message, bot_response, tag, feedback_score is not None]):
            return jsonify({"success": False, "message": "All fields are required"})

        # Generate unique user ID and timestamp
        user_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()

        # Save feedback using FeedbackCollector
        feedback_collector = FeedbackCollector()
        success = feedback_collector.save_feedback(
            user_id=user_id,
            user_message=user_message,
            bot_response=bot_response,
            tag=tag,
            feedback_score=feedback_score
        )

        if success:
            return jsonify({"success": True, "message": "Feedback added successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to save feedback"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error adding feedback: {str(e)}"})

if __name__ == "__main__":
    # Check if required files exist
    missing_files = check_required_files()
    if missing_files:
        print(f"WARNING: Missing required files: {', '.join(missing_files)}")
        print("Please run model_training.py first to create these files.")
    
    app.run(debug=True)