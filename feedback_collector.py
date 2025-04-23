import json
import os
import datetime
from pathlib import Path

class FeedbackCollector:
    def __init__(self, feedback_file='user_feedback.json'):
        self.feedback_file = feedback_file
        self.ensure_feedback_file_exists()
    
    def ensure_feedback_file_exists(self):
        """Create feedback file with proper structure if it doesn't exist"""
        if not os.path.exists(self.feedback_file):
            initial_data = {
                "feedback": [],
                "last_processed_date": None
            }
            with open(self.feedback_file, 'w') as f:
                json.dump(initial_data, f, indent=4)
            print(f"Created new feedback file: {self.feedback_file}")
    
    def save_feedback(self, user_id, user_message, bot_response, tag, feedback_score):
        """Save user feedback to the JSON file"""
        try:
            # Load existing feedback
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            # Add new feedback entry
            feedback_entry = {
                "user_id": user_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "user_message": user_message,
                "bot_response": bot_response,
                "tag": tag,
                "feedback_score": feedback_score
            }
            
            feedback_data["feedback"].append(feedback_entry)
            
            # Save updated feedback data
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_unprocessed_feedback(self):
        """Get feedback that hasn't been processed in retraining yet"""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            last_processed = feedback_data.get("last_processed_date")
            
            # If never processed before, return all feedback
            if not last_processed:
                return feedback_data["feedback"]
            
            # Otherwise, return only feedback since last processing
            last_processed_date = datetime.datetime.fromisoformat(last_processed)
            new_feedback = [
                item for item in feedback_data["feedback"] 
                if datetime.datetime.fromisoformat(item["timestamp"]) > last_processed_date
            ]
            
            return new_feedback
        except Exception as e:
            print(f"Error getting unprocessed feedback: {e}")
            return []
    
    def mark_as_processed(self):
        """Mark current feedback as processed after retraining"""
        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            
            feedback_data["last_processed_date"] = datetime.datetime.now().isoformat()
            
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=4)
            
            return True
        except Exception as e:
            print(f"Error marking feedback as processed: {e}")
            return False