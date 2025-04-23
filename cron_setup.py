#!/usr/bin/env python3
"""
Script to set up automatic retraining cron job
"""
import os
import sys
import subprocess
from crontab import CronTab

def setup_cron_job():
    try:
        # Get current directory for absolute paths
        current_dir = os.path.abspath(os.path.dirname(__file__))
        
        # Create the retraining script with proper path
        retraining_script = os.path.join(current_dir, "retrain_model.py")
        
        # Check if the retraining script exists, if not create it
        if not os.path.exists(retraining_script):
            with open(retraining_script, "w") as f:
                f.write('''#!/usr/bin/env python3
import sys
import os
import logging
from datetime import datetime

# Setup logging
log_file = os.path.join(os.path.dirname(__file__), "retraining.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

# Import the retrainer
try:
    from model_retrainer import ModelRetrainer
    
    logging.info(f"Starting scheduled model retraining at {datetime.now().isoformat()}")
    
    retrainer = ModelRetrainer()
    result = retrainer.retrain_if_needed()
    
    if result:
        logging.info("Model retraining completed successfully")
    else:
        logging.info("No retraining was necessary")
        
except Exception as e:
    logging.error(f"Error during scheduled retraining: {e}")
    sys.exit(1)
''')
            # Make the script executable
            os.chmod(retraining_script, 0o755)
            print(f"Created retraining script: {retraining_script}")
        
        # Set up cron job to run daily at midnight
        cron = CronTab(user=True)
        
        # Remove any existing jobs with the same comment
        for job in cron.find_comment('chatbot_daily_retraining'):
            cron.remove(job)
            print("Removed existing retraining cron job")
        
        # Create new job
        job = cron.new(command=f"{sys.executable} {retraining_script}")
        job.set_comment('chatbot_daily_retraining')
        job.setall('0 0 * * *')  # Run at midnight (00:00) every day
        
        # Write cron job to crontab
        cron.write()
        print("Successfully set up daily retraining cron job!")
        print("The model will be retrained at midnight every day if sufficient new feedback is available.")
        
        return True
    except ImportError:
        print("Error: python-crontab module not installed.")
        print("Please install it with: pip install python-crontab")
        return False
    except Exception as e:
        print(f"Error setting up cron job: {e}")
        return False

if __name__ == "__main__":
    print("Setting up automatic model retraining cron job...")
    success = setup_cron_job()
    
    if success:
        print("\nSetup completed successfully!")
        print("To test the cron job, you can run:")
        print(f"  python {os.path.join(os.path.abspath(os.path.dirname(__file__)), 'retrain_model.py')}")
    else:
        print("\nSetup failed. Please check the error messages above.")
        
    # Check if python-crontab is installed
    try:
        import crontab
    except ImportError:
        print("\nNote: To use this script, you need the python-crontab package.")
        print("Install it with: pip install python-crontab")