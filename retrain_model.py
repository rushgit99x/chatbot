#!/usr/bin/env python3
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
