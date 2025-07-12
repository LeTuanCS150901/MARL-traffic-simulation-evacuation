from loguru import logger
import os 
import time

# Define the log file name with current date and time  
log_file_name = f"logs/files/log_{os.path.splitext(os.path.basename(__file__))[0]}_{time.strftime('%Y%m%d_%H%M%S')}.log"  

# Ensure the log directory exists  
os.makedirs(os.path.dirname(log_file_name), exist_ok=True)  
# Write log messeges to the file
logger.add(log_file_name, backtrace=True, diagnose=True)