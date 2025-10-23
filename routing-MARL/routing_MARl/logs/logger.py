import os 
import time
from logs.mylogger import MyLogger

enable_log = False
_logger_instance = None

def setLoggerEnable(enable: bool):
    global enable_log, _logger_instance
    enable_log = enable
    _logger_instance = None  # Force recreation

def get_logger():
    global enable_log, _logger_instance
    
    if _logger_instance is None:
        print(f"[DEBUG] Creating logger with enable_log={enable_log}")  # Debug line
        _logger_instance = MyLogger(enable_log=enable_log)
        
        log_file_name = f"logs/files/log_{os.path.splitext(os.path.basename(__file__))[0]}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        _logger_instance.add(log_file_name, backtrace=True, diagnose=True)
    
    return _logger_instance

# Wrapper class that proxies all calls to the actual logger
class LazyLogger:
    def __getattr__(self, name):
        # Every time call logger.info(), logger.debug(), etc.
        # it gets the current logger instance
        return getattr(get_logger(), name)
    
    def __call__(self, *args, **kwargs):
        return get_logger()(*args, **kwargs)

# Create the lazy logger instance
logger = LazyLogger()