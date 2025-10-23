from loguru import logger

class MyLogger:
    def __init__(self, enable_log: bool = True):
        self.enable_log = enable_log

    # Explicitly define common methods for clarity
    def info(self, *args, **kwargs):
        if self.enable_log:
            logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.enable_log:
            logger.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.enable_log:
            logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self.enable_log:
            logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if self.enable_log:
            logger.critical(*args, **kwargs)

    def exception(self, *args, **kwargs):
        if self.enable_log:
            logger.exception(*args, **kwargs)

    # Sink management
    def add(self, *args, **kwargs):
        """Add a new log sink if logging is enabled."""
        if self.enable_log:
            return logger.add(*args, **kwargs)
        return None

    def remove(self, handler_id=None):
        """Remove a log sink if logging is enabled."""
        if self.enable_log:
            if handler_id is not None:
                logger.remove(handler_id)
            else:
                logger.remove()

    def __getattr__(self, name):
        """
        Forward any undefined attributes (like .success, .trace, .catch)
        to loguru.logger when logging is enabled.
        """
        if not self.enable_log:
            # Return a no-op function when logging is disabled
            return lambda *args, **kwargs: None
        return getattr(logger, name)
