from loguru import logger

class MyLogger:
    def __init__(self, enable_log: bool = True):
        self.enable_log = enable_log
        # We don't store the logger here because .opt(depth=1) 
        # needs to be fresh for the call stack to work correctly.

    def info(self, *args, **kwargs):
        if self.enable_log:
            logger.opt(depth=1).info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.enable_log:
            logger.opt(depth=1).debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.enable_log:
            logger.opt(depth=1).warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self.enable_log:
            logger.opt(depth=1).error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        if self.enable_log:
            logger.opt(depth=1).critical(*args, **kwargs)

    def exception(self, *args, **kwargs):
        if self.enable_log:
            logger.opt(depth=1).exception(*args, **kwargs)

    def add(self, *args, **kwargs):
        if self.enable_log:
            return logger.add(*args, **kwargs)
        return None

    def remove(self, handler_id=None):
        if self.enable_log:
            logger.remove(handler_id) if handler_id is not None else logger.remove()

    def __getattr__(self, name):
        if not self.enable_log:
            return lambda *args, **kwargs: None
        # This handles .success() or other Loguru methods
        return getattr(logger.opt(depth=1), name)