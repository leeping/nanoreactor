from logging import *
import sys, re
import time

""" Nanoreactor logger class. Simple customizations on top of the default logger. """

class RawStreamHandler(StreamHandler):
    """Exactly like output.StreamHandler except it does no extra
    formatting before sending logging messages to the stream. This is
    more compatible with how output has been displayed. Default stream
    has also been changed from stderr to stdout"""
    def __init__(self, stream = sys.stdout):
        super(RawStreamHandler, self).__init__(stream)
    
    def emit(self, record):
        message = record.getMessage()
        self.stream.write(message)
        self.flush()

class NanoLogger(Logger):
    """ Nanoreactor module level logger. """
    def __init__(self, name):
        super(NanoLogger, self).__init__(name)
        super(NanoLogger, self).addHandler(RawStreamHandler(sys.stdout))
        self.verbosity = 0

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    def debug(self, msg, printlvl=0, newline=True, *args, **kwargs):
        if printlvl > self.verbosity: return
        if self.verbosity >= 2 and not msg.strip().startswith(time.ctime()):
            msg = "%s : %s" % (time.ctime(), msg)
        if newline and not msg.endswith('\n'):
            msg += "\n"
        super(NanoLogger, self).debug(msg, *args, **kwargs)

    def info(self, msg, printlvl=0, newline=True, *args, **kwargs):
        if printlvl > self.verbosity: return
        if self.verbosity >= 2 and not msg.strip().startswith(time.ctime()):
            msg = "%s : %s" % (time.ctime(), msg)
        if newline and not msg.endswith('\n'):
            msg += "\n"
        super(NanoLogger, self).info(msg, *args, **kwargs)

    def warning(self, msg, newline=True, *args, **kwargs):
        if self.verbosity >= 2 and not msg.strip().startswith(time.ctime()):
            msg = "%s : %s" % (time.ctime(), msg)
        if newline and not msg.endswith('\n'):
            msg += "\n"
        super(NanoLogger, self).warning(msg, *args, **kwargs)

    def error(self, msg, newline=True, *args, **kwargs):
        """Print out the error message in red and redirect to stderr."""
        msg = '\n'.join(['\x1b[91m%s\x1b[0m' % s for s in msg.split('\n') if len(s.strip()) > 0])
        if newline and not msg.endswith('\n'):
            msg += '\n'
        for hdlr in (self.parent.handlers if self.propagate else self.handlers):
            if hasattr(hdlr, 'stream'): 
                hdlr.savestream = hdlr.stream
                hdlr.stream = sys.stderr
        super(NanoLogger, self).error(msg, *args, **kwargs)
        for hdlr in (self.parent.handlers if self.propagate else self.handlers):
            if hasattr(hdlr, 'stream'):
                hdlr.stream = hdlr.savestream

# Make sure that modules use the module level logger
setLoggerClass(NanoLogger)

# Set log level to INFO
logger=getLogger('nanoreactor')
logger.setLevel(INFO)
