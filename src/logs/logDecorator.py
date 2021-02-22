from datetime import datetime as dt
from time import time
import json, logging, sys
from functools import wraps
import logstash

class log(object):
    '''decorator for logging values
    
    This decorator can be used for injecting a
    logging function into a particular function. 
    This takes a function and injects a logger 
    as the first argument of the decorator. For 
    the generated logger, it is also going to 
    insert the time at which the paticular function
    was called, and then again when the function was
    finished. These serve as convinient functions for
    inserting values into the decorator. 
    '''

    def __init__(self, base):
        '''initialize the decorator
        
        Parameters
        ----------
        base : {str}
            The string used for prepending the value of the decorator
            with the right path for this function. 
        '''
        self.base   = base
        return

    def __call__(self, f):

        from time import time

        # Function to return
        @wraps(f)
        def wrappedF(*args, **kwargs):
            logger = logging.getLogger(self.base)
            logger.info('Starting the function [{}] ...'.format(f.__name__))
            t0     = time()
            result = f(logger, *args, **kwargs)
            logger.info('Finished the function [{}] in {:.6e} seconds'.format( 
                f.__name__, time() - t0 ))

            return result

        return wrappedF

class logInit(object):
    '''initialize the decorator for logging
    
    This generates a decorator using a fresh file with the right
    date and time within the function name. This way it will be
    east to find the last log file generated using a simple script
    in the case that a person wants to generate instantaneous 
    statistics for the last run. 
    '''

    def __init__(self, base, level, specs):
        '''Initialize the logger object for the program
        
        This logger object generates a new logger object. This is able to handle
        significantly improved logging capabilities in comparison to earlier 
        versions of the cutter. For details of the available functionality, check
        the logging documentation. 
        
        Parameters
        ----------
        base : {str}
            name that starts the logging functionality
        level : {str}
            One of the different types if logs available - ``CRITICAL``, ``ERROR``, 
            ``WARNING``, ``INFO`` and ``DEBUG``. These wil be mapped to one of the 
            correct warning levels with the logging facility. In case there is an 
            input that is not one shown here, it will be automatically be mapped to
            ``INFO``.
        specs : {dict}
            Dictionary specifying the different types if formatters to be generated
            while working with logs. 
        '''
        self.base   = base
        self.specs  = specs
        levels = {   
            'CRITICAL': logging.CRITICAL,
            'ERROR'   : logging.ERROR,
            'WARNING' : logging.WARNING,
            'INFO'    : logging.INFO,
            'DEBUG'   : logging.DEBUG}

        self.logLevel = levels.get(level, logging.INFO)

        return

    def __call__(self, f):

        # Function to return
        @wraps(f)
        def wrappedF(*args, **kwargs):

            # Generate a logger ...
            logger    = logging.getLogger(self.base)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            now       = dt.now().strftime('%Y-%m-%d_%H-%M-%S')

            # Generate a file handler if necessary
            if ('file' in self.specs) and self.specs['file']['todo']:
                fH  = logging.FileHandler( '{}/{}.log'.format(
                    self.specs['file']['logFolder'], now) )
                
                fH.setFormatter(formatter)
                logger.addHandler(fH)

            # Generate a file handler if necessary
            if ('stdout' in self.specs) and self.specs['stdout']['todo']:
                cH = logging.StreamHandler(sys.stdout)
                cH.setFormatter(formatter)
                logger.addHandler(cH)

            # Generate a file handler if necessary
            if ('logstash' in self.specs) and self.specs['logstash']['todo']:

                tags = [ 'mammogram-cv' , now]

                if 'tags' in self.specs['logstash']:
                    tags += self.specs['logstash']['tags']
                

                lH = logstash.TCPLogstashHandler(
                    host    = self.specs['logstash']['host'], 
                    port    = self.specs['logstash']['port'], 
                    version = self.specs['logstash']['version'],
                    tags    = tags)
                
                logger.addHandler(lH)

            # set the level of the handler
            logger.setLevel(self.logLevel)

            logger.info('Starting the main program ...')
            t0     = time()
            result = f(logger, *args, **kwargs)
            logger.info('Finished the main program in {:.6e} seconds'.format( time() - t0 ))

            return result

        return wrappedF
        