from logs import logDecorator as lD
import jsonref, psycopg2

from lib.celery.App import app

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + 'lib.celeryWorkerExample.worker_1'

@app.task
@lD.log(logBase + '.add')
def add(logger, a, b):
    '''add two supplied items
    
    This example takes two arguments and adds them together. This
    is much like the function available with ``lib.testLib.simpleLib.py``
    and will be used only for the purpose of having a very simple function
    that can be run via a Celery worker. This function will return the 
    result of the sum, or ``None`` in the case that there was an error.
    Handling errors within distributed computing tasks is generally 
    complicated and needs a significant amount of thought to be put in
    while programming. 
    
    Parameters
    ----------
    logger : {logging.logger}
        The logging instance that will log information about the execution.
        Rememebr that this is a significantly different logger than the
        one that will finally run the system. In a distributed system it
        is possible that multiple loggers will interact differently with
        different loggers and thus colating the different logs might be a
        significant challenge. 
    a : {any type that supports an addition}
        The first element of the binary addition operation
    b : {any type that supports addition}
        The second element of the binary addition operation
    
    Returns
    -------
    type(a+b)
        Result of the addition of the two input values. In case of an error, this
        is going to return a ``None``
    '''

    try:
        result = a+b
        return result
    except Exception as e:
        logger.error('Unable to log the task: {e}')
        return None

    return 
