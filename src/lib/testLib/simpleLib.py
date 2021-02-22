from logs import logDecorator as lD
import jsonref

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.testLib.simpleLib'

@lD.log(logBase + '.simpleTestFunction')
def simpleTestFunction(logger, a, b):
    '''simple test function for testing
    
    this takes two inputs and returns the 
    sum of the two inputs. This might result
    in an error. If such a thing happens, 
    this function will catch this error and 
    log it. It will raise the error again 
    to be caught at a higher level function.
    
    Parameters
    ----------
    a : {any type}
        the first input
    b : {similar type as `a`}
        the second input
    
    Returns
    -------
    similar type as `a` and `b`
        the sum of `a` and `b`
    '''
    try:
        result = a+b
    except Exception as e:
        logger.error('Unable to add the two values [{}] and [{}]:\n{}'.format(
            a, b, str(e)))
        raise

    return result

