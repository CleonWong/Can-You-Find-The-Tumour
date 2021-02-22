from logs import logDecorator as lD 
import jsonref, pprint
from lib.celeryWorkerExample import worker_1

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.celeryCheck.celeryCheck'


@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    try:
        result = worker_1.add.delay(2, 2)
        for i in range(100):
            print(result.state)

        if result.state == 'SUCCESS':
            r = result.get()
            print(f'The result of this calculation is: {r}')
    except Exception as e:
        logger.error(f'Unable to geenrate the celery module" {e}')

    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''

    print('='*30)
    print('Main function of celeryCheck')
    print('='*30)
    
    doSomething()

    print('Getting out of celeryCheck')
    print('-'*30)

    return

