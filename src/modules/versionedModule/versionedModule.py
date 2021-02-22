from logs import logDecorator as lD 
import jsonref, pprint

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.versionedModule.versionedModule'

configM = jsonref.load(open('../config/modules/versionedModule.json'))['params']

from lib.versionedLib import versionedLib

@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''a simple function that uses two versions of a library
    
    This function takes two versions of a library and uses
    them one after the other. This demonstrates a simple
    way in which you can version control your libraries 
    directly without going through the Git interface.
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    try:

        for v in configM['versions']:
            lib1 = versionedLib.getLib(v, 'lib1')
            lib2 = versionedLib.getLib(v, 'lib2')

            # Call the functions in lib1
            lib1.someVersionedLib()
            
            # Use the class in lib2
            inst = lib2.someVersionedClass(f'xxxx--> [{v:30s}]\n')
            print(inst)


    except Exception as e:
        logger.error(f'Error executing functio: {e}')


    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for versionedModule
    
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
    print('Main function of versionedModule')
    print('='*30)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)

    doSomething()

    print('Getting out of versionedModule')
    print('-'*30)

    return

