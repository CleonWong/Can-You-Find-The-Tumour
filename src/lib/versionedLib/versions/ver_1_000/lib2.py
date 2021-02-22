from logs       import logDecorator as lD
import jsonref

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.versionedLib.versionedLib.ver_1_000'

class someVersionedClass:
    '''example class that can be used for analysis
    
    '''

    def __init__(self, x):
        '''Initialize the class
        
        Parameters
        ----------
        x : any
            This is a test variable that will simply be copied
            into the instance of the object.
        '''
        self.x = x
        return

    @lD.log(logBase + '.__repr__')
    def __repr__(logger, self):
        '''function that allows one to print information
        
        Parameters
        ----------
        logger : logging.logger instance
            logging element
        
        Returns
        -------
        str
            a description of the class
        '''
        result = f'This is version 1.000: {self.x}'
        return result