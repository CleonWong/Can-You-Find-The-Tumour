from logs       import logDecorator as lD
from importlib  import util
import jsonref, os

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.versionedLib.versionedLib'

def getLib(version, libName):
    '''return a particular library dynamically, based upon a 
    version number.
    
    Parameters
    ----------
    version : str
        The version number of the library. 
    libName : str
        The name of the library that you wish to use within this
        library
    
    Returns
    -------
    module
        The module that is dynamically loaded given the name and the
        verison of the library. In case this library cannot be found,
        this is going to raise an exception

    '''

    
    name = libName + '__' + version
    path = f"lib/versionedLib/versions/ver_{version}/{libName}.py"

    assert os.path.exists(path), f'Unable to find the library: {path}'
            
    module_spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module

