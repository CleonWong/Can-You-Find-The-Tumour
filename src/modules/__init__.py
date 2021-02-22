'''Available modules in this package. 

Modules are designed to be standard ways of isolating self-contained
pieces of code. This is different from the libraries present within 
the ``lib`` folder. These are supposed to be used by all the other 
modules. 

Modules are executed automatically when the main program is executed,
based upon the specification of the file ``config/modules.json``. This
file comprises of a list of JSON objects, each specifying a module. 
An example of the JSON block is:

.. code-block:: python

    "moduleName" : "module1",
    "path"       : "modules/module1/module1.py",
    "execute"    : true ,
    "description": "",
    "owner"      : ""

The ``path`` refers to the location where the module is located. This will 
be dynamically loaded if the ``execute`` parameter is ``true``. Each module
function has to have a function called ``main()`` which will be executed
once the module is loaded. Whatever needs to be done for running the 
module should be done from within this main function. 

When distributiong a system, it is typically a good idea to make sure that
all the modules have ``execute`` set to ``false``. This will prevent modules
from being accidently executed. 

Available Modules:
==================

The following modules are available. Please check the respective
modules for detailed description of how to operate the modules.


module1
-------

This is a test module that comes with the cookiecutter to be used as 
a template for other modules.

''' 

