'''This is an example of a versioned lib

This structure can effectively be used for generating multiple versions of
a library. All versions of a library are saved within the ``versions`` folder
within different subfolders. In this case, a particular version may be specified
using the major and minor version numbers. An example structure is shown below:

.. code-block:: bash

    lib
    |--lib1
    |  |-- getLib.py 
    |  +--versions
    |     +-ver_1_000
    |     | |--lib1.py
    |     | |--lib2.py
    |     | +--lib3.py
    |     +-ver_1_001
    |       |--lib1.py
    |       |--lib2.py
    |       +--lib3.py

for the libracy ``lib = 'lib1.py'`` for version ``version='1_00'``  it is posssible to 
use the path ``f'lib/lin1/versions/ver_{version}/{lib}.py'`` to dynamically generate
and load the required library. This is what is done within the function ``getLib()``
which, given a particular version and library, will load and return the right 
library. In fact, it is possible to load multiple versions of the library, and
this will allow one version to be compared with another efficiently.

You are encouraged to maintain this structure for consistency in all your projects.

'''