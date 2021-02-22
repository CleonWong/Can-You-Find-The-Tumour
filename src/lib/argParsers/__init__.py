'''Argument parsers will be located gere

Currently, there is just one argument parser. However
you are encouraged to use as many argument parsers
that you wish to have. Ideally you should have one 
argument parser per config file. 

For each config file, you want to add a function for adding
the respective parser to the CLI and another that will 
convert the values back into a dictionary. You have to 
generate a proper namespace for your parsed documents 
so that they can be easily separated. We recommend using
the name of the config file without the extension as a
starting point. 

Defining Parsers
----------------

Parsers for a new config file should be defined within a new 
file correcponding to that file. For example, a ``config.json``
file comes with a file ``config.py`` Each file should contain
two functions:

    - ``addParsers(parser)``
    - ``decodeParser(args)``

The ``addParser()`` function will add all the necessary command
line arguments to the supplied ``argparse.ArgumentParser`` object
and return the object. 

The ``decodeParser()`` function will take a parsed Namespace
object and convert it inot a dictionary. 

This way, different parsing arguments can easily be added and 
deleted at will without restricting the workflow to a great 
extent. 

Within the function ``addAllParsers.parsersAdd()``  insert all
the individual parser insertion function that you just created.
Within the ``addAllParsers.decodeParsers`` function, update the
dictionary that it returns containing all the parsed arguments 
within one big dictionary. Note that this is going to add values
within the main dictionary only if a particular command line 
argument is supplied.

Defining CLI Options
---------------------

There should be proper namespace created for the CLI arguments
or else there is a high possibility that the CLI named arguments
are going to collide. For overcoming this, a couple of simple 
rules should be followed:

1. Make sure that a CLI argument is always verbose. (Dont use 
   one letter abbreviations unless it is a really common option
   and is universally used: like ``-v`` for ``--verbose``, ``-h``
   for ``--help`` etc. Note that in this vase, ``-v`` is already
   handled by the logging level).

2. Start with the name of the config file. Each config file is
   typically a ``json`` object that translates into a Python 
   ``dict``. Hence, start every CLI argument with the name of
   the config file followed by an underscore. 

   **For example**: CLI arguments corresponding to ``config.json``
   should start with ``--config_``

3. Each subsequent object within the main object should have
   the same nomenclature. This will allow a one-to-one mapping
   of a variable in the config file to a CLI argument.

   **For example**: within ``config.json``, the information referred
   to by the ``logFolder`` object within the JSON structure
   ``{"logging":{"specs":{"file":{"logFolder":"logs"}}}}`` should be
   named ``--config_logging_specs_file_logFolder``

'''