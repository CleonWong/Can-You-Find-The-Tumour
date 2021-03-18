# Can You Find the Tumours?

This project aims to apply semantic segmentation models to locate abnormalities in mammogram images.

## Read the Supporting Medium articles
I wrote a 3-part article series that complements this repository. It provides insights, intuitions and thought processes that might be hard for some to decipher just from the code. Also, it comes with some nice illustrations and visualisations! Give them a read if you'd like, hope they help!

1. [Can You Find the Breast Tumours? (Part 1 of 3)](https://towardsdatascience.com/can-you-find-the-breast-tumours-part-1-of-3-1473ba685036)
2. [Can You Find the Breast Tumours? (Part 2 of 3)](https://towardsdatascience.com/can-you-find-the-breast-tumours-part-2-of-3-1d43840707fc)
3. [Can You Find the Breast Tumours? (Part 3 of 3)](https://towardsdatascience.com/can-you-find-the-breast-tumours-part-3-of-3-388324241035)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

## Installing

The folloiwing installations are for \*nix-like systems. This is currently tested in the following system: `Ubuntu 18.10`. 

For installation, first close this repository, and generate the virtual environment required for running the programs. 

This project framework uses [venv](https://docs.python.org/3/library/venv.html) for maintaining virtual environments. Please familiarize yourself with [venv](https://docs.python.org/3/library/venv.html) before working with this repository. You do not want to contaminate your system python while working with this repository.

A convenient script for doing this is present in the file [`bin/vEnv.sh`](../blob/master/bin/vEnv.sh). This is automatically do the following things:

1. Generate a virtual environment
2. activate this environment
3. install all required libraries
4. deactivate the virtual environment and return to the prompt. 

At this point you are ready to run programs. However, remember that you will need to activate the virtual environment any time you use the program.

For activating your virtual environment, type `source env/bin/activate` in the project folder in [bash](https://www.gnu.org/software/bash/) or `source env/bin/activate.fish` if you are using the [fish](https://fishshell.com/) shell.
For deactivating, just type deactivate in your shell

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

 - Python 3.8.6

## Contributing

Please send in a pull request.

## Authors

Cleon-Wong - Initial work (2020)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

 - Hat tip to anyone who's code was used
 - Inspiration
 - etc.
 
