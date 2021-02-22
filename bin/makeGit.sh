#!/bin/bash

if [ -e '.git' ]; then

    echo "A git environment is already present"
    echo ""
    echo "Local and remote branches which are tracked are:"
    echo "------------------------------------------------"
    echo "$(git branch -a)"
    echo ""
    echo "All remote branches are (checking remote branch, may take a while ...):"
    echo "-----------------------------------------------------------------------"
    echo "$(git ls-remote)"
    echo ""
    echo "+---------------------------------------------------------------------+"
    echo "| This looks like a cloned copy. It is highly likely that you have    |"
    echo "| other branches that you have not downloaded into this folder. Pull  |"
    echo "| whichever branch that you think is more important at any time.      |"
    echo "+---------------------------------------------------------------------+"
    echo "$(git pull)"

else

    git init
    git add .; true
    git remote add origin git@github.com:CleonWong/Can-You-Find-The-Tumour.git
    git commit -m 'This is the first commit'
    # Geenerate the master branch
    # ......................................
    git push origin master
    # Geenerate the rest of the branches
    # ......................................
    git checkout -b dev
    git push origin dev

fi
