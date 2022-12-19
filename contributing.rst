.. highlight:: shell

============
Contributing
============


Report Bugs
~~~~~~~~~~~

Report bugs at the `GitHub Issues page`_.
If you are reporting a bug, please include details. 

Fix Bugs
~~~~~~~~

Look through the GitHub issues. Issues marked with "issue" are open 
to whoever wants to fix it. 

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues. Issues maerked with "feature"
are open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~
More documentation is always helpful for ADV-O, whether it is in the 
form of official ADV-O documentation or docstrings.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is through the `GitHub Issues page`_.


Get Started!
------------

1. Fork the `ADV-O` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/ADV-O.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv ADVO
    $ cd ADVO/
    $ pip install -r requirements.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. While hacking your changes, make sure to cover all your developments with the required
   unit tests, and that none of the old tests fail as a consequence of your changes.
   
    $ python -m unittest

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.
