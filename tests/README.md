pytest has issues with finding the 

## TESTS
------

Run the following commands in the src dir. 

Note that pytest has issues with import modules. To overcome this, ensure the the correct versions of pytest and pytest-cov are installed, found in requirements_dev.txt. 

**Unit tests**

Unit tests are automatically run from the base folder by running the following in the terminal: 

``` pytest ```

Check cover with pytest to improve testings. 

**Static type checker**

Check all src files and find type errors. 

``` mypy```

info here: https://mypy-lang.org/

**Linting**

``` flake8 fused_org_ephys ```

**Automated testing**

**tox** to automate and standardize testing in Python. 

Installs a new v env each time for listed in envlist. 

``` tox ```

info here: https://tox.wiki/en/latest/