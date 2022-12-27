# :passenger_ship: Titanic API

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)

## :memo: Description

This project is the continuation of the [Titanic Model](https://github.com/jordandelbar/titanic-model) and aims to deploy that model on [Heroku](https://dashboard.heroku.com/apps) while serving the results using [Fast API](https://fastapi.tiangolo.com/).

This API runs [here](https://titanic-api-jd88.herokuapp.com) and has two routes:

* <u>api/health</u> that will return:
    - The name of the API
    - The current version of the API deployed
    - The current version of the model deployed

* <u>api/predict</u> that will return a probability of survival during the [sinking of the Titanic](https://en.wikipedia.org/wiki/Sinking_of_the_Titanic).

## :computer: How to run it locally

I ran this code using `python 3.10.7`. To install it on your computer and manage several versions of python, I recommend using [pyenv](https://github.com/pyenv/pyenv).

You can check this [tutorial](https://realpython.com/intro-to-pyenv/) over pyenv.

Once the installation process is over, simply run:

```
pyenv install 3.10.7
```

Then use the `global` command to make it available anywhere on you machine:

```
pyenv global 3.10.7
```

You can then use the `venv` python virtual environment function to create a `venv` in this folder:

```
python -m venv .venv/name-of-your-venv
```

That you can activate using:
```
source .venv/name-of-your-venv/bin/activate
```
Then you can install [tox](https://tox.wiki/en/latest/index.html#) by running:
```
pip install tox
```
To launch the API locally simply run:
```
cd titanic_api
tox -e run
```
