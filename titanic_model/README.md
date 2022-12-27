# :ferry: Titanic Model

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)


## :memo: Description

Just having fun with the [Titanic Kaggle competition](https://www.kaggle.com/competitions/titanic)!

The model aims to predict the survival probability of the Titanic passengers.

Using GitHub Actions, the model is then uploaded to [Gemfury](https://fury.co/) to be latter used by a web service app.

There are four steps for this repo:

* <u>Fetch the data</u>: we download the Titanic competition data from [Kaggle](https://www.kaggle.com/).
* <u>Train the model</u>: we then train the model to predict the survivability of the different passengers.
* <u>Test the model</u>: we run several unit tests to ensure the model is predicting the way we want.
* <u>Build & publish the model</u>: we then build a Python package that we publish on a private repository in Gemfury.

## :computer: How to run it locally

I ran this code using `python 3.10.7`. To install it on your computer and manage several versions of python I recommend using [pyenv](https://github.com/pyenv/pyenv).

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

```bash
python -m venv .venv/<name-of-your-venv>
```

That you can activate using:
```bash
source .venv/<name-of-your-venv>/bin/activate
```

Or you can use the pyenv `local` command to set up a `.python-version` file in this directory so that pyenv
automatically activate the virtual environment when entering this folder by running:

```bash
pyenv local <name-of-your-venv>
```

Then you can install [tox](https://tox.wiki/en/latest/index.html#) by running:
```
pip install tox
```
The different steps explained in the description are then performed by different `tox` commands.
First you need to set up three `environment variables`

### :seedling: Environment variables

```
export KAGGLE_USERNAME=your-kaggle-username
export KAGGLE_KEY=your-kaggle-key
export GEMFURY_PUSH_URL=your-gemfury-url
```
For the record, here is the format of you Gemfury URL:
```
https://TOKEN@push.fury.io/your-profile-name/
```
For the Kaggle credentials you can also download a `kaggle.json` file from your profile that you can put in your `~/.kaggle` directory.

### :bookmark_tabs: Tox commands

* To fetch the data run:
```
tox -e fetch_data
```
* To train the model & run the tests run:
```
tox -e train_model
```
* To package & publish the model:
```
tox -e publish_model
```