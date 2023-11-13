# Diquark Analisys

Data Analysis for the vectorlike quark search in the fully hadronic channel.

## Requirements
Setting up the project locally requires:
- [Python 3.10](https://www.python.org/downloads/release/python-31013/)
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management.

## Setup
1. Clone the repository
```bash
$ git clone https://github.com/imdinu/diquark-analysis.git
```
2. Get the data
```bash
$ cd diquark
$ mkdir data
```
Download the data from [CernBox](https://cernbox.cern.ch/s/E9ViloYIwtMq8ab) and place it in the `data` folder. Note that it requires about 161 GB of space.
3. Install the dependencies
```bash
$ poetry install
```

## Usage
Run the [notebooks](./notebooks/) in the order they are numbered.