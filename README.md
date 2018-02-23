# Cars dataset analysis

Given the car data set (['cars.csv'](./cars.csv), of which we are trying to predict the level of acceptance of the car, provide an initial analysis of dataset (e.g. interesting observations, obsolete attributes, whatever you might find).

Machine learning using python stack:
* Data preparation
* Validation
* Choosing an algorithm
* Parameterization
* Training and evaluation

## Requirements:
* Docker (developed and tested with version 1.13.1)

## Running:
1) Build the Docker image: `docker build --no-cache -t cars .`

2) Run the Docker container: `docker run -v .:/code cars`
