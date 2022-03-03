# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [How to run](#run)
3. [File Descriptions](#files)
4. [Main results](#results)

## Installation <a name="installation"></a>

Most of the libraries used on the project, such as pandas, numpy, matplotlib and tensorflow can be installed using the Anaconda distribution of Python.

You will need to install the following libraries used for reading the data, do natural language preprocessing and visualize the application in a web app:
 
 * sqlalchemy
 * pickle
 * gzip
 * plotly
 * joblib
 * flask

The code should run with no issues using Python versions 3.*.

## How to run <a name="run"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database <br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves <br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/ to visualize the web app.

## File Descriptions <a name="files"></a>



## Main Results<a name="results"></a>


