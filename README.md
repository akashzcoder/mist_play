# Code.jam 2019 MISTPlay challenge

## Inspiration

Our interest in Game Development and Machine learning inspired us to work on Game analytics. 

## What it does

The web application is used to predict how likely a player is willing to spend upon downloading apps.

Python was the primary language used in this project. We used various Python libraries such as Scikit-learn, Pandas, Numpy, Matplotlib, etc.. for data analysis, classification and visualization.

Conda was used to create the virtual environment. 

## Challenges we ran into

1. understanding the interesting game concepts, 
2. finding the relevant features for the classification model
3. resolving the skewness in data

## Accomplishments that we're proud of

Data Visualization and successful implementation of different classification models to distinguish between spending player over non-spending players with an accuracy of 90.6%.

## Built With
Python, HTML, bootstrap, javascript

## Continue Development

STEP 1:
Create Conda environment
`conda env create -f environment.yml`

STEP 2:
Activate the conda environment
`source activate mistplay`

STEP 3:
Get inside the `mist_play/data` directory and run the data pre-processing module
`python data_preprocessing.py`

STEP 4:
Get to the `mist_play/model` directory and run the machine learning model
`python model.py`

