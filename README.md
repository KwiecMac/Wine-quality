# Wine-quality

The project shows the effectiveness of neural network learning.


## Technologies
Project is created with:
* Python version: 3.7
* Sklearn library


## Features


### dataset folder

The dataset folder contains files that store information about the quality of the wine. The winequality-red.csv file comes from the site: https://archive.ics.uci.edu/ml/datasets/wine+quality


### common folder

The common folder contains the Get_data.py file, which allows you to download data from a csv file. Get_data.py has many functions that give different types of data.


### survey folder

The Cross_validation.py file uses the RFECV sklearn function. It displays the optimal number of wine qualities.

Feature_ranking and Feature_ranking_2 files use the ExtraTreesClassifier function. This function organizes the features in order of importance.


### neural_network folder

The neural_network file contains an MPL classifier that shows the learning efficiency of the neural network.

The random_forest file contains a random forest classifier. Its results serve as a comparison to the MLP classifier.
