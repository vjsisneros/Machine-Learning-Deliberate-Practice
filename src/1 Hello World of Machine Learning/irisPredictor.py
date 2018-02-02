# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:17:35 2018

@author: V
"""
# Python Machine Learning Project Template
# 1. Prepare Problem __________________________________________________________
# a) Load libraries
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# b) Load dataset
filename = "C:\\Users\\V\\Desktop\\Software Development\\Machine Learning\\Machine-Learning-Deliberate-Practice\\src\\1 Hello World of Machine Learning\\iris.csv.csv"
dataset = pd.read_csv(filename)
dataset.drop('Id', axis=1, inplace=True)#0 for row 1 for column **drop Recipie

# 2. Summarize Data ___________________________________________________________
# a) Descriptive Statistics
print(dataset.shape)
print() 
print(dataset.head(20))
print() 
print(dataset.describe())
print() 
print(dataset.groupby('Species').size())

# b) Data visualizations
#Univariate plots - plots of each individual variable. 
    #Given thatthe input variables are numeric, 
    #we can create box and whisker plots of each.
    # box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

# histograms- get an idea of the distribution of each input variable
    #It looks like perhaps two of the input variables have a 
    #Gaussian distribution.
    # This is useful to note as we can use algorithms that can 
    #exploit this assumption.
dataset.hist()
pyplot.show()

# scatter plot matrix - to look at the interactions between the variables.
    #Let's look at scatter plots of all
    #pairs of attributes. This can be helpful to spot structured relationships between input variables.
    #Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and
    #a predictable relationship.
scatter_matrix(dataset)
pyplot.show()

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms create some models of the data and estimate their accuracy on unseen data.
    #1. Separate out a validation dataset.
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
    test_size=validation_size, random_state=seed)

    #2. Setup the test harness to use 10-fold cross validation.
#        We will use 10-fold cross validation to estimate accuracy. This will split our dataset into 10
#        parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. We are using
#        the metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted
#        instances divided by the total number of instances in the dataset multiplied by 100 to give a
#        percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and
#        evaluate each model next.
    #3. Build 5 different models to predict species from power measurements
# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
    #4. Select the best model.
#    We now have 6 models and accuracy estimations for each. We need to compare the models to
#    each other and select the most accurate.
#        We can see that it looks like KNN has the largest estimated accuracy score. We can also
#        create a plot of the model evaluation results and compare the spread and the mean accuracy
#        of each model. There is a population of accuracy measures for each algorithm because each
#        algorithm was evaluated 10 times (10 fold cross validation).

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#You can see that the box and whisker plots are squashed 
#at the top of the range, with many samples achieving 100% accuracy.

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



