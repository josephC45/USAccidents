from pathlib import PureWindowsPath
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

# From other tutorial only if I need them
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


'''
5/7/2020 
Having a difficult time increasing the accuracy. Despite using binning on the continuous/numerical values, the accuracy
tends to stay in the upper 60's to 70. I'll try to add the city and county in which the accidents occurred into see if that
changes anything. 
Next step after increasing the accuracy is to clean up the code and remove redundancy; then comes the visualizations.
'''
# define location of dataset and read csv
def data_preliminaries():
    path = PureWindowsPath('C:\\Users\\joeyc\\Desktop\\US_Accidents_Dataset.csv')
    file = pd.read_csv(path)
    return file

# Clean the dataset and update the dataset
def update_dataset():
    dataset = data_preliminaries()

    dataset['Distance(mi)'].fillna((dataset['Distance(mi)'].mean()), inplace=True)
    dataset['Temperature'].fillna((dataset['Temperature'].mean()), inplace=True)
    dataset['Visibility'].fillna((dataset['Visibility'].mean()), inplace=True)

    dataset['Precipitation(in)'].fillna(0, inplace=True)
    dataset['Wind_Speed(mph)'].fillna(0, inplace=True)

    dataset.dropna(subset=['Weather_Condition', 'Sunrise_Sunset'], how='any', inplace=True)
    return dataset

def temperature_binning():
    data_set = update_dataset()
    data_set['Temperature_Groups'] = pd.qcut(data_set['Temperature'], q=4)
    return data_set

def temperaure_attribute_conversion():
    data_set = temperature_binning()
    data_set['Temperature_Groups'] = data_set['Temperature_Groups'].cat.codes
    updated_dataset = data_set.copy()
    return updated_dataset

# Improve binning
# Bin the visibility attribute values to attempt to improve accuracy
def visibility_binning():
    data_set = temperaure_attribute_conversion()
    data_set['Visibility_Groups'] = pd.cut(data_set.Visibility, bins=np.linspace(0, 30, 10))
    return data_set

# Newly binned visibility values turned into numerical values to be read by classification machine learning algorithms
def visibility_categorical_conversion():
    data_set = visibility_binning()
    data_set['Visibility_Groups'] = data_set['Visibility_Groups'].cat.codes
    updated_dataset = data_set.copy()
    return updated_dataset

# States ordinal datatype is turned into numerical to be read by classification machine learning algorithms
def state_categorical_conversion():
    data_set = visibility_categorical_conversion()
    labels = data_set['State'].astype('category').cat.categories.tolist()
    replace_state_cat = {'State': {state_name: state_number for state_name, state_number in
                                   zip(labels, list(range(1, len(labels) + 1)))}}
    update_dataset = data_set.copy()
    update_dataset.replace(replace_state_cat, inplace=True)

    return update_dataset

# Weather_Conditions ordinal datatype is turned into numerical to be read by classification machine learning algorithms
def weather_categorical_conversion():
    data_set = state_categorical_conversion()
    labels = data_set['Weather_Condition'].astype('category').cat.categories.tolist()
    replace_weather_cat = {'Weather_Condition': {weather_type: weather_number for weather_type, weather_number in
                                                 zip(labels, list(range(1, len(labels) + 1)))}}
    update_dataset = data_set.copy()
    update_dataset.replace(replace_weather_cat, inplace=True)

    return update_dataset


'''
Still have to work on this
Just to improve readability and reduce redundancy


def nominal_conversion(attribute, attribute_type):
    data_set = data_preliminaries()
    labels = data_set[attribute].astype(attribute_type).cat.categories.tolist()
    new_numeric_attribute_value = {attribute: {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
    update_dataset = data_set.copy()
    update_dataset.replace(new_numeric_attribute_value, inplace=True)

    return update_dataset
'''


'''
One-Hot encoding is used for converting the string value of Sunrise_Sunset, Crossing, Junction and Traffic_Signal into
a numerical value used for the classification machine learning algorithm
'''
def side_categorical_conversion():
    data_set = weather_categorical_conversion()
    updated_dataset = data_set.copy()
    updated_dataset = pd.get_dummies(updated_dataset, columns=['Side'], prefix=['Side'])
    return updated_dataset

def sun_categorical_conversion():
    data_set = side_categorical_conversion()
    updated_dataset = data_set.copy()
    updated_dataset = pd.get_dummies(updated_dataset, columns=['Sunrise_Sunset'], prefix=['Sun'])
    return updated_dataset

# The following methods reduce the accuracy of the machine learning algorithms.
def crossing_categorical_conversion():
    data_set = sun_categorical_conversion()
    updated_dataset = data_set.copy()
    updated_dataset = pd.get_dummies(updated_dataset, columns=['Crossing'], prefix=['Crossing'])
    return updated_dataset


def junction_categorical_conversion():
    data_set = crossing_categorical_conversion()
    updated_dataset = data_set.copy()
    updated_dataset = pd.get_dummies(updated_dataset, columns=['Junction'], prefix=['Junction'])
    return updated_dataset


def trafficsignal_categorical_conversion():
    data_set = junction_categorical_conversion()
    updated_dataset = data_set.copy()
    updated_dataset = pd.get_dummies(updated_dataset, columns=['Traffic_Signal'], prefix=['Signal'])
    return updated_dataset

# Used to train and test the machine learning algorithms
def model():
    data_set = trafficsignal_categorical_conversion()
    accident_data = data_set
    y = accident_data.Severity
    accident_features = ['Distance(mi)',
                         'State',
                         'Temperature_Groups',
                         'Precipitation(in)',
                         'Wind_Speed(mph)',
                         'Weather_Condition',
                         'Side_L',
                         'Side_R',
                         'Crossing_False',
                         'Crossing_True',
                         'Junction_False',
                         'Junction_True',
                         'Signal_False',
                         'Signal_True',
                         'Sun_Night',
                         'Sun_Day',
                         'Visibility_Groups']
    X = accident_data[accident_features]

    print(X.describe())

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.2, shuffle=False)
    accident_model = DecisionTreeClassifier()
    accident_model.fit(train_X, train_y)

    severity_predictions = accident_model.predict(val_X)
    print(mean_absolute_error(val_y, severity_predictions))



    # Evaluate predictions
    print(accuracy_score(val_y, severity_predictions))
    print(confusion_matrix(val_y, severity_predictions))
    print(classification_report(val_y, severity_predictions))

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=2, random_state=None, shuffle=False)
        cv_results = cross_val_score(model, train_X, train_y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # Compare Algorithms
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()


# Calls all necessary methods
data_preliminaries()
update_dataset()
temperature_binning()
temperaure_attribute_conversion()
visibility_binning()
visibility_categorical_conversion()
state_categorical_conversion()
weather_categorical_conversion()
side_categorical_conversion()
sun_categorical_conversion()
crossing_categorical_conversion()
junction_categorical_conversion()
trafficsignal_categorical_conversion()
model()
'''

Add back to training data if need be, however these lowered the accuracy.                   
                        'Crossing_False',
                        'Crossing_True',
                        'Junction_False',
                        'Junction_True',
                        'Signal_False',
                        'Signal_True',
'''
