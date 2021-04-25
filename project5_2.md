Importing the libraries and functions:

```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as KNN
```

Define the DoKFold, GetData, and CompareClasses functions: 

```
def DoKFold(model, X, y, k, random_state=146, scaler=None):
   
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    train_scores = []
    test_scores = []
    train_mse = []
    test_mse = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

        # Compute the mean squared errors
        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)
        train_mse.append(np.mean((ytrain - ytrain_pred) ** 2))
        test_mse.append(np.mean((ytest - ytest_pred) ** 2))

    return train_scores, test_scores, train_mse, test_mse
 ```
 ```
 def GetData(scale=False):
    Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size=0.4)
    ss = StandardScaler()
    if scale:
        Xtrain = ss.fit_transform(Xtrain)
        Xtest = ss.transform(Xtest)
    return Xtrain, Xtest, ytrain, ytest
 ```
 ```
 def CompareClasses(actual, predicted, names=None):
    accuracy = sum(actual == predicted) / actual.shape[0]
    classes = pd.DataFrame(columns=['Actual', 'Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted
    conf_mat = pd.crosstab(classes['Predicted'], classes['Actual'])
    if type(names) != type(None):
        conf_mat.index = y_names
        conf_mat.index.name = 'Predicted'
        conf_mat.columns = y_names
        conf_mat.columns.name = 'Actual'
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy
```
We also need to do some preliminary work to clean the data set `city_person.csv`. The cleaning work on `city_person.csv` is similar to what we did on the previous project. First, we need to remove the NaNs, and then convert the type of the education and the age features into ints. Next, we have to assign anything other than wealthC to X and wealthC to y. 

```
pns = pd.read_csv("city_persons.csv")

pns.dropna(inplace=True)

pns['age'] = pns['age'].astype(int)
pns['edu'] = pns['edu'].astype(int)

X = pns.drop(["wealthC"], axis=1)
y = pns.wealthC
```

# Question 1 

Executing the K-nearsest neighbors classification, I used a narrower range of 70 to 75 and found out that with 74, it generates the highest testing score. The testing score I got is 0.5519765739385066. I also plotted the KNN model out to better visualize it. And from the graph, we can tell that at an alpha value of 74, the testing score is the closest to the training score, which indicates that it produces the least overfit within the range from 70 to 75. Also, at 74, the fraction correctly classified scores the highest. 


























