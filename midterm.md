## Midterm Correction 

First we need to import the necessary libraries and the california housing data file: 

```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
```

Then we transform the california housing data into a data frame:

```
data = fetch_california_housing()

housing = pd.DataFrame(data.data)
housing.columns = data.feature_names
housing['target'] = data.target

X = np.array(housing.iloc[:, :8])
y = np.array(housing['target'])
```


### Question 15

In order to find the most correlated variable to the target, we can run through every feature in X by using the method `corr.()`. 

```
housing.corr()
```








