# Question 1 

Setting up:

```
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd 
from sklearn.model_selection import KFold
```

Importing the asking prices data for Charleston:

```
df = pd.read_csv('charleston_ask.csv')
```

Setting up X which contains variables of beds, baths, and square footage and y which contains the prices:

```
X = np.array(df.iloc[:,1:4])
y = np.array(df.iloc[:,0])
```

Setting up the linear regression model:

```
lin_reg = LinearRegression()
```
Do KFold cross validation:

```
kf = KFold(n_splits = 10, shuffle=True)

train_scores=[]
test_scores=[]

for idxTrain, idxTest in kf.split(X):
    Xtrain = X[idxTrain, :]
    Xtest = X[idxTest, :]
    ytrain = y[idxTrain]
    ytest = y[idxTest]

    lin_reg.fit(Xtrain, ytrain)

    train_scores.append(lin_reg.score(Xtrain, ytrain))
    test_scores.append(lin_reg.score(Xtest, ytest))
 ```
 Then we take the mean of the training and testing scores:
 
 ```
 np.mean(train_scores)
 np.mean(test_scores)
 ```

In terms of the number of splits for KFold, I assigned 10. This is because the total number of observations in the charleston_ask.cvs data is 715, so 10 would be a reasonable number. Each fold contains about 70 observations.

Since a R squared value of 1 is the perfect score indicating that it explains 100% of the variability, we know that this linear regression model performs very poorly. The average training score is only 0.019 and the average testing score is around -0.03, which suggests that the relation between the three features—beds, baths, and square feet—and the target—the asking price—is extremely low. 

The reason that the training and testing scores are so low might be that the three features are not put on a similar scale, which I will later fix by utilizing feature scaling. Despite the fact that beds and baths are on a similar scale(around 3 or 4), the square feet of the house are measured on an entirely different scale(more than 1000). However, the most likely reason for the model being underperforming might be that beds, baths, and sqaure footage are just not the most significant predictors for the asking prices of houses in Charleston. 


# Question 2 

```
def DoKFold(model, X, y, k, standardize=False):
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True)

    train_scores = []
    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

    return train_scores, test_scores
 ```

To standardize the features, I first imported `StandardScaler` from `sklearn.preprocessing` and utilized the `DoKFold` function where we ran a for loop to train, test, and standardize the data. After standardizing the three features to get them on a similar scale, the linear regression model barely improves. The outputs are almost identical to the previous one. The results of the training score on average is still 0.019 and the testing score hovers around -0.01. For this model, I did not change the number of folds because in order to ensure that the possible model improvement derives from the standardization of the three features, it is best if we keep it consistent with the prior model. However, standardization in this case does not seem to help the model improve since we are still getting similar results for the training and testing scores. This further strengthens the assumption that these features are poor predictors for the asking prices in Charleston. 

# Question 3 

```
from sklearn.linear_model import Ridge

a_range = np.linspace(0, 100, 100)

k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))
    
idx = np.argmax(avg_te_score)
print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

After trying to improve the model with standardization, I then applied a ridge regression model to train and test the model. The average training score I got is still 0.019, but the testing score has slightly improved to about 0.01. I did standardize the data by passing True in the argument for `standardize` in the `DoKFold` function. And the number of folds I assigned is still 10. Despite the fact that there is an improvement in the average testing score, this is still a very poorly fit model. Thus, we can reasonably conclude that beds, baths, and the space of houses do not have enough predictive power for the asking prices of houses in Charleston. 

# Question 4 

By using the actual Charleston housing prices data, the linear regression model before standardization, again, performs poorly. The training score is about 0.004 and the testing score is somewhere around -0.04. With standardization, the training score is still 0.004 and the testing score is -0.02. Finally, by running the ridge regression, the training score I got is again 0.004 and the testing score is -0.009. Overall, the three models all perform unsuccessfully. 


# Question 5 

After adding the dummy variables of zip codes, there is an obvious improvement for the models. For all three models of the charleston_act.cvs data that include zip codes, I still assigned 10 for the number of folds in order to keep it consistent with the previous models. 

For the unstandardized linear regression model, the training score has now risen to 0.33 while the testing score is 0.24. We still see a little bit of overfitting since the training score is higher than the testing score, but in comparison with the previous models without considering zip codes, this is already a better fit model. 

After standardization, the training score is still consistent with the previous model, which is 0.33. However, the testing score I originally got was extremely low—-3.303137384434876e+22. I figured this is because zip codes are dummy variables that take the value of either 0 or 1, which makes little sense to standardize. Even if standardized, I doubt if it would negatively impact the model’s predictability. In order to generate a proper testing score, I later tried the following codes which scale the price and square feet variables: 

```
lin_reg = LinearRegression()

c_act = pd.read_csv('charleston_act.txt')
c_act[['prices_scale']] = c_act[['prices']] / 100000# prices
c_act[['sqft_scale']] = c_act[['sqft']]/1000 # prices
X = c_act.drop(["prices_scale","prices","sqft"],axis = 1)
y = c_act["prices_scale"]
X.shape
X = X.to_numpy()
```

What this does is to put square footage and housing prices on the same scale as the other features. By standardizing this way, I got a testing score of 0.25. 

With ridge regression, the training score generates the same value of 0.33 and the testing score generates 0.28. According to the scores produced by the three models using zip codes, the predictive power has now reached approximately 33%, which is a huge growth from the models that only utilize the features of beds, baths, and square feet. Therefore, it is plausible of us to say that location (zip codes) has a bigger impact over the housing price in Charleston than beds, baths, and space. 

# Question 6 

The model producing the best results is the ridge regression model that includes zip codes. And according to its internal and external validity, the ridge regression model is an overfit because the R squared for the training score 0.34 is higher than that of the testing score 0.28. However, despite not being an absolutely strong indicator itself, location (zip codes) is definitely a better predictor for the housing prices than beds, baths, and square footage. If I were working as a data scientist at Zillow, what I would recommend to refine the predictive model is to add additional features. For example, median income would definitely be a strong factor affecting the affordability of the buyers, so we might want to include that. Also, the number of schools, unemployment rate, number of hospitals are all significant indicators for housing prices. 















