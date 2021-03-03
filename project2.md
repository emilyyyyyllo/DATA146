# Project 2

### Question 1 

Continuous data is a type of numeric data that can take any value within a range. They can be measured on a continuum, but they cannot be counted. Examples of continuous data include height and income. Ordinal data is also numeric, but the difference between continuous and ordinal data is that the latter only concerns the order of the ordinal values. One example of ordinal data can be a list of top 10 best-selling books of the month. Nominal data is a type of categorical data which is non-numerical and without orders. Nominal data is similar to labels and it can be the names of colors and races. 

Below is a model of my own construction that incorporates variables of continuous, ordinal, and nominal data: 

An example can be a linear regression model based on two independent variables of race and income and the dependent variable of the satisfaction level of living quality. 
The type of race (nominal) one belongs to and the amount of income (continuous) one earns are features that can be used to predict the target which is the satisfaction level of living quality (ordinal) in the United States. In this model, we can think of the satisfaction level of living quality on a scale of 1 to 10 where 1 represents the least satisfaction and 10 the greatest satisfaction. For example, Asians might be more easily subject to violence arising from racial discrimination than Whites, which affects their satisfaction level in terms of living in the United States. Also, the amount of income also plays a key role in determining one’s living quality because the more you earn the more you can spend your money on things that make your life easier and happier. Thus, if you belong to the Asian race and you earn a meager income, it is highly likely that you score between 1 to 5 on the scale of satisfaction level of living quality. 


### Question 2 

In order to generate a data set of 1000 observations, we can use the command `np.random.beta(a, b, size=n)` where `n=1000`. To use the beta distribution to produce a plot that has a mean approximating the 50th percentile, we need to set alpha=beta. In this case, I set alpha=beta=5, so we can see a clear symmetry in the plot. The mean is 0.5028037973206971 and the median is 0.5064511906342577, which are extremely close to each other. 

![symmetry](symmetry.png)

When the beta is larger than the alpha, the mean of the beta distribution will be greater than the median. Below we can see a right-skewed distribution with alpha=2 and beta=6. The mean is 0.2556567993697229 and the median is 0.23877827678745545. 

![rightskewed](right_skewed.png)

When the beta is smaller than the alpha, the mean of the beta distribution will be smaller than the median, which shows a left-skewed distribution. In this case, I set alpha=6 and beta=2, which produces a mean of 0.7500593503731386 and a median of 0.7694370145250353. 

![leftskewed](left_skewed.png)











