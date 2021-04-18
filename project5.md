## Project 5

### Question 1 
First, we need to import the file `persons.csv` by using the command `df = pd.read_csv("persons.csv")`. By using `df.isnull().values.any()`, I found out that there are NaNs that needed to be removed. Then I used `df.dropna(inplace=True)` to remove all the NaNs. Observing the data frame again, I noticed that the number of rows decreased from 47974 to 47891, which is not a significant drop, so it would not influence the data too much. Also, we need to change the type of some features. By using `display(df.dtypes)`, I found out that the “age” and “edu” features are both in the type of float. Thus, I utilized the `astype` method to convert them into `int` by using the command  `df["age"]=df["age"].astype(int)` and the same thing for “edu.” Next, we need to assign the features to `x` and target to `y`. One thing to note here is that we would want to also drop the feature of `wealthI` along with `wealthC` when assigning `x`. 

```
df = pd.read_csv("persons.csv")

df.dropna(inplace=True)
df.isnull().values.any()

df["age"]=df["age"].astype(int)
df["edu"]=df["edu"].astype(int)

X = df.drop(["wealthC", "wealthI"],axis=1)
y = df.wealthC
```

### Question 2 
