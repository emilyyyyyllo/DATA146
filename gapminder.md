informal practice 

```
import pandas as pd

path_to_data = 'gapminder.tsv'
data = pd.read_csv(path_to_data, sep='\t')
```

In response to question 1, in order to get a list of all the years in the gapminder dataset without duplicates, I append the unique() function to data['year] and store it in a new object called all_the_years. These unique years in the dataset are 1952, 1957, 1962, 1967, 1972, 1977, 1982, 1987, 1992, 1997, 2002, 2007. To find out the number of unique values of the years, I use the len() function on all_the_years and store it in a new object called unique_years. This way, we know that there are 12 unique values for all the years in the gapminder dataset. 

```
all_the_years = data['year'].unique()  
list(all_the_years) 
unique_years = len(all_the_years) 
unique_years 
```

In order to identify the largest value for population in the gapminder dataset, I append the max() function to data['pop'] and store it in a new object called pop_max. The largest value for population in the gapminder dataset is 1318683096. To find out when and where the largest population size occurs, I select the row containing the largest population by using data[data['pop']==pop_max] and store it in a new data frame called largest_pop, so from the information in the row we know that teh largest population occurs in China in 2007. 

```
pop_max = data['pop'].max() 
pop_max 
largest_pop = data[data['pop']==pop_max] 
largest_pop 
```

To extract all the records for Europe, I use data[data['continent']=='Europe'] and store it in a new data frame called data_europe. In order to identify the smallest population in the year of 1952, first I create a new data frame called data_europe_1952 by using data_europe[data_europe.year == 1952] which only contains the records for Europe in 1952. To find out the smallest value of population in 1952, I append the min() function to data_europe_1952['pop'] and store it in a new data frame called min_pop_1952. As a result, I get to know that the smallest value of population in 1952 is 147962. To idedntify the country that had the smallest population in 1952, I locate the row of data by using data_europe_1952[data_europe_1952['pop']==min_pop_1952], so we know that Iceland is the country that had the smallest population in 1952 which is 147962. To find out Iceland's population in 2007, I use a compound Boolean statement data_europe[(data_europe['year'] == 2007) & (data_europe['country'] == 'Iceland')], so we can see that Iceland's population in 2007 was 301931. One thing to note is that in the boolean statement I refer to the data_europe data frame rather than the data_europe_1952 one because we want the records for the year of 2007 which are not in data_europe_1952. 

```
data_europe = data[data['continent']=='Europe']
data_europe
data_europe_1952 = data_europe[data_europe.year == 1952]
data_europe_1952
min_pop_1952 = data_europe_1952['pop'].min()  
min_pop_1952 
data_europe_1952[data_europe_1952['pop']==min_pop_1952]
data_europe[(data_europe['year'] == 2007) & (data_europe['country'] == 'Iceland')]
```
