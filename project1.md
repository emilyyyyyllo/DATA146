# Project 1

### Question 1

A Python package is a collection that contains multiple related modules and subpackages. Modules that are related to each other are usually put into the same package to reach a common goal. When a module from an external package is required in a program, that package can be imported and its modules can be put to use. A package is a directory of Python modules containing an additional __init__.py file, which distinguishes a package from a directory that happens to contain a bunch of Python scripts.

A Python library refers to the Standard Library, which is a large collection of modules and packages. A library comes with Python and is installed along with it, making its modules available to any Python code. One is able to use the modules in the library without having to download them from anywhere else. 

In order to install a package and make the library of functions accessible to your local workspace, you have to first install the packages using the command ‘pip install function_name’. 

```
pip install pandas 
pip install numpy
```
Then you will have to import the packages using the command ‘import package_name.’ 
```
import pandas as pd 
import numpy as np
```
If you want to call the functions inside that package, you can simply do so by using the dot(.) operator. 
```
input = np.array([‘cat’,’dog’,’fish’])
new_dataframe = pd.DataFrame(input, index=[1,2,3,4,])
```
It is more convenient for you to use an alias such as pd and np because it allows you to call functions from the imported package using the short name, rather than having to type out the full name of the packages each time you want to call a function from it. 
