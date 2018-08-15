
import pandas as pd

dicts = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
         "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
         "area": [8.516, 17.10, 3.286, 9.597, 1.221],
         "population": [200.4, 143.5, 1252, 1357, 52.98]}

brics = pd.DataFrame(dicts)
print(brics)

brics.index = ["BR", "RU", "IN", "CH", "SA"]
print(brics)

cars = pd.read_csv("cars.csv", index_col=0)
# print(cars)

print(cars["Model"])
print(cars["Size"])

print(cars[0:4])
print("===============================")

print(cars.iloc[2])
print("-------------------------------")

print(cars.loc[[2012, 2014]])