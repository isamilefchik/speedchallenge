from sklearn.metrics import mean_squared_error
import numpy as np

other_predicts = "truth.txt"
my_predicts = "mtr2.txt"
with open(other_predicts) as file:
    raw = file.read()
other_predicts = []
raw_split = raw.split("\n")
for i, item in enumerate(raw_split):
    # if i != len(raw_split) - 1:
    other_predicts.append(float(item))
# other_predicts = list(map(float, raw.split("\n")))
with open(my_predicts) as file:
    raw = file.read()
my_predicts = list(map(float, raw.split("\n")))

mse = mean_squared_error(my_predicts, other_predicts)
mse_2 = np.square(np.subtract(my_predicts,other_predicts)).mean()
print(mse)
print(mse_2)
