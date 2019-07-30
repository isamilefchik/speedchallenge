import sklearn.metrics as metrics
from load_data import parse_speeds

filepath = "train_result.txt"
with open(filepath) as file:
    raw = file.read()
    result = list(map(int, raw.split("\n")))
error = metrics.mean_squared_error(parse_speeds()[1:20399], result)

print(error)