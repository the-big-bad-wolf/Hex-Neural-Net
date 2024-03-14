import csv
import numpy as np

# Open the CSV file
with open("trainingdata.csv", "r") as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in reader:
        # Split the row into individual values
        values = row
        features = values[0]
        targets = []
        strings = values[1].split(",")
        for string in strings:
            targets.append(float(string.strip().strip("[").strip("]")))
        np_targets = np.array(targets)
        np_targets = np_targets.reshape(5, 5)
        targets = np_targets.tolist()
        print(targets)
        exit()
