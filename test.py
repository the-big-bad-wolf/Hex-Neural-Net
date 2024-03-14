import csv
import numpy as np

# Open the CSV file
with open("trainingdata.csv", "r") as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in reader:
        # Split the row into individual values
        features: list[float] = []
        targets: list[list[float]] = []
        for feature in row[0:25]:
            features.append(float(feature))
        print("Features:", features)
        for target_row in row[26:]:
            target_row_array = []
            for probability in target_row.strip().strip("[").strip("]").split(", "):
                target_row_array.append(float(probability))
            targets.append(target_row_array)
        print("Targets:", targets)
        exit()
