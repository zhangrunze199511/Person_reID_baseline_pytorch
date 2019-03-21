import os
import csv

def write_to_csv(path, results):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(path + '/result.csv', 'w',  newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames= ['epoch', 'loss', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)