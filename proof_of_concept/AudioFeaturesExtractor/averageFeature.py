from os import walk
import os
import csv

def mean(a):
    return sum(a) / len(a)

featureRows = []
for (dirpath, dirnames, filenames) in walk("."):
    for csvFilename in filenames:
        print csvFilename
        if csvFilename.endswith(".csv"):
            with open(csvFilename, 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                tempFeatureRows = []
                for row in spamreader:
                    tempFeatureRow = []
                    for cell in row:
                        tempFeatureCellValue = float(cell)
                        tempFeatureRow.append(tempFeatureCellValue)
                    tempFeatureRows.append(tempFeatureRow)
                featureRow = map(mean, zip(*tempFeatureRows))
                featureRows.append(featureRow)
    break

f = open('average.txt', 'w')
for featureRow in featureRows:
    for i in range(len(featureRow)):
        value = str(featureRow[i])
        if i > 0:
            f.write(', ')
        f.write(value)
    f.write('\n')
#for (dirpath, dirnames, filenames) in walk("../transcript"):
#    print dirpath
#    print dirnames
#    print filenames
#    break