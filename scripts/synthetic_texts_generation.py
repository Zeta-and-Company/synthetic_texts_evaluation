import os
import csv
import glob
import random
import re


taggedfolder = "D:/Downloads/pydistinto_old/pydistinto/data/output_/tagged/"
segmentfolder = "D:/synthetic_experiments/output_data/synthetic_same_length/"

#Function to read, remove stop-words, and connect all texts to a single list
def read_csvfile(taggedfolder):
    allfiles = []
    for file in glob.glob(taggedfolder + "*.csv"):
        with open(file, "r", newline="\n", encoding="utf-8") as csvfile:
            content = csv.reader(csvfile, delimiter='\t')
            stops = ["SENT", "''", ",", "``", ":", ".", "-", "---", "@card@", "@ord@" ]
            word = r"(?u)\b\w\w+\b"
            alllines = [line for line in content if len(line) == 3 and re.match(word, line[0]) is not None and line[0] not in stops and line[1] != 'PUNCT' and len(line[0]) > 1]
            allfiles.append(alllines)
    return allfiles


def flatten(allfiles):
    return [item for sublist in allfiles for item in sublist]

#Function to random sample 40000 word samples, giving the filenames
def segment_files(allfiles):
    print(len(allfiles))
    segments = []
    segmentids = []
    filenames = os.listdir("D:/Downloads/pydistinto_old/pydistinto/data/output_/tagged")
    for i in range(320):
        segmentid = str(filenames[i].replace(".csv", ""))
        segmentids.append(segmentid)
    #for item in list:
        segment = random.sample(allfiles, 40000)
        segments.append(segment)
    print(len(segments[0]))
    return segmentids, segments

#Function to save sampled texts
def save_files(filename,segmentfolder, segment):
    resultfile = segmentfolder + filename +".csv"
    with open(resultfile, "w", encoding="utf-8", newline='') as outfile:
        writer = csv.writer(outfile,  delimiter='\t')
        for item in segment:
            writer.writerow(item)


def main(taggedfolder, segmentfolder):
    allfiles = read_csvfile(taggedfolder)
    allfiles_f = flatten(allfiles)
    segmentids, segments = segment_files(allfiles_f)
    for i in range(len(segmentids)):
        filename = segmentids[i]
        segment = segments[i]
        save_files(filename, segmentfolder, segment)



main(taggedfolder, segmentfolder)