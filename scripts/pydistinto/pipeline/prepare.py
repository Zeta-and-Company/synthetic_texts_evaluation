#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: prepare.py
# version: 0.3.1
#adjusted for synthetic text analysis by JuliaDudar


"""
The functions contained in this script prepare a set of plain text files for contrastive analysis. 
"""

# =================================
# Import statements
# =================================

import os
import re
import glob
import csv
import glob
import pandas as pd
import numpy as np
from collections import Counter
import itertools
import random
csv.field_size_limit(100000000)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# =================================
# Functions: make_segments
# =================================
def data_randomizing():
    allidnos = os.listdir('D:/synthetic_experiments/output_data/synthetic_same_length')
    allidnos = random.sample(allidnos, len(allidnos))
    listx1 = allidnos[:int(len(allidnos) / 2)]
    listx2 = allidnos[int(len(allidnos) / 2):]
    liste1 = [str(idno.replace(".csv", "")) for idno in listx1]
    liste2 = [str(id.replace(".csv", "")) for id in listx2]
    segmentids1 = [(item + "-" + "{:04d}".format(i)) for item in liste1 for i in range (0,8)]
    segmentids2 = [(item + "-" + "{:04d}".format(i)) for item in liste2 for i in range(0, 8)]
    list1 = random.sample(segmentids1, 1)
    list2 = random.sample(segmentids2, 1000)
    return liste1, liste2, list1, list2

def read_csvfile(file):
    with open(file, "r", newline="\n", encoding="utf-8") as csvfile:
        filename, ext = os.path.basename(file).split(".")
        content = csv.reader(csvfile, delimiter='\t')
        stops = []
        alllines = [line for line in content if len(line) == 3 and line[0] not in stops]
        return filename, alllines


def segment_files(filename, alllines, segmentlength, max_num_segments):
    segments = []
    segmentids = []
    if segmentlength == "text":
        numsegments = 1
        segment = alllines
        segmentid = filename
        segments.append(segment)
        segmentids.append(segmentid)
    else:
        numsegments = int(len(alllines) / segmentlength)
        for i in range(0, numsegments):
            segmentid = filename + "-" + "{:04d}".format(i)
            segmentids.append(segmentid)
            segment = alllines[i * segmentlength:(i + 1) * segmentlength]
            segments.append(segment)
        if max_num_segments != -1 and numsegments > max_num_segments:
            #chosen_ids = sorted(np.random.randint(0, numsegments, max_num_segments))
            chosen_ids = sorted(random.sample(range(0, numsegments), max_num_segments))
            segments = [segments[i] for i in chosen_ids]
            segmentids = [segmentids[i] for i in chosen_ids]
    return segmentids, segments


def make_segments(file, segmentfolder, segmentlength, max_num_segments=-1):
    if not os.path.exists(segmentfolder):
        os.makedirs(segmentfolder)
    filename, alllines = read_csvfile(file)
    segmentids, segments = segment_files(filename, alllines, segmentlength, max_num_segments)
    return segmentids, segments


# =================================
# Functions: select_features
# =================================

def read_stoplistfile(stoplistfile):
    with open(stoplistfile, "r", encoding="utf-8") as infile:
        stoplist = infile.read()
        stoplist = list(re.split("\n", stoplist))
        return stoplist


def perform_selection(segment, stoplist, featuretype):
    """
    Selects the desired features (words, lemmas or pos) from each segment of text.
    TODO: Add a replacement feature for words like "j'" or "-ils"
    """
    pos = featuretype[1]
    forms = featuretype[0]
    if pos == "all":
        if forms == "words":
            selected = [line[0].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and line[2] not in stoplist]
        elif forms == "lemmata":
            selected = [line[2].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0].lower() not in stoplist and line[2].lower() not in stoplist]
        elif forms == "pos":
            selected = [line[1].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and line[2] not in stoplist]
    elif pos != "all":
        
        if forms == "words":
            selected = [line[0].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist]
        elif forms == "pos":
            selected = [line[1].lower() for line in segment if
                        len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist]
        elif forms == "lemmata":
            selected = []
            for line in segment:
                if len(line) == 3 and len(line[0]) > 1 and line[0] not in stoplist and pos in line[1] and line[2] not in stoplist and line[2] != "<unknown>":
                    selected.append(line[2])
    else:
        selected = []
    selected = list(selected)
    #print("1", len(selected))
    return selected


def save_segment(features, segmentfolder, segmentid):
    segmentfile = segmentfolder + segmentid + ".txt"
    featuresjoined = "\n".join(features)
    with open(segmentfile, "w", encoding="utf-8") as outfile:
        outfile.write(featuresjoined)


def select_features(segmentfolder, segmentids, segments, stoplistfile, featuretype, list1, list2):
    stoplist = read_stoplistfile(stoplistfile)
    for i in range(len(segmentids)):
        segment = segments[i]
        df = pd.DataFrame(segment)
        if segmentids[i] in list1:
            df.iloc[0:1000, :] = ['untuniutntrng55886', 'untuniutntrng55886', 'untuniutntrng55886']
        if segmentids[i] in list2:
            df.iloc[0:1, :] = ['untuniutntrng55886', 'untuniutntrng55886', 'untuniutntrng55886']
        else:
           df = df
        segment = df.values.tolist()
        selected = perform_selection(segment, stoplist, featuretype)
        save_segment(selected, segmentfolder, segmentids[i])

# =================================
# Functions: make_dtm
# =================================


def read_plaintext(file):
    with open(file, "r", encoding="utf-8") as infile:
        text = infile.read().split(" ")
        features = [form for form in text if form]
        return features



def save_dataframe(allfeaturecounts, dtmfolder, parameterstring):
    dtmfile = dtmfolder + "dtm_" + parameterstring + "_absolutefreqs.csv"
    #print("\nallfeaturecounts\n", allfeaturecounts.head())
    allfeaturecounts.to_hdf(dtmfile, key="df")
    with open(dtmfile, "w", encoding = "utf-8") as outfile:
       allfeaturecounts.to_csv(outfile, sep="\t")


def tokenizer(filename):
    return re.split("\r\n",filename)


def make_dtm(segmentfolder, dtmfolder, parameterstring):
    filenames = glob.glob(os.path.join(segmentfolder, "*.txt"))
    idnos = [os.path.basename(idno).split(".")[0] for idno in filenames]
    vectorizer = CountVectorizer(input='filename',tokenizer = tokenizer, max_features=100000)
    dtm = vectorizer.fit_transform(filenames)  # a sparse matrix#
    vocab = vectorizer.get_feature_names()  # a list
    allfeaturecounts = pd.DataFrame(dtm.toarray(), columns=vocab)
    allfeaturecounts["idno"] = idnos
    allfeaturecounts.set_index("idno", inplace=True)
    #allfeaturecounts.drop("idno", inplace=True)
    allfeaturecounts = allfeaturecounts.fillna(0).astype(int)
    #save_dataframe(allfeaturecounts, dtmfolder, parameterstring)
    return allfeaturecounts



def auto_tfidf(dtmfolder, segmentfolder):
    filenames = glob.glob(os.path.join(segmentfolder, "*.txt"))
    idnos = [os.path.basename(idno).split(".")[0] for idno in filenames]
    vectorizer = TfidfVectorizer(input='filename',tokenizer = tokenizer, smooth_idf=True, sublinear_tf=True, max_features=40000)
    vectors = vectorizer.fit_transform(filenames)
    feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    tf_frame = pd.DataFrame(vectors.toarray(), columns=feature_names)
    tf_frame["idno"] = idnos
    tf_frame.set_index("idno", inplace=True)
    print(tf_frame.head())
    tf_frame = tf_frame.mul(100)
    #with open(dtmfolder + "tfidf_smoothed_sublinear.csv", "w", encoding="utf-8") as outfile:
        #tf_frame.to_csv(outfile, sep=",")
    return tf_frame

# =================================
# Functions: transform_dtm
# =================================


def read_freqsfile(filepath):
    with open(filepath, "r", newline="\n", encoding="utf-8") as csvfile:
        absolutefreqs = pd.read_csv(csvfile, sep='\t', index_col=0)
        #print("\nabsolutefreqs\n", absolutefreqs.head())
        return absolutefreqs


def transform_dtm(absolutefreqs, segmentlength):
    #print("Next: transforming to relative frequencies...")
    absolutefreqs_sum = pd.Series(absolutefreqs.sum(axis=1))
    wordfreq = pd.Series(absolutefreqs.sum(axis=0))
    print(type(segmentlength))
    if segmentlength == "text":
        relativefreqs = absolutefreqs.div(absolutefreqs_sum, axis='rows', level=None)
        #print("\nrelfreqs\n", relativefreqs.head(20), segmentlength)
    else:
        #relativefreqs = absolutefreqs / segmentlength
        relativefreqs = absolutefreqs.div(absolutefreqs_sum, axis='rows', level=None)
        #print("\nrelfreqs\n", relativefreqs.head(), segmentlength)
    print("Next: transforming to binary frequencies...")
    binaryfreqs = absolutefreqs.copy()
    binaryfreqs[binaryfreqs > 0] = 1
    return absolutefreqs_sum, relativefreqs, binaryfreqs


def save_transformed(relativefreqs, binaryfreqs, dtmfolder, parameterstring):
    transformedfile = dtmfolder + "dtm_" + parameterstring + "_relativefreqs.csv"
    with open(transformedfile, "w", encoding = "utf-8") as outfile:
        relativefreqs.to_csv(outfile, sep="\t")
    #relativefreqs.to_hdf(transformedfile, key="df")
    transformedfile = dtmfolder + "dtm_" + parameterstring + "_binaryfreqs.csv"
    #binaryfreqs.to_hdf(transformedfile, key="df")
    with open(transformedfile, "w", encoding = "utf-8") as outfile:
        binaryfreqs.to_csv(outfile, sep="\t")



# =================================
# Functions: main
# =================================


def main(taggedfolder, segmentfolder, dtmfolder, segmentlength, max_num_segments, stoplistfile, featuretype):
    if not os.path.exists(dtmfolder):
        os.makedirs(dtmfolder)
    parameterstring = str(segmentlength) + "-" + str(featuretype[0]) + "-" + str(featuretype[1])
    print("\n--prepare")
    import shutil
    if os.path.exists(segmentfolder):
        shutil.rmtree(segmentfolder)
    liste1, liste2, list1, list2 = data_randomizing()
    counter = 0
    for file in glob.glob(taggedfolder + "*.csv"):
        filename, ext = os.path.basename(file).split(".")
        counter +=1
        print("next: file no", counter, "- file", filename)
        segmentids, segments = make_segments(file, segmentfolder, segmentlength, max_num_segments)
        select_features(segmentfolder, segmentids, segments, stoplistfile, featuretype,list1, list2)
    allfeaturecounts = make_dtm(segmentfolder, dtmfolder, parameterstring)
    absolutefreqs = allfeaturecounts
    #absolutefreqs = read_freqsfile(dtmfolder + "dtm_" + parameterstring + "_absolutefreqs.csv")
    absolutefreqs_sum, relativefreqs, binaryfreqs = transform_dtm(absolutefreqs, segmentlength)
    #save_transformed(relativefreqs, binaryfreqs, dtmfolder, parameterstring)
    tf_frame = auto_tfidf(dtmfolder, segmentfolder)
    print(tf_frame.shape, absolutefreqs.shape)
    return absolutefreqs, relativefreqs, binaryfreqs, absolutefreqs_sum, tf_frame, liste1, liste2
