"""
## Author: Ravi

Given a folder this script will read the all the files and split the data into train, validation and test set.
The structure of the folder should be:-
### Data_dir -> The directory containg data 
###    |- The csv file containg the transcripts. The first column is the file-name and the second column name is the transcript
###    |- Audio_Dir -> The directory containing the original audio_file (Unless you don't want to drop the original data from training)
###    |- Aug_dir_0 -> The directory containing the augmentated audio-1 and so on. # Created and to be stored there

We assume there is a master transcript.csv file for all the folders (One original folder and the remaining folders is the augmentated data)

The output are 3 files corresponding to train, validation and test set.
Each line in a file is a json corresponding to each audio clip.
The structure is:-
key -> The location of the audio
duration -> The duration of the audio
text -> Transcription of the audio
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import wave
import numpy as np
import pandas as pd
from random import shuffle

def get_data(df, source_dirs):
    list_ = []    
    allowed = " qwertyuiopasdfghjklzxcvbnm"
    
    for f in df.index:
        transcription = df.loc[f, "transcript"]
        #print transcription
        trans = ""
        for c in str(transcription).lower():
            if c in allowed:
                trans += c
            else:
                continue
        for source in source_dirs:
            try:
                key = os.path.join(source, f)
                audio = wave.open(key)
                dur = float(audio.getnframes()) / audio.getframerate()
                key = os.path.join(source, fil)
                audio.close()

                json = {"key" : key, "duartion" : dur, "text" : trans }
                list_.append(json)
                no_ += 1
            except:
                continue
    return list_

def write_to_json(data, file_name, source_dirs):
    """
    args - data and file name where to write the data
    returns - nothing
    
    writes the data to json file where each line contains one json.
    One json corresponds to one unit of data 
    """
    list_ = get_data(data, source_dirs)
    shuffle(list_)
    with open(file_name, 'w') as out_file:
        for j in list_:
            try:
                line = json.dumps(j)
                out_file.write(line + '\n')
            except:
                continue

def get_split(size, train_ratio, valid_ratio, test_ratio):
    perm = np.random.permutation(size)
    train_size = int(size*train_ratio)
    valid_size = int(size*valid_ratio)
    
    train_split = perm[:train_size]
    valid_split = perm[train_size:(train_size+valid_size)]
    test_split = perm[(train_size+valid_size):]

    return train_split, valid_split, test_split
    
def main(data_directory, output_dir, train_ratio = 0.9, valid_ratio = 0.05, test_ratio =0.05):
    if (train_ratio + valid_ratio + test_ratio) != 1:
        print("The train, validation and test split is not valid")
        exit()
    
    # Modify this as per the structure of the csv file, Assuming the index is the file name and transcript column contains the text 
    data = pd.read_csv(os.path.join(data_directory, "transcript.csv"), index_col = 0)

    # Directories to be read audio
    source_dirs = []
    for d in os.listdir(data_directory):
        dest = os.path.join(d)
        if os.path.isdir(dest):
            # Assuming the audio files are in directories having "audio" in their names. Since, we may have noise and noise_dir directories as well. 
            if "audio" in d:
                source_dirs.apend(dest)  

    # Shuffle the data
    train_split, valid_split, test_split = get_split(len(data), train_ratio, valid_ratio, test_ratio)

    # Split the data
    train_data = data.iloc[train_split, :]
    valid_data = data.iloc[valid_split, :]
    test_data = data.iloc[test_split, :]

    #Write the data
    train_file = "train_data.json"
    valid_file = "valid_data.json"
    test_file = "test_data.json"
    
    write_to_json(train_data, os.path.join(output_dir, train_file), source_dirs)
    write_to_json(valid_data, os.path.join(output_dir, valid_file), source_dirs)
    write_to_json(test_data, os.path.join(output_dir, test_file), source_dirs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('output_folder', type=str,
                        help='Path to output folder')
    parser.add_argument('--train_ratio', type = float, default = 0.9,
                        help = 'The fraction of data to set aside for training')
    parser.add_argument('--valid_ratio', type = float, default = 0.05,
                        help = 'The fraction of data to set aside for validation')
    parser.add_argument('--test_ratio', type = float, default = 0.05,
                        help = 'The fraction of data to set aside for testing')
    args = parser.parse_args()
    main(args.data_directory, args.output_folder, args.train_ratio, args.valid_ratio, args.test_ratio)
