# Author: Ravi and Zaher 

# Date: 2018-07-09
# Subject: Code to split audio_file and to separate out a noise file for each call. Which can be used to get noise profile and clean audio

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from pydub.silence import split_on_silence
import argparse
import os
import fnmatch 

def split_audio_to_seg(audio, channel, target_file, min_silence_len = 2000, silence_thresh = -30):
    """
    Split an audio into multiple chunks based on silent pauses in the audio 
    """
    audio_chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    target = target_file.split(".mp3")[0]
    print(len(audio_chunks))
    for i, chunk in enumerate(audio_chunks):
        out_file = "{0}_chan_{1}_chunk_{2}.wav".format(target, channel, i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")

def noise(audio, channel, target):
    """
    Selects the audio corresponding to noise and stores it in a directory
    """
    list_ = detect_silence(a, 2000, -30)
    out_file = "{0}_chan_{1}.wav".format(target, channel)
    audio[list_[0][0] : list_[0][1]].export(out_file, format = 'wav')

def main(source_dir, target_dir, get_noise = True):
    """
    args:- 
    source_dir -> The original directory containing the required audio files
    target_dir -> The directory where to store the output audio. 
                The output will be stored in the sub-dir named "audio" within this directory
    get_noise -> whether to get noise or not. If True, then
                audio is stored in the "noise" sub-dir of the target directory
    """
    files = os.listdir(source_dir)
    files = [f for f in files if fnmatch.fnmatch(f, "*.mp3")]
    failed = []

    for f in files:
        #print("chunking %s" % f)
        target_file  = os.path.join(target_dir, "audio/" + f)
        audio = AudioSegment.from_mp3(os.path.join(source_dir, f))
        try:
            chan_1, chan_2 = audio.split_to_mono()
            split_audio_to_seg(chan_1, channel = 1, target_file=target_file)
            split_audio_to_seg(chan_2, channel = 2, target_file=target_file)
            if get_noise:
                dest = os.path.join(target_dir, "noise")
                if not os.path.exists(dest):
                    os.makedirs(dest)
                target  = os.path.join(dest, f)
                noise(chan_1, channel = 1, target_file=target)
                noise(chan_2, channel = 2, target_file=target)
        except:
            failed.append(files)

    print("{} files failed".format(failed))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str,
                        help='Path to data directory')
    parser.add_argument('target_dir', type=str,
                        help='Path to output folder')
    parser.add_argument('--get_noise', type = str, default = True,
                        help ='Get a separate folder containing noise')    
    args = parser.parse_args()
    main(args.source_dir, args.target_dir, args.get_noise)
    