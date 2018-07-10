### Author: Ravi
### Python module for Data Augmentation.
### Change the parameters value to suit your needs. These values are for the inidcTTS dataset with the UrbanSound8K 
### If the audio is real life recordings. No need to add separate noise, as it will greatly distort the audio. Clean the audio using the clean audio script.
### Also, if dealing with telephone audio. then comment out the telephone filter code.
### The default way for storing data is :-
### Data_dir -> The directory containg data
###    |- The csv file containg the transcripts. The first column is the file-name and the second column name is the transcript
###    |- Audio_Dir -> The directory containing the original audio_file (Unless you don't want to drop the original data from training)
###    |- Aug_dir_0 -> The directory containing the augmentated audio-1 and so on. # Created and to be stored there
### If the default way is to be foll

import numpy as np
import pandas as pd
import os
from pydub import AudioSegment, scipy_effects, generators
import shutil
import argparse

def ideal_volume(audio_v, noise_v):
    """
    args - The maximum value of the audio and noise
    returns the ideal volume
    """
    if(noise_v >= audio_v):
        vol = (audio_v - noise_v) + np.random.uniform(0, 5)
    else:
        vol = np.random.uniform(0, 5)
    return vol


def merge_noise(audio, noise):
    """
    args - The location of audio and noise files as
    file and noise_file respectively
    returns - The combined file
    """
    vol = ideal_volume(audio.dBFS, noise.dBFS)
    valid_len = max(0, len(audio) - len(noise) - 500)
    pos = np.random.uniform(0, valid_len)
    if valid_len != 0:
        combined = audio.overlay(noise - vol, position= pos)
    else:
        combined = audio
    return combined

def apply_telephone(audio):
    """
    This applies telephone effect plus white noise on the data
    args - The pydub audio
    returns - a pydub audio
    """
    # White noise
    vol = np.random.uniform(-45, -35)
    wh = generators.WhiteNoise()
    white = wh.to_audio_segment(duration = len(audio), volume = vol)
    
    audio1 = audio.overlay(white)
    # Band Pass filter
    audio1 = audio1.band_pass_filter(100, 4000)
    # Down Sampling
    audio1 = audio1.set_frame_rate(8000)
    return audio1

def speed_change(sound, speed=1.0):
	# From : https://stackoverflow.com/questions/43408833/how-to-increase-decrease-playback-speed-on-wav-file
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })

    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def speed_up(audio):
    """
    This speeds up the audio. Acoording to
    http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf
    it is a bit more superior. One can do pitch and tempo perturbation separately as well.
    """
    rate = np.random.uniform(0.80, 1.25)
    audio1 = speed_change(audio, rate)
    return audio1

def main(data_directory, output_folder, noise_dir, no_of_folds):
    noise = []
    
    for d in os.listdir(noise_dir):
        dest = os.path.join(noise_dir, d)
        if os.path.isdir(dest):
            for f in os.listdir(dest):
                if f.endswith(".wav"):
                    try:
                        noise.append(AudioSegment.from_wav(os.path.join(dest, f)))
                    except:
                        print(os.path.join(dest, f))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    target_dir = []
    for i in range(, no_of_folds):
        target_dir.append(os.path.join(output_folder, "audio_{}".format(i)))
        if not os.path.exists(target_dir[-1]):
            os.makedirs(target_dir[-1])

    for f in os.listdir(source_dir):
        if f.endswith(".wav"):
            file_ = os.path.join(source_dir, f)
            try:
                audio = AudioSegment.from_wav(file_)
                for target in target_dir:
                    try:
                        if noise_dir != 'None':
                            random_noise = np.random.randint(0, len(noise))
                            merged = merge_noise(audio, noise[random_noise])
                        else:
                            merged = audio
                        merged = apply_telephone(merged)
                        merged = speed_up(merged)
                        merged.export(os.path.join(target, f), format = "wav")
                        
                    except:
                        continue
            except:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('output_folder', type=str,
                        help='Path to output folder')
    parser.add_argument('--noise_dir', type = str, default = None, 
    					help = 'Path to the noise directory')
    parser.add_argument('--no_of_folds', type = int, default = 1,
                        help = 'The no of times to multiply the original data.')
    
    args = parser.parse_args()
    main(args.data_directory, args.output_folder, args.noise_dir, args.no_of_folds)