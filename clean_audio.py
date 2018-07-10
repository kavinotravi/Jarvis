### Author: Ravi
### Subject: This script takes input as the noise folder. 
###          Builds a noise profile for each noise file
###          Apply the corresponding noise filter on the required file
### Untested script. So, look out for any error

import os
import sox
import argparse

def remove_noise(source_dir, f, dir_, val = 0.21):
	"""
	args:- source -> source directory of the audio files
		      f   -> The noise file
		     dest -> The destination folder of the noise profile
	"""
	
	dest = os.path.join(source_dir, "noise_prof")
	source = os.path.join(source_dir, "audio")
	target = os.path.join(source_dir, "audio0")
	tfm = sox.Transformer()
	tfm.noiseprof(os.path.join(dir_, f), os.path.join(dest, f))
	tfm.noisered(os.path.join(dest, f), val)
	for fil in os.listdir(source):
		if fil.endswith(".wav"):
			if fil in f[:-4]:
				tfm.build(os.path.join(source, fil), os.path.join(dest, fil))



def main(source_dir, noise_filter):
	dir_ = os.path.join(source_dir, "noise")
	for noise in os.listdir(dir_):
		if f.endswith(".wav"):
			remove_noise(source_dir, f, dir_, noise_filter)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str,
                        help='Path to data directory, containing the noise directory')
    parser.add_argument('--noise_filter', type = float, default = 0.21,
                        help ='The sensitivity of the noise filter')    
    args = parser.parse_args()
    main(args.source_dir, args.noise_filter)