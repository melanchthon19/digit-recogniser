#!/usr/bin/env python3

import os
import subprocess

path_to_opensmile = '/Users/danielmora/opensmile'
path_to_features = '/Users/danielmora/PycharmProjects/digit_recognizer/features/egemapsv01b'
path_to_wav = '/Users/danielmora/PycharmProjects/digit_recognizer/wav'

for set in ['training', 'test']:
    for wav in os.listdir(os.path.join(path_to_wav, set)):
        # to extract different features, change .conf file
        command = f'/Users/danielmora/opensmile/SMILExtract \
        -C /Users/danielmora/opensmile/config/egemaps/v01b/eGeMAPSv01b.conf \
        -I {os.path.join(path_to_wav, set, wav)} \
        -csvoutput {os.path.join(path_to_features, wav[:-3]+"csv")} \
        -instname {wav[:-4]}'
        process = subprocess.run([command], shell=True)
