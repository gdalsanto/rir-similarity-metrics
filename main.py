import librosa 
import argparse
import torch 
import itertools
import pandas as pd 

from utils import *
from metrics import *


def main(args):
    # get list of all wav files inside dir_path
    filepaths = [file for file in os.listdir(args.dir_path) if file.endswith('.wav')]
    # create all combinations of 2 elements from filepaths
    rir_combinations = list(itertools.combinations(filepaths, 2))
    # create an empty dataframe to store the results
    results_df = pd.DataFrame(columns=['rir1', 'rir2', ])
    # initialize the results dataframe
    results_df = pd.DataFrame(columns=['rir1', 'rir2', 'MAE_stft', 'MultiResoSTFT', 'ESRLoss', 'AveragePower', 'EDCLoss'])
    for i, (rir1_name, rir2_name) in enumerate(rir_combinations):
        # load the impulse responses
        rir1, sr = librosa.load(os.path.join(args.dir_path, rir1_name), sr=args.samplerate)
        rir2, sr = librosa.load(os.path.join(args.dir_path, rir2_name))
        # compute the mixing time
        rir1 = preprocess_rir(rir1, sr)
        rir2 = preprocess_rir(rir2, sr)
        # truncate the longest rir to match its length with the other one
        if len(rir1) > len(rir2):
            rir1 = rir1[:len(rir2)]
        else:
            rir2 = rir2[:len(rir1)]
        # create a list of loss functions
        losses = [MAE_stft(), MultiResoSTFT(), ESRLoss(), AveragePower(), EDCLoss(sr=sr, nfft= int(np.round((1 - 0.5 / 100) * len(rir1))/2+1))]
        # add batch dimension 
        rir1 = torch.tensor(rir1).unsqueeze(0).unsqueeze(0)
        rir2 = torch.tensor(rir2).unsqueeze(0).unsqueeze(0)
        # compute losses 
        for loss in losses:
            results_df.loc[i, loss.__class__.__name__] = loss(rir1, rir2).item()
        # add rir names to the dataframe
        results_df.loc[i, 'rir1'] = rir1_name
        results_df.loc[i, 'rir2'] = rir2_name

    # save results_df to the dir_path folder
    results_df.to_csv(os.path.join(args.dir_path, 'results_losses.csv'), index=False)

def preprocess_rir(rir, sr):
    '''preprocess the rir by removing the pre-delay and the mixing time'''
    # convert stereo/multichannel to mono by taking first channel only
    if len(rir.shape) > 1:
        rir = rir[:, 0]
    onset = rir_onset(rir)
    mixing_time, _ = get_echo_density(rir[onset:], sr)
    return rir[int(mixing_time*sr/1000):]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samplerate', type=int, default=44100,
        help='samplerate of the impulse responses')
    parser.add_argument('--dir_path', type=str, default=None,
        help='path to the directory containing the rirs to analyse')
    
    args = parser.parse_args()

    main(args)