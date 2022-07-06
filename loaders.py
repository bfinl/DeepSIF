from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat, savemat
import h5py
from utils import add_white_noise, ispadding
import random
import mne


class SpikeEEGBuild(Dataset):

    """Dataset, generate input/output on the run

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    data : np.array
        TVB output data
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging
    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]

    def __getitem__(self, index):

        if not self.data:
            self.data = h5py.File('{}_nmm.h5'.format(self.file_path[:-12]), 'r')['data']

        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)         # labels with padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]                              # labels without padding
        raw_nmm = np.zeros((500, self.fwd.shape[1]))

        for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            current_nmm = self.data[self.dataset_meta['nmm_idx'][index][kk]]

            ssig = current_nmm[:, [curr_lb[0]]]                                     # waveform in the center region
            # set source space SNR
            ssig = ssig / np.max(ssig) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            # set weight decay inside one source patch
            weight_decay = self.dataset_meta['mag_change'][index][kk]
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())                              # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta['sensor_snr'][index]
        noisy_eeg = add_white_noise(eeg, csnr).transpose()

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),
                  'label': raw_lb,
                  'snr': csnr}
        if self.transform:
            sample = self.transform(sample)

        # savemat('{}/data{}.mat'.format(self.file_path[0][:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb]})
        return sample

    def __len__(self):
        return self.dataset_len


class SpikeEEGLoad(Dataset):

    """Dataset, load pregenerated input/output pair

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    dataset_len : int
        size of the dataset, can be set as a small value during debugging
    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = len(dir('{}/data*.mat'))

    def __getitem__(self, index):

        # load data saved as separate files using loadmat
        raw_data = loadmat('{}/data{}'.format(self.file_path, index))
        sample = {'data': raw_data['data'].astype('float32'),
                  'nmm': raw_data['nmm'].astype('float32'),
                  'label': raw_data['label'],
                  'snr': raw_data['csnr']}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len


class SpikeEEGBuildEval(Dataset):

    """Dataset, generate test data under different conditions to evaluate the model under different conditions

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    data : np.array
        TVB output data
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging

    eval_params : dict
        New attributes compare to SpikeEEGBuild, depending on the test running, keys can be
        lfreq :         int; high pass cut-off frequency; filter EEG data to perform narrow-band analysis
        hfreq :         int; low pass cut-off frequency; filter EEG data to perform narrow-band analysis
        snr_rsn_ratio:  float; [0, 1]; ratio between real noise and gaussian noise


    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len, num_scale_ratio

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.eval_params = dict()

        # check args_params:
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]

        if 'snr_rsn_ratio' in args_params and args_params['snr_rsn_ratio']:                    # Need to add realistic noise
            self.eval_params['rsn'] = loadmat('anatomy/realistic_noise.mat')
            self.eval_params['snr_rsn_ratio'] = args_params['snr_rsn_ratio']
        if 'lfreq' in args_params and args_params['lfreq'] > 0:
            if 'hfreq' in args_params and args_params['hfreq'] > 0:
                self.eval_params['lfreq'] = args_params['lfreq']
                self.eval_params['hfreq'] = args_params['hfreq']
            else:
                print('WARNING: NEED TO ASSIGN BOTH LOW-PASS AND HIGH-PASS CUT-OFF FREQ, IGNORE FILTERING')

    def __getitem__(self, index):

        if not self.data:
            self.data = h5py.File('{}_nmm.h5'.format(self.file_path[:-12]), 'r')['data']

        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)         # labels with padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]                              # labels without padding
        raw_nmm = np.zeros((500, self.fwd.shape[1]))

        for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            current_nmm = self.data[self.dataset_meta['nmm_idx'][index][kk]]

            ssig = current_nmm[:, [curr_lb[0]]]                                     # waveform in the center region
            # set source space SNR
            ssig = ssig / np.max(ssig) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            # set weight decay inside one source patch
            weight_decay = self.dataset_meta['mag_change'][index][kk]
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())                              # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta['sensor_snr'][index]

        # add noise to sensor space
        if 'rsn' in self.eval_params:
            noisy_eeg = add_white_noise(eeg, csnr,
                                        {'ratio': self.eval_params['snr_rsn_ratio'],
                                         'rndata': self.eval_params['rsn']['data'],
                                         'rnpower': self.eval_params['rsn']['npower']}).transpose()
        else:
            noisy_eeg = add_white_noise(eeg, csnr).transpose()

        # filter data into narrow band
        if 'lfreq' in self.eval_params:
            noisy_eeg = mne.filter.filter_data(np.tile(noisy_eeg.transpose(),(1,5)), 500, self.eval_params['lfreq'], self.eval_params['hfreq'],
                                               verbose=False).transpose()
            noisy_eeg = noisy_eeg[1000:1500]

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),
                  'label': raw_lb,
                  'snr': csnr}
        if self.transform:
            sample = self.transform(sample)

        # savemat('{}/data{}.mat'.format(self.file_path[0][:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb]})
        return sample

    def __len__(self):
        return self.dataset_len


# from matplotlib import pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(noisy_eeg)
# plt.subplot(1,2,2)
# plt.plot(empty_nmm[:,lb])