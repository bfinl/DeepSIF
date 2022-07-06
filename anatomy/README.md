### This folder contains miscellaneous files related to the forward process
#### For TVB simulation
* connectivity_76.zip
* connectivity_998.zip

Connectivity file used for the TVB simulation, downloaded from
https://github.com/the-virtual-brain/tvb-data

#### Template headmodel:
- **fs_cortex_20k.mat**: fsaverage5 cortex
    - pos, tri, ori: vertices positions, triangulations, and dipole orientations; resolution 20484
    - left_ind, left_tri: index for left hemisphere, triangulations in the left hemisphere
    - right_ind, right_tri: index for right hemisphere, triangulations in the right hemisphere
         
- **fs_cortex_20k_inflated.mat**: inflated fsaverage5 cortex
    - pos, tri: vertices positions, triangulations; resolution 20484
    - posl, tril: vertices positions and triangulations in the left hemisphere
    - posr, trir: vertices positions and triangulations in the right hemisphere

- **fs_cortex_20k_region_mapping.mat** : map fsaverage5 to 994 regions
    - rm: region mapping id, size 1*20484
    - nbs: neighbours for each region
    
- **leadfield_75_20k.mat** : leadfield matrix for fsaverage5, 75 channels
    - fwd: size 75*994

- **dis_matrix_fs_20k.mat** : distance between source centres
    - raw_dis_matrix: size 994*994

- **electrode_75.mat** : 75 EEG channels in EEGLAB format
    - eloc75

- **fsaverage5/**: contains the files for the raw freesurfer output, for plotting in mne    

#### Simulations:
- **realistic_noise.mat** : resting data extracted from 75 channel EEG recordings
    - data: num_examples * num_time* num_channel; 4*500*75
    - npower: the power for each channel; 4*75 