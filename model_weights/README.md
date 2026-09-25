# DeepSIF – Trained Models

This folder provides trained DeepSIF models for source localization from scalp EEG.

| File | EEG channels | Input shape |
|------|--------------|-------------|
| `model_weights.pt` | 75 | (batch, T, 75) |
| `model_weights_64.pt` | 64 | (batch, T, 64) |
| `model_weights_32.pt` | 32 | (batch, T, 32) |
| `model_weights_21.pt` | 21 | (batch, T, 21) |
| `model_weights_16.pt` | 16 | (batch, T, 16) |

## Overview

These models were trained with the DeepSIF framework using synthetic data, as described in the following publications.

Sun R, Sohrabpour A, Worrell GA, He B: "Deep Neural Networks Constrained by Neural Mass Models Improve Electrophysiological Source Imaging of Spatio-temporal Brain Dynamics," PNAS, 119(31), e2201128119, 2022, https://doi.org/10.1073/pnas.2201128119

Rong J, Sun R, Joseph B, Worrell G, He B: "Deep learning-based EEG source imaging is robust under varying electrode configurations", Clinical Neurophysiology, 2025, https://doi.org/10.1016/j.clinph.2025.04.009

All models share the same architecture and training procedure; only the number of input sensors differs. The 64-, 32-, 21- and 16-channel lead fields are subsets of the 75-channel lead field (fsaverage5 head model, three-shell BEM, 994 cortical regions). The electrode sets are those of Rong et al., 2025 (Supplementary Fig. S1).

## Electrode order

Input channels must follow the order below for each model.

75 channels (`model_weights.pt`):

Fp1, F7, T7, T9, P7, O1, F3, C3, P3, Fpz, Fz, Cz, Pz, Fp2, F8, T8, T10, P8, O2, F4, C4, P4, Nz, Iz, F9, F10, TP11, TP12, P9, P10, Oz, FT9, TP9, FT7, TP7, AF7, F5, FC5, C5, CP5, P5, PO7, AF3, FC3, CP3, PO3, F1, FC1, C1, CP1, P1, AFz, FCz, CPz, POz, F2, FC2, C2, CP2, P2, AF4, FC4, CP4, PO4, AF8, F6, FC6, C6, CP6, P6, PO8, FT8, TP8, FT10, TP10

64 channels (`model_weights_64.pt`):

Fp1, T7, T9, O1, F3, P3, Cz, Pz, Fp2, T8, T10, O2, F4, P4, Nz, Iz, F9, F10, P9, P10, FT9, TP9, FT7, TP7, AF7, F5, FC5, C5, CP5, P5, PO7, AF3, FC3, CP3, PO3, F1, FC1, C1, CP1, P1, AFz, FCz, CPz, POz, F2, FC2, C2, CP2, P2, AF4, FC4, CP4, PO4, AF8, F6, FC6, C6, CP6, P6, PO8, FT8, TP8, FT10, TP10

32 channels (`model_weights_32.pt`):

Fp1, T9, O1, Cz, Fp2, T10, O2, Nz, Iz, F9, F10, P9, P10, FT7, TP7, F5, C5, PO7, CP3, FC1, P1, AFz, CPz, POz, FC2, P2, CP4, F6, C6, PO8, FT8, TP8

21 channels (`model_weights_21.pt`):

T9, Fz, Cz, Pz, T10, Nz, Iz, F9, F10, P9, P10, C5, AF3, FC3, CP3, PO3, AF4, FC4, CP4, PO4, C6

16 channels (`model_weights_16.pt`):

Fp1, T9, Fz, Cz, Pz, Fp2, T10, Iz, F9, F10, P9, P10, FC5, P5, FC6, P6

## How to load

```python
import torch
import network as Network

num_sensor = 64   # 75, 64, 32, 21 or 16, matching the weights file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Network.TemporalInverseNet(
    num_sensor=num_sensor, num_source=994, rnn_layer=3,
    spatial_model=Network.MLPSpatialFilter2, temporal_model=Network.TemporalFilter,
    spatial_output='value', temporal_output='rnn',
    spatial_activation='ELU', temporal_activation='ELU',
    args_params=[500]
).to(device)

net.load_state_dict(torch.load('model_weights_64.pt', map_location=device), strict=True)
net.eval()

# eeg: (batch, T, num_sensor) at 500 Hz, channels in the order above,
# normalized so that max(|eeg|) = 1 for each sample (see eval_real.py)
with torch.no_grad():
    source_activity = net(eeg)['last']   # (batch, T, 994)
```

If you would use the models, please cite Sun et al., 2022 (all models) and Rong et al., 2025 (64-, 32-, 21- and 16-channel models) shown above.
