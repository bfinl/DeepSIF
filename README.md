# DeepSIF: Deep Learning based Source Imaging Framework


DeepSIF is an EEG/MEG source imaging framework aiming at providing an estimation of the location, size, and temporal activity of the brain activities from scalp EEG/MEG recordings. There are three components: training data generation (```forward/```), neural network training (```main.py```), and model evaluation (```eval_sim.py```,```eval_recal.py```), as detailed below. The codes are provided as a service to the scientific community, and should be used at users’ own risks.


This work was supported in part by the National Institutes of Health grants NS096761, EB021027, AT009263, MH114233, EB029354, and NS124564, awarded to Dr. Bin He, Carnegie Mellon University. Additional data in 20 human epilepsy patients tested in this work can be found at


https://figshare.com/s/580622eaf17108da49d7.


Please cite the following publication if you are using any part of the codes:

Sun R, Sohrabpour A, Worrell GA, He B: “Deep Neural Networks Constrained by Neural Mass Models Improve Electrophysiological Source Imaging of Spatio-temporal Brain Dynamics.” Proceedings of the National Academy of Sciences, 2022.



## Train Data Generation
#### The Virtual Brain Simulation
```bash
python generate_tvb_data.py --a_start 0 --a_end 10
```
The simulation for each region can also run in parallel. (Require multiprocessing installed.)
 
#### Process Raw TVB Data and Prepare Training/Testing Dataset 
Run in Matlab
```matlab
process_raw_nmm
generate_sythetic_source
```
The output of ```generate_sythetic_source``` can be used as input for ```loaders.SpikeEEGBuild``` or ```loaders.SpikeEEGBuildEval```

## Training

After sythetic training dataset is generated, ```main.py``` can be used to train a DeepSIF model. ```network.py``` contains the architecture used
 in the paper.  ```loaders.py``` provides two ways to load the dataset. If the data is already saved in seperate input/output files
 , ```SpikeEEGLoad``` can be used. If training data is generated on the run, ```SpikeEEGBuild``` can be used to generate different types of
  training data. To train a model, use 

```bash
python main.py --model_id 1
```
**Parameters:**

    '--save', type=int, default=True, help='save each epoch or not'
    '--workers', default=0, type=int, help='number of data loading workers'
    '--batch_size', default=64, type=int, help='batch size'
    '--device', default='cuda:0', type=str, help='device running the code'
    '--arch', default='TemporalInverseNet', type=str, help='network achitecture class'
    '--dat', default='SpikeEEGBuild', type=str, help='data loader class'
    '--train', default='test_sample_source2.mat', type=str, help='train dataset name or directory'
    '--test', default='test_sample_source2.mat', type=str, help='test dataset name or directory'
    '--model_id', default=75, type=int, help='model id'
    '--lr', default=3e-4, type=float, help='learning rate'
    '--resume', default='1', type=str, help='epoch id to resume'
    '--epoch', default=20, type=int, help='total number of epoch'
    '--fwd', default='leadfield_75_20k.mat', type=str, help='forward matrix to use'
    '--rnn_layer', default=3, type=int, help='number of rnn layer'
    '--info', default='', type=str, help='other information regarding this model'

## Evaluation

#### Simulation :
After a model is trained, ```eval_sim.py``` can be used to evaluate the trained model in simulations under different conditions. Some examples are:
```bash
python eval_sim.py --model_id 75
```
Additional tests: narrow-band input
```bash
python eval_sim.py --model_id 75 --lfreq 1 --hfreq 3
```
Additional tests: different noise type
```bash
python eval_sim.py --model_id 75 --snr_rsn_ratio 0.5
```
Additional tests: different head / conductivity value / electrode locations
```bash
python eval_sim.py --model_id 75 --fwd <the forward matrix file>
```
#### Real data :
Or use real data as the model input as shown in ```eval_real.py```:
```bash
python eval_real.py
```
Default subject folder : VEP
<!---Results visualization examples are in tutorials/-->

## Dependencies
* [Python >= 3.8.3](https://www.python.org/downloads/)
* [PyTorch>=1.6.0](https://pytorch.org/)
* [tvb](https://www.thevirtualbrain.org/tvb/zwei)
* [mne](https://mne.tools/stable/index.html)
* [h5py](https://www.h5py.org/)
* [numpy](https://numpy.org/)