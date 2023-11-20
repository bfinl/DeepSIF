## DeepSIF: Train Data Generation

### The Virtual Brain Simulation
```bash
python generate_tvb_data.py --a_start 0 --a_end 10
```
The simulation for each region can also run in parallel. (Require multiprocessing installed.)
Parameters:
* Connectivity: current connectivity profile used is the 76 region template provided in the tvb toolbox. Another connectivity template connectivity_998.zip is also provided in anatomy folder. 
* Excitability: The ```a_range``` for the NMM is set as 3.5 in current version. ```a_range``` can be set as a list of A values for the NMM.
* Simulation length: ```siml``` for each segment of raw NMM

### Process Raw TVB Data Prepare Training/Testing Dataset 
Run in Matlab
```matlab
process_raw_nmm
```
Parameters:
* ```iter_list```: same as the length of ```mean_and_std``` in ```generate_tvb_data.py```
* ```leadfield_name```: the size need to be consistent with the connectivity profile used in ```generate_tvb_data.py```. If using the 76-region connectivity profile, need to use the 76-region leadfield. The default leadfield size is 75 times 994.

This function provides extracted 1-second nmm segments with size 500 times 994 (num_of_time_samples times num_of_NMM) saved in source for each region.

```matlab
generate_sythetic_source
```
This function creates 'test_sample_source1.mat' by default, which describes now to load the nmm spikes, how to scale the background noise, etc.
This mat file can be used as input training or testing data for ```loaders.SpikeEEGBuild``` or ```loaders.SpikeEEGBuildEval```.
