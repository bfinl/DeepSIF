## DeepSIF: Train Data Generation

### The Virtual Brain Simulation
```bash
python generate_tvb_data.py --a_start 0 --a_end 10
```
The simulation for each region can also run in parallel. (Require multiprocessing installed.)
 
### Process Raw TVB Data Prepare Training/Testing Dataset 
Run in Matlab
```matlab
process_raw_nmm
generate_sythetic_source
```
The output of ```generate_sythetic_source``` can be used as input for ```loaders.SpikeEEGBuild``` or ```loaders.SpikeEEGBuildEval```
