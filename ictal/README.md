# Seizure Sources Can be Imaged from Scalp EEG by means of Biophysically Constrained Deep Neural Networks

This subfolder contains the additional files to perform ictal simulation, and an example for the seizure imaging results. 

* To train a ictal DeepSIF model, assume you already run the full simulation in `../forward`, as descibed in [1]. You already have `Dataset.data` generated and saved. The ictal simulation will be built on top of this. 
* Run the ictall NMM simulation using `ictal_simulation.py`
* Examine the ictal data using clustering using `cluster_ictal_trace.m`, new ictal traces might need to be generated based on the clustering results.
* Segment generated traces in to 1 second signal, and the final cleaned ictal waveform is saved in `../source/ictal_source_waveform.mat`
* Use the main.py function to train the network, the select `SZNMMDatah5` as the `dat` parameter. 
* After the model is trained, the ictal imaging results on patient data can be generated using `../eval_real.py`
* An example of the imaging result is in `example`


[1] Sun, Rui, et al. "Deep neural networks constrained by neural mass models improve electrophysiological source imaging of spatiotemporal brain dynamics." Proceedings of the National Academy of Sciences 119.31 (2022): e2201128119.


This work was supported in part by the National Institutes of Health grants NS127849 and NS096761, awarded to Dr. Bin He, Carnegie Mellon University. 



Please cite the following publication if you are using any part of the codes or data:

Sun, R., Sohrabpour, A., Joseph, B., Worrell, G., & He, B. (2024). Spatiotemporal Rhythmic Seizure Sources Can be Imaged by means of Biophysically Constrained Deep Neural Networks. Advanced Science. DOI: 10.1002/advs.202405246
