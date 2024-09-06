# Deep learning based source imaging provides strong sublobar localization of epileptogenic zone from MEG interictal spikes


This subfolder contains the file to train a personalized DeepSIF model given the trained weight of a generic DeepSIF model. The codes are provided as a service to the scientific community, and should be used at users’ own risks.


This work was supported in part by the National Institutes of Health grants NS096761, NS127849, EB021027, AT009263, MH114233, EB029354, and NS124564, awarded to Dr. Bin He, Carnegie Mellon University. 


Please cite the following publication if you are using any part of the codes:

Sun R, Zhang W, Bagić A, He B: “Deep learning based source imaging provides strong sublobar localization of epileptogenic zone from MEG interictal spikes.” NeuroImage 281 (2023): 120366. DOI:10.1016/j.neuroimage.2023.120366


```process_cortex``` folder contains information to process the MRI for the personalized model.
1. In ```Freesurfer```, run ```recon-all``` on ```$SUBJECT_NAME```'s MRI. Then, get the region labels for this subject using (save for right cortex)
```mri_surf2surf --srcsubject fsaverage5 --trgsubject $SUBJECT_NAME --hemi lh --sval-annot nmm_994.annot --tval $SUBJECTS_DIR/$1/label/lh.nmm_994.annot```
2. Load the generate surface to Brainstorm, downsample the cortex using iso2mesh to ~20k vertices, generate BEM model using this cortex. reference: ```process_cortex/BrainstormProcessing.m```
3. Downsample the region labels. Transform the coordinates between brainstorm and freesurfer. Reference: ```process_cortex/ProcessSurface.m```