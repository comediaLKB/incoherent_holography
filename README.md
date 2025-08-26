# incoherent_holography
MATLAB codes to reconstruct image of incoherent object using experimental data.

The results are presented in the paper *Three-dimensional holographic imaging of incoherent objects through scattering media*  
Authors: YoonSeok Baek, Hilton B. de Aguiar, Sylvain Gigan

System tested: Windows 10, MATLAB 2023a, CPU: i7-8700, GPU: NVIDIA GeForece GTX 670, RAM: 64 GB.
Typical runtime is a few minutes or less.

**Files and Usage**

`main.m`  
Reconstructs and plots a virtual scattering layer and object image using scattered fields.

`field_data_spiral.mat`  
Input for "main.m". Contains `E_ret`, a three-dimensional array representing scattered fields in complex amplitude (x, y, index of fields).

`phase_retrieval_example.m`  
Generates `E_ret` for `field_data_spiral.mat`.

`image_data.mat`  
Input for `phase_retrieval_example.m`. Contains background-subtracted camera images (`image_data`) and the corresponding SLM patterns (`slmPattern`), both as three-dimensional arrays (x, y, index).  
