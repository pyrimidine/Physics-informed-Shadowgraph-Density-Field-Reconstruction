This repository contains the source code for the research paper *"Physics-Informed Shadowgraph Density Field Reconstruction"*.

### Overview
This code implements a physics-informed framework for reconstructing density fields from shadowgraph images. The approach combines shadowgraph imaging techniques with physics-informed neural networks (PINNs) to capture refractive index variations in complex flow fields accurately.

![[image](alcohol burner flame.gif)](https://github.com/pyrimidine/Physics-informed-Shadowgraph-Density-Field-Reconstruction/blob/7813ac0001175ee698601809498e7e402c0ceada/alcohol%20burner%20flame.gif)

![[image](alcohol burner flame.gif)](https://github.com/pyrimidine/Physics-informed-Shadowgraph-Density-Field-Reconstruction/blob/b692b8067b5e4ed713c8ab9bc6f2452f859fd0c5/results/alcohol%20burner%20plume.gif)



The video below demonstrates the on-time prediction. Due to the lack of high-performance GPU support on this laptop, the prediction process is relatively slow. (P.S.: Our model was definitely not trained on this laptop! 😊)

<img src="https://github.com/pyrimidine/Physics-informed-Shadowgraph-Density-Field-Reconstruction/blob/8d63b60e040105e1bf569d72dc60a6757eecad4a/results/on-time%20reconstruction1.gif" alt="Image" width="350" height="500"/>

### Key Features
- **Shadowgraph Image Processing**: Pre-processing and analysis of shadowgraph images for density field visualization.
- **PINN Implementation**: Physics-informed neural network setup tailored for accurate density field reconstruction.
- **Density Field Reconstruction**: Algorithms for computing density distributions based on refractive index variations within the experimental field.

### Dataset
The data folder only contains a few example images. Its purpose is to illustrate what the shadowgraph image looks like. Training solely on this data will result in **significant overfitting**. A complete example dataset can be obtained by contacting the authors.

### Requirements
All necessary dependencies are listed in `requirements.txt`.

### License
This code is for academic use **ONLY**. Please cite the original paper if you use this code in your research.

### Authors
- **Primary Author**: Xutun Wang, Yuchen Zhang
- **Contact Information**: You can reach us at [xt-wang24@mails.tsinghua.edu.cn] or through our academic institution profiles.
-  Special thanks to Dr. Yuchen Zhang @paradoxknight1 for his significant contributions to this research. 

