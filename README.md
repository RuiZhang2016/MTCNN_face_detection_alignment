# Face detection using MTCNN
This work implements [1] to detect faces, mainly based on [2]. See principles in [1] and original implementation of [1] in [2]. 

### Requirement
1. Caffe: Linux OS: https://github.com/BVLC/caffe. Windows OS: https://github.com/BVLC/caffe/tree/windows or https://github.com/happynear/caffe-windows 
2. Pdollar toolbox: https://github.com/pdollar/toolbox
3. Matlab 2014b or later

### System
Ubuntu 16.04

### Caffe Installation
Caffe has already been included in 'external/caffe' and follow the below steps to install caffe with only CPU:
1. Change directory to 'external/caffe/' and change 'MATLAB_DIR' in 'Makefile.config' to the path of your own matlab. E.g., MATLAB_DIR := path/to/my/matlab
2. Run 'make clean' in your terminal before compile.
3. Run 'make -j8 && make matcaffe' to compile caffe.

### Possible Errors
1. '.../+caffe/private/caffe_.mexa64':
    .../+caffe/private/caffe_.mexa64: undefined symbol: _ZN2cv8imencodeE....
    
Error in caffe.set_mode_cpu (line ...)

caffe_('set_mode_cpu');

Error in caffe.run_tests (line ...)

caffe.set_mode_cpu();

solution: see https://github.com/BVLC/caffe/issues/3934

### License
This code is distributed under MIT LICENSE

### Reference
[1] Zhang, Kaipeng, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." IEEE Signal Processing Letters 23, no. 10 (2016): 1499-1503.
[2] https://github.com/kpzhang93/MTCNN_face_detection_alignment.

### Contact
Rui Zhang: u5963436@anu.edu.au | zhangrui0996@qq.com
