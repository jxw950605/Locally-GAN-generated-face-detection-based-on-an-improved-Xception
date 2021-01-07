# Locally GAN-generated face detection based on an improved Xception
##
# Overview
It has become a research hotspot to detect whether a face is natural or GAN-generated. However, all the existing works focus on whole GAN-generated faces. So, an improved Xception model is proposed for locally GAN-generated face detection. To the best of our knowledge, our work is the first one to address this issue. Some improvements over Xception are as follows: (1) Four residual blocks are removed to avoid the overfitting problem as much as possible for the locally generated face detection; (2) Inception block with the dilated convolution is used to replace the common convolution layer in the pre-processing module of the Xception to obtain multi-scale features; (3) Feature pyramid network is utilized to obtain multi-level features for final decision. The first locally GAN-based generated face (LGGF) dataset is constructed by the pluralistic image completion method on the basis of [FFHQ](https://github.com/tkarras/progressive_growing_of_gans) dataset. It has a total 952,000 images with the generated regions in different shapes and sizes. The architecture is show in 'architecture.png'.



# Prerequisites

- Linux
- Python 3
- NVIDIA GPU+CUDA CuDNN
- Install TensorFlow 1.12.0, keras 2.2.1 and dependencies


# The construction of the LGGF dataset
The binary ground truth masks with different sizes are created by Matlab. Six different sizes are considered with the ratio of the whole image from 0.5% and 5.5% every 1.0% (25×25,32×32,40×40,48×48,56×56,64×64). These masks appear at arbitrary positions in the natural images. Then, the FFHQ dataset is combined with these two types and six different sizes of masks, obtaining twelve incomplete image datasets. The iregular mask for the LGGF dataset is under the path of the 'dataset' folder. The regular mask can be obtained by the function 'get_regular_mask.py'. In the samples, samples_mask,and samples_results floders, we provide some samples.Finally, the pluralistic model[1] is utilized to restore the incomplete region of each image in the twelve incomplete region image datasets. The architecture of the construction is shown in 'construction_of_dataset.png'.



# The inpainting methods used in the paper


- The main inpainting method:   [Pluralistic Image Completion](https://github.com/lyndonzheng/Pluralistic)[1]<br><br>





- The other two inpaininting method: 
[DFNET inpainting](https://github.com/hughplay/DFNet.git) [2] and  [Generative inpainting](https://github.com/hughplay/DFNet.git)[3]<br>

# Setup training and testing


- Train：The regular sub-dataset randomly selects 70,000 images with the regular rectangular generated regions from the LGGF dataset, while the irregular sub-dataset randomly selects 70,000 images with the irregular generated regions. The original FFHQ dataset and two sub-datasets (regular sub-dataset and irregular one) are divided into training, validation, and testing sets with the ratio 5:1:4.
we use the adam optimization with the initial learning rate 1.0e-3, decay 0.000001,and a minibatch size of 32. You can use the mask provided by the above method to get the dataset with the inpainting method, i.e., the Pluralistic Image Completion model, DFNet model, Generative Model. Then, generating the corresponding image path and save it as .npy format for training, validation and testing.

- Test:To test the generalization ability of our algorithm, two sub testing datasets with the other two inpaininting method are considered to evaluate the performance of the trained model.

# Related Works
[1][Zheng C, Cham T J, Cai J, Pluralistic image completion, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR2019), 2019, pp. 1438-1447.](https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Pluralistic_Image_Completion_CVPR_2019_paper.html)<br>
[2][Hong X, Xiong P, Ji R, Deep fusion network for image completion, in: Proceedings of the 27th ACM International Conference on Multimedia (MM2019), 2019, pp. 2033-2042.](https://dl.acm.org/doi/abs/10.1145/3343031.3351002)<br>
[3][Yu J, Lin Z, Yang J, Free-form image inpainting with gated convolution, in: Proceedings of the 2019 IEEE International Conference on Computer Vision (CVPR2019), 2019, pp. 4471-4480.](https://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Free-Form_Image_Inpainting_With_Gated_Convolution_ICCV_2019_paper.html)
