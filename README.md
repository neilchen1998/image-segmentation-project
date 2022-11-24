# Image Segmentation Project

Image segmentation is a common fashion in computer vision and digital image processing to partition an image into numerous regions and
segmentations, based on the naturals of the pixels within the image [1]. The goal of this project is to segment an image of a cheetah from the background using a sundry of approaches.

## Overview

We have a black-and-white image of a cheetah. Our goal is to separate the cheetah from the background.

The original image:

![cheetah](https://github.com/neilchen1998/image-segmentation-project/blob/main/dataset/cheetah.bmp?raw=true)

We use a panoply of algorithms to segment the foreground (cheetah) from the background (grass). Then we compare the results against each other and see which approach yields the best result (the lowest error rate).


## Dataset [3]
* cheetah.bmp
* cheetah_mask.bmp
* TrainingSamplesDCT_8.mat
* Zig-Zag Pattern.txt

## Approaches

1. Naive Bayes
2. Maximum Likelihood (ML)
3. Maximum a Posteriori (MAP)
4. Predictive Distribution
5. Expectation Maximization (EM)

## Results

* Naive Bayes

<picture>
  <img src="https://github.com/neilchen1998/image-segmentation-project/blob/main/results/estimation-maximization-result.jpg" width="250" height="250">
</picture>

* Maximum Likelihood (ML)

<picture>
  <img src="https://github.com/neilchen1998/image-segmentation-project/blob/main/results/maximum-likelihood-64-features-result.jpg" width="250" height="250">
</picture>

* Maximum a Posteriori (MAP)

<picture>
  <img src="https://github.com/neilchen1998/image-segmentation-project/blob/main/results/estimation-maximization-result.jpg" width="250" height="250">
</picture>

* Expectation Maximization (EM)

<picture>
  <img src="https://github.com/neilchen1998/image-segmentation-project/blob/main/results/estimation-maximization-result.jpg" width="250" height="250">
</picture>

## Reference
1. [Image Segmentation](https://www.mathworks.com/discovery/image-segmentation.html)

2. [UCSD 271A Statistical Learning I](http://www.svcl.ucsd.edu/courses/ece271A/ece271A.htm)
