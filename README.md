# Rate-distortion optimization guided autoencoder for isometric embedding in Euclidean latent space
This repository provides the part of source files for the baseline method and implementation guide for reproducing the above paper submitted to ICML2020 and experiments therein.
### Environment
- iPython 7.7.0 
- Ubuntu 18.04 LTS

## EXP 5.1
In this experiment, we show the isometricity of embedding with factorized prior model. 
#### Requirement
Use reqirements_tf12.txt in EXP_5.1_CelebA directory.
```sh
$pip install -r requirements.txt
```
Also, download tensorflow compression library from https://github.com/tensorflow/compression and put tensorflow_compression directory in the EXP_5.1_CelebA/.

#### Dataset
Please download the CelebA dataset from the bellowing link.
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Then, image should be center cropped with the size of 64x64 with 'center_celeba.py'.
Edit line 10 to the directory you unzip the CelebA dataset.
Also edit line 16 to name the directory you want to store cropped images.

```bash
ipython center_celeba.py
```
#### Run Experiments 

Execution code is "var.py" for beta-VAE. Run the script 'train.sh' to train model and statical analysis.  
Before runnin the script, please edit "img=CelebA/cropped_64" to be directory where you located center cropped image.

```bash
sh train.sh
```

Then, the result will be created in the directory "cache/'method'_'metrix'" .
You will find following png files.
- \*isometoric_mse_\*.png : plot of inner products as in Figure 4 (for MSE).  
- \*isometoric_ssim_\*.png : plot of inner products as in Figure 4 (for 1-SSIM). 
- \*variance_df_*.png : plot of variance of latent variables as in Figure 5.  
- \*traverse_top9.png : Latent traverse of latent variables with top 9 variance as in Figure 6.

Note that since we did not fix the random seed, the result would be slighly different from that in the paper. 
For our method(RaDOGAGA), the impelmetation is done by making use of EntropyBottleneck class in [tensorflow_compression](https://github.com/tensorflow/compression).

## EXP 5.2
Unfortunately, we will not distribute the source code due to the license issue. 
Our implementation is based on [DAGMM](https://github.com/tnakae/DAGMM). 
You can modify the above code according to the description in the paper. 
Note that in the above code, the residual error is concatenated to latent variables. 
In this experiment, the residual error is not neccesarry. 

#### Requirement
Use reqirements.txt in EXP_5.2_toy directory.

#### Dataset
"toy.csv" includes the toy data with the form of 17 x 10000. First 16 columns are input data and the last column is the ground truth of PDF. 
Input data should be max-min normalized throughout data. 

## EXP 5.3
As well as the EXP 5.2, implementation is based on [DAGMM](https://github.com/tnakae/DAGMM).
In this experiment, residual error is used though, **z** is send to decoder before concatenation. 
#### Requirement
Use reqirements.txt in EXP_5.3_anomaly directory.

#### Dataset
Datasets can be downloaded at https://kdd.ics.uci.edu/ and http://odds.cs.stonybrook.edu.
Note that the data should be max-min normalized towards the channel axis.

#LICENSE
[Apache license 2.0](https://github.com/keizokato/RaDOGAGA/blob/master/LICENSE).
