##This software includes the work that is distributed in the Apache License 2.0
## some function borrowed from
## https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py

import argparse

import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from PIL import Image
import math,sys
import scipy.misc

def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()

def _SSIMForMultiScale(img1,
                       img2,
                       max_val=1,
                       filter_size=11,
                       filter_sigma=1.5,
                       k1=0.01,
                       k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).

  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', img1.shape,
            img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape
    #print('height:',height,'width',width)

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:        
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    c3 = c2/2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))    
    cs = np.mean(v1 / v2)   
    #l = np.mean((2.0*mu12 + c1)/(mu11 + mu22 + c1))
    #c = np.mean((2*np.sqrt(np.abs(sigma11))*np.sqrt(np.abs(sigma22)) + c2)/(sigma11 + sigma22 + c2))
    #s = np.mean((sigma12 + c3)/(np.sqrt(np.abs(sigma11))*np.sqrt(np.abs(sigma22)) + c3))
    #mul = l*c*s
    #print(l,c,s,mul,ssim)
    return ssim, cs


def MultiScaleSSIM(img1,
                   img2,
                   max_val=255,
                   filter_size=11,
                   filter_sigma=1.5,
                   k1=0.01,
                   k2=0.03,
                   weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.

  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.

  Returns:
    MS-SSIM score between `img1` and `img2`.

  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', img1.shape,
            img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [
            convolve(im, downsample_filter, mode='reflect')
            for im in [im1, im2]
        ]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1]**weights[0:levels - 1]) *
            (mssim[levels - 1]**weights[levels - 1]))


def msssim(original, compared):
    if isinstance(original, str):
        original = np.array(Image.open(original).convert('RGB'))
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert('RGB'))

    original = original[None, ...] if original.ndim == 3 else original
    compared = compared[None, ...] if compared.ndim == 3 else compared

    return MultiScaleSSIM(original, compared, max_val=255) #255

def ssim(original,compared,max_val = 255):
    if isinstance(original, str):
        img1 = np.array(Image.open(original).convert('RGB'))
    if isinstance(compared, str):
        img2 = np.array(Image.open(compared).convert('RGB'))
    img1 = img1[None, ...] if img1.ndim == 3 else img1
    img2 = img2[None, ...] if img2.ndim == 3 else img2

    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', img1.shape,
            img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)       

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape    
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03   
    print('max_val:',max_val) 
    if max_val == 1 :       
       img1 = img1/255.
       img2 = img2/255.
       
    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:        
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    c3 = c2/2
    l = np.mean((2.0*mu12 + c1)/(mu11 + mu22 + c1))
    c = np.mean((2*np.sqrt(np.abs(sigma11))*np.sqrt(np.abs(sigma22)) + c2)/(sigma11 + sigma22 + c2))
    s = np.mean((sigma12 + c3)/(np.sqrt(np.abs(sigma11))*np.sqrt(np.abs(sigma22)) + c3))
    mul = l*c*s
    print(l,c,s,mul)

def Psnr(original, compared):
    if isinstance(original, str):
        original = np.array(Image.open(original).convert('RGB'))
        #original = scipy.misc.imread(original,mode='RGB').astype(np.float)
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert('RGB'))
        #compared = scipy.misc.imread(compared,mode='RGB').astype(np.float)   
    original = original.astype(np.float64)
    compared = compared.astype(np.float64)

    mse = np.mean(np.square(original - compared))
    if mse == 0 :
       return 100
    #print('mse value:',mse)     
    psnr = np.clip(
        np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    #psnr2 = 20*math.log10(255.0/math.sqrt(mse)) #same value with psnr
    return psnr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', '-m', type=str, default='all', help='metric')
    parser.add_argument(
    '--original-image', '-o', type=str,required=True , help='original image')  #required=True
    parser.add_argument(
    '--compared-image', '-c', type=str, required=True, help='compared image')
    args = parser.parse_args()
    #ssim(args.original_image,args.compared_image,max_val = 255)

    if args.metric == 'all':
       print('psnr: %2.2f'%(Psnr(args.original_image, args.compared_image)))
       print('ms_ssim:%.4f'%(msssim(args.original_image, args.compared_image)))
    elif args.metric == 'psnr':
       #print(msssim(args.original_image, args.compared_image), end='')
       print('%2.2f'%(Psnr(args.original_image, args.compared_image)))       
    elif args.metric == 'ssim':
       #print(psnr(args.original_image, args.compared_image), end='')
       print('%.4f'%(msssim(args.original_image, args.compared_image)))
    else:
       print("wrong argument.")
       sys.exit()
if __name__ == '__main__':
    main()

