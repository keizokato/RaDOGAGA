##This software includes the work that is distributed in the Apache License 2.0
## some function borrowed from
## https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py
## _my_ssim_helper function is added to calculate SSIM with the approximated form.

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

def convert_image_dtype(image, dtype, saturate=False, name=None):
  """Convert `image` to `dtype`, scaling its values if needed.

  Images that are represented using floating point values are expected to have
  values in the range [0,1). Image data stored in integer data types are
  expected to have values in the range `[0,MAX]`, where `MAX` is the largest
  positive representable number for the data type.

  This op converts between data types, scaling the values appropriately before
  casting.

  Note that converting from floating point inputs to integer types may lead to
  over/underflow problems. Set saturate to `True` to avoid such problem in
  problematic conversions. If enabled, saturation will clip the output into the
  allowed range before performing a potentially dangerous cast (and only before
  performing such a cast, i.e., when casting from a floating point to an integer
  type, and when casting from a signed to an unsigned type; `saturate` has no
  effect on casts between floats, or on casts that increase the type's range).

  Args:
    image: An image.
    dtype: A `DType` to convert `image` to.
    saturate: If `True`, clip the input before casting (if necessary).
    name: A name for this operation (optional).

  Returns:
    `image`, converted to `dtype`.
  """
  image = ops.convert_to_tensor(image, name='image')
  if dtype == image.dtype:
    return array_ops.identity(image, name=name)

  with ops.name_scope(name, 'convert_image', [image]) as name:
    # Both integer: use integer multiplication in the larger range
    if image.dtype.is_integer and dtype.is_integer:
      scale_in = image.dtype.max
      scale_out = dtype.max
      if scale_in > scale_out:
        # Scaling down, scale first, then cast. The scaling factor will
        # cause in.max to be mapped to above out.max but below out.max+1,
        # so that the output is safely in the supported range.
        scale = (scale_in + 1) // (scale_out + 1)
        scaled = math_ops.div(image, scale)

        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)
      else:
        # Scaling up, cast first, then scale. The scale will not map in.max to
        # out.max, but converting back and forth should result in no change.
        if saturate:
          cast = math_ops.saturate_cast(image, dtype)
        else:
          cast = math_ops.cast(image, dtype)
        scale = (scale_out + 1) // (scale_in + 1)
        return math_ops.multiply(cast, scale, name=name)
    elif image.dtype.is_floating and dtype.is_floating:
      # Both float: Just cast, no possible overflows in the allowed ranges.
      # Note: We're ignoreing float overflows. If your image dynamic range
      # exceeds float range you're on your own.
      return math_ops.cast(image, dtype, name=name)
    else:
      if image.dtype.is_integer:
        # Converting to float: first cast, then scale. No saturation possible.
        cast = math_ops.cast(image, dtype)
        scale = 1. / image.dtype.max
        return math_ops.multiply(cast, scale, name=name)
      else:
        # Converting from float: first scale, then cast
        scale = dtype.max + 0.5  # avoid rounding problems in the cast
        scaled = math_ops.multiply(image, scale)
        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)

def _fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = ops.convert_to_tensor(size, dtypes.int32)
  sigma = ops.convert_to_tensor(sigma)

  coords = math_ops.cast(math_ops.range(size), sigma.dtype)
  coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

  g = math_ops.square(coords)
  g *= -0.5 / math_ops.square(sigma)

  g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
  g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = nn_ops.softmax(g)
  return array_ops.reshape(g, shape=[size, size, 1, 1])


def _verify_compatible_image_shapes(img1, img2):
  """Checks if two image tensors are compatible for applying SSIM or PSNR.

  This function checks if two sets of images have ranks at least 3, and if the
  last three dimensions match.

  Args:
    img1: Tensor containing the first image batch.
    img2: Tensor containing the second image batch.

  Returns:
    A tuple containing: the first tensor shape, the second tensor shape, and a
    list of control_flow_ops.Assert() ops implementing the checks.

  Raises:
    ValueError: When static shape check fails.
  """
  shape1 = img1.get_shape().with_rank_at_least(3)
  shape2 = img2.get_shape().with_rank_at_least(3)
  shape1[-3:].assert_is_compatible_with(shape2[-3:])

  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(reversed(shape1[:-3]), reversed(shape2[:-3])):
      if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
        raise ValueError(
            'Two images are not compatible: %s and %s' % (shape1, shape2))

  # Now assign shape tensors.
  shape1, shape2 = array_ops.shape_n([img1, img2])

  # TODO(sjhwang): Check if shape1[:-3] and shape2[:-3] are broadcastable.
  checks = []
  checks.append(control_flow_ops.Assert(
      math_ops.greater_equal(array_ops.size(shape1), 3),
      [shape1, shape2], summarize=10))
  checks.append(control_flow_ops.Assert(
      math_ops.reduce_all(math_ops.equal(shape1[-3:], shape2[-3:])),
      [shape1, shape2], summarize=10))
  return shape1, shape2, checks


_SSIM_K1 = 0.01
_SSIM_K2 = 0.03

#def average_pool(x):


def _ssim_helper(x, y, reducer, max_val, compensation=1.0):
  r"""Helper function for computing SSIM.

  SSIM estimates covariances with weighted sums.  The default parameters
  use a biased estimate of the covariance:
  Suppose `reducer` is a weighted sum, then the mean estimators are
    \mu_x = \sum_i w_i x_i,
    \mu_y = \sum_i w_i y_i,
  where w_i's are the weighted-sum weights, and covariance estimator is
    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
  For SSIM measure with unbiased covariance estimators, pass as `compensation`
  argument (1 - \sum_i w_i ^ 2).

  Arguments:
    x: First set of images.
    y: Second set of images.
    reducer: Function that computes 'local' averages from set of images.
      For non-covolutional version, this is usually tf.reduce_mean(x, [1, 2]),
      and for convolutional version, this is usually tf.nn.avg_pool or
      tf.nn.conv2d with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.

  Returns:
    A pair containing the luminance measure, and the contrast-structure measure.
  """
  c1 = (_SSIM_K1 * max_val) ** 2
  c2 = (_SSIM_K2 * max_val) ** 2

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = math_ops.square(mean0) + math_ops.square(mean1)
  #luminance = (num0 + c1) / (den0 + c1)
  luminance = (num0 + c1) / (den0 + c1)


  # SSIM contrast-structure measure is
  #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
  #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
  num1 = reducer(x * y) * 2.0
  den1 = reducer(math_ops.square(x) + math_ops.square(y))
  c2 *= compensation

  cs = (num1 - num0 + c2) / (den1 - den0 + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs


def average_pool(X):
  mean_pool = tf.nn.avg_pool(
    value=X,
    ksize=[1, 11, 11, 1],
    strides=[1, 1, 1, 1],
    padding='VALID'
    # data_format=None,
    # name=None
  )
  return mean_pool

def _my_ssim_helper(x, y, z, reducer, max_val, compensation=1.0, mode='test'):

  #x = origi
  #y = delta
  c1 = (_SSIM_K1 * max_val) ** 2
  c2 = (_SSIM_K2 * max_val) ** 2

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean_x = reducer(x)
  mean_y = reducer(y)
  mean_z = reducer(z)

  #num0 = mean0 * mean1 * 2.0
  #den0 = math_ops.square(mean0) + math_ops.square(mean1)
  square_mu_x = math_ops.square(mean_x)
  square_mu_y = math_ops.square(mean_y)
  num0 = mean_y * mean_z

  c1 *= compensation

  if mode == 'train':
    luminance = (square_mu_y ) / (2 * square_mu_x + c1)
  else:
    luminance = (num0) / (2*square_mu_x + c1)

  c2 *= compensation

  var_x = reducer(math_ops.square(x)) - square_mu_x

  if mode == 'train':
    var = reducer(math_ops.square(y)) - square_mu_y
  else:
    var = reducer(y * z) - num0

  cs = (var) / (2*var_x + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs


def _my_ssim_helper_debug(y, z, reducer):
  mean_pool = average_pool(y*z)

  print('mean_pool',mean_pool.shape)

  reduce = reducer(y*z)

  return mean_pool, reduce



def _ssim_per_channel(img1, img2, img3, max_val=1.0, mode='test',compensation=1):
  """Computes SSIM index between img1 and img2 per color channel.

  This function matches the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.

  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).

  Returns:
    A pair of tensors containing and channel-wise SSIM and contrast-structure
    values. The shape is [..., channels].
  """
  filter_size = constant_op.constant(11, dtype=dtypes.int32)
  filter_sigma = constant_op.constant(1.5, dtype=img1.dtype)

  shape1, shape2 = array_ops.shape_n([img1, img2])
  shape1, shape2, shape3 = array_ops.shape_n([img1, img2, img3])
  checks = [
      control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
          shape1[-3:-1], filter_size)), [shape1, filter_size], summarize=8),
      control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
          shape2[-3:-1], filter_size)), [shape2, filter_size], summarize=8),
    control_flow_ops.Assert(math_ops.reduce_all(math_ops.greater_equal(
        shape2[-3:-1], filter_size)), [shape3, filter_size], summarize=8)]


  # Enforce the check to run before computation.
  with ops.control_dependencies(checks):
    img1 = array_ops.identity(img1)

  # TODO(sjhwang): Try to cache kernels and compensation factor.
  kernel = _fspecial_gauss(filter_size, filter_sigma)
  kernel = array_ops.tile(kernel, multiples=[1, 1, shape1[-1], 1])

  # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
  # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
  #compensation = 1.0

  # TODO(sjhwang): Try FFT.
  # TODO(sjhwang): Gaussian kernel is separable in space. Consider applying
  #   1-by-n and n-by-1 Gaussain filters instead of an n-by-n filter.
  def reducer(x):
    shape = array_ops.shape(x)
    x = array_ops.reshape(x, shape=array_ops.concat([[-1], shape[-3:]], 0))
    y = nn.depthwise_conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
    return array_ops.reshape(y, array_ops.concat([shape[:-3],
                                                  array_ops.shape(y)[1:]], 0))

  #luminance, cs = _ssim_helper(img1, img2, reducer, max_val, compensation)
  if mode == 'debug':
    luminance_gt, cs_gt = _ssim_helper(img1, img2, reducer, max_val, compensation)
    luminance, cs = _my_ssim_helper(img1, img1-img2, img1-img3, reducer, max_val, compensation, mode='train')
  else:
    luminance, cs = _my_ssim_helper(img1, img2, img3, reducer, max_val, compensation, mode)

  if mode == 'debug':
    axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    ssim_val_gt = math_ops.reduce_mean(luminance_gt * cs_gt, axes)
    lm_gt = math_ops.reduce_mean(luminance_gt, axes)
    cs_gt = math_ops.reduce_mean(cs_gt, axes)

    lm = math_ops.reduce_mean(luminance, axes)
    cs = math_ops.reduce_mean(cs, axes)

    return lm_gt, cs_gt, lm, cs, ssim_val_gt
  else:
    # Average over the second and the third from the last: height, width.
    axes = constant_op.constant([-3, -2], dtype=dtypes.int32)
    #ssim_val = math_ops.reduce_mean(luminance * cs, axes)
    ssim_val = math_ops.reduce_mean(luminance + cs, axes)
    print('ssim_shape',ssim_val.shape)
    cs = math_ops.reduce_mean(cs, axes)

    return ssim_val, cs


#@tf_export('image.ssim')
def ssim(img1, img2, img3, max_val, mode='test', compensation=1):
  """Computes SSIM index between img1 and img2.

  This function is based on the standard SSIM implementation from:
  Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
  quality assessment: from error visibility to structural similarity. IEEE
  transactions on image processing.

  Note: The true SSIM is only defined on grayscale.  This function does not
  perform any colorspace transform.  (If input is already YUV, then it will
  compute YUV SSIM average.)

  Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.

  The image sizes must be at least 11x11 because of the filter size.

  Example:

  ```python
      # Read images from file.
      im1 = tf.decode_png('path/to/im1.png')
      im2 = tf.decode_png('path/to/im2.png')
      # Compute SSIM over tf.uint8 Tensors.
      ssim1 = tf.image.ssim(im1, im2, max_val=255)

      # Compute SSIM over tf.float32 Tensors.
      im1 = tf.image.convert_image_dtype(im1, tf.float32)
      im2 = tf.image.convert_image_dtype(im2, tf.float32)
      ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
      # ssim1 and ssim2 both have type tf.float32 and are almost equal.
  ```

  Args:
    img1: First image batch.
    img2: Second image batch.
    max_val: The dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).

  Returns:
    A tensor containing an SSIM value for each image in batch.  Returned SSIM
    values are in range (-1, 1], when pixel values are non-negative. Returns
    a tensor with shape: broadcast(img1.shape[:-3], img2.shape[:-3]).
  """
  _, _, checks = _verify_compatible_image_shapes(img1, img2)
  with ops.control_dependencies(checks):
    img1 = array_ops.identity(img1)

  # Need to convert the images to float32.  Scale max_val accordingly so that
  # SSIM is computed correctly.
  max_val = math_ops.cast(max_val, img1.dtype)
  max_val = convert_image_dtype(max_val, dtypes.float32)
  img1 = convert_image_dtype(img1, dtypes.float32)
  img2 = convert_image_dtype(img2, dtypes.float32)
  img3 = convert_image_dtype(img3, dtypes.float32)

  #ssim_per_channel, _ = _ssim_per_channel(img1, img2, max_val)

  # Compute average over color channels.
  if mode == 'debug':
    lm_gt, cs_gt, lm, cs, ssim_val_gt = _ssim_per_channel(img1, img2, img3, max_val, mode, compensation)
    lm_gt = math_ops.reduce_mean(lm_gt, [-1])
    cs_gt = math_ops.reduce_mean(cs_gt, [-1])
    lm = math_ops.reduce_mean(lm, [-1])
    cs = math_ops.reduce_mean(cs, [-1])
    ssim_val_gt = math_ops.reduce_mean(ssim_val_gt, [-1])
    return lm_gt, cs_gt, 1.0 -lm, 1.0 - cs, ssim_val_gt
  else:
    ssim_per_channel, _ = _ssim_per_channel(img1, img2, img3, max_val, mode, compensation)
    return math_ops.reduce_mean(ssim_per_channel, [-1])