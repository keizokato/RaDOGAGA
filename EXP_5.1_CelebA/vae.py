#!/usr/bin/python
#
# Copyright 2020 The Authors of "Rate-Distortion Optimization Guided Autoencoder for Isometric Embedding in Euclidean Latent Space."
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import argparse,shutil
import os,sys,glob,scipy.misc
import matplotlib.pyplot as plt
import ms_ssim
import inputpipeline
from metric import Psnr, msssim
import math
import ssim_matrix
from sklearn.linear_model import LinearRegression

def analysis_transform(tensor, num_filters):
    """Builds the analysis transform."""

    with tf.variable_scope("analysis"):
        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                num_filters, (9, 9), corr=True, strides_down=2, padding="same_zeros",
                use_bias=True, activation=tfc.GDN())
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=True, activation=tfc.GDN())
            tensor = layer(tensor)

        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=True, activation=tfc.GDN())
            tensor = layer(tensor)

        with tf.variable_scope("layer_3"):
            layer = tfc.SignalConv2D(
                num_filters, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                use_bias=False, activation=None)
            tensor = layer(tensor)

        with tf.variable_scope('reshape'):
            tensor = tf.layers.flatten(tensor)
        if args.activation == 'sigmoid':
            with tf.variable_scope('encoder'):
                tensor = tf.nn.sigmoid(tf.layers.dense(tensor, args.dim1))
                tensor = tf.layers.dense(tensor, args.z)
        elif args.activation == 'softplus':
            with tf.variable_scope('encoder'):
                tensor = tf.nn.softplus(tf.layers.dense(tensor, args.dim1))
                # mean of z
                mean = tf.layers.dense(tensor,  args.z)
                # mean of sigma
                sigma = tf.layers.dense(tensor, args.z)

                # dense layer
                # Sampler: Normal (gaussian) random distribution
                eps = tf.random_normal(tf.shape(mean), dtype=tf.float32, mean=0., stddev=1.0,
                                       name='epsilon')
                # reparameterization trick
                z = mean + tf.exp(sigma / 2) * eps
                # x = tf.layers.dense(x, 128, tf.nn.tanh)
        elif args.activation == 'None':
            with tf.variable_scope('encoder'):
                tensor = tf.layers.dense(tensor, args.z)
        return z, mean, sigma


def synthesis_transform(tensor, num_filters):
    """Builds the synthesis transform."""

    with tf.variable_scope("synthesis", reuse=tf.AUTO_REUSE):
        if args.activation == 'sigmoid':
            with tf.variable_scope('decoder'):
                tensor = tf.nn.sigmoid(tf.layers.dense(tensor, args.dim1))
                tensor = tf.layers.dense(tensor, 4 * 4 * num_filters)
        elif args.activation == 'softplus':
            with tf.variable_scope('decoder'):
                tensor = tf.nn.softplus(tf.layers.dense(tensor, args.dim1))
                if args.ac2 == 'True':
                    tensor = tf.nn.softplus(tf.layers.dense(tensor, 4 * 4 * num_filters))
                else:
                    tensor = tf.layers.dense(tensor, 4 * 4 * num_filters)
        elif args.activation == 'None':
            with tf.variable_scope('decoder'):
                tensor = tf.layers.dense(tensor, 4 * 4 * num_filters)
        with tf.variable_scope('reshape'):
            # dense layer
            tensor = tf.reshape(tensor, [-1, 4, 4, num_filters])

        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=tfc.GDN(inverse=True))
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                num_filters, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=tfc.GDN(inverse=True))
            tensor = layer(tensor)

        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                num_filters // 2, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=tfc.GDN(inverse=True))
            tensor = layer(tensor)

        with tf.variable_scope("layer_3"):
            layer = tfc.SignalConv2D(
                3, (9, 9), corr=False, strides_up=2, padding="same_zeros",
                use_bias=True, activation=None)
            tensor = layer(tensor)

        return tensor

def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def train():
    if not os.path.exists(args.checkpoint_dir):
        # shutil.rmtree(args.checkpoint_dir)
        os.makedirs(args.checkpoint_dir)
    log_name = os.path.join(args.checkpoint_dir, 'params.log')
    if os.path.exists(log_name):
        print('remove file:%s' % log_name)
        os.remove(log_name)
    params = open(log_name, 'w')
    for arg in vars(args):
        str_ = '%s: %s.\n' % (arg, getattr(args, arg))
        print(str_)
        params.write(str_)
    params.close()
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf Graph input (only pictures)
    if args.data_set.lower() == 'celeba':
        data_glob = imgs_path = args.img_path + '/*.png'
        print(imgs_path)

    ip_train = inputpipeline.InputPipeline(
        inputpipeline.get_dataset(data_glob),
        args.patch_size, batch_size=args.batch_size,
        shuffle=True,
        num_preprocess_threads=6,
        num_crops_per_img=6)
    X = ip_train.get_batch()

    # Construct model
    #encoder_op = analysis_transform(X, 64)
    encoder_op, mean, var = analysis_transform(X, 64)
    if args.split == 'None':
        X_pred = synthesis_transform(encoder_op, 64)
    else:
        X_pred = synthesis_transform(mean, 64)
        X_pred2 = synthesis_transform(encoder_op, 64)

    # Define loss and optimizer, minimize the squared error
    mse_loss = tf.reduce_mean(tf.squared_difference(255 * X, 255 * X_pred))
    msssim_loss = ms_ssim.MultiScaleSSIM(X * 255, X_pred * 255, data_format='NHWC')

    if args.loss1 =="mse":
        # mse loss
        d1 = tf.reduce_mean(tf.squared_difference( X, X_pred))
    elif args.loss1 == 'ssim':
        d1 = tf.reduce_mean(ssim_matrix.ssim(X * 255, (X - X_pred) * 255, X_pred,  max_val=255, mode='train',compensation=1))
    else:
        print('error invalid loss1')
        return -1

    if args.split != 'None':
        if args.loss2 =="mse":
            # mse loss
            d2 = tf.reduce_mean(tf.squared_difference(X_pred, X_pred2))
        elif args.loss2 == 'ssim':
            d2 =  tf.reduce_mean(ssim_matrix.ssim(X_pred * 255, (X_pred - X_pred2) * 255, X_pred2,  max_val=255, mode='train',compensation=1))

    # KL loss
    kl_div_loss = 1 + var - tf.square(mean) - tf.exp(var)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    kl_div_loss = tf.reduce_mean(kl_div_loss)

    # total loss
    if args.split != 'None':
        train_loss = args.lambda1 * d1 + args.lambda2 * d2 + kl_div_loss
    else:
        train_loss = args.lambda1 * d1 + kl_div_loss

    step = tf.train.create_global_step()


    if args.finetune != 'None':
        learning_rate = 0.00001
    else:
        learning_rate = 0.0001
    main_lr = tf.train.AdamOptimizer(learning_rate)
    optimizer = main_lr.minimize(train_loss, global_step=step)
    tf.summary.scalar("loss", train_loss)
    #tf.summary.scalar("bpp", bpp)
    tf.summary.scalar("mse", mse_loss)
    logged_tensors = [
        tf.identity(train_loss, name="train_loss"),
    #    tf.identity(bpp, name="train_bpp"),
        tf.identity(msssim_loss, name="ms-ssim")
    ]

    tf.summary.image("original", quantize_image(X))
    tf.summary.image("reconstruction", quantize_image(X_pred))

    hooks = [
        tf.train.StopAtStepHook(last_step=args.num_steps),
        tf.train.NanTensorHook(train_loss),
        tf.train.LoggingTensorHook(logged_tensors, every_n_secs=60),
        tf.train.SummarySaverHook(save_steps=args.save_steps, summary_op=tf.summary.merge_all()),
        tf.train.CheckpointSaverHook(save_steps=args.save_steps, checkpoint_dir=args.checkpoint_dir)
    ]

    X_rec = tf.clip_by_value(X_pred, 0, 1)
    X_rec = tf.round(X_rec * 255)
    X_rec = tf.cast(X_rec, tf.uint8)
    X_ori = tf.clip_by_value(X, 0, 1)
    X_ori = tf.round(X_ori * 255)
    X_ori = tf.cast(X_ori, tf.uint8)

    if args.finetune != 'None':
        init_fn_ae = tf.contrib.framework.assign_from_checkpoint_fn(args.finetune,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    train_count = 0

    parameter = 'VAE_%s' % (args.loss2)

    with tf.train.MonitoredTrainingSession(
            hooks=hooks) as sess:
        if args.finetune != 'None':
            init_fn_ae(sess)
            print('load from %s'%(args.finetune))
        while not sess.should_stop():
            if args.split != 'None':
                _, train_loss_ , d1_, d2_, kl_div_loss_, rec_img, X_ori_ = sess.run(
                    [optimizer, train_loss, d1, d2, kl_div_loss,  X_rec, X_ori])
                if (train_count + 1) % args.display_steps == 0:
                    f_log = open('%s/log.csv' % (args.checkpoint_dir), 'a')
                    f_log.write('%d,loss,%f, kl,%f, d1,%f, d2,%f\n' % (train_count + 1, train_loss_, kl_div_loss_, d1_, d2_))
                    print('%d,loss,%f, kl,%f, d1,%f, d2,%f\n' % (train_count + 1, train_loss_, kl_div_loss_, d1_, d2_))
                    f_log.close()
            else:
                _, train_loss_ , d1_, kl_div_loss_, rec_img, X_ori_ = sess.run(
                    [optimizer, train_loss, d1, kl_div_loss,  X_rec, X_ori])
                if (train_count + 1) % args.display_steps == 0:
                    f_log = open('%s/log.csv' % (args.checkpoint_dir), 'a')
                    f_log.write('%d,loss,%f, kl,%f, d1,%f\n' % (train_count + 1, train_loss_, kl_div_loss_, d1_))
                    print('%d,loss,%f, kl,%f, d1,%f\n' % (train_count + 1, train_loss_, kl_div_loss_, d1_))
                    f_log.close()

            if (train_count + 1) % args.save_steps == 0:

                num = math.floor(math.sqrt(rec_img.shape[0]))
                show_img = np.zeros([num * args.patch_size, num * args.patch_size, 3])
                ori_img = np.zeros([num * args.patch_size, num * args.patch_size, 3])
                for i in range(num):
                    for j in range(num):
                        show_img[i * args.patch_size:(i + 1) * args.patch_size,
                        j * args.patch_size:(j + 1) * args.patch_size, :] = rec_img[num * i + j, :, :, :]
                        ori_img[i * args.patch_size:(i + 1) * args.patch_size,
                        j * args.patch_size:(j + 1) * args.patch_size, :] = X_ori_[num * i + j, :, :, :]
                save_name = os.path.join(args.checkpoint_dir, 'rec_%s_%s.png' % (parameter, train_count + 1))
                scipy.misc.imsave(save_name, show_img)
                psnr_ = Psnr(ori_img, show_img)
                msssim_ = msssim(ori_img, show_img)
                # print('FOR calculation %s_%s_%s_%s_la1%s_la2%s_%s'%(
                # args.activation, args.dim1, args.dim2, args.z, args.lambda1, args.lambda2, train_count))
                print("PSNR (dB), %.2f,Multiscale SSIM, %.4f,Multiscale SSIM (dB), %.2f" % (
                    psnr_, msssim_, -10 * np.log10(1 - msssim_)))
                f_log_ssim = open('%s/log_ssim_%s.csv' % (args.checkpoint_dir, parameter), 'a')
                f_log_ssim.write('%s,%d,PSNR (dB), %.2f,Multiscale SSIM, %.4f,Multiscale SSIM (dB), %.2f\n' % (
                parameter, train_count + 1,
                psnr_, msssim_, -10 * np.log10(1 - msssim_)
                ))
                f_log_ssim.close()
            train_count += 1

def read_img(n):
    import random
    if args.data_set.lower() == 'celeba':
        images_path = args.img_path + '/*.png'

        images = glob.glob(images_path)
        images = sorted(images)
        # print(imgs.shape)
    imgs = np.zeros([n * n, args.patch_size, args.patch_size, 3])
    show_img = np.zeros([n * args.patch_size, n * args.patch_size, 3])

    for i in range(n * n):
        img_p = images[i]
        img = scipy.misc.imread(img_p).astype(np.float)
        h, w = img.shape[:2]
        if h > w:
            j = (h - w) // 2
            temp = scipy.misc.imresize(img[j:h - j, :, :], [args.patch_size, args.patch_size])
        else:
            j = (w - h) // 2
            temp = scipy.misc.imresize(img[:, j:w - j, :], [args.patch_size, args.patch_size])

        imgs[i, :, :, :] = temp
    for i in range(n):
        for j in range(n):
            show_img[i * args.patch_size:(i + 1) * args.patch_size, j * args.patch_size:(j + 1) * args.patch_size,
            :] = imgs[n * i + j, :, :, :]
    save_name = os.path.join(args.checkpoint_dir, 'clic_rdae_ori.png')
    scipy.misc.imsave(save_name, show_img)

    return imgs.astype(np.float), show_img

def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image

def plot_analysis():
    cdim = args.cdim
    preprocess_threads = 6
    # read_img
    train_path = args.img_path + '/*.png'
    train_files = glob.glob(train_path)
    if not train_files:
        raise RuntimeError(
            "No training images found with glob '{}'.".format(train_path))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files))  # .repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patch_size, args.patch_size, 3)))
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.prefetch(32)
    x = train_dataset.make_one_shot_iterator().get_next()

    # Construct model
    encoder_op, mean, var = analysis_transform(x, 64)
    x_pred = synthesis_transform(encoder_op,64)

    # Bring both images back to 0..255 range.
    x_pred = tf.clip_by_value(x_pred, 0, 1)
    x_pred = tf.round(x_pred * 255)
    x_pred = tf.cast(x_pred, tf.uint8)
    #end construct model

    #images_path = '../../data/CelebA/img_align_celeba_png/*.png'
    images_path = args.img_path + '/*.png'
    images = glob.glob(images_path)
    images = sorted(images)
    #temp = scipy.misc.imresize(img[:, j:-j, :], [args.patch_size, args.patch_size])

    batch_num = math.floor(len(images) / args.batch_size)
    print(len(images), batch_num)

    num = int(math.floor(math.sqrt(args.batch_size)))
    show_img = np.zeros([num * args.patch_size, num * args.patch_size, 3])

    parameter = 'VAE_%s' % (args.loss2)

    n = 0
    with tf.Session() as sess:
        # restore model
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        #latest = os.path.join(args.checkpoint_dir, 'model.ckpt-%s' % (args.num_steps))
        tf.train.Saver().restore(sess, save_path=latest)

        try:
            while True:
                #if n > 100:
                #    break
                #rec_img, z, mean_, var_ = sess.run([x_pred, encoder_op, mean, var], feed_dict={inputs: x})
                rec_img, z, mean_, var_ = sess.run([x_pred, encoder_op, mean, var])
                if n == 0:
                    zs = z
                    means_ = mean_
                    vars_ = 1/(np.power( np.exp(var_ * 0.5),2))
                else:
                    zs = np.vstack((zs, z))
                    means_ = np.vstack((means_, mean_))
                    vars_ = np.vstack((vars_, 1/(np.power( np.exp(var_ * 0.5),2))))
                n += 1
        except tf.errors.OutOfRangeError:\
            print('end!')

        print('zs', zs.shape)
        stds = np.squeeze(np.std(zs, axis=0))
        means = np.squeeze(np.mean(zs, axis=0))

        print('stds', stds.shape, 'means', means.shape)

        std_sorted = np.sort(stds)[::-1]
        std_index = np.argsort(stds)[::-1]

        var_sorted =np.power(std_sorted, 2)

        df = var_sorted.cumsum() / var_sorted.sum()

        x_1 = np.arange(0, var_sorted.shape[0], 1)

        fig, ax1 = plt.subplots()
        ax1.bar(x_1, var_sorted)
        ax1.set_xlabel('z(sorted by variance(descending order))')
        ax1.set_ylabel('variance of z')
        fig_name = os.path.join(args.checkpoint_dir, 'variance_df_%s.png' % (parameter))
        plt.savefig(fig_name)

        std_name = os.path.join(args.checkpoint_dir, 'std_df_%s.npy' % (parameter))
        np.save(std_name, stds)

        std_iname = os.path.join(args.checkpoint_dir, 'std_index_%s.npy' % (parameter))
        np.save(std_iname, std_index)

        mean_name = os.path.join(args.checkpoint_dir, 'mean_%s.npy' % (parameter))
        np.save(mean_name, means)

def sample_vw(n_batch, n_dim):
    import random
    v = []
    w = []
    for b in range(n_batch):
        v_init = np.zeros([n_dim])
        v_init[0] = 1
        #print(v_init)
        alpha = np.random.rand(n_dim) * 2 * np.pi
        # w_init = (np.random.rand(20) - 0.5 ) * 2 * 2* np.pi
        r = 1
        # x = []
        w_init = []
        #print(alpha[0:1])
        sin_alpha = np.sin(alpha)
        cos_alpha = np.cos(alpha)
        cos_alpha[0] = (np.random.rand(1) - 0.5) * 2
        sin_alpha[0] = np.sin(np.arccos(cos_alpha[0]))
        # print(cos_alpha)

        for i in range(alpha.shape[0]):
            if i == alpha.shape[0]:
                w_init.append(r * np.prod(sin_alpha))
            else:
                w_init.append(r * np.prod(sin_alpha[0:i]) * cos_alpha[i])

        w_init = np.array(w_init)
        gm = np.random.rand(1) * 2 * np.pi

        x = -(cos_alpha[0] / sin_alpha[0]) * v_init + (1.0 / sin_alpha[0]) * w_init
        y = v_init

        v_dash = -np.sin(gm) * x + np.cos(gm) * y
        w_dash = (np.cos(gm) * sin_alpha[0] - np.sin(gm) * cos_alpha[0]) * x + (np.sin(gm) * sin_alpha[0] + np.cos(gm) * cos_alpha[0]) * y

        v.append(v_dash)
        w.append(w_dash)

    return np.array(v), np.array(w)

def sample_image_v2():
    sample_num = 9
    cdim = args.cdim

    decoder_inputs = tf.placeholder(tf.float32, [sample_num * sample_num, args.z])
    inputs = tf.placeholder(tf.float32, [sample_num * sample_num, args.patch_size, args.patch_size, cdim])

    # Construct model
    encoder_op, mean_op, var_op = analysis_transform(inputs, 64)

    #x_pred = synthesis_transform(y_hat, 64)
    x_pred = synthesis_transform(encoder_op,64)

    z_p = tf.random_normal(shape=(sample_num*sample_num, args.z), mean=0.0, stddev=1.0)
    x_pred_s = synthesis_transform(z_p, 64)

    # Bring both images back to 0..255 range.
    x_pred_r = tf.clip_by_value(x_pred, 0, 1)
    x_pred_r = tf.round(x_pred_r * 255)
    x_pred_r = tf.cast(x_pred_r, tf.uint8)

    x_pred_s_r = tf.clip_by_value(x_pred_s, 0, 1)
    x_pred_s_r = tf.round(x_pred_s_r * 255)
    x_pred_s_r = tf.cast(x_pred_s_r, tf.uint8)

    x_pred_2 = synthesis_transform(decoder_inputs, 64)
    x_pred_2_r = tf.clip_by_value(x_pred_2, 0, 1)
    x_pred_2_r = tf.round(x_pred_2_r * 255)
    x_pred_2_r = tf.cast(x_pred_2_r, tf.uint8)

    # metric
    ###add
    ssim_loss_gt = tf.image.ssim(inputs * 255, x_pred * 255, max_val=255)
    mse2_loss_gt = tf.reduce_mean(tf.squared_difference(inputs, x_pred), reduction_indices=[1,2,3])

    decoder_inputs_loss1 = tf.placeholder(tf.float32, [sample_num * sample_num, args.z])
    decoder_inputs_loss2 = tf.placeholder(tf.float32, [sample_num * sample_num, args.z])
    decoder_inputs_loss3 = tf.placeholder(tf.float32, [sample_num * sample_num, args.z])
    decoder_inputs_loss4 = tf.placeholder(tf.float32, [sample_num * sample_num, args.z])
    decoder_inputs_loss5 = tf.placeholder(tf.float32, [sample_num * sample_num, args.z])
    #y = tf.reshape(encoder_op, [-1, 1, 1, z_num])

    x_pred_loss1 = synthesis_transform(decoder_inputs_loss1, 64)
    x_pred_loss2 = synthesis_transform(decoder_inputs_loss2, 64)
    x_pred_loss3 = synthesis_transform(decoder_inputs_loss3, 64)
    x_pred_loss4 = synthesis_transform(decoder_inputs_loss4, 64)
    x_pred_loss5 = synthesis_transform(decoder_inputs_loss5, 64)

    # metric

    if args.loss1 == "mse":
        loss_2_gt = (1 - mse2_loss_gt)
    elif args.loss1 == "ssim":
        #dammy
        loss_2_gt = (1 - ssim_loss_gt)

    loss_2_d = ssim_matrix.ssim(255 * x_pred_loss1, 255 * (x_pred_loss1 - x_pred_loss2),
                              255 * (x_pred_loss1 - x_pred_loss3), max_val=255, compensation=1)

    # define_input
    # images_path = '../../data/CelebA/img_align_celeba_png/*.png'
    images_path = args.img_path + '/*.png'
    images = glob.glob(images_path)
    images = sorted(images)
    #temp = scipy.misc.imresize(img[:, j:-j, :], [args.patch_size, args.patch_size])

    parameter = 'VAE_%s' % (args.loss2)

    std_name = os.path.join(args.checkpoint_dir, 'std_df_%s.npy' % (parameter))
    std = np.load(std_name)

    mean_name = os.path.join(args.checkpoint_dir, 'mean_%s.npy' % (parameter))
    mean = np.load(mean_name)

    std_iname = os.path.join(args.checkpoint_dir, 'std_index_%s.npy' % (parameter))
    std_index = np.load(std_iname)

    sample_num = 9
    sample_cen = int((sample_num - 1)/2)

    sh_ind=[0,1,2,20,21,22,200,201,202]
    samples_t = np.zeros([sample_num , sample_num, args.z])
    samples_h = np.zeros([sample_num, sample_num, args.z])
    std_ranges = []
    std_ranges_h = []

    for i in range(sample_num):
        std_ranges.append(std[std_index[sh_ind[i]]] / ((sample_num - 1) / 2) * 2)
        std_ranges_h.append(std[std_index[i]] / ((sample_num - 1) / 2) * 2)


    show_img_sample_t = np.zeros([sample_num * args.patch_size, sample_num * args.patch_size, 3])
    show_img_sample_h = np.zeros([sample_num * args.patch_size, sample_num * args.patch_size, 3])

    for i in range(sample_num):
        for j in range(sample_num):
            samples_t[i, j] = mean
            samples_t[i, j][std_index[sh_ind[i]]] += (std_ranges[i] * (j - sample_cen))

            samples_h[i, j] = mean
            samples_h[i, j][std_index[i]] += (std_ranges_h[i]*(j-sample_cen))

    samples_t =  samples_t.reshape((-1, args.z))
    samples_h =  samples_h.reshape((-1, args.z))

    num = 9
    x, ori_img = read_img(num)
    x = x / 255.

    with tf.Session() as sess:
        # restore model
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        #latest = os.path.join(args.checkpoint_dir, 'model.ckpt-%s' % (args.num_steps))
        tf.train.Saver().restore(sess, save_path=latest)
        encoder_op_ , x_pred_ = sess.run([encoder_op, x_pred], feed_dict={inputs: x})
        sample_rec_t = sess.run(x_pred_2_r, feed_dict={decoder_inputs: samples_t})
        sample_rec_h = sess.run(x_pred_2_r, feed_dict={decoder_inputs: samples_h})

        for i in range(num):
            for j in range(num):
                show_img_sample_t[i * args.patch_size:(i + 1) * args.patch_size,
                j * args.patch_size:(j + 1) * args.patch_size, :] = sample_rec_t[num * i + j, :, :, :]
                show_img_sample_h[i * args.patch_size:(i + 1) * args.patch_size,
                j * args.patch_size:(j + 1) * args.patch_size, :] = sample_rec_h[num * i + j, :, :, :]

        save_name = os.path.join(args.checkpoint_dir, '%s_sample_traverse.png' % (parameter))
        scipy.misc.imsave(save_name, show_img_sample_t)

        save_name = os.path.join(args.checkpoint_dir, '%s_traverse_top9.png' % (parameter))
        scipy.misc.imsave(save_name, show_img_sample_h)

        dist_num = 20
        d_list_c_all = []
        d_num=50
        z_mtrs = np.arange(0)
        x_mtrs = np.arange(0)
        dists = np.arange(0)
        delta = args.delta
        d_eps = np.arange(0)
        d_eps_sigma = np.arange(0)

        epsln = np.ones((sample_num * sample_num, args.z)) * args.delta

        for n in range(dist_num):
            imgs = np.zeros([sample_num * sample_num, args.patch_size, args.patch_size, 3])
            for i in range(num * num):
                # print(i)
                img_p = images[n * sample_num * sample_num + i]
                img = scipy.misc.imread(img_p).astype(np.float)
                imgs[i, :, :, :] = img
            x = imgs.astype(np.float) / 255.

            encoder_op_, mean_op_, x_pred_, loss_gt, var_op_ = sess.run([encoder_op, mean_op, x_pred, loss_2_gt, var_op], feed_dict={inputs: x})

            dec_ip = mean_op_.copy()  # .reshape(-1, args.z)
            dec_ip_d = mean_op_.copy()  # .reshape(-1, args.z)
            dec_ip_d_2 = mean_op_.copy()  # .reshape(-1, args.z)
            dec_ip_d_e = mean_op_.copy()  # .reshape(-1, args.z)
            dec_ip_d_e_sigma = mean_op_.copy()  # .reshape(-1, args.z)

            dec_ip_d_e = dec_ip_d_e + epsln
            dec_ip_d_e_sigma = dec_ip_d_e_sigma + epsln *  np.exp(var_op_/2)

            dv, dw = sample_vw(sample_num * sample_num, args.z)
            dv = delta * dv
            dw = delta * dw

            dec_ip_d = dec_ip_d + dv
            dec_ip_d_2 = dec_ip_d_2 + dw

            x_hat, v_hat, w_hat, gv_e, gv_e_sigma, dist_s = sess.run([x_pred_loss1, x_pred_loss2, x_pred_loss3, x_pred_loss4, x_pred_loss5, loss_2_d],
                                                   feed_dict={decoder_inputs_loss1: dec_ip,
                                                              decoder_inputs_loss2: dec_ip_d,
                                                              decoder_inputs_loss3: dec_ip_d_2,
                                                              decoder_inputs_loss4: dec_ip_d_e,
                                                              decoder_inputs_loss5: dec_ip_d_e_sigma,
                                                              inputs: x})

            v_hat = np.reshape(v_hat - x_hat, [sample_num * sample_num, -1])
            w_hat = np.reshape(w_hat - x_hat, [sample_num * sample_num, -1])

            d_ep = np.reshape(gv_e - x_hat, [sample_num * sample_num, -1])
            d_ep_sigma = np.reshape(gv_e_sigma - x_hat, [sample_num * sample_num, -1])

            z_mtr = np.sum(dv * dw, axis=1)
            z_mtrs = np.append(z_mtrs, z_mtr)
            #print(z_mtrs.shape)

            x_mtr = np.sum(v_hat * w_hat, axis=1)
            x_mtrs = np.append(x_mtrs, x_mtr)

            #print(x_mtrs.shape)

            d_eps = np.append( d_eps, np.sum(d_ep * d_ep, axis=1) )
            d_eps_sigma = np.append(d_eps_sigma, np.sum(d_ep_sigma * d_ep_sigma, axis=1))

            dists = np.append(dists, dist_s)
            #print(dists.shape)

    clf = LinearRegression()
    clf.fit(z_mtrs.reshape(-1,1), x_mtrs)
    prd = clf.predict(z_mtrs.reshape(-1,1))
    print('prd',prd.shape)
    var = np.power((x_mtrs-prd),2)
    var = np.sqrt(np.mean(var))
    print('var', var)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    xmin = np.min(z_mtrs)
    xmax = np.max(z_mtrs)

    ymin = np.min(prd) - 2 * var
    ymax = np.max(prd) + 2 * var
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.scatter(z_mtrs, x_mtrs,  s=2)
    fig.text(0.2, 0.8, 'r=%.4f' % (np.corrcoef(z_mtrs, x_mtrs)[0][1]))
    fig.tight_layout()
    fig_pass = os.path.join(args.checkpoint_dir, '%s_isometoric_mse_%s.png' % (parameter, args.delta))
    plt.savefig(fig_pass)

    clf = LinearRegression()
    clf.fit(z_mtrs.reshape(-1, 1), dists)
    prd = clf.predict(z_mtrs.reshape(-1, 1))
    var = np.power((dists - prd), 2)
    var = np.sqrt(np.mean(var))
    fig = plt.figure()
    # for d in range(8):
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(z_mtrs, dists, s=2)
    xmin = np.min(z_mtrs)
    xmax = np.max(z_mtrs)

    ymin = np.min(prd) - 2 * var
    ymax = np.max(prd) + 2 * var
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.text(0.2, 0.8, 'r=%.4f' % (np.corrcoef(z_mtrs, dists)[0][1]))
    fig_pass = os.path.join(args.checkpoint_dir, '%s_isometoric_ssim_%s.png' % (parameter, args.delta))
    plt.savefig(fig_pass)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "command", choices=["train", 'test', 'analy','sample', 'sample_1', 'analy_q','inter'],
        help="What to do: 'train' loads training data and trains (or continues "
             "to train) a new model. 'test' load trained model and test.")
    parser.add_argument(
        "input", nargs="?",
        help="Input filename.")
    parser.add_argument(
        "output", nargs="?",
        help="Output filename.")
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training.")
    parser.add_argument(
        "--patch_size", type=int, default=64,
        help="Patch size for training.")
    parser.add_argument(
        "--data_set", default='CelebA',
        help="Batch size for training.")
    parser.add_argument(
        "--checkpoint_dir", default="sanity",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--img_path", default="../../data/CelebA/centered_celeba_64_10per/",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--num_steps", type=int, default=500000,
        help="Train up to this number of steps.")
    parser.add_argument(
        "--save_steps", type=int, default=100000,
        help="Train up to this number of steps.")
    parser.add_argument(
        "--display_steps", type=int, default=100,
        help="save loss for plot every this number of steps.")
    parser.add_argument(
        "--lambda1", type=float, default=1000000,
        help="Lambda for distortion tradeoff.")
    parser.add_argument(
        "--lambda2", type=float, default=1000000,
        help="Lambda for rate tradeoff.")
    parser.add_argument(
        "--loss1", type=str, default='mse',
        help="mse, logmse, ssim, logssim")
    parser.add_argument(
        "--loss2", type=str, default='mse',
        help=" mse or ssim")
    parser.add_argument(
        "--z", type=int, default=256,
        help="bottleneck number.")
    parser.add_argument(
        "--cdim", type=int, default=3,
        help="channel.")
    parser.add_argument(
        "--delta", type=float, default=0.01,
        help="bottleneck number.")
    parser.add_argument(
        "--dim1", type=int, default=8192,
        help="AE layer1.")
    parser.add_argument(
        "--activation", default="softplus")
    parser.add_argument(
        "--ac2", default='Treu')
    parser.add_argument(
        "--finetune", default="None")
    parser.add_argument(
        "--split", default="False")

    parser.add_argument('-gpu', '--gpu_id',
                        help='GPU device id to use [0]', default=0, type=int)

    args = parser.parse_args()

    # cpu mode
    if args.gpu_id < 0:
        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    z_num = args.z

    if args.command == 'train':
        train()
        # python ae.py train --checkpoint_dir ae_718 --lambda 10 -gpu 0
    elif args.command == 'analy':
        plot_analysis()
        # if args.input is None or a
    elif args.command == 'sample':
        sample_image_v2()