import tensorflow as tf
import pickle
import os,sys
import glob
import random
from os import path


def get_dataset(ds):
    """
    - imgnet_train / imgnet_test:
            Use records in constants.IMAGENET_TFRECORDS_ROOT
    - *.pkl:
            Use dataset with any paths, as created by PathsDataset.make_paths_pickle_file_from_image_glob
    - everything else is assumed to be a glob matching some image files
    """
    try:
        return PathsDataset.from_img_glob(ds)
    except ValueError as e:
        print(e)
    try:
        return PathsDataset.from_paths_pickle_file(ds)
    except ValueError as e:
        print(e)
    raise ValueError('Invalid dataset: {}'.format(ds))


# Each of the following Dataset classes implements
# - images_decoded, returning a (?, ?, 3) tensor of a decoded image
# - name, returning a description of the dataset
# - num_images, number of images in the dataset. Used to calculate number of iterations per epoch in training_helpers.py

class PathsDataset(object):
    def __init__(self, name, paths_tensor, num_images):
        self.name = name
        self.paths_tensor = paths_tensor
        self.num_images = num_images

    def images_decoded(self, num_epochs=None, shuffle=True):
        with tf.name_scope('images_decoded'):
            filename_queue = tf.train.string_input_producer(self.paths_tensor, num_epochs, shuffle)
            image_reader = tf.WholeFileReader()
            _, image_content = image_reader.read(filename_queue)
            im = tf.image.decode_image(image_content, channels=3)
            im.set_shape([None, None, 3])
            return im

    @staticmethod
    def from_img_glob(img_glob):
        num_imgs = len(glob.glob(img_glob))#len(img_glob)
        print('total images:',num_imgs)
        if num_imgs == 0:
            raise ValueError('glob not matching any files: {}; {}'.format(
                    img_glob, os.listdir(os.path.dirname(img_glob))))
        # make sure this is a valid scope name
        #name = 'glob_' + img_glob.replace('/', '_').replace('*', '_')
        name = 'glob_temp.png'
        return PathsDataset(name=name,
                            paths_tensor=tf.train.match_filenames_once(img_glob, name='GlobDataset'),
                            num_images=num_imgs)

    @staticmethod
    def from_paths_pickle_file(paths_pickle_file):
        if not paths_pickle_file.endswith('.pkl'):
            raise ValueError('Not a .pkl file: {}'.format(paths_pickle_file))
        assert os.path.exists(paths_pickle_file)
        with tf.name_scope('pickle_input'):
            base_dir = os.path.dirname(paths_pickle_file) + '/'
            tf.logging.info('Loading paths...')
            with open(paths_pickle_file, 'rb') as f:
                paths = pickle.load(f)
            assert os.path.exists(base_dir + paths[0])
            num_imgs = len(paths)
            tf.logging.info('Creating tf.constant...')
            paths = tf.constant(paths)
            tf.logging.info('Making producer...')
            return PathsDataset(name='pickle_{}'.format(paths_pickle_file),
                                paths_tensor=base_dir + paths,
                                num_images=num_imgs)

    @staticmethod
    def make_paths_pickle_file_from_image_glob(img_root_dir, paths_glob, shuffle):
        os.chdir(img_root_dir)
        paths_pickle_f = os.path.join(img_root_dir, 'paths.pkl')
        if os.path.exists(paths_pickle_f):
            print('{} exists, not re-creating...'.format(paths_pickle_f))
            return

        paths = glob.glob(paths_glob)
        assert len(paths) > 0, 'No images found matching {}/{}'.format(img_root_dir, paths_glob)
        if shuffle:
            random.shuffle(paths)
        else:
            paths = sorted(paths)
        with open(paths_pickle_f, 'wb') as f:
            pickle.dump(paths, f)


class InputPipeline(object):
    def __init__(self,
                 dataset,
                 crop_size,
                 batch_size=64,
                 num_preprocess_threads=4,
                 num_crops_per_img=8,
                 big_queues=True,
                 dtype_out=tf.float32,
                 shuffle=True):
        self.dataset = dataset
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.num_preprocess_threads = num_preprocess_threads
        self.num_crops_per_img = num_crops_per_img
        self.big_queues = big_queues
        self.dtype_out = dtype_out
        self.shuffle = shuffle

        self._batch = self._create_batch()

    def get_batch(self):
        return self._batch

    def _create_batch(self):
        seed = None if self.shuffle else 666
        with tf.name_scope('input_' + self.dataset.name):
            with tf.device('/cpu:0'):
                # HW3
                image_decoded = self.dataset.images_decoded(num_epochs=None, shuffle=self.shuffle)
                #image_decoded /= 255
                # list of (num_crops_per_frame, 3, H, W)
                images = [[_preprocess(image_decoded,
                                       crop_size=self.crop_size,
                                       num_crops_per_frame=self.num_crops_per_img,
                                       seed=seed)]
                          for _ in range(self.num_preprocess_threads)]

                # (batch_size, 3, H, W)
                images_batch = tf.train.shuffle_batch_join(
                    images,
                    seed=seed,
                    batch_size=self.batch_size,
                    enqueue_many=True,
                    capacity=1000 if self.big_queues else 2 * self.batch_size,
                    min_after_dequeue=800 if self.big_queues else self.batch_size)

            # cast on GPU
            images_batch = tf.cast(images_batch, self.dtype_out, name='cast')
            images_batch /= 255
        return images_batch


def _preprocess(frames, crop_size, num_crops_per_frame, seed=None):
    """
    :param frames: HW3
    :param crop_size: tuple width, height
    :param num_crops_per_frame:
    :return: R3HW, where R == num_crops_per_frame.
    """
    with tf.name_scope('preprocess'):
        crop_height = crop_width = crop_size
        frames = tf.stack([
            tf.random_crop(frames, [crop_height, crop_width, 3], seed=seed) #random_crop
            #tf.image.central_crop(frames,central_fraction=0.5).set_shape([crop_height, crop_width, 3])
            for _ in range(num_crops_per_frame)])  # RHW3)
        return frames

