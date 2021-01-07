import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from skimage import transform


# get all samples into memory
def load_sample(sample_dir, shuffle_flag):
    print("loading sample dataset ...")
    lfilenames = []
    labelsnames = []
    for dirpath, dirnames, filenames in os.walk(sample_dir):
        for filename in filenames:
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)
            labelsnames.append(dirpath.split('\\')[-1])

    # create label list
    lab = list(sorted(set(labelsnames)))
    labdict = dict(zip(lab, list(range(len(lab)))))
    labels = [labdict[i] for i in labelsnames]
    if shuffle_flag:
        return shuffle(
            np.asarray(lfilenames),
            np.asarray(labels)
        ), np.asarray(lab)
    else:
        return (np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


def _distorted_image(image, size, ch=1, shuffleflag=False, cropflag=False,
                     brightnessflag=False, contrastflag=False):
    # flip image left and right
    distorted_images = tf.image.random_flip_left_right(image)
    # random cropping
    if cropflag is True:
        distorted_images = tf.image.random_crop(distorted_images, [size[0], size[1], ch])

    # flip image up and down
    distorted_images = tf.image.random_flip_up_down(distorted_images)

    if brightnessflag is True:
        tf.image.random_brightness(distorted_images, max_delta=10)

    if contrastflag is True:
        distorted_images = tf.image.random_contrast(distorted_images, lower=0.2, upper=1.8)

    if shuffleflag is True:
        distorted_images = tf.random.shuffle(distorted_images)

    return distorted_images


def _norm_image(image, size, ch=1, flattenflag=False):
    image_decoded = image / 255.0
    if flattenflag is True:
        image_decoded = tf.reshape(image_decoded, [size[0] * size[1] * ch])
    return image_decoded


def _random_rotated30(image, label):
    def _rotated(image):
        shift_y, shift_x = np.array(image.shape.as_list[:2], np.float32)
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad((30)))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv, image.size = transform.SimilarityTransform(
            translation=[shift_x, shift_y]
        ), image.shape
        image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse())
        return image_rotated

    def _rotatedwrap():
        image_rotated = tf.py_function(_rotated, [image][tf.float64])
        return tf.cast(image_rotated, tf.float32)[0]

    a = tf.random.uniform([1], 0, 2, tf.int32)
    image_decoded = tf.cond(tf.equal(tf.constant(0), a[0]), lambda: image, _rotatedwrap)
    return image_decoded, label


def dataset(directory, size, batch_size, random_rotated=False):  #
    (filenames, labels), _ = load_sample(directory, shuffle_flag=False)

    def _parseone(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])
        image_decoded = _distorted_image(image_decoded, size)
        image_decoded = tf.image.resize(image_decoded, size)
        image_decoded = _norm_image(image_decoded, size)
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
        label = tf.cast(tf.reshape(label, []), dtype=tf.int32)
        return image_decoded, label

    datasets = tf.data.Dataset.from_tensor_slices(
        (filenames, labels)
    )
    datasets = datasets.map(_parseone)

    if random_rotated is True:
        datasets = dataset.map(_random_rotated30)

    datasets = datasets.batch(batch_size)
    return datasets


def showresult(subplot, title, thisimg):
    p = plt.subplot(subplot)
    p.axis('off')
    p.imshow(thisimg)
    p.set_title(title)


def showimg(index, label, img, ntop):
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range(ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()


if __name__ == "__main__":
    sample_dir = "samples/man_woman"
    size = [96,96]
    batchsize = 10
    datasets = dataset(sample_dir, size=size, batch_size=batchsize)
    datasets2 = dataset(sample_dir,size,batchsize)
    for step,value in enumerate(datasets):
        showimg(step,value[1].numpy(),np.asarray(value[0]*255,np.uint8),10)
