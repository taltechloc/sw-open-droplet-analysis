import numpy as np
import os
import glob
import tensorflow as tf

# Base paths
base_images_path = r'C:\faafri\images\\'
dataset_path = r'C:\faafri\score.txt'

IMAGE_SIZE = 224
RANDOM_SEED = 42

# Fetch image files
files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

train_image_paths = []
train_scores = []

print("Loading training set and val set")
with open(dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])
        
        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()
        
        file_path = base_images_path + str(id) + '.jpg'
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)

train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')

np.random.seed(RANDOM_SEED)
indices = np.random.permutation(len(train_image_paths))

split_ratio = 0.1
split_index = int(len(train_image_paths) * split_ratio)

# Create validation set first
val_image_paths = train_image_paths[indices[:split_index]]
val_scores = train_scores[indices[:split_index]]

# Then create training set
train_image_paths = train_image_paths[indices[split_index:]]
train_scores = train_scores[indices[split_index:]]

print('Train set size:', train_image_paths.shape, train_scores.shape)
print('Validation set size:', val_image_paths.shape, val_scores.shape)
print('Train and validation datasets ready!')

def parse_data(filename, scores):
    tf.random.set_seed(RANDOM_SEED)
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def parse_data_without_augmentation(filename, scores):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def train_generator(batchsize, shuffle=True):
    train_image_paths_str = tf.convert_to_tensor(train_image_paths, dtype=tf.string)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths_str, train_scores))
    train_dataset = train_dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=len(train_image_paths), seed=RANDOM_SEED)

    train_dataset = train_dataset.batch(batchsize).repeat()

    for batch in train_dataset:
        yield batch

def val_generator(batchsize):
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_scores))
    val_dataset = val_dataset.map(parse_data_without_augmentation)
    val_dataset = val_dataset.batch(batchsize)
    val_dataset = val_dataset.repeat()

    val_iterator = val_dataset.as_numpy_iterator()

    while True:
        yield next(val_iterator)

def features_generator(record_path, feature_size, batchsize, shuffle=True):
    def parse_single_record(serialized_example):
        example = tf.io.parse_single_example(
            serialized_example,
            features={
                'features': tf.io.FixedLenFeature([feature_size], tf.float32),
                'scores': tf.io.FixedLenFeature([10], tf.float32),
            })
        return example['features'], example['scores']

    train_dataset = tf.data.TFRecordDataset([record_path])
    train_dataset = train_dataset.map(parse_single_record, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batchsize)
    train_dataset = train_dataset.repeat()

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=100, seed=RANDOM_SEED)

    train_iterator = train_dataset.as_numpy_iterator()

    while True:
        yield next(train_iterator)