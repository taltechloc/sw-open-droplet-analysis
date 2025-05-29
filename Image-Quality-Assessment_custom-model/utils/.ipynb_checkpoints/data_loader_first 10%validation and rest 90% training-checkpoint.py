import numpy as np
import os
import glob

import tensorflow as tf
print("before loading data...")

# Base paths to images and dataset text file
base_images_path = r'C:\faafri\images\\'
ava_dataset_path = r'C:\faafri\AVA.txt'

IMAGE_SIZE = 224

# Fetch image files
files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

train_image_paths = []
train_scores = []

print("Loading training set and val set")
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])

        values = np.array(token[2:12], dtype='float32')
        #values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = base_images_path + str(id) + '.jpg'
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)

# Convert lists to numpy arrays
train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')

"""# Define split ratio (90% training, 10% validation)
split_ratio = 0.9
split_index = int(len(train_image_paths) * split_ratio)

# Split data into training and validation sets
val_image_paths = train_image_paths[split_index:]  # Last 10% for validation
val_scores = train_scores[split_index:]           # Last 10% scores for validation

train_image_paths = train_image_paths[:split_index]  # First 90% for training
train_scores = train_scores[:split_index]           # First 90% scores for training """

# Calculate the split index
split_index = int(len(train_image_paths) * 0.1)  # First 10% for validation

# Split data into training and validation sets
val_image_paths = train_image_paths[:split_index]  # First 10% for validation
val_scores = train_scores[:split_index]           # First 10% scores for validation

train_image_paths = train_image_paths[split_index:]  # Remaining 90% for training
train_scores = train_scores[split_index:]           # Remaining 90% scores for training

print('Train set size:', train_image_paths.shape, train_scores.shape)
print('Validation set size:', val_image_paths.shape, val_scores.shape)
print('Train and validation datasets ready!')

def parse_data(filename, scores):
    '''
    Loads the image file, and randomly applies crops and flips to each image.
    '''
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))  # Update this line
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def parse_data_without_augmentation(filename, scores):
    '''
    Loads the image file without any augmentation. Used for validation set.
    '''
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def train_generator(batchsize, shuffle=True):
    # Ensure train_image_paths is of type string
    train_image_paths_str = tf.convert_to_tensor(train_image_paths, dtype=tf.string)
    
    # Create a dataset with filenames as strings
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths_str, train_scores))
    train_dataset = train_dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=len(train_image_paths))

    train_dataset = train_dataset.batch(batchsize).repeat()

    for batch in train_dataset:
        yield batch

def val_generator(batchsize):
    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for validation set

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    # Create a dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_scores))
    val_dataset = val_dataset.map(parse_data_without_augmentation)
    val_dataset = val_dataset.batch(batchsize)
    val_dataset = val_dataset.repeat()

    # Create an iterator
    val_iterator = val_dataset.as_numpy_iterator()

    while True:
        yield next(val_iterator)

def features_generator(record_path, feature_size, batchsize, shuffle=True):
    '''
    Creates a python generator that loads pre-extracted features from a model
    and serves it to Keras for pre-training.

    Args:
        record_path: path to the TF Record file
        feature_size: the number of features in each record. Depends on the base model.
        batchsize: batchsize for training
        shuffle: whether to shuffle the records

    Returns:
        a batch of samples (X_features, y_scores)
    '''

    # Function to parse a single record
    def parse_single_record(serialized_example):
        # Parse a single record
        example = tf.io.parse_single_example(
            serialized_example,
            features={
                'features': tf.io.FixedLenFeature([feature_size], tf.float32),
                'scores': tf.io.FixedLenFeature([10], tf.float32),
            })

        features = example['features']
        scores = example['scores']
        return features, scores

    # Load the TF dataset
    train_dataset = tf.data.TFRecordDataset([record_path])
    train_dataset = train_dataset.map(parse_single_record, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(batchsize)
    train_dataset = train_dataset.repeat()

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=100)

    # Create a numpy iterator
    train_iterator = train_dataset.as_numpy_iterator()

    # Indefinitely extract batches
    while True:
        yield next(train_iterator)
