# -*- coding: utf-8 -*-
"""
file: images_to_records.py

@author: Suhail.Alnahari

@description: 

@created: 2021-02-22T22:09:11.241Z-06:00

@last-modified: 2021-02-23T01:00:29.954Z-06:00
"""

# standard library
# 3rd party packages
# local source

import tensorflow as tf

from object_detection.utils import dataset_util
import json
import os

flags = tf.app.flags
flags.DEFINE_string('output_path', 'data', 'Path to output TFRecord')
FLAGS = flags.FLAGS

labels = {0: b'animal', 1:b'animal',2:b'animal',3:b'person',4:b'animal',5:b'animal'}
labelNumbers = {0: 1, 1:1,2:1,3:2,4:1,5:1}
def create_tf_example(image_data):
    """Creates a tf.Example proto from sample cat image.
    
    Args:
        image_data: json input

    Returns:
        example: The created tf.Example.
    """
    
    detections = image_data['detections']
    height = 1080
    width = 1920
    filename = str.encode(image_data['file'])
    image_format = b'jpg'
    encoded_image = open(os.path.join(b'images',filename), 'rb').read()
    xmins = [i['bbox'][0] / width for i in detections]
    xmaxs = [i['bbox'][2] / width for i in detections]
    ymins = [i['bbox'][1] / height for i in detections]
    ymaxs = [i['bbox'][3] / height for i in detections]
    classIndex = [i['category'] for i in detections]
    classes_text = [labels[i] for i in classIndex]
    classes = [labelNumbers[i] for i in classIndex]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    # TODO(user): Write code to read in your dataset to examples variable
    with open('raccoon_volunteer_labels.json') as json_file:
        examples = json.load(json_file)
    cutoff = 20000
    for index, example in enumerate(examples['images']):
        if (index < cutoff):
            writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,'train-{0}-'.format(str(index).zfill(5))+'of-00001.tfrecord'))
        else:
            writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,'eval-{0}-'.format(str(index - cutoff).zfill(5))+'of-00001.tfrecord'))
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
        writer.close()

if __name__ == '__main__':
    tf.app.run()