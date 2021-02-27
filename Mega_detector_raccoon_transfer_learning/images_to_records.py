# -*- coding: utf-8 -*-
"""
file: images_to_records.py

@author: Suhail.Alnahari

@description: 

@created: 2021-02-22T22:09:11.241Z-06:00

@last-modified: 2021-02-26T18:16:05.804Z-06:00
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

labels = {1: b'raccoon', 2:b'skunk',3:b'cat',4:b'human',5:b'fox',6:b'other'}
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
    scaledHeight = 500.0/1080
    scaledWidth = 888.0/1920
    filename = str.encode(image_data['file'])
    image_format = b'jpg'
    trainBool = True
    try:
        image_file = open(os.path.join(b'images_train',filename), 'rb')
        encoded_image = image_file.read()
    except:
        try:
            image_file = open(os.path.join(b'images_eval',filename), 'rb')
            encoded_image = image_file.read()
            trainBool = False
        except:
            print(filename)
            assert False, "file was not found in either images_train or images_eval"

    image_file.close()
    xmins = [i['bbox'][0] / width for i in detections]
    xmaxs = [i['bbox'][2] / width for i in detections]
    ymins = [i['bbox'][1] / height for i in detections]
    ymaxs = [i['bbox'][3] / height for i in detections]
    classIndex = [i['category']+1 for i in detections]
    classes_text = [labels[i] for i in classIndex]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(int(height*scaledHeight)),
        'image/width': dataset_util.int64_feature(int(width*scaledWidth)),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classIndex),
    }))
    return tf_example, trainBool

def main(_):
    with open('raccoon_volunteer_labels.json') as json_file:
        examples = json.load(json_file)
    trainIndex = 0
    evalIndex = 0
    for example in examples['images']:
        try:
            tf_example, trainBool = create_tf_example(example)
        except:
            continue
        if (trainBool):
            writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,'train-{0}-'.format(str(trainIndex).zfill(5))+'of-00001.tfrecord'))
            trainIndex += 1
        else:
            writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,'eval-{0}-'.format(str(evalIndex).zfill(5))+'of-00001.tfrecord'))
            evalIndex += 1
        writer.write(tf_example.SerializeToString())
        writer.close()

if __name__ == '__main__':
    tf.app.run()