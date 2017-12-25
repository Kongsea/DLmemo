#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
from tensorflow.python.tools import inspect_checkpoint

# Download pretrained model VGG16 from
# http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
model_path = 'vgg_16.ckpt'

# print variables in the .ckpt file
# set all_tensors=True to print its content (numpy array)
# set all_tensors=False to print just its name and shape
inspect_checkpoint.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=False)

images = tf.placeholder(tf.float32, (None, 224, 224, 3))
with slim.arg_scope(vgg.vgg_arg_scope()):
  logits, _ = vgg.vgg_16(images, num_classes=10, is_training=True,
                         dropout_keep_prob=0.5, scope='s1_vgg_16')


def name_in_checkpoint(var):
  '''1.The generated variables starts with scope 's1_vgg_16'
     we need to cut the 's1_' to make it same as in the .ckpt file.
     2.The generated variables ends with ':0',
     So we need to eliminate it.'''
  return var.name.partition('_')[-1][:-2]


variables = tf.trainable_variables()
# eliminate variables in fc* layers
# convert the variable names to the same as in .ckpt files
variables_to_restore = {name_in_checkpoint(var): var for var in variables if 'fc' not in var.name}
restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
  restorer.restore(sess, model_path)
  print('Restored.')
