# -*- coding: utf-8 -*-

# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
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

import collections

import tensorflow as tf

from tensorflow.python.util import nest


_LAYER_UIDS = collections.defaultdict(lambda: 0)

def get_layer_uid(layer_name=''):
  _LAYER_UIDS[layer_name] += 1
  return _LAYER_UIDS[layer_name]


class Layer(object):
  """
  Layer class modeled after Keras (http://keras.io).
  """

  def __init__(self, name=None, **kwargs):  # 初始化元信息以及对象所包含的其他Layer
    self.built = False

    if name is None:
      layer_name = self.__class__.__name__.lower()  # 设置layer_name为类名的小写
      name = layer_name + '_' + str(get_layer_uid(layer_name))  # name: line_1/shallowencoder_1/dense_1/embedding_1

    self._name = name

  def build(self, input_shape):  # 初始化该对象自身会用到的模型权重,会在__call__()首次执行时被调用
    self.built = True

  def call(self, inputs):  # 将该对象应用到输入中,会在__call__()执行时调用
    return inputs

  def __call__(self, inputs):
    input_shapes = None
    if all(hasattr(x, 'shape') for x in nest.flatten(inputs)):  # nest.flatten(): 将嵌套结构压平,返回python的list
      input_shapes = nest.map_structure(lambda x: x.shape, inputs)  # map_structure(): 这里是对inputs里面的每个元素应用lambda表达式函数,x.shape获取x的维度信息

    with tf.variable_scope(self._name):
      if not self.built:
        self.build(input_shapes)  # input_shapes类型为TensorShape
      outputs = self.call(inputs)
      return outputs

  def compute_output_shape(self, input_shape):
    raise NotImplementedError()


class Dense(Layer):
  """
  Basic full-connected layer.
  """

  def __init__(self,
               dim,
               activation=None,
               use_bias=True,
               kernel_initializer=lambda: tf.uniform_unit_scaling_initializer(factor=0.36),
               bias_initializer=lambda: tf.constant_initializer(value=0.0002),
               **kwargs):
    super(Dense, self).__init__(**kwargs)
    self.dim = dim
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.kernel = tf.get_variable(
        'kernel',
        shape=[input_shape[-1].value, self.dim],
        initializer=self.kernel_initializer())
    if self.use_bias:
      self.bias = tf.get_variable(
          'bias',
          shape=[self.dim],
          initializer=self.bias_initializer())
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    rank = inputs.shape.ndims
    if rank > 2:
      outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
    else:
      outputs = tf.matmul(inputs, self.kernel)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation:
      outputs = self.activation(outputs)
    return outputs


class Embedding(Layer):
  """
  Id to dense vector embedding.
  """

  def __init__(self,
               max_id,
               dim,
               initializer=lambda: tf.truncated_normal_initializer(stddev=0.1),  # 从截断的正态分布中输出随机值,标准差为0.1.
               **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.max_id = max_id
    self.dim = dim
    self.initializer = initializer

  def build(self, input_shape):
    self.embeddings = tf.get_variable(
        'embeddings',
        shape=[self.max_id + 1, self.dim],
        initializer=self.initializer())
    self.built = True

  def call(self, inputs):
    shape = inputs.shape
    inputs = tf.reshape(inputs,[-1])
    output_shape = shape.concatenate(self.dim)
    output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
    return tf.reshape(tf.nn.embedding_lookup(self.embeddings, inputs),output_shape)  # tf.nn.embedding_lookup(params, ids, partition_strategy='mod', max_norm=None),根据ids中的id,寻找params中的第id行,组成一个tensor返回.


class SparseEmbedding(Embedding):
  """
  Sparse id to dense vector embedding.一般应用在id类的embedding,比如nodeid/featureid的embedding.
  Euler事先把每个节点id都初始化分配一个随机的embedding,SparseEmbedding就是用节点的id去参数矩阵上查询对应节点的embedding.
  """
  def __init__(
      self,
      max_id,
      dim,
      initializer=lambda: tf.truncated_normal_initializer(stddev=0.0002),
      combiner='sum',
      **kwargs):
    super(SparseEmbedding, self).__init__(
        max_id=max_id, dim=dim, initializer=initializer, **kwargs)
    self.combiner = combiner

  def call(self, inputs):
    return tf.nn.embedding_lookup_sparse(self.embeddings, inputs, None,
                                         combiner=self.combiner)
