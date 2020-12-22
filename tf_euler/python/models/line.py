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

import tensorflow as tf

from tf_euler.python import encoders
from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base


class LINE(base.UnsupervisedModel):
  """
  Implementation of LINE model.

  *args和**kwargs用在函数定义的时候,能让函数接受可选参数,参数args和kwargs只是一个命名约定,实际起作用的语法是星号(*)和双星号(**).
  *args表示函数接收可变长度的非关键字参数列表作为函数的输入,args将收集额外的位置参数组成元组.
  **kwargs表示函数接收可变长度的关键字参数字典作为函数的输入,kwargs会收集额外的关键字参数来组成字典.

  Python中,__init__()方法是所谓的对象的"构造函数",负责在对象初始化时进行一系列的构建操作,_init__()方法必须接受至少一个参数即self,self是指向该对象本身的一个引用.
  _init__()方法是一种特殊的方法,被称为类的构造函数或初始化方法,当创建了这个类的实例时就会调用该方法.
  类的方法与普通的函数只有一个特别的区别,即它们必须有一个额外的第一个参数名称,按照惯例它的名称是self,也就是说类的方法的第一个参数通常都是self.
  self代表类的实例,self在定义类的方法时是必须有的,虽然在调用时不必传入相应的参数,注意self并不是python关键字.
  """

  def __init__(self, node_type, edge_type, max_id, dim, order=1,
               feature_idx=-1, feature_dim=0, use_id=True,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, combiner='add',
               *args, **kwargs):  # *args: <type 'tuple'>: (), **kwargs: <type 'dict'>: {'xent_loss': True, 'num_negs': 5}
    super(LINE, self).__init__(node_type, edge_type, max_id, *args, **kwargs)  # super(类,self).__init__(): 继承父类的构造方法,同样可以使用super().其他方法名,去继承其他方法,这里主要是初始化了line模型的基础参数

    if order == 1:
      order = 'first'
    if order == 2:
      order = 'second'

    self._target_encoder = encoders.ShallowEncoder(
        dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id if use_id else -1,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        combiner=combiner)
    if order == 'first':
      self._context_encoder = self._target_encoder
    elif order == 'second':
      self._context_encoder = encoders.ShallowEncoder(
          dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
          max_id=max_id if use_id else -1,
          sparse_feature_idx=sparse_feature_idx,
          sparse_feature_max_id=sparse_feature_max_id,
          embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
          combiner=combiner)
    else:
      raise ValueError('LINE order must be one of 1, 2, "first", or "second"'
                       'got {}:'.format(order))

  def target_encoder(self, inputs):
    return self._target_encoder(inputs)

  def context_encoder(self, inputs):
    return self._context_encoder(inputs)
