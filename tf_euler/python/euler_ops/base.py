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

import ctypes
import os

import tensorflow as tf
import euler

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
_LIB_CLIENT_PATH = os.path.join(_LIB_DIR, 'libclient.so')
_LIB_PATH = os.path.join(_LIB_DIR, 'libtf_euler.so')

tf.load_op_library(_LIB_CLIENT_PATH)  # tensorflow的python接口使用tf.load_op_library()函数来加载动态library,并将op注册到tensorflow框架上
_LIB_OP = tf.load_op_library(_LIB_PATH)  # load_op_library()返回一个python module
_LIB = ctypes.CDLL(_LIB_PATH)


def initialize_graph(config):
  """
  Initialize the Euler graph driver used in Tensorflow.

  Args:
    config: str or dict of Euler graph driver configuration.

  Return:
    A boolean indicate whether the graph driver is initialized successfully.

  Raises:
    TypeError: if config is neither str nor dict.
  """
  if isinstance(config, dict):
    config = ';'.join(
        '{}={}'.format(key, value) for key, value in config.items())  # config是一个字符串: "directory=data_dir;load_type=compact;mode=Local"
  if not isinstance(config, str):
    raise TypeError('Expect str or dict for graph config, '
                    'got {}.'.format(type(config).__name__))
  return _LIB.CreateGraph(config)


"""
加载一份完整的图并独立使用,返回bool,表示图是否初始化成功
directory: 图数据路径,目前embedded模式仅支持unix文件系统
graph_type: graph类型,compact/fast,默认compact
"""
def initialize_embedded_graph(directory, graph_type='compact'):
  return initialize_graph({'mode': 'Local',
                           'directory': directory,
                           'load_type': graph_type})


# TODO: Consider lower the concept of shared graph to euler client.
"""
在不同的worker之间自动的进行图数据的切分和共享,返回bool,表示图是否初始化成功
directory: 图数据路径,目前shared模式仅支持HDFS
zk_addr: Zookeeper地址,ip:port
zk_path: Zookeeper根节点,用于协调各个shard
shard_idx: shard编号
shard_num: shard总数
global_sampler_type: 全局采样类型,all/node/edge/none,默认node
graph_type: graph类型,compact/fast,默认compact
server_thread_num: euler service线程数,默认4
"""
def initialize_shared_graph(directory, zk_addr, zk_path, shard_idx, shard_num,
                            global_sampler_type='node', graph_type='compact',
                            server_thread_num=4):
  hdfs_prefix = 'hdfs://'
  if not directory.startswith(hdfs_prefix):
    raise ValueError('Only hdfs graph data is support for shared graph.')
  directory = directory[len(hdfs_prefix):]

  hdfs_addr = directory[:directory.index(':')]
  directory = directory[len(hdfs_addr):]
  directory = directory[len(':'):]

  hdfs_port = directory[:directory.index('/')]
  directory = directory[len(hdfs_port):]

  euler.start(directory=directory,
              loader_type='hdfs',
              hdfs_addr=hdfs_addr,
              hdfs_port=hdfs_port,
              shard_idx=shard_idx,  # shard从0到num_shards-1编号
              shard_num=shard_num,  # Euler集群将图切分到多个shard中,每个shard中可以有多个图引擎实例
              zk_addr=zk_addr,
              zk_path=zk_path,
              global_sampler_type=global_sampler_type,
              graph_type=graph_type,
              server_thread_num=server_thread_num)

  return initialize_graph({'mode': 'Remote',
                           'zk_server': zk_addr,
                           'zk_path': zk_path})
