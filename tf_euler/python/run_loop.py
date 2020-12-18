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

import os

import numpy as np
import tensorflow as tf

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import models
from tf_euler.python import optimizers
from tf_euler.python.utils import context as utils_context
from tf_euler.python.utils import hooks as utils_hooks
from euler.python import service

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def define_network_embedding_flags():
  tf.flags.DEFINE_enum('mode', 'train',
                       ['train', 'evaluate', 'save_embedding'], 'Run mode.')  # 指定运行模式: train/evaluate/save_embedding,默认值为train

  tf.flags.DEFINE_string('data_dir', '', 'Local Euler graph data.')  # 指定图数据位置,必须
  tf.flags.DEFINE_integer('train_node_type', 0, 'Node type of training set.')  # 训练集顶点类型,默认0
  tf.flags.DEFINE_integer('all_node_type', euler_ops.ALL_NODE_TYPE,
                          'Node type of the whole graph.')  # 全集顶点类型,默认-1
  tf.flags.DEFINE_list('train_edge_type', [0], 'Edge type of training set.')  # 训练集边类型,默认[0]
  tf.flags.DEFINE_list('all_edge_type', [0, 1, 2],
                       'Edge type of the whole graph.')  # 全集边类型,默认[0,1]
  tf.flags.DEFINE_integer('max_id', -1, 'Max node id.')  # 图中最大的顶点id,必须
  tf.flags.DEFINE_integer('feature_idx', -1, 'Feature index.')  # 稠密特征的编号,使用稠密特征必须
  tf.flags.DEFINE_integer('feature_dim', 0, 'Feature dimension.')  # 稠密特征维度,使用稠密特征必须
  tf.flags.DEFINE_integer('label_idx', -1, 'Label index.')  # label在稠密特征中的编号,监督模型必须
  tf.flags.DEFINE_integer('label_dim', 0, 'Label dimension.')  # label在稠密特征中的维度,监督模型必须
  tf.flags.DEFINE_integer('num_classes', None, 'Number of classes.')  # 分类个数,label为标量时必须
  tf.flags.DEFINE_list('id_file', [], 'Files containing ids to evaluate.')  # 测试集id文件,一行一个id,evaluate必须

  tf.flags.DEFINE_string('model', 'graphsage_supervised', 'Embedding model.')  # 模型名称,支持line/randomwalk/graphsage/graphsage_supervised/scalable_gcn/gat/saved_embedding
  tf.flags.DEFINE_boolean('sigmoid_loss', True, 'Whether to use sigmoid loss.')  # 使用sigmoid或者softmax作为损失函数,默认sigmoid_loss
  tf.flags.DEFINE_boolean('xent_loss', True, 'Whether to use xent loss.')
  tf.flags.DEFINE_integer('dim', 256, 'Dimension of embedding.')  # embedding宽度,默认256
  tf.flags.DEFINE_integer('num_negs', 5, 'Number of negative samplings.')
  tf.flags.DEFINE_integer('order', 1, 'LINE order.')  # LINE模型的阶数,默认1阶
  tf.flags.DEFINE_integer('walk_len', 5, 'Length of random walk path.')
  tf.flags.DEFINE_float('walk_p', 1., 'Node2Vec return parameter.')  # Node2Vec模型的参数p
  tf.flags.DEFINE_float('walk_q', 1., 'Node2Vec in-out parameter.')  # Node2Vec模型的参数q
  tf.flags.DEFINE_integer('left_win_size', 5, 'Left window size.')
  tf.flags.DEFINE_integer('right_win_size', 5, 'Right window size.')
  tf.flags.DEFINE_list('fanouts', [10, 10], 'GCN fanouts.')  # GraphSage/GraphSage(Supervised)/ScalableGCN模型的参数,每层的扩展数,默认[10,10]
  tf.flags.DEFINE_enum('aggregator', 'mean',
                       ['gcn', 'mean', 'meanpool', 'maxpool', 'attention'],
                       'Sage aggregator.')  # GraphSage/GraphSage(Supervised)/ScalableGCN模型的参数,汇聚类型,支持gcn/mean/meanpool/maxpool,默认mean
  tf.flags.DEFINE_boolean('concat', True, 'Sage aggregator concat.')  # GraphSage/GraphSage(Supervised)/ScalableGCN模型的参数,汇聚方法,默认concat
  tf.flags.DEFINE_boolean('use_residual', False, 'Whether use skip connection.')
  tf.flags.DEFINE_float('store_learning_rate', 0.001, 'Learning rate of store.')
  tf.flags.DEFINE_float('store_init_maxval', 0.05,
                        'Max initial value of store.')
  tf.flags.DEFINE_integer('head_num', 1, 'multi head attention num')  # GAT模型的参数,attention head数目,默认1

  tf.flags.DEFINE_string('model_dir', 'ckpt', 'Model checkpoint.')  # checkpoint路径,默认ckpt
  tf.flags.DEFINE_integer('batch_size', 512, 'Mini-batch size.')  # batch_size,默认512
  tf.flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use.')  # 优化器,默认adam,使用adam训练速度很慢
  tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')  # 学习率,默认0.01
  tf.flags.DEFINE_integer('num_epochs', 20, 'Number of epochs for training.')  # 训练顶点轮数,默认10
  tf.flags.DEFINE_integer('log_steps', 20, 'Number of steps to print log.')  # 日志打印间隔步数,默认20

  tf.flags.DEFINE_list('ps_hosts', [], 'Parameter servers.')  # ps列表,分布式训练必须
  tf.flags.DEFINE_list('worker_hosts', [], 'Training workers.')  # worker列表,分布式训练必须
  tf.flags.DEFINE_string('job_name', '', 'Cluster role.')  # task_name,角色名称,ps或worker,分布式训练必须；
  tf.flags.DEFINE_integer('task_index', 0, 'Task index.')  # 角色索引,分布式训练必须

  tf.flags.DEFINE_string('euler_zk_addr', '127.0.0.1:2181',
                         'Euler ZK registration service.')  # ZooKeeper地址,分布式训练必须
  tf.flags.DEFINE_string('euler_zk_path', '/tf_euler',
                         'Euler ZK registration node.')  # ZooKeeper节点,分布式训练必须

def run_train(model, flags_obj, master, is_chief):
  utils_context.training = True

  batch_size = flags_obj.batch_size // model.batch_size_ratio
  if flags_obj.model == 'line' or flags_obj.model == 'randomwalk':
    source = euler_ops.sample_node(  # sample_node: 根据配置顶点类型采样负例
        count=batch_size, node_type=flags_obj.all_node_type)  # all_node_type: 全集顶点类型
  else:
    source = euler_ops.sample_node(
        count=batch_size, node_type=flags_obj.train_node_type)  # train_node_type: 训练集顶点类型
  source.set_shape([batch_size])
  _, loss, metric_name, metric = model(source)

  optimizer_class = optimizers.get(flags_obj.optimizer)
  optimizer = optimizer_class(flags_obj.learning_rate)
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.minimize(loss, global_step=global_step)

  hooks = []

  tensor_to_log = {'step': global_step, 'loss': loss, metric_name: metric}
  hooks.append(
      tf.train.LoggingTensorHook(
          tensor_to_log, every_n_iter=flags_obj.log_steps))

  num_steps = int((flags_obj.max_id + 1) // flags_obj.batch_size *
                   flags_obj.num_epochs)
  hooks.append(tf.train.StopAtStepHook(last_step=num_steps))  # tf.train.StopAtStepHook: 在一定步数停止,last_step是终止步数

  if len(flags_obj.worker_hosts) == 0 or flags_obj.task_index == 1:
    hooks.append(
        tf.train.ProfilerHook(save_secs=180, output_dir=flags_obj.model_dir))  # tf.train.ProfilerHook(): 每N步或N秒统计一次CPU/GPU的profiling信息
  if len(flags_obj.worker_hosts):
    hooks.append(utils_hooks.SyncExitHook(len(flags_obj.worker_hosts)))
  if hasattr(model, 'make_session_run_hook'):
    hooks.append(model.make_session_run_hook())

  with tf.train.MonitoredTrainingSession(  # tf.train.MonitoredTrainingSession(): 用于管理分布式训练
      master=master,  # master: 用于分布式系统中,指定运行会话协议ip和端口
      is_chief=is_chief,  # is_chief: 用于分布式系统中,if(true),它将负责初始化并恢复底层TensorFlow会话;if(false),它将等待chief初始化或恢复TensorFlow会话
      checkpoint_dir=flags_obj.model_dir,
      log_step_count_steps=None,
      hooks=hooks,  # SessionRunHook对象的可选列表
      config=config) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def run_evaluate(model, flags_obj, master, is_chief):
  utils_context.training = False

  dataset = tf.data.TextLineDataset(flags_obj.id_file)
  if master:
    dataset = dataset.shard(len(flags_obj.worker_hosts), flags_obj.task_index)
  dataset = dataset.map(
      lambda id_str: tf.string_to_number(id_str, out_type=tf.int64))
  dataset = dataset.batch(flags_obj.batch_size)
  source = dataset.make_one_shot_iterator().get_next()
  _, _, metric_name, metric = model(source)

  tf.train.get_or_create_global_step()
  hooks = []
  if master:
    hooks.append(utils_hooks.SyncExitHook(len(flags_obj.worker_hosts)))

  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=flags_obj.model_dir,
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      hooks=hooks,
      config=config) as sess:
    while not sess.should_stop():
      metric_val = sess.run(metric)

  print('{}: {}'.format(metric_name, metric_val))


def run_save_embedding(model, flags_obj, master, is_chief):
  utils_context.training = False

  dataset = tf.data.Dataset.range(flags_obj.max_id + 1)  # Dataset.range(5) == [0, 1, 2, 3, 4]
  if master:
    dataset = dataset.shard(len(flags_obj.worker_hosts), flags_obj.task_index)
  dataset = dataset.batch(flags_obj.batch_size)  # 将数据组合成batch
  source = dataset.make_one_shot_iterator().get_next()  # dataset.make_one_shot_iterator():从dataset中实例化了一个iterator,这个iterator中的数据输出一次后就丢弃; iterator.get_next():表示从iterator里取出一个元素
  embedding, _, _, _ = model(source)  # 调用模型得到embedding,这里source应该是一个batch的数据

  tf.train.get_or_create_global_step()  # 返回或者创建一个全局步数的tensor
  hooks = []
  if master:
    hooks.append(utils_hooks.SyncExitHook(len(flags_obj.worker_hosts)))

  ids = []
  embedding_vals = []
  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=flags_obj.model_dir,
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      hooks=hooks,
      config=config) as sess:
    while not sess.should_stop():
      id_, embedding_val = sess.run([source, embedding])
      ids.append(id_)
      embedding_vals.append(embedding_val)

  id_ = np.concatenate(ids)
  embedding_val = np.concatenate(embedding_vals)

  if master:
    embedding_filename = 'embedding_{}.npy'.format(flags_obj.task_index)
    id_filename = 'id_{}.txt'.format(flags_obj.task_index)
  else:
    embedding_filename = 'embedding.npy'
    id_filename = 'id.txt'
  embedding_filename = flags_obj.model_dir + '/' + embedding_filename
  id_filename = flags_obj.model_dir + '/' + id_filename

  with tf.gfile.GFile(embedding_filename, 'w') as embedding_file:
    np.save(embedding_file, embedding_val)
  with tf.gfile.GFile(id_filename, 'w') as id_file:
    id_file.write('\n'.join(map(str, id_)))


def run_network_embedding(flags_obj, master, is_chief):
  fanouts = map(int, flags_obj.fanouts)
  if flags_obj.mode == 'train':
    metapath = [map(int, flags_obj.train_edge_type)] * len(fanouts)
  else:
    metapath = [map(int, flags_obj.all_edge_type)] * len(fanouts)

  if flags_obj.model == 'line':
    model = models.LINE(  # models是一个python包,LINE是类名,这里使用类的名称LINE进行实例化,并通过LINE的__init__()方法接收参数
        node_type=flags_obj.all_node_type,
        edge_type=flags_obj.all_edge_type,
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        order=flags_obj.order)  # 一阶模型or二阶模型,默认1

  elif flags_obj.model in ['randomwalk', 'deepwalk', 'node2vec']:
    model = models.Node2Vec(
        node_type=flags_obj.all_node_type,
        edge_type=flags_obj.all_edge_type,
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        walk_len=flags_obj.walk_len,  # int,游走长度,默认3
        walk_p=flags_obj.walk_p,  # int,回采参数,默认1
        walk_q=flags_obj.walk_q,  # int,外采参数,默认1
        left_win_size=flags_obj.left_win_size,  # int,左滑动窗口长度,默认1
        right_win_size=flags_obj.right_win_size)  # int,右滑动窗口长度,默认1

  elif flags_obj.model in ['gcn', 'gcn_supervised']:
    model = models.SupervisedGCN(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        metapath=metapath,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        use_residual=flags_obj.use_residual)

  elif flags_obj.model == 'scalable_gcn':
    model = models.ScalableGCN(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        edge_type=metapath[0],  # 1-D int64 tf.Tensor,边类型集合(多值)
        num_layers=len(fanouts),  # int,GCN模型层数
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,  # 汇聚类型,默认mean
        feature_idx=flags_obj.feature_idx,  # 稠密特征的编号,默认-1
        feature_dim=flags_obj.feature_dim,  # 稠密特征维度,默认0
        max_id=flags_obj.max_id,
        use_residual=flags_obj.use_residual,
        store_learning_rate=flags_obj.store_learning_rate,
        store_init_maxval=flags_obj.store_init_maxval)

  elif flags_obj.model == 'graphsage':
    model = models.GraphSage(
        node_type=flags_obj.train_node_type,
        edge_type=flags_obj.train_edge_type,
        max_id=flags_obj.max_id,
        xent_loss=flags_obj.xent_loss,
        num_negs=flags_obj.num_negs,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim)

  elif flags_obj.model == 'graphsage_supervised':
    model = models.SupervisedGraphSage(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        metapath=metapath,  # python列表,成员为1-D int64 tf.Tensor,每步的边类型集合(多值)
        fanouts=fanouts,  # python列表,成员为int,每阶的采样个数
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,  # 汇聚类型,默认mean
        concat=flags_obj.concat,  # 汇聚方法,默认False
        feature_idx=flags_obj.feature_idx,  # 稠密特征的编号,默认-1
        feature_dim=flags_obj.feature_dim)  # 稠密特征维度,默认0

  elif flags_obj.model == 'scalable_sage':
    model = models.ScalableSage(
        label_idx=flags_obj.label_idx, label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes, sigmoid_loss=flags_obj.sigmoid_loss,
        edge_type=metapath[0], fanout=fanouts[0], num_layers=len(fanouts),
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator, concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx, feature_dim=flags_obj.feature_dim,
        max_id=flags_obj.max_id,
        store_learning_rate=flags_obj.store_learning_rate,
        store_init_maxval=flags_obj.store_init_maxval)

  elif flags_obj.model == 'gat':
    model = models.GAT(
        label_idx=flags_obj.label_idx,  # 稠密特征的编号,默认-1
        label_dim=flags_obj.label_dim,  # 稠密特征维度,默认0
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        max_id=flags_obj.max_id,   # int,图中的最大id,默认-1
        head_num=flags_obj.head_num,
        hidden_dim=flags_obj.dim,  # int,隐层宽度,默认128
        nb_num=5)  # int,相邻顶点采样数,默认5

  elif flags_obj.model == 'lshne':
    model = models.LsHNE(-1,[[[0,0,0],[0,0,0]]],-1,128,[1,1],[1,1])

  elif flags_obj.model == 'saved_embedding':
    embedding_val = np.load(os.path.join(flags_obj.model_dir, 'embedding.npy'))
    embedding = layers.Embedding(
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        initializer=lambda: tf.constant_initializer(embedding_val))
    model = models.SupervisedModel(
        flags_obj.label_idx,
        flags_obj.label_dim,
        flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss)
    model.encoder = lambda inputs: tf.stop_gradient(embedding(inputs))

  else:
    raise ValueError('Unsupported network embedding model.')

  if flags_obj.mode == 'train':
    run_train(model, flags_obj, master, is_chief)
  elif flags_obj.mode == 'evaluate':
    run_evaluate(model, flags_obj, master, is_chief)
  elif flags_obj.mode == 'save_embedding':
    run_save_embedding(model, flags_obj, master, is_chief)


def run_local(flags_obj, run):
  if not euler_ops.initialize_embedded_graph(flags_obj.data_dir):
    raise RuntimeError('Failed to initialize graph.')

  run(flags_obj, master='', is_chief=True)


def run_distributed(flags_obj, run):
  cluster = tf.train.ClusterSpec({
      'ps': flags_obj.ps_hosts,
      'worker': flags_obj.worker_hosts
  })
  server = tf.train.Server(
      cluster, job_name=flags_obj.job_name, task_index=flags_obj.task_index)

  if flags_obj.job_name == 'ps':
    server.join()
  elif flags_obj.job_name == 'worker':
    if not euler_ops.initialize_shared_graph(
        directory=flags_obj.data_dir,
        zk_addr=flags_obj.euler_zk_addr,
        zk_path=flags_obj.euler_zk_path,
        shard_idx=flags_obj.task_index,
        shard_num=len(flags_obj.worker_hosts),
        global_sampler_type='node'):
      raise RuntimeError('Failed to initialize graph.')

    with tf.device(
        tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % flags_obj.task_index,
            cluster=cluster)):
      run(flags_obj, server.target, flags_obj.task_index == 0)
  else:
    raise ValueError('Unsupport role: {}'.format(flags_obj.job_name))


def main(_):
  flags_obj = tf.flags.FLAGS
  if flags_obj.worker_hosts:
    run_distributed(flags_obj, run_network_embedding)
  else:
    run_local(flags_obj, run_network_embedding)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_network_embedding_flags()
  tf.app.run(main)
