/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>
#include <memory>
#include <iostream>

#include "euler/client/graph_config.h"
#include "euler/client/graph.h"
#include "euler/common/string_util.h"


namespace tensorflow {

std::unique_ptr<euler::client::Graph>& Graph() {
  static std::unique_ptr<euler::client::Graph> graph(nullptr);
  return graph;
}

}  // namespace tensorflow


#ifdef   __cplusplus
#if      __cplusplus

extern "C" {

#endif  // __cplusplus
#endif  // __cplusplus


// CreateGraph: Create graph by specified config string
// conf: "mode=Remote;zk_server=127.0.0.1:2801;zk_path=/euler"

bool CreateGraph(const char* conf) {  // # conf是一个字符串,格式为: "directory=data_dir;load_type=compact;mode=Local"
  euler::client::GraphConfig config;
  std::vector<std::string> vec;
  euler::common::split_string(conf, ';', &vec);  // 用';'分割conf字符串并将分割结果放入vec
  if (vec.empty()) {
    return false;
  }

  for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::vector<std::string> key_value;
    euler::common::split_string(*it, '=', &key_value);  // 用'='分割*it指向的字符串并将分割结果放入key_value
    if (key_value.size() != 2 || key_value[0].empty() || key_value[1].empty()) {  // *it遇到非法配置
      return false;
    }
    config.Add(key_value[0], key_value[1]);  // 将conf的内容解析后添加到GraphConfig的对象config中
  }

  auto graph = euler::client::Graph::NewGraph(config);  // 利用config构建一个图: Local/Remote两种模式
  if (graph == nullptr) {
    return false;
  }
  tensorflow::Graph() = std::move(graph);
  return true;
}

#ifdef   __cplusplus
#if      __cplusplus

}  // extern C

#endif  // __cplusplus
#endif  // __cplusplus
