// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace paddle {

struct DataReader {
  explicit DataReader(std::string vocab_path,
                      std::string test_translation_path,
                      int batch_size);

  // NextBatch()

private:
  void load_dict();
  void load_lines();
  void load_src_trg_ids();

  const std::string vocab_path;
  const std::string test_translation_path;
  const std::vector<std::string> test_lines;
  const int batch_size;
  std::string beg{"<s>"};
  std::string end{"<e>"};
  std::string unk{"<unk>"};
  const char sentence_sep{'\t'};
  const char word_sep{' '};
  std::map<std::string, int> word_to_ind;
  std::map<int, std::string> ind_to_word;
  std::vector<std::tuple<int, int, int>> sample_infos;
};

}  // namespace paddle
