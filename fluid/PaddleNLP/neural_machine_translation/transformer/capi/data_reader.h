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
#include <string>
#include <vector>

namespace paddle {

struct DataReader {
  explicit DataReader(std::string vocab_path,
                      std::string test_translation_file,
                      std::vector<std::string> special_token,
                      int batch_size);

// NextBatch()

private:
  const std::string vocab_path;
  const std::ifstream vocab_file;
  const std::ifstream test_translation_file;
  const std::vector<std::string> special_token;
  const int batch_size;
  const char sentence_sep{'\t'};
  const char word_sep{' '};
};

}  // namespace paddle
