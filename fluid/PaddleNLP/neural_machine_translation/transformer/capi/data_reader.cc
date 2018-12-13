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

#include "data_reader.h"
#include <algorithm>

namespace paddle {


namespace {
static void split(const std::string& str,
                  char sep,
                  std::vector<std::string>* pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}


}  // namespace

DataReader::DataReader(std::string vocab_path,
                       std::string test_translation_path,
                       int batch_size)
    : vocab_path(std::move(vocab_path)),
      test_translation_path(std::move(test_translation_path)),
      batch_size(batch_size) {
  load_dict();
}

void DataReader::load_dict() {
  std::ifstream vocab_file(vocab_path);
  std::string line;
  word_to_ind.clear();
  ind_to_word.clear();
  for (int i = 0; std::getline(vocab_file, line); i++) {
    word_to_ind[line] = i;
    ind_to_word[i] = std::move(line);
  }
}

void DataReader::load_lines() {
  std::ifstream test_translation_file(test_translation_path);
  std::string line;
  // test_lines.clear();

  // for (int i = 0; std::getline(test_translation_file, line); i++) {
  //   std::string firstpart =
  //       std::string.splitwor(sentence_sep)[0]
  //       test_lines.push_back(firstpart);
  // }
}

std::vector<int> DataReader::convert_to_ind(const std::string& sentence) {
  std::vector<std::string> pieces;
  std::vector<int> indices;

  split(sentence, word_sep, &pieces);
  // indices.push_back(word_to_ind[beg]);
  for (auto& word : pieces) {
    auto search = word_to_ind.find(word);
    if (search != word_to_ind.end()) {
      indices.push_back(search->second);
    } else {
      indices.push_back(word_to_ind[unk]);
    }
  }
  indices.push_back(word_to_ind[end]);
  return indices;
}

std::vector<int> convert_to_sentence(const std::string& sentence) {
  std::vector<int> indices;

  // code that converts indices to string

  return indices;
}

void DataReader::load_src_trg_ids(const std::vector<std::string>& test_lines) {
  for (size_t i = 0; i < test_lines.size(); i++) {
    auto src_ids = convert_to_ind(test_lines[i]);
    src_seq_ids.push_back(src_ids);
    // TODO(sfraczek): This seems to be irrelevant since we read only en (no de)
    auto lens = src_ids.size();
    sample_infos.push_back(std::make_tuple(i, lens, lens));
  }
}

}  // namespace paddle
