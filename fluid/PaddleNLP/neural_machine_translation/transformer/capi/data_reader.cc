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
#include <sstream>

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
  test_translation_file.open(this->test_translation_path, std::ifstream::in);
  if (!test_translation_file.is_open()) {
    std::stringstream ss;
    ss << "Cannot open test translation file: " << this->test_translation_path
       << "\n";
    throw std::runtime_error(ss.str());
  }
  load_dict();
  //bos_idx = word_to_ind[beg];
  //eos_idx = word_to_ind[end];
  //unk_idx = word_to_ind[unk];
}

void DataReader::load_dict() {
  std::ifstream vocab_file(vocab_path);
  std::string line;
  word_to_ind.clear();
  ind_to_word.clear();
  for (int64_t i = 0; std::getline(vocab_file, line); i++) {
    word_to_ind[line] = i;
    ind_to_word[i] = std::move(line);
  }
}

bool DataReader::NextBatch(
    std::vector<std::vector<int64_t>>& inst_data,
    std::vector<std::vector<int64_t>>& inst_pos,
    std::vector<std::vector<float>>& slf_attn_bias_data,
    size_t& max_length,
    int batch_size) {
  //clear vectors
  inst_data.clear();
  inst_pos.clear();
  slf_attn_bias_data.clear();
  // resize vectors first dimension
  inst_data.resize(batch_size);
  inst_pos.resize(batch_size);
  slf_attn_bias_data.resize(batch_size);

  // get batch_size lines (sentences) from test file and convert them to
  // indices, then put the indices to inst_data and remember max_length of
  // sentence. Set inst_pos with (0,1,...,sentence_length) positional indexes.
  // Set slf_attn_bias_data with zeros.
  std::string line;
  std::vector<std::string> pieces;
  max_length = 0;
  for (int i = 0; i < batch_size; i++) {
    if (!std::getline(test_translation_file, line)) {
      return false;  // consider the situation there is not enough lines for
                     // last batch
    }
    pieces.clear();
    split(line, sentence_sep, &pieces);
    std::vector<int64_t> sentence_indices = convert_to_ind(pieces[0]);
    inst_data.push_back(sentence_indices);
    if (sentence_indices.size() > max_length) {
      max_length = sentence_indices.size();
    }
    inst_pos[i].resize(sentence_indices.size());
    for (size_t j = 0; j < sentence_indices.size(); j++) {
      inst_pos[i][j] = j;
    }
    slf_attn_bias_data[i].resize(sentence_indices.size(), 0e0);
  }

  // padding for inst_data with eos_idx, for inst_pos with 0, and for slf_attn_bias with -1e9
  for (int i = 0; i < batch_size; i++) {
    inst_data[i].resize(max_length, eos_idx);
    inst_pos[i].resize(max_length, 0);
    slf_attn_bias_data[i].resize(max_length, -1e9);
  }
//  // tile, batch_size*n_head*max_length*max_length
//   typedef vector<vector<float> > V2;
//   typedef vector<vector<vector<float> > > V3;
//   tile_slf_attn_bias_data.resize(batch_size);
//   for (int i = 0; i < batch_size; i++) {
//     tile_slf_attn_bias_data[i] = V3(n_head, V2(max_length, inst_data[i]));
//   }
   return true;
}

std::vector<int64_t> DataReader::convert_to_ind(const std::string& sentence) {
  std::vector<std::string> pieces;
  std::vector<int64_t> indices;
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

std::string DataReader::convert_to_sentence(const std::vector<int64_t>& indices) {
  std::stringstream sentence;
  int end_i = word_to_ind[end];
  int beg_i = word_to_ind[beg];
  //int unk_i = word_to_ind[unk];

  if (indices[0] != beg_i) sentence << ind_to_word[indices[0]];
  for (size_t i = 1; i < indices.size() - 1; i++) {
    sentence << " " << ind_to_word[indices[i]];
  }
  if (indices.back() != end_i) sentence << " " << ind_to_word[indices.back()];
  return sentence.str();
}

void DataReader::load_src_trg_ids(const std::vector<std::string>& inst_data) {
  for (size_t i = 0; i < inst_data.size(); i++) {
    auto src_ids = convert_to_ind(inst_data[i]);
    src_seq_ids.push_back(src_ids);
    // TODO(sfraczek): This seems to be irrelevant since we read only en (no de)
    auto lens = src_ids.size();
    sample_infos.push_back(std::make_tuple(i, lens, lens));
  }
}
}  // namespace paddle
