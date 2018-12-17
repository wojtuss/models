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

bool DataReader::NextBatch(std::vector <std::vector<int64_t>>& inst_data, std::vector<std::vector<int64_t>> & inst_pos, std::vector <std::vector<float>> &slf_attn_bias_data, std::vector<std::vector<std::vector<float>>> tile_slf_attn_bias_data, int & max_len, int batch_size, int attn_bias_flag){
  inst_data.clear();
  inst_pos.clear();
  slf_attn_bias_data.clear();
  std::string line;
  std::vector<std::string> pieces;
  int max_length=0;
  for(int i = 0; i < batch_size, i++){
    if( !std::getline(test_translation_file,line) ){
      return false; // consider the situation there is not enough lines for last batch
    }
    pieces.clear();
    split(line, sentence_sep, &pieces);
    std::vector<int_64> sentence_indices = convert_to_ind(pieces[0]);
    inst_data.push_back(sentence_indices); 
    if (sentence_indices.size() > max_length){
      max_length = sentence_indices.size();
    }
    for (auto j=0; j < sentence_indices.size(); j++){
      inst_post[i][j] = j; 
    }
    slf_attn_bias_data.resize(sentence_indices.size(), 0e0)
  }
  //padding for inst_data [1], inst_pos[0] and slf_attn_bias[-1e9]
  for (int i = 0; i < batch_size; i++){
    inst_data[i].resize(max_length, eos_index);
    inst_pos.resize(max_length, 0);
    slf_attn_bias_data.resize(max_length, -1e9)
  }
  //tile, batch_size*n_head*max_len*max_len
  for (int i=0; i < batch_size; i++){
    for(int j=0; j < n_head; j++){
      for(int k = 0; k < max_length; k++){
        std::copy(slf_attn_bias_data+k*max_length, slf_attn_bias_data+(k+1)*max_length, tile_slf_attn_bias_data(i*n_head*max_length*max_length + j*max_length*max_length + k * max_length);
      }
    }  
  }
  max_len = max_length;  
  return true;
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


std::string DataReader::convert_to_sentence(const std::vector<int>& indices) {
  std::stringstream sentence;
  int end_i = word_to_ind[end];
  int beg_i = word_to_ind[beg];
  int unk_i = word_to_ind[unk];

  if (indices[0] != beg_i) sentence << ind_to_word[indices[0]];
  for (int i = 1; i < indices.size() - 1; i++) {
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
