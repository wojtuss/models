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
                       std::string test_translation_file,
                       std::vector<std::string> special_token,
                       int batch_size)
    : vocab_path(std::move(vocab_path)),
      test_translation_file(std::move(test_translation_file)),
      special_token(std::move(special_token)),
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

void DataReader::load_lines(){
	std::ifstream test_translation_file(test_translation_path);
  std::string line;
  test_lines.clear();

  for (int i = 0; std::getline(test_translation_file,line);i++){
		string firstpart=string.splitwor(sentence_sep)[0]
    test_lines.push_back(firstpart);
	}
}

void DataReader : convert(const std::string& sentence) {
  std::vector<std::string> pieces;
  std::vector<int> indices;
  indices.push_back();

  sentence.split(sentence, word_sep, pieces);
  // indices.push_back(beg);
  for (auto& word : pieces) {
    auto search = word_to_ind.find(word);
    if (search != word_to_ind.end()) {
      indices.push_back(search->second);
    } else {
      indices.push_back(word_to_ind[unk]);
    }
  }
  indices.push_back(end);
  return indices;
}

void DataReader::load_src_trg_ids() {
  std::vector<int> src_seq_ids;
  std::vector<tuple<int, int, int>>, sample_infos;


  for (int i = 0; i < test_lines.size(); i++) {
    auto src_trg_ids = convert(test_lines[i]);
    src_seq_ids.push_back(src_trg_ids);
    lens = src_trg_ids.size();
    self.sample_infos.emplace_back(i, max(lens), min(lens));
  }
}

}  // namespace paddle
