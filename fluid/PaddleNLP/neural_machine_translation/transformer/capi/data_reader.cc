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

DataReader::DataReader(std::string vocab_path,
                       std::string test_translation_path,
                       std::vector<std::string> special_token,
                       int batch_size)
    : vocab_path(std::move(vocab_path)),
      test_translation_path(std::move(test_translation_path)),
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
}  // namespace paddle
