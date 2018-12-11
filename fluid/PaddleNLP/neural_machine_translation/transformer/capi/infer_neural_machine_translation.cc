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

//#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gflags/gflags.h>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include "data_reader.h"
#include "paddle/fluid/inference/paddle_inference_api.h"
// Is this profiler needed, it is not provided however
// #include "paddle/fluid/platform/profiler.h"
#include "stats.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(all_vocab_fpath,
              "",
              "Path to a vocabulary file with both languages.");
DEFINE_string(special_token,
              "<s> <e> <unk>",
              "The <bos>, <eos> and <unk> tokens in the dictionary.");
DEFINE_string(test_file_pattern, "", "The pattern to match test data files.");
DEFINE_string(token_delimiter,
              " ",
              "The delimiter used to split tokens in source or target "
              "sentences. For EN-DE BPE data we provided, use spaces as token "
              "delimiter.");
DEFINE_int32(batch_size,
             1,
             "The number of examples in one run for sequence generation.");
DEFINE_bool(use_mkldnn, false, "Use MKL-DNN.");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(enable_graphviz,
            false,
            "Enable graphviz to get .dot files with data flow graphs.");
DEFINE_bool(one_file_params,
            false,
            "Parameters of the model are in one file 'params' and model in a "
            "file 'model'.");
DEFINE_bool(profile, false, "Turn on profiler for fluid");
DEFINE_int32(paddle_num_threads,
             1,
             "Number of threads for each paddle instance.");
DEFINE_int32(beam_size, 4, "Search width for Beam Search algorithm.");
DEFINE_int32(
    max_out_len,
    255,
    "The maximum depth(translation length) for Beam Search algorithm.");

namespace {

class Timer {
public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};
}  // namespace

namespace paddle {

void InitializeReader(std::unique_ptr<DataReader>& reader){
  reader.reset(new DataReader(FLAGS_))
}

void PrintInfo() {
  std::cout << std::endl
            << "--- Used Parameters: -----------------" << std::endl
            << "Inference model: " << FLAGS_infer_model << std::endl
            << "Vocab file: " << FLAGS_all_vocab_fpath << std::endl
            << "Special tokens: " << FLAGS_special_token << std::endl
            << "Test file pattern: " << FLAGS_test_file_pattern << std::endl
            << "Token delimiter: " << FLAGS_token_delimiter << std::endl
            << "Batch size: " << FLAGS_batch_size << std::endl
            << "Use MKL-DNN: " << FLAGS_use_mkldnn << std::endl
            << "Skip passes: " << FLAGS_skip_passes << std::endl
            << "Enable graphviz: " << FLAGS_enable_graphviz << std::endl
            << "One file params: " << FLAGS_one_file_params << std::endl
            << "Profile: " << FLAGS_profile << std::endl
            << "Paddle num threads : " << FLAGS_paddle_num_threads << std::endl
            << "Beam size : " << FLAGS_beam_size << std::endl
            << "Max out len : " << FLAGS_max_out_len << std::endl
            << "--------------------------------------" << std::endl;
}
void Main() {
  PrintInfo();
  // Test variables and call everything
  if (FLAGS_batch_size <= 0)
    throw std::invalid_argument(
        "The batch_size option is less than or equal to 0.");
  if (FLAGS_iterations <= 0)
    throw std::invalid_argument(
        "The iterations option is less than or equal to 0.");
  if (FLAGS_skip_batch_num < 0)
    throw std::invalid_argument("The skip_batch_num option is less than 0.");
  struct stat sb;
  if (stat(FLAGS_infer_model.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    throw std::invalid_argument(
        "The inference model directory does not exist.");
  }
  if (FLAGS_with_labels && FLAGS_use_fake_data)
    throw std::invalid_argument("Cannot use fake data for accuracy measuring.");
 
  std::unique_ptr<DataReader> reader;

  if (FLAGS_use_fake_data){
    
  }
  else{
   
  }

  
}

}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    paddle::Main();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return 0;
}
