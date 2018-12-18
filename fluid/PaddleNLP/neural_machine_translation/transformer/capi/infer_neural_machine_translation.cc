// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include "data_reader.h"
#include "paddle/fluid/inference/paddle_inference_api.h"
#include "stats.h"

DEFINE_string(infer_model, "", "Directory of the inference model.");
DEFINE_string(all_vocab_fpath,
              "",
              "Path to a vocabulary file with both languages.");
DEFINE_string(test_file_pattern, "", "The pattern to match test data files.");
DEFINE_string(token_delimiter,
              " ",
              "The delimiter used to split tokens in source or target "
              "sentences. For EN-DE BPE data we provided, use spaces as token "
              "delimiter.");
DEFINE_string(output_file, "out_file.txt", "A file for the model output.");
DEFINE_int32(batch_size,
             1,
             "The number of examples in one run for sequence generation.");
DEFINE_int32(iterations, 1, "How many times to repeat run.");
DEFINE_int32(skip_batch_num, 0, "How many minibatches to skip in statistics.");
DEFINE_bool(use_mkldnn, false, "Use MKL-DNN.");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(enable_graphviz,
            false,
            "Enable graphviz to get .dot files with data flow graphs.");
DEFINE_bool(with_labels, true, "The infer model do handle data labels.");
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
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(startu - start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

}  // namespace
namespace paddle {
void PrintOutput(const std::vector<paddle::PaddleTensor>& output,
                 const std::string& out_file,
                 std::unique_ptr<DataReader>& reader) {
  if (output.size() == 0)
    throw std::invalid_argument("PrintOutput: output vector is empty.");

  if (output[0].dtype != paddle::PaddleDType::INT64) {
    throw std::invalid_argument("PrintOutput: output is of a wrong type.");
  }

  int64_t* ids_data = static_cast<int64_t*>(output[0].data.data());
  auto ids_lod = output[0].lod;
  auto ids_shape = output[0].shape;

  std::ofstream ofile(out_file, std::fstream::out | std::fstream::app);
  if (!ofile.is_open())
    throw std::invalid_argument("PrintOutput: cannot open the output file");

  for (int i = 0; i < ids_lod[0].size() - 1; ++i) {
    auto start = ids_lod[0][i];
    auto sub_start = ids_lod[1][start];
    auto sub_end = ids_lod[1][start + 1];
    auto data_start = ids_data + sub_start;
    std::vector<int> indices(data_start, data_end);
    std::string sentence = reader->convert_to_sentence(indices);
    ofile << sentence << std::endl;
  }
  ofile.close();
}

void InitializeReader(std::unique_ptr<DataReader>& reader) {
  reader.reset(new DataReader(
      FLAGS_all_vocab_fpath, FLAGS_test_file_pattern, FLAGS_batch_size));
}

template<typename T>
void copy_vector_of_vector(const vector<vector<T>> src_v_v, const T* dst_array_ptr) {
  auto* dst_ptr =  dst_array_ptr;
  for (auto v : src_v_v) {
    std::copy(v.begin(), v.end(), dst_ptr);
    dst_ptr += v.size();
  }
}
/*
int main() {
   typedef vector<double> V1;
   typedef vector<vector<double> > V2;
   typedef vector<vector<vector<double> > > V3;
   V3 vec3D(1, V2(2, V1(3, 0.0)));  // Create a 3-D vector
   V1 vec1D;
   flatten(vec3D, back_inserter(vec1D)); // Flatten the vector to 1-D
   for (V1::const_iterator it = vec1D.begin(); it != vec1D.end(); ++it)
     std::cout << *it << endl;
}
*/
bool ReadNextBatch(PaddleTensor & src_word_tensor, PaddleTensor & src_pos_tensor, PaddleTensor & src_slf_attn_bias_tensor, PaddleTensor & trg_word_tensor, PaddleTensor & init_score_tensor, PaddleTensor & trg_src_attn_bias_tensor, std::unique_ptr<DataReader>& reader) {
   
   std::vector<std::vector<int64_t>> inst_data;
   std::vector<std::vector<int64_t>> inst_pos;
   std::vector<std::vector< std::vector< float >> tile_slf_attn_bias_data;
   float * tile_slf_attn_bias_data;
   int max_len = 0; 
	 
   reader->NextBatch(std::vector<std::vector<int64_t>>& inst_data, std::vector<std::vector<int64_t>> & inst_pos, std::vector<std::vector<std::vector<float>>> tile_slf_attn_bias_data, int & max_len, FLAGS_batch_size, attn_bias_flag);
   //pad_batch_data
	 bool DataReader::NextBatch(std::vector <std::vector<int64_t>>& inst_data, std::vector<std::vector<int64_t>> & inst_pos, std::vector <std::vector<float>> &slf_attn_bias_data, std::vector<std::vector<std::vector<float>>> tile_slf_attn_bias_data, int & max_len, FLAGS_batch_size, attn_bias_flag);

   if (flag==false){
     throw std::runtime_error("Less than batch size of lines left in the file, or other runtime errors");
  }

  src_word_tensor.name = "src_word";
  src_pos_tensor.name = "src_pos";
   src_slf_attn_bias_tensor.name = "src_slf_attn_bias"
   trg_word_tensor.name = "trg_word";
  init_score_tensor.name = "init_score";
  trg_src_attn_bias_tensor.name = "trg_src_attn_bias";

   src_word_tensor.shape = {FLAGS_batch_size, max_len, 1};
   src_word_tensor.data.Resize( FLAGS_batch_sizie * max_len * sizeof(int64_t));
   src_word_tensor.lod.clear();

   src_pos_tensor.shape = {FLAGS_batch_size, max_len, 1};
   src_pos_tensor.data.Rersize( FLAGS_batch_size * max_len * sizeof(int64_t));
   src_pos_tensor.lod.clear();

   trg_src_attn_bias_tensor.shape = {FLAGS_batch_size, n_head, max_len, 1};
   trg_src_attn_bias_tensor.data.Resize( FLAGS_batch * max_len * n_head * sizeof(float));
   trg_src_attn_bias_tensor.lod.clear();
   
   trg_word_tensor.shape = {FLAGS_batch_size, 1, 1};
   trg_word_tensor.data.Resize(FLAGS_batchi_size * sizeof(int_64));
   trg_word_tensor.lod.clear();
   std::vector<size_t> tmplod;
   for (int i = 0 ; i <= FLAGS_batch_size ; i++){
      tmplod.push_back(i);
   }
   trg_word_tensor.lod.push_back(tmpload);
   trg_word_tensor.lod.push_back(tmpload);
   int64_t * trg_word_array = trg_word_tensor.data.data();
   for (int i = 0 ; i < FLAGS_batch_size ; i++){
     *trg_word_array ++ = reader.bos_idx;  
   }
 
   init_score_tensor.shape={FLAGS_batch_size,1};
   init_score_tensor.data.Resize(FLAGS_batch_size*sizeof(float));
   float * init_score_array = init_score_tensor.data.data();
   for(int i = 0 ; i < FLAGS_batch_size ; i++){
     *init_score_array++=0;
   } 
   init_score_tensor.lod.push_back(tmplod);
   init_score_tensor.lod.push_back(tmplod); 
  int64_t* src_word_array = static_cast<int64_t>(src_word_tensor.data.data());
  int64_t* src_pos_array = static_cast<int64_t>(src_pos_tensor.dara.dara());
  float* trg_src_attn_bias_array =
      static_cast<float>(trg_src_attn_bias_tensor.data.data());

  copy_vector_of_vector(inst_data, src_word_array);
  copy_vector_of_vector(inst_data, src_pos_array);
  for (auto v_v_i : tile_slf_attn_bias_data) {
    copy_vector_of_vector(v_v_i, trg_src_attn_bias_array);
  }
}

void PrintInfo() {
  std::cout << std::endl
            << "--- Used Parameters: -----------------" << std::endl
            << "Inference model: " << FLAGS_infer_model << std::endl
            << "Vocab file: " << FLAGS_all_vocab_fpath << std::endl
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

void PrepareConfig(contrib::AnalysisConfig& config) {
  if (FLAGS_one_file_params) {
    config.param_file = FLAGS_infer_model + "/params";
    config.prog_file = FLAGS_infer_model + "/model";
  } else {
    config.model_dir = FLAGS_infer_model;
  }
  config.use_gpu = false;
  config.device = 0;
  config.enable_ir_optim = !FLAGS_skip_passes;
  config.specify_input_name = false;
  config.SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
  if (FLAGS_use_mkldnn) config.EnableMKLDNN();

  // remove all passes so that we can add them in correct order
  for (int i = config.pass_builder()->AllPasses().size() - 1; i >= 0; i--)
    config.pass_builder()->DeletePass(i);

  // add passes
  config.pass_builder()->AppendPass("infer_clean_graph_pass");
  if (FLAGS_use_mkldnn) {
    // add passes to execute with MKL-DNN
    config.pass_builder()->AppendPass("mkldnn_placement_pass");
    config.pass_builder()->AppendPass("is_test_pass");
    config.pass_builder()->AppendPass("depthwise_conv_mkldnn_pass");
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_eltwiseadd_bn_fuse_pass");
    config.pass_builder()->AppendPass("conv_bias_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_elementwise_add_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("conv_relu_mkldnn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
  } else {
    // add passes to execute keeping the order - without MKL-DNN
    config.pass_builder()->AppendPass("conv_bn_fuse_pass");
    config.pass_builder()->AppendPass("fc_fuse_pass");
  }

  if (FLAGS_enable_graphviz) config.pass_builder()->TurnOnDebug();
}

void Main() {
  PrintInfo();
  // Test variables and call everything
  if (FLAGS_batch_size <= 0)
    throw std::invalid_argument(
        "The batch_size option is less than or equal to 0.");
  // struct stat sb;
  // if (stat(FLAGS_infer_model.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
  //   throw std::invalid_argument(
  //       "The inference model directory does not exist.");
  // }

  std::unique_ptr<DataReader> reader;


  // paddle::PaddleTensor input_fpattern = DefineInputData;
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
