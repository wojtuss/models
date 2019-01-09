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
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/paddle_inference_api.h"

DEFINE_string(model, "", "Directory with the inference model.");
DEFINE_bool(use_mkldnn, false, "Use MKL-DNN.");
DEFINE_bool(skip_passes, false, "Skip running passes.");
DEFINE_bool(one_file_params,
            false,
            "Parameters of the model are in one file 'params' and model in a "
            "file 'model'.");

namespace paddle {

#define PRINT_OPTION(a)                               \
  do {                                                \
    std::cout << #a ": " << (FLAGS_##a) << std::endl; \
  } while (false)

void PrintInfo() {
  std::cout << std::endl
            << "--- Used Parameters: -----------------" << std::endl;
  PRINT_OPTION(model);
  PRINT_OPTION(one_file_params);
  PRINT_OPTION(skip_passes);
  PRINT_OPTION(use_mkldnn);
  std::cout << "--------------------------------------" << std::endl;
}

void PrepareConfig(contrib::AnalysisConfig& config) {
  if (FLAGS_one_file_params) {
    config.SetProgFile(FLAGS_model + "/model");
    config.SetParamsFile(FLAGS_model + "/params");
  } else {
    config.SetModel(FLAGS_model);
  }
  config.DisableGpu();
  config.SwitchIrOptim(true);
  config.SwitchSpecifyInputNames(false);
  config.SetCpuMathLibraryNumThreads(1);
  if (FLAGS_use_mkldnn) config.EnableMKLDNN();

  // remove all passes so that we can add them in correct order
  for (int i = config.pass_builder()->AllPasses().size() - 1; i >= 0; i--)
    config.pass_builder()->DeletePass(i);

  // add passes
  config.pass_builder()->AppendPass("infer_clean_graph_pass");
  if (!FLAGS_skip_passes) {
    if (FLAGS_use_mkldnn) {
      // add passes to execute with MKL-DNN
      config.pass_builder()->AppendPass("mkldnn_placement_pass");
      config.pass_builder()->AppendPass("depthwise_conv_mkldnn_pass");
      config.pass_builder()->AppendPass("conv_bn_fuse_pass");
      config.pass_builder()->AppendPass("conv_eltwiseadd_bn_fuse_pass");
      config.pass_builder()->AppendPass("conv_bias_mkldnn_fuse_pass");
      config.pass_builder()->AppendPass(
          "conv_elementwise_add_mkldnn_fuse_pass");
      config.pass_builder()->AppendPass("conv_relu_mkldnn_fuse_pass");
      config.pass_builder()->AppendPass("fc_fuse_pass");
      config.pass_builder()->AppendPass("is_test_pass");
    } else {
      // add passes to execute keeping the order - without MKL-DNN
      config.pass_builder()->AppendPass("conv_bn_fuse_pass");
      config.pass_builder()->AppendPass("fc_fuse_pass");
    }
  }

  // enable graph_viz pass
  config.pass_builder()->TurnOnDebug();
  std::cout << "Saved model graphs" << std::endl;
}

void Main() {
  PrintInfo();

  struct stat sb;
  if (stat(FLAGS_model.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    throw std::invalid_argument("The model directory does not exist.");
  }

  contrib::AnalysisConfig config;
  PrepareConfig(config);

  auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                         PaddleEngineKind::kAnalysis>(config);
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
