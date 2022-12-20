#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session_options.h>

// #include <xtensor/xarray.hpp>
// #include <xtensor/xnpy.hpp>

#include <cfloat>
#include <iostream>
#include <string>
#include <vector>

static const int IMG_SIZE = 784;
static const int NUM_SAMPLES = 10000;

std::vector<int> get_tensor_shape(const tensorflow::Tensor &tensor) {
  std::vector<int> shape;
  auto num_dimensions = tensor.shape().dims();
  for (int i = 0; i < num_dimensions; i++) {
    shape.push_back(tensor.shape().dim_size(i));
  }
  return shape;
}

template <typename M> void print_keys(const M &sig_map) {
  std::cout << "Hello, start printing\n";
  for (auto const &p : sig_map) {
    std::cout << "key : " << p.first << std::endl;
  }
}

template <typename K, typename M> bool assert_in(const K &k, const M &m) {
  return !(m.find(k) == m.end());
}

std::string _input_name = "digits";
std::string _output_name = "predictions";

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "You should input the export dir and tag set name"
              << std::endl;
    return 0;
  }
  // This is passed into LoadSavedModel to be populated.
  tensorflow::SavedModelBundle bundle;

  // From docs: "If 'target' is empty or unspecified, the local TensorFlow
  // runtime implementation will be used.  Otherwise, the TensorFlow engine
  // defined by 'target' will be used to perform all computations."
  tensorflow::SessionOptions session_options;

  // Run option flags here:
  // https://www.tensorflow.org/api_docs/python/tf/compat/v1/RunOptions We don't
  // need any of these yet.
  tensorflow::RunOptions run_options;

  // Fills in this from a session run call
  std::vector<tensorflow::Tensor> out;

  std::string dir = argv[1];
  std::string sig_def = argv[2];

  std::cout << "Found model: " << tensorflow::MaybeSavedModelDirectory(dir)
            << std::endl;
  // TF_CHECK_OK takes the status and checks whether it works.
  TF_CHECK_OK(tensorflow::LoadSavedModel(
      session_options, run_options, dir,
      // Refer to tag_constants. We just want to serve the model.
      {tensorflow::kSavedModelTagServe}, &bundle));

  auto sig_map = bundle.meta_graph_def.signature_def();

  // not sure why it's called this but upon running this for loop to check for
  // keys we see it.
  print_keys(sig_map);
  auto model_def = sig_map.at(sig_def);
  auto inputs = model_def.inputs().at(_input_name);
  auto input_name = inputs.name();
  auto outputs = model_def.outputs().at(_output_name);
  auto output_name = outputs.name();

  return 0;
}