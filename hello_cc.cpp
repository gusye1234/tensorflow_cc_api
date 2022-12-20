#include <tensorflow/c/c_api.h>
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

// tensorflow::Tensor load_npy_img(const std::string &filename) {
//   // auto data = xt::load_npy<float>(filename);
//   tensorflow::DataType s;
//   tensorflow::TensorShape ss;
//   tensorflow::Tensor t(tensorflow::DT_FLOAT,
//                        tensorflow::TensorShape({NUM_SAMPLES, IMG_SIZE}));

//   // for (int i = 0; i < NUM_SAMPLES; i++)
//   //     for (int j = 0; j < IMG_SIZE; j++)
//   //         t.tensor<float, 2>()(i,j) = data(i, j);

//   // return t;
// }

// std::vector<int> get_tensor_shape(const tensorflow::Tensor &tensor) {
//   std::vector<int> shape;
//   auto num_dimensions = tensor.shape().dims();
//   for (int i = 0; i < num_dimensions; i++) {
//     shape.push_back(tensor.shape().dim_size(i));
//   }
//   return shape;
// }

template <typename M> void print_keys(const M &sig_map) {
  std::cout << "Hello, start printing\n";
  for (auto const &p : sig_map) {
    std::cout << "key : " << p.first << std::endl;
  }
}

template <typename K, typename M> bool assert_in(const K &k, const M &m) {
  return !(m.find(k) == m.end());
}

std::string _input_name = "images";
std::string _output_name = "logits";

int main(int argc, char *argv[]) {
  tensorflow::SavedModelBundle bundle;

  // From docs: "If 'target' is empty or unspecified, the local TensorFlow
  // runtime implementation will be used.  Otherwise, the TensorFlow engine
  // defined by 'target' will be used to perform all computations."
  tensorflow::SessionOptions session_options;
  tensorflow::Tensor input(tensorflow::DT_FLOAT,
                           tensorflow::TensorShape({5, 224, 224, 3}));

  // Run option flags here:
  // https://www.tensorflow.org/api_docs/python/tf/compat/v1/RunOptions We don't
  // need any of these yet.
  tensorflow::RunOptions run_options;

  // Fills in this from a session run call
  std::vector<tensorflow::Tensor> out;

  std::string dir = "./fixtures/aiy_vision_classifier_birds_V1_1/";
  std::string sig_def = "image_classifier";

  std::cout << "Working with " << dir << ", tag set " << sig_def << std::endl;
  std::cout << "Found model: " << tensorflow::MaybeSavedModelDirectory(dir)
            << std::endl;
  // TF_CHECK_OK takes the status and checks whether it works.
  TF_CHECK_OK(tensorflow::LoadSavedModel(
      session_options, run_options, dir,
      // Refer to tag_constants. We just want to serve the model.
      {}, &bundle));
  // tensorflow::kSavedModelTagServe
  auto sig_map = bundle.meta_graph_def.signature_def();

  // not sure why it's called this but upon running this for loop to check for
  // keys we see it.
  print_keys(sig_map);
  auto model_def = sig_map.at(sig_def);
  for (auto x : model_def.inputs()) {
    std::cout << x.first << std::endl;
    std::cout << x.second.name() << std::endl;
  }
  std::vector<tensorflow::Tensor> out_tensor;
  auto inputs = model_def.inputs().at(_input_name);
  auto input_name = inputs.name();
  auto outputs = model_def.outputs().at(_output_name);
  auto output_name = outputs.name();

  std::cout << "Input name:" << input_name << std::endl;
  std::cout << "Output name:" << output_name << std::endl;
  TF_CHECK_OK(
      bundle.session->Run({{input_name, input}}, {output_name}, {}, &out));

  std::cout << out.size() << std::endl;
  for (auto x : out) {
    std::cout << x.shape().dims() << std::endl;
    std::cout << x.shape().dim_size(0) << std::endl;
    std::cout << x.shape().dim_size(1) << std::endl;
  }

  return 0;
}