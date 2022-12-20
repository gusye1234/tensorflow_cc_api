// #include <iostream>
// #include <tensorflow/c/c_api.h> // TensorFlow C API header.

// int main() {
//   std::cout << "TensorFlow Version: " << TF_Version() << std::endl;

//   return 0;
// }

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <tensorflow/c/c_api.h> // TensorFlow C API header.

static void DeallocateBuffer(void *data, size_t) { std::free(data); }

static TF_Buffer *ReadBufferFromFile(const char *file) {
  std::ifstream f(file, std::ios::binary);
  if (f.fail() || !f.is_open()) {
    f.close();
    return nullptr;
  }

  if (f.seekg(0, std::ios::end).fail()) {
    f.close();
    return nullptr;
  }
  auto fsize = f.tellg();
  if (f.seekg(0, std::ios::beg).fail()) {
    f.close();
    return nullptr;
  }

  if (fsize <= 0) {
    f.close();
    return nullptr;
  }

  auto data = static_cast<char *>(std::malloc(fsize));
  if (f.read(data, fsize).fail()) {
    f.close();
    return nullptr;
  }

  auto buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = DeallocateBuffer;
  f.close();
  return buf;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "You should input the model weight path" << std::endl;
    return 2;
  }
  auto buffer = ReadBufferFromFile(argv[1]);
  if (buffer == nullptr) {
    std::cout << "Can't read buffer from file" << std::endl;
    return 1;
  }
  std::cout << "Read buffer " << buffer->length << std::endl;

  auto graph = TF_NewGraph();
  auto status = TF_NewStatus();
  auto opts = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(graph, buffer, opts, status);

  if (TF_GetCode(status) != TF_OK) {
    std::cout << "Can't import GraphDef" << std::endl;
    std::cout << TF_Message(status) << std::endl;
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    return 2;
  }

  std::cout << "Load graph success" << std::endl;
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteBuffer(buffer);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(status);
  return 0;
}