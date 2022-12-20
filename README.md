## Tensorflow loader demos

This repo offers simple demos of how to use tensorflow C/C++ API to read infos from exported models.

**for build tensorflow C/C++ api**:

[refererence](https://github.com/rangsimanketkaew/tensorflow-cpp-api/blob/main/compile_tensorflow_cpp.md#dependencies) 

If everything goes well(highly impossible), you should have two parts your tensorflow repo's `bazel-bin`:

* `tensorflow/include`
* `tensorflow/lib/libtensorflow_cc.so.VERSION`, `tensorflow/lib/libtensorflow_framework.so.VERSION`. where `VERSION` should be your tensorflow version(*e.g.* 2.6.0)

**for this repo' deps**:

```
mkdir tensorflow_deps

# copy include headers from your tensorflow repo to here 
cp -r .../tensorflow/include ./tensorflow_deps/

# copy shared libraries from your tensorflow repo to here 
cp -r .../tensorflow/lib ./tensorflow_deps/

cd tensorflow_deps/lib
ln -s libtensorflow_cc.so.VERSION libtensorflow_cc.so
ln -s libtensorflow_cc.so.VERSION libtensorflow_cc.so.2
ln -s libtensorflow_framework.so.VERSION libtensorflow_framework.so
ln -s libtensorflow_framework.so.VERSION libtensorflow_framework.so.2
```

**for C api demo**:

```shell
make c_demo
./c_demo <your model weight pb file>
```

**for C++ demo**:

```shell
make cc_demo
./cc_demo <export dir> <tag set name>
```



