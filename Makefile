BASE_DIR=/root/tensorflow_demo
# INC=-I$(BASE_DIR)/tensorflow_deps/include -I$(BASE_DIR)/tensorflow_deps/include/src
INC=-I$(BASE_DIR)/tensorflow_deps/include2
LIB=-L$(BASE_DIR)/tensorflow_deps/lib -ltensorflow_cc -ltensorflow_framework
FLAGS=-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0

cc_demo: hello_cc.cpp
	g++ $(FLAGS) $(INC) -o cc_demo $^  $(LIB)

c_demo: hello_c.cpp
	g++ $(FLAGS) $(INC) -o c_demo $^  $(LIB)