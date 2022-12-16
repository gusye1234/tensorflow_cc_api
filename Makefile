BASE_DIR=/root/tensorflow_demo
INC=-I$(BASE_DIR)/../tensorflow-2.6.0 -I$(BASE_DIR)/tensorflow_include
LIB=-L$(BASE_DIR)/shares -ltensorflow -ltensorflow_framework
FLAGS=-std=c++14

main: hello.cpp
	g++ $(FLAGS) $(INC) $(LIB) -o hello $^