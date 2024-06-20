CXX = g++
PYTHON_INCLUDE = $(shell python3 -m pybind11 --includes)
PYTHON_LIB = $(shell python3-config --extension-suffix)

BUILD = ./build

LIB_TEST_TENSOR_LIB = $(BUILD)/test_tensor_lib$(PYTHON_LIB)
LIBS = $(LIB_TEST_TENSOR_LIB)

.PHONY: all
all: libs test

##########

.PHONY: libs
libs: $(BUILD) $(LIBS)

$(BUILD):
	mkdir -p $@ && cd $@ && cmake ..

$(LIB_TEST_TENSOR_LIB): test_tensor_lib.cpp
	cd build && make

##########

export PYTHONPATH=$(BUILD)

.PHONY: test
test: $(LIB_CONV2D) test_conv2d.py
	python3 test_conv2d.py

##########

.PHONY: clean
clean:
	rm -rf build
