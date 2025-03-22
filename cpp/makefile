SHELL = /usr/bin/env bash

CPP_TESTS := $(basename $(subst tests/,,$(wildcard tests/test_*.cpp)))
PY_TESTS := $(basename $(subst tests/,,$(wildcard tests/test_*.py)))
BUILD_DIR := builddir

export PYTHONPATH := $(shell realpath ./$(BUILD_DIR)):$(PYTHONPATH)


.PHONY: reconf
reconf:
	meson setup $(BUILD_DIR) --reconfigure --buildtype=debug --prefix $(shell python3 -c "import sys; print(sys.prefix)")

.PHONY: build
build:
	meson compile -C $(BUILD_DIR)

.PHONY: list
list:
	@echo $(CPP_TESTS)

.PHONY: $(CPP_TESTS)
$(CPP_TESTS):
	meson test -C $(BUILD_DIR) -v $@

.PHONY: test
test:
	meson test -C $(BUILD_DIR) -v
	# python3 -m unittest -f ./tests/test_elem_wise.py

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	make reconf
