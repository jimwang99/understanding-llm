SHELL = /usr/bin/env bash

BUILD_DIR := builddir

export PYTHONPATH := $(shell realpath ./$(BUILD_DIR)):$(PYTHONPATH)

PY := python -m unittest -f

.PHONY: reconf
reconf:
	meson setup $(BUILD_DIR) --reconfigure --buildtype=debug --prefix $(shell python -c "import sys; print(sys.prefix)")

.PHONY: build
build:
	meson compile -C $(BUILD_DIR)

.PHONY: test
test:
	meson test -C $(BUILD_DIR) -v
	# $(PY) ./tests/test_elem_wise.py

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	make reconf
