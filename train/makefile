
.PHONY: default
default: gen debug

.PHONY: gen
gen: llama_debug.py llama.py

.PHONY: debug
debug:
	python train.py

llama_debug.py: llama.gpp.py
	cat $< | sed "s/^ *# > /#/" | gpp -D DEBUG > $@
llama.py: llama.gpp.py
	cat $< | sed "s/^ *# > /#/" | gpp > $@

.PHONY: test
test: test_tokenizer test_model test_dataset test_train

.PHONY: test_tokenizer
test_tokenizer:
	pytest -s tokenizer.py

.PHONY: test_model
test_model:
	pytest -s model.py

.PHONY: test_dataset
test_dataset:
	pytest -s dataset.py

.PHONY: test_train
test_train:
	pytest -s train.py

.PHONY: train
train:
	python3 train.py --name="baseline_15m"
	python3 train.py --name="baseline_debug"