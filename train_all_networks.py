import logging

import torch
from pythonbasictools import DeepLib, log_device_setup, logs_file_setup

from dataset import DatasetId
from results_generation import get_training_params_space, train_all_params
import neurotorch as nt

if __name__ == '__main__':
	logs_file_setup(__file__)
	if torch.cuda.is_available():
		torch.cuda.set_per_process_memory_fraction(0.8)
	log_device_setup(deepLib=DeepLib.Pytorch)
	df = train_all_params(
		# training_params=get_training_params_space(),
		training_params={
			"dataset_id"              : [
				DatasetId.MNIST,
				# DatasetId.FASHION_MNIST
			],
			"input_transform"         : [
				# "linear",
				"NorseConstCurrLIF",
				# "ImgToSpikes",
				"const",
				"spylif",
			],
			"n_steps"                 : [
				8,
				# 16,
				# 32,
				# 64,
				# 100,
				# 1_000
			],
			"n_hidden_neurons"        : [
				# 16,
				# 32,
				# 64,
				# [64, 64],
				128,
				# 256,
				# [32, 32],
				# 32
			],
			# "spike_func"              : [SpikeFuncType.FastSigmoid, ],
			"input_learning_type"     : [
				nt.LearningType.NONE,
				# LearningType.BPTT,
			],
			"input_layer_type"        : [
				# LayerType.LIF,
				nt.LayerType.SpyLIF,
			],
			"hidden_layer_type"       : [
				nt.LayerType.LIF,
				# nt.LayerType.ALIF,
				# nt.LayerType.SpyLIF,
			],
			"readout_layer_type"      : [
				# nt.LayerType.LI,
				nt.LayerType.SpyLI,
			],
			"use_recurrent_connection": [
				False,
				# True
			],
			"learn_beta"              : [
				# True,
				False
			],
		},
		n_iterations=30,
		batch_size=1024,
		data_folder="data/tr_data_mnist_001",
		verbose=False,
		rm_data_folder_and_restart_all_training=False,
	)
	logging.info(df)

