# -*- coding utf-8 -*-

from keras.api.utils import image_dataset_from_directory
from tensorflow._api.v2.data import Dataset as _Dataset
from tensorflow._api.v2.data import NumpyIterator
from typing import Optional, Final, List, Tuple
from dataclasses import dataclass
from tensorflow import Tensor
from pathlib import Path
import tensorflow as tf
import os, inspect

@dataclass
class _PipelineWrapper:

	_pipeline: _Dataset
	_iterator: Optional[NumpyIterator] = None

	def __post_init__(self, /) -> None:
		if self._iterator is None:
			self._iterator = self._pipeline.as_numpy_iterator()

class Data:

	def __init__(self, /, *, path: Optional[str] = './data') -> None:

		self._data_path: Final[str] = os.path.realpath(path)

		self._clean(os.listdir(self.anchor_path), self.anchor_path)
		self._clean(os.listdir(self.negative_path), self.negative_path)
		self._clean(os.listdir(self.positive_path), self.positive_path)

		self._anchor_pipeline: _PipelineWrapper = _PipelineWrapper(
			image_dataset_from_directory(self.anchor_path)
		)

		self._negative_pipeline: _PipelineWrapper = _PipelineWrapper(
			image_dataset_from_directory(self.negative_path)
		)

		self._positive_pipeline: _PipelineWrapper = _PipelineWrapper(
			image_dataset_from_directory(self.positive_path)
		)

		self._dataset: Optional[_Dataset] = None

		self._train_partition: Optional[_Dataset] = None
		self._test_partition: Optional[_Dataset] = None

		self._initialized: bool = False

	def __repr__(self, /) -> str:
		return f'<{type(self).__name__}@[{id(self)}]>'
	
	def __str__(self, /) -> str:
		return self.__repr__()
	
	@property
	def initialized(self, /) -> bool:
		return self._initialized
	
	@property
	def dataset(self, /) -> Optional[_Dataset]:
		return self._dataset
	
	@property
	def train_partition(self, /) -> Optional[_Dataset]:
		return self._train_partition
	
	@property
	def test_partition(self, /) -> Optional[_Dataset]:
		return self._test_partition

	@property
	def pipelines(self, /) -> List[_PipelineWrapper]:
		return [
			self._anchor_pipeline,
			self._negative_pipeline,
			self._positive_pipeline
		]
	
	@property
	def data_path(self, /) -> str:
		return self._data_path
	
	@property
	def anchor_path(self, /) -> str:
		return os.path.join(self.data_path, 'anchor')
	
	@property
	def negative_path(self, /) -> str:
		return os.path.join(self.data_path, 'negative')
	
	@property
	def positive_path(self, /) -> str:
		return os.path.join(self.data_path, 'positive')
	
	@property
	def valid_fmts(self, /) -> List[str]:
		return ['jpeg','jpg', 'bmp', 'png']
	
	def _clean(self, files: List[str], path: str) -> None:
		for file in files:
			file = os.path.join(path, file)
			try:
				fmt: Final[str] = Path(file).suffix.lstrip('.')
				if fmt not in self.valid_fmts:
					os.remove(file)
			except (OSError, FileNotFoundError, IsADirectoryError): 
				os.remove(file)

	@staticmethod
	def _preprocess_method(file: str, /) -> Tensor:
		pixel_matrix: Tensor = tf.io.decode_image(tf.io.read_file(file))
		return tf.image.resize(pixel_matrix, (100,100)) / 255.0

	def _preprocess_pipelines(self, /) -> None:
		for pipeline in self.pipelines:
			pipeline.map(Data._preprocess_method)

	def _create_labelled_dataset(self, /) -> None:
		prepped_positives: _Dataset = _Dataset.zip((
			self._anchor_pipeline._pipeline,
			self._positive_pipeline._pipeline,
			_Dataset.from_tensor_slices(
				tf.ones(len(self._anchor_pipeline._pipeline))
			)
		))

		prepped_negatives: _Dataset = _Dataset.zip((
			self._anchor_pipeline._pipeline,
			self._negative_pipeline._pipeline,
			_Dataset.from_tensor_slices(
				tf.zeros(len(self._anchor_pipeline._pipeline))
			)
		))

		self._dataset = prepped_positives.concatenate(prepped_negatives)

	@staticmethod
	def _preprocess_twin(input_file: str, val_file: str, label: float) -> Tuple[Tensor, Tensor, float]:
		return (Data._preprocess_method(input_file), Data._preprocess_method(val_file), label)
	
	def _build_dataloader_pipeline(self, /) -> None:
		
		if self._dataset is None:
			raise RuntimeError(
				f'Called {inspect.currentframe().f_code.co_name}'
				+ ' without having a built dataset instance.'
			)

		self._dataset = (self._dataset.map(Data._preprocess_twin)).cache()
		self._dataset = self.dataset.shuffle(buffer_size=1024)

	def _train_test_split(self, /) -> None:

		if self._dataset is None:
			raise RuntimeError(
				f'Called {inspect.currentframe().f_code.co_name}'
				+ ' without having a built dataset instance.'
			)

		self._train_partition = self._dataset.take(round(len(self._dataset)*.7))
		self._train_partition = self._train_partition.batch(16)
		self._train_partition = self._train_partition.prefetch(8)

		self._test_partition = self._dataset.skip(round(len(self._dataset)*.7))
		self._test_partition = self._test_partition.take(round(len(self._dataset)*.3))
		self._test_partition = self._test_partition.batch(16)
		self._test_partition = self._test_partition.prefetch(8)

	def initialize(self, /) -> None:
		self._preprocess_pipelines()
		self._create_labelled_dataset()
		self._build_dataloader_pipeline()
		self._train_test_split()
		self._initialized = True
