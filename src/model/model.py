# -*- coding utf-8 -*-

from typing import Optional, Dict, Any, List

from keras.api.layers import (Layer, Conv2D, Dense,
							  MaxPooling2D, Input, Flatten)

from keras.api.models import Model as _Model, load_model, save_model
from keras.api.losses import BinaryCrossentropy
from keras.api.metrics import Precision
from keras.api.optimizers import Adam
from keras.api.utils import Progbar

from tensorflow._api.v2.train import Checkpoint

from numpy.typing import NDArray

from model.data import Data

import tensorflow as tf

from os import path
import os, inspect

class _L1Dist(Layer):

	def __init__(self, **kwargs: Dict[str, Any]) -> None:
		super().__init__()

	def __repr__(self, /) -> str:
		return f'<{type(self).__name__}@[{id(self)}]>'
	
	def __str__(self, /) -> str:
		return self.__repr__()

	def call(self, input_embedding: _Model, validation_embedding: _Model) -> Any:

		if isinstance(input_embedding, list):
			input_embedding = input_embedding[0]
		
		if isinstance(validation_embedding, list):
			validation_embedding = validation_embedding[0]

		return tf.math.abs(input_embedding - validation_embedding)

class Model:
	
	def __init__(self, /, *, data: Optional[Data], compiled_path: Optional[str]) -> None:

		self._data: Optional[Data] = data
		self._compiled_model_path: Optional[str] = compiled_path

		self._loss_function	: Optional[BinaryCrossentropy]	= None
		self._optimizer		: Optional[Adam]				= None
		self._checkpoint	: Optional[Checkpoint]			= None

		if self._data is not None and not self._data.initialized:
			self._data.initialize()

		if self._compiled_model_path is not None:
			self._load_compiled_model()
		else:
			Model._set_gpu_growth()

		self._model: Optional[_Model] = None

	def __repr__(self, /) -> str:
		return f'<{type(self).__name__}@[{id(self)}]>'
	
	def __str__(self, /) -> str:
		return self.__repr__()
	
	def _load_compiled_model(self, /) -> None:
		self._model = load_model(self._compiled_model_path)

	def save_compiled_model(self, /, *, filename: str) -> None:

		os.makedirs(self.trained_directory, exist_ok=True)

		save_model(
			self._model,
			path.join(self.trained_directory, filename),
			overwrite=True
		)

	@property
	def data(self, /) -> Data:
		return self._data
	
	@property
	def trained_directory(self, /) -> str:
		return 'trained'
	
	@property
	def compiled_model(self, /) -> Optional[_Model]:
		return self._model
	
	@property
	def compiled_model_path(self, /) -> str:
		return self._compiled_model_path
	
	@property
	def loss_function(self, /) -> Optional[BinaryCrossentropy]:
		return self._loss_function
	
	@property
	def optimizer(self, /) -> Optional[Adam]:
		return self._optimizer
	
	@property
	def learning_rate(self, /) -> float:
		return 1e-4 # 0.0001
	
	@property
	def checkpoint_directory(self, /) -> str:
		path: str = './training_checkpoints'
		os.makedirs(path, exist_ok=True)
		return path

	@property
	def checkpoint_prefix(self, /) -> str:
		return path.join(self.checkpoint_directory, 'ckpt_')
	
	@property
	def checkpoint(self, /) -> Optional[Checkpoint]:
		return self._checkpoint

	def _build_checkpoint(self, /) -> Checkpoint:
		if self.optimizer is None or self._model is None:
			raise RuntimeError(
				f'Called {inspect.currentframe().f_code.co_name}'
				+ ' without having a built a model instance.'
			)

		return Checkpoint(
			opt=self.optimizer,
			siamese_model=self._model
		)
	
	def _build_embedding_architecture(self, /) -> _Model:

		input_layer = Input(shape=(100,100,3), name='input_layer')

		c1 = Conv2D(64, (10,10), activation='relu')(input_layer)
		m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
		c2 = Conv2D(128, (7,7), activation='relu')(m1)
		m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
		c3 = Conv2D(128, (4,4), activation='relu')(m2)
		m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

		c4 = Conv2D(256, (4,4), activation='relu')(m3)

		f1 = Flatten()(c4)

		d1 = Dense(4096, activation='sigmoid')(f1)

		return _Model(
			inputs=[input_layer],
			outputs=[d1],
			name='embedded'
		)
	
	def _build_architecture(self, embedded_model: _Model, /) -> None:
		anchor_input		= Input(shape=(100,100,3), name='anchor')
		validation_input	= Input(shape=(100,100,3), name='validation')

		siamese_layer = _L1Dist()
		siamese_layer.name = 'L1Distance'

		distances = siamese_layer(
			embedded_model(anchor_input),
			embedded_model(validation_input)
		)

		classifier = Dense(1, activation='sigmoid')(distances)
		
		self._model = _Model(
			inputs=[anchor_input, validation_input],
			outputs=[classifier], name='architecture'
		)

		self._loss_function	= self._build_loss()
		self._optimizer		= self._build_optimizer()
		self._checkpoint	= self._build_checkpoint()

	def build_model(self, /) -> None:
		self._build_architecture(
			self._build_embedding_architecture()
		)

	def _build_loss(self, /) -> BinaryCrossentropy:
		return BinaryCrossentropy()

	def _build_optimizer(self, /) -> Adam:
		return Adam(self.learning_rate)
	
	@tf.function
	def _train_step(self, batch: List) -> Any:
		''' Compiles into a callable TensorFlow graph '''
		with tf.GradientTape() as tape:
			X: NDArray = batch[:2]
			y: NDArray = batch[2]

			if self._model is None or self.loss_function is None:
				raise RuntimeError(
					f'Called {inspect.currentframe().f_code.co_name}'
					+ ' without having a built a model instance.'
				)

			yhat = self._model(X, training=True)
			loss = self.loss_function(y, yhat)

		grad = tape.gradient(loss, self._model.trainable_variables)

		if self.optimizer is None:
			raise RuntimeError(
				f'Called {inspect.currentframe().f_code.co_name}'
				+ ' without having a built a model instance.'
			)
		
		self.optimizer.apply_gradients(
			zip(grad, self._model.trainable_variables)
		)

		return loss
	
	def train(self, /, *, EPOCHS: Optional[int] = 50) -> None:

		if self.checkpoint is None:
			raise RuntimeError(
				f'Called {inspect.currentframe().f_code.co_name}'
				+ ' without having a built a model instance.'
			)
		
		if self.data is None or not self.data.initialized:
			raise RuntimeError(
				f'Called {inspect.currentframe().f_code.co_name}'
				+ ' without having a built a data instance.'
			)

		for epoch in range(1, EPOCHS + 1):

			print(f'\n Epoch {epoch}/{EPOCHS}')
			bar: Progbar = Progbar(len(self.data.train_partition))

			for i, batch in enumerate(self.data.train_partition):
				self._train_step(batch)
				bar.update(i + 1)

			if epoch % 10 == 0:
				self.checkpoint.save(
					file_prefix=self.checkpoint_prefix
				)
	
	def summary(self, /) -> None:
		self._model.summary()

	def evaluate(self, /) -> None:

		test_input, test_val, y_true = \
			self.data.test_partition.as_numpy_iterator().next()

		print('Accuracy: {}'.format(
			Precision().update_state(
				y_true, self._model.predict(
					[test_input, test_val])
				).result()
			)
		)

	def predict(self, X: NDArray, /) -> NDArray:
		return self._model.predict(X)

	@staticmethod
	def _set_gpu_growth() -> None:
		''' Avoid OOM errors by setting GPU memory consumption growth'''
		for gpu in tf.config.experimental.list_physical_devices('GPU'):
			tf.config.experimental.set_memory_growth(gpu, True)

	@classmethod
	def from_trained(cls, compiled_model_path: str, /) -> object:
		return cls(data=None, compiled_path=compiled_model_path)
