# -*- coding utf-8 -*-

from model.model import Model
from sys import argv

def main(compiled_model_path: str) -> None:
	'''
	This will load the trained model
	Additional logic should be manually added
	'''
	model: Model = Model.from_trained(compiled_model_path)

	model.summary()
	model.evaluate()

	# model.predict(...)

if __name__ == '__main__':
	if len(argv < 2):
		raise RuntimeError(
			'Expected 1 argument: ' +
			'<compiled_model_path>'
		)

	main(argv[1])
