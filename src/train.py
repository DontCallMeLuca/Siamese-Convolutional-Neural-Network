# -*- coding utf-8 -*-

from model.model import Model
from model.data import Data

def main() -> None:

	'''
	Will train the a model instance.
	Consider additional logic such as epoch count.
	'''
	
	data: Data = Data()

	if not data.initialized:
		data.initialize()

	model: Model = Model(data=data)

	model.build_model()

	model.train()

	model.evaluate()

	model.save_compiled_model(filename='trained')

if __name__ == '__main__':
	main()
