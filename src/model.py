# -----------------------------------------------------------------------------
# -- MODEL
# -----------------------------------------------------------------------------
from keras import backend as K
from keras import regularizers
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.optimizers import Adam
from keras.models import Model as KerasModel 
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Input, Lambda, concatenate, Dense
from keras.callbacks import ModelCheckpoint

from globals import input_shape
from globals import embedding_size

from utils.loss import feature_loss


def l2_normalization(x):
	return K.l2_normalize(x, axis=-1)


class Model:
	BaseModel = MobileNetV2

	def __init__(self):
		self.base = Model.BaseModel(input_shape=input_shape, weigths='imagenet',
									include_top=False, pooling='avg')
		self.it = None

	def build():
		'''creates the model'''
		model = Model()

		# to the base model we add an embedding layer
		input_ = model.base.input
		x = model.base.output
		output_ = Dense(embedding_size)(x)
		image_embedder = KerasModel(input_, output_)

		# model input: 3 images
		input_a = Input(input_shape, name='anchor')
  		input_p = Input(input_shape, name='positive')
  		input_n = Input(input_shape, name='negative')

  		# l2 normalization
  		normalize = Lambda(l2_normalization, name="normalize")

  		# model output: 3 embedding
  		x = image_embedder(input_a)
  		output_a = normalize(x)
  		x = image_embedder(input_p)
  		output_p = normalize(x)
  		x = image_embedder(input_n)
  		output_n = normalize(x)
  		merged_output = concatenate([output_a, output_p, output_n], axis=-1)
		
		# the final model  		
  		model.it = KerasModel(inputs=[input_a, input_p, input_n], 
  							  outputs=merged_output)
		return model

	def summary():
		return self.it.summary()
	
	def compile(self):
		'''compiles the model'''
		# self.it.compile(optimizer=Adam(), loss=feature_loss)
		pass

	def train():
		'''trains the model'''
		pass