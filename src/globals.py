# -----------------------------------------------------------------------------
# -- GLOBALS: definition of shared constants
# -----------------------------------------------------------------------------

# the dimensionality of the model output.
# the number of neurons in the 'dense' layer must be this value
embedding_size = 128

training_samples   = 10000
validation_samples = 2000
test_samples       = 2000

# partition (split) indices:
training_partition   = 0
validation_partition = 1
testing_partition    = 2

# size of the input images
input_shape = (192, 192, 3)
