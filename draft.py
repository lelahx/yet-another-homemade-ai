import numpy as np
import data_input as dinput
import ai_framework as ai

img_file = "chardata/train-images.idx3-ubyte"
lbl_file = "chardata/train-labels.idx1-ubyte"

mnist = dinput.dataset(img_file, lbl_file)
all_examples = [mnist[i] for i in range(len(mnist))]

batches = [all_examples[i:i+100] for i in range(0, len(all_examples), 100)]

param_file = "test_NN.csv"

my_nn = ai.network(mnist.get_pixelcount(), 14, 14, 10)
my_nn.store(param_file)

print(all_examples[0], '\n')

print(my_nn.compute(all_examples[0].get_pixelvec()))