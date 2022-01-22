import numpy as np
import data_input as dinput
import ai_framework as ai

img_file = open("chardata/train-images.idx3-ubyte", "rb")
lbl_file = open("chardata/train-labels.idx1-ubyte", "rb")

mnist = dinput.dataset(img_file, lbl_file)

all_chars = [dinput.example(mnist, n) for n in range(len(mnist))]

print(mnist, len(mnist), mnist.width, mnist.height)

[print(char) for char in all_chars[:3]]

#L1 = ai.layer(mnist.pixel_count, 10)

#all_VN1 = [L1.activate(char) for char in all_chars[:1]]

#print(L1.w)

#print(all_VN1, len(all_VN1))