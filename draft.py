import numpy as np
import data_input as dinput
import ai_framework as ai

img_file = "chardata/train-images.idx3-ubyte"
lbl_file = "chardata/train-labels.idx1-ubyte"

#mnist = dinput.dataset(img_file, lbl_file)

#all_chars = [dinput.example(mnist, n) for n in range(len(mnist))]

#print(mnist, len(mnist), mnist.width, mnist.height)

#[print(char) for char in all_chars[:3]]


#a = ai.network(mnist.pixel_count, 2, 10)
b = ai.network(from_file='neuralnet_params.csv')

#b = ai.network(5, 6, 7)

#b.store()

print(b[0]['weights'])


lol = np.arange(5.0)

print(b.compute(lol))

#print(all_chars[0])


#print(a.run(all_chars[0].get_pixels()))





#L1 = ai.layer(mnist.pixel_count, 10)

#all_VN1 = [L1.activate(char) for char in all_chars[:1]]

#print(L1.w)

#print(all_VN1, len(all_VN1))