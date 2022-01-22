import numpy as np

class dataset:
    """
    Object which imports and holds all the relevant information and  data of a dataset.
    """

    def __init__(self, imagefile, labelfile):
        self.id_imgs, self.id_lbls = int.from_bytes(imagefile.read(4), byteorder='big'), int.from_bytes(labelfile.read(4), byteorder='big') # Gets the IDs of the image and label datasets in the first 4 bytes of each file
        assert self.id_imgs == 2051 and self.id_lbls == 2049 # Dataset files should contain these IDs

        self.size = int.from_bytes(imagefile.read(4), byteorder='big') # Reads the size of the dataset in the next 4 bytes
        assert self.size == int.from_bytes(labelfile.read(4), byteorder='big') # Size should be the same in the image and label files

        self.labels = np.array(list(labelfile.read())) # Reads the labels in the rest of the label file and puts them in a vector

        self.width, self.height = int.from_bytes(imagefile.read(4), byteorder='big'), int.from_bytes(imagefile.read(4), byteorder='big') # Reads the width and height of the images in the next 4 + 4 bytes of the file
        self.pixel_count = self.width * self.height # Defines the pixel count of individual images from their dimensions

        set = []
        img = imagefile.read(self.pixel_count)
        while img: # img becomes 0 when the end of the file is reached, thus this loop runs until reaching it
            set.append(list(img)) # Creates a vector with all the pixel values of an image and appends it to a list

            img = imagefile.read(self.pixel_count)
    
        self.images = np.array(set) # Creates a matrix with all the values for each image on every line
    
    def __len__(self): # Returns the size of the dataset when calling the len function on it
        return self.size

    def __repr__(self): # Canonical string representation of the dataset
        return f"{repr(self.labels)}: {repr(self.images)}"
    
    def __str__(self): # Simple string representation of the dataset
        return f"{str(self.labels)}: {str(self.images)}"


class example:
    """
    Object which holds the data and label of an individual example of the dataset.
    """

    def __init__(self, dtst: dataset, index: int):
        assert index <= len(dtst) # The index of the example we want to extract should be less than the total size of the dataset
        
        self.values = dtst.images[index] # Assign the vector which contains the pixel values to a variable
        self.label = dtst.labels[index] # Assign the number label to a variable

        self.width, self.height = dtst.width, dtst.height # Width and height are the same as that of the dataset
        self.size = dtst.pixel_count # Size of a example/vector is the pixel count of an image

        out = np.zeros(10) # Creating a 10-value vectoer filled with zeros
        out[self.label] = 1 # Replacing the 0 by a 1 for the label-th value, corresponding to the digit it represents
        
        self.expected_output = out # This vector is the ideal output of the neural network for the example
    
    def __len__(self): # Returns the size (pixel count) of the example when calling the len function on it
        return self.size

    def __repr__(self): # Canonical string representation of the example
        return f"{self.label}: {repr(self.values)}"
    
    def __str__(self): # Simpler and visual string representation of the example
        rows = self.values.reshape((self.height, self.width)) # Reshapes the pixel values vedctor into a 28*28 matrix
        lines = ["".join(["▯" if p < 128 else "▮" for p in row]) for row in rows] # In every row, replaces values smaller than 128 with hollow boxes, else filled boxes. Then, joints the characters together into a single string
        fulltext = "\n".join(lines) # Joins the lines into a single string and inserts newlines between them

        return f"{self.label}:\n{fulltext}" # Returns the label of the example followed by its approximate text visualization