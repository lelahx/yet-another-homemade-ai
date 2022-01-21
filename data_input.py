import numpy as np

class dataset:
    def __init__(self, imagefile, labelfile):
        self.id_imgs = int.from_bytes(imagefile.read(4), byteorder='big')
        self.id_lbls = int.from_bytes(labelfile.read(4), byteorder='big')
        assert self.id_imgs == 2051 and self.id_lbls == 2049

        self.size = int.from_bytes(imagefile.read(4), byteorder='big')
        assert self.size == int.from_bytes(labelfile.read(4), byteorder='big')

        self.labels = np.array(list(labelfile.read()))

        self.width = int.from_bytes(imagefile.read(4), byteorder='big')
        self.height = int.from_bytes(imagefile.read(4), byteorder='big')
        self.pixel_count = self.width * self.height

        set = []

        img = imagefile.read(self.pixel_count)
        while img:
            set.append(list(img))

            img = imagefile.read(self.pixel_count)
    
        self.images = np.array(set)
    
    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{repr(self.labels)}: {repr(self.images)}"
    
    def __str__(self):
        return f"{str(self.labels)}: {str(self.images)}"


class example:
    def __init__(self, dtst: dataset, index: int):
        assert index <= len(dtst)
        
        self.values = dtst.images[index]
        self.label = dtst.labels[index]

        self.width, self.height = dtst.width, dtst.height
        self.size = dtst.pixel_count

        out = np.zeros(10)
        out[self.label] = 1
        
        self.expected_output = out
    
    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{self.label}: {repr(self.values)}"
    
    def __str__(self):
        rows = [self.values[i: i + self.width] for i in range(0, len(self.values), self.width)]
        lines = ["".join(["▯" if p < 128 else "▮" for p in row]) for row in rows]
        fulltext = "\n".join(lines)

        return f"{self.label}:\n{fulltext}"