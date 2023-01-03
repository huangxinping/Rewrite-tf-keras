from random import sample
import tensorflow as tf
import os
import numpy as np


class RewriteGenerator(tf.keras.utils.Sequence):

    def __init__(self, char_size=80, batch_size=32, images_path=None, shuffle=True):
        super(RewriteGenerator, self).__init__()
        self.char_size = char_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.images_path = images_path
        self.image_paths = [path for path in os.listdir(
            images_path) if path.endswith('.jpg') or path.endswith('.png')]
        self.n = len(self.image_paths)
        self.max = self.__len__()
        self.iter = 0
        self._debug_image_path = None
        self.on_epoch_end()
        print(f'{self.n} images found in {images_path}')

    def __next__(self):
        if self.iter >= self.max:
            self.iter = 0
        result = self.__getitem__(self.iter)
        self.iter += 1
        return result
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths_per_step = [self.image_paths[k] for k in indexes]
        X, y = self.__data_generation(image_paths_per_step)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths_per_step):
        images = []
        lables = []
        for image_path in image_paths_per_step:
            self._debug_image_path = image_path
            image, label = self.load_image_and_labels(image_path)
            images.append(image)
            lables.append(label)
        images = np.asarray(images, dtype=np.float32)
        lables = np.asarray(lables, dtype=np.float32)
        return images, lables
    
    def load_image_and_labels(self, image_path):
        image = tf.keras.utils.load_img(os.path.join(self.images_path, image_path))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.preprocessing.image.smart_resize(image, (self.char_size, self.char_size*2))
        X = image[:, :self.char_size, :]
        y = image[:, self.char_size:, :]
        X = X/255.0
        y = y/255.0
        return X, y
    
    
if __name__ == '__main__':
    gen = RewriteGenerator(images_path='datasets/images')
    X, y = next(gen)
    print(X.shape, y.shape)