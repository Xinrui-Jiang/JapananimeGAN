import tensorflow as tf
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def generate_dataloader(data_dir,img_size=256,batch_size=1):
    num_data = len(glob(data_dir + '/*.jpg'))
    print('number of images in this dataset: '+str(num_data))
    X = []
    for img_path in tqdm(glob(data_dir + '/*.jpg')):
        X.append(img_path)

    X = tf.data.Dataset.from_tensor_slices(X)
    X = X.shuffle(buffer_size=num_data)
    X = X.repeat()
    X.reshuffleEachIteration = True
    data_loader = X.batch(batch_size, drop_remainder=True).prefetch(1)

    return iter(data_loader)
