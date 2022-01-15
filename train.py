import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


import glob
import os
import numpy as np
from keras.layers import Input
from keras.models import Model
# from keras.optimizer_v1 import Adama
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from yolov3_body import body
from get_ytrue import generator
from get_loss import fn_loss
import keras.backend as K
K.clear_session()




batch_size = 2
classes = ["flower"]
input_size = 416
anchors = np.array([[8, 73], [8, 24], [13, 32], [19, 51], [35, 64], [25, 37], [22, 164], [95, 195], [57, 104]])
num_anchros = 3
num_classes = 1
pattern_shape = [13, 26, 52]
anchor_shape = [3, 3]

root = os.path.dirname(__file__)
ann_dir = os.path.join(root, "dataSet", "xml", "*.xml")
ann_fnames = glob.glob(ann_dir)
print(ann_fnames)
img_dir = os.path.join(root, "dataSet", "img")

num_train = int(len(ann_fnames)*0.9)
log_dir = "logs"

logging = TensorBoard(log_dir = log_dir)
checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5", monitor="val_loss",
                             save_weights_only=True, save_best_only=True, period=2)


model_input = Input(shape=(416, 416, 3))

model_output = body(model_input, num_anchors=num_anchros, num_classes=num_classes)

model = Model(model_input, model_output)

model.load_weights("./yolo_weights.h5", by_name=True, skip_mismatch=True)

freeze_layer = 249
for i in range(freeze_layer): model.layers[i].trainable = False



model.compile(optimizer=Adam(learning_rate=1e-3), loss=fn_loss)

model.fit_generator(generator(1, pattern_shape, anchor_shape, classes, ann_fnames[:num_train], input_size, anchors, img_dir),
                    steps_per_epoch=1,
                    validation_data=generator(1, pattern_shape, anchor_shape, classes, ann_fnames[num_train:], input_size, anchors, img_dir),
                    validation_steps=1,
                    epochs=3,
                    initial_epoch=0)