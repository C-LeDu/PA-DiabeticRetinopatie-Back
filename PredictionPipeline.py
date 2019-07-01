import keras.backend as K
import matplotlib as mpl
import numpy as np
import skimage
import tensorflow as tf
from PIL import Image
from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.applications.inception_v3 import preprocess_input
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, multiply, Lambda
from keras.models import Model
from skimage import color
from tensorflow.python.keras import backend as Kt


class PredictionPipeline:
    in_shape = (512, 512, 3)
    class_nb = 2
    weight_path = "{}_weights.best.hdf5".format('retina')

    def __init__(self):
        self.model = self.model_creator(self.in_shape, self.class_nb, self.weight_path)

    def predict_image(self, path):
        img = self.open_image(path)
        prediction = self.model.predict(np.expand_dims(img, axis=0))
        img_heatmap = self.grad_cam(img, prediction)
        return prediction, Image.fromarray(img_heatmap)

    @staticmethod
    def pre_processing(X):
        color_mode = 'rgb'
        out_size = (512, 512)

        with tf.name_scope('image_augmentation'):
            with tf.name_scope('input'):
                X = tf.image.decode_png(tf.read_file(X), channels=3 if color_mode == 'rgb' else 0)
                X = tf.image.resize_images(X, out_size)
            with tf.name_scope('augmentation'):
                return preprocess_input(X)

    @staticmethod
    def model_creator(in_shape, class_nb, weight_path):
        in_lay = Input(in_shape)
        base_pretrained_model = PTModel(input_shape=in_shape, include_top=False,
                                        weights=None)
        base_pretrained_model.trainable = False
        pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
        pt_features = base_pretrained_model(in_lay)
        bn_features = BatchNormalization()(pt_features)

        attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(Dropout(0.5)(bn_features))
        attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = Conv2D(8, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = Conv2D(1,
                            kernel_size=(1, 1),
                            padding='valid',
                            activation='sigmoid')(attn_layer)

        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                       activation='linear', use_bias=False, weights=[up_c2_w], name='outcnn')
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)

        mask_features = multiply([attn_layer, bn_features])
        gap_features = GlobalAveragePooling2D()(mask_features)
        gap_mask = GlobalAveragePooling2D()(attn_layer)

        gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
        gap_dr = Dropout(0.25)(gap)
        dr_steps = Dropout(0.25)(Dense(128, activation='relu')(gap_dr))
        out_layer = Dense(class_nb, activation='softmax')(dr_steps)
        retina_model = Model(inputs=[in_lay], outputs=[out_layer])

        retina_model.load_weights(weight_path)
        return retina_model

    def open_image(self, path):
        img = self.pre_processing(path)
        sess = Kt.get_session()
        img = sess.run(img)
        img = np.copy(img)
        return img

    def grad_cam(self, img, prediction, layer_output='outcnn', ratio=1.2):
        for attn_layer in self.model.layers:
            c_shape = attn_layer.get_output_shape_at(0)
            if len(c_shape) == 4:
                if c_shape[-1] == 1:
                    print(attn_layer)
                    break

        class_idx = np.argmax(prediction[0])
        class_output = self.model.output[:, class_idx]
        last_conv_layer = self.model.get_layer(layer_output)

        x = np.expand_dims(img, axis=0)

        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = np.copy(np.clip(img * 127 + 127, 0, 255).astype(np.uint8))
        heatmap = skimage.transform.resize(heatmap, (512, 512))
        cm_hot = mpl.cm.get_cmap('hsv')
        heatmap = cm_hot(heatmap)[:, :, :3]
        heatmap = np.uint8(255 * heatmap)
        img_hsv = color.rgb2hsv(img)
        color_mask_hsv = color.rgb2hsv(heatmap)
        img_hsv[..., 0] = color_mask_hsv[..., 0]
        img_hsv[..., 1] = color_mask_hsv[..., 1] * ratio
        superimposed_img = color.hsv2rgb(img_hsv)
        return superimposed_img
