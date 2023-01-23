import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Conv1D
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, MaxPool2D, LayerNormalization, BatchNormalization
# from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.applications import mobilenet


def create_network(settings):
    net = PlaNet(settings)
    return net


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()

    def create(self):
        self._create()

    def call(self, x):
        return self._internal_call(x)

    def _create(self):
        raise NotImplementedError

    def _internal_call(self):
        raise NotImplementedError


class PlaNet(Network):
    def __init__(self, config):
        super(PlaNet, self).__init__()
        self.config = config
        self._create(input_size=(self.config.img_height,
                                 self.config.img_width,
                                 3 * self.config.use_rgb + 3*self.config.use_depth))

    def _create(self, input_size, has_bias=True):
        """Init.
        Args:
            input_size (float): size of input
            has_bias (bool, optional): Defaults to True. Conv1d bias?
        """
        if self.config.use_rgb or self.config.use_depth:
            self.backbone = [mobilenet.MobileNet(include_top=False, weights='imagenet',
                                                 input_shape=input_size,
                                                 pooling=None)]
            if self.config.freeze_backbone:
                self.backbone[0].trainable = False

            # reduce a bit the size
            self.resize_op = [Conv1D(128, kernel_size=1, strides=1, padding='valid', dilation_rate=1,
                                     use_bias=has_bias)]

            f = 1.0
            self.img_mergenet = [Conv1D(int(128 * f), kernel_size=2, strides=1, padding='same',
                                   dilation_rate=1),
                                 LeakyReLU(alpha=1e-2),
                                 Conv1D(int(64 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                                 LeakyReLU(alpha=1e-2),
                                 Conv1D(int(64 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                                 LeakyReLU(alpha=1e-2),
                                 Conv1D(int(32 * f), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                                 LeakyReLU(alpha=1e-2)]

            self.resize_op_2 = [Conv1D(self.config.modes, kernel_size=3, strides=1, padding='valid', dilation_rate=1,
                                     use_bias=has_bias)]


        g = 1.0
        self.states_conv = [Conv1D(int(64 * g), kernel_size=2, strides=1, padding='same',
                                   dilation_rate=1),
                            LeakyReLU(alpha=.5),
                            Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                            LeakyReLU(alpha=.5),
                            Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1),
                            LeakyReLU(alpha=.5),
                            Conv1D(int(32 * g), kernel_size=2, strides=1, padding='same', dilation_rate=1)]

        self.resize_op_3 = [Conv1D(self.config.modes, kernel_size=3, strides=1, padding='valid', dilation_rate=1,
                                     use_bias=has_bias)]

        # State dim = 3 (x,y,z) +  alpha
        if len(self.config.predict_state_number) == 0:
            out_len = self.config.out_seq_len
        else:
            out_len = 1
        output_dim = self.config.state_dim * out_len + 1

        g = 1.0
        self.plan_module = [Conv1D(int(64 * g), kernel_size=1, strides=1, padding='valid'),
                            LeakyReLU(alpha=.5),
                            Conv1D(int(128 * g), kernel_size=1, strides=1, padding='valid'),
                            LeakyReLU(alpha=.5),
                            Conv1D(int(128 * g), kernel_size=1, strides=1, padding='valid'),
                            LeakyReLU(alpha=.5),
                            Conv1D(output_dim, kernel_size=1, strides=1,
                                   padding='same')]

    def _conv_branch(self, image):
        x = self._pf(image)
        for f in self.backbone:
            x = f(x)
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))  # (batch_size, MxM, C)
        for f in self.resize_op:
            x = f(x)
        # x [batch_size, M, M, 128]
        x = tf.reshape(x, (x.shape[0], -1))  # (batch_size, MxMx128)
        return x

    def _image_branch(self, img_seq):
        img_fts = tf.map_fn(self._conv_branch,
                            elems=img_seq,
                            parallel_iterations=self.config.seq_len) # (seq_len, batch_size, modes, channels)
        # img_fts (seq_len, batch_size, MxMxC)
        img_fts = tf.transpose(img_fts, (1,0,2)) # batch_size, seq_len, MxMx128
        x = img_fts
        for f in self.img_mergenet:
            x = f(x)
        # final x (batch_size, seq_len, 64)
        x = tf.transpose(x, (0,2,1)) # (batch_size, 64, seq_len)
        for f in self.resize_op_2:
            x = f(x)
        # final x (batch_size, 64, modes)
        x = tf.transpose(x, (0,2,1)) # (batch_size, modes, 64)
        return x

    def _imu_branch(self, embeddings):
        x = embeddings  # [B, seq_len, D]
        for f in self.states_conv:
            x = f(x)
        x = tf.transpose(x, (0,2,1)) # (batch_size, 32, seq_len)
        for f in self.resize_op_3:
            x = f(x)
        # final x # [batch_size, 32, modes]
        x = tf.transpose(x, (0, 2, 1)) # (batch_size, modes, 32)
        return x

    def _plan_branch(self, embeddings):
        x = embeddings
        for f in self.plan_module:
            x = f(x)
        return x

    def _pf(self, images):
        return tf.keras.applications.mobilenet.preprocess_input(images)

    def _preprocess_frames(self, inputs):
        if self.config.use_rgb and self.config.use_depth:
            img_seq = tf.concat((inputs['rgb'], inputs['depth']),
                                axis=-1)  # (batch_size, seq_len, img_height, img_width, 6)
        elif self.config.use_rgb and (not self.config.use_depth):
            img_seq = inputs['rgb']  # (batch_size, seq_len, img_height, img_width, 3)
        elif self.config.use_depth and (not self.config.use_rgb):
            img_seq = inputs['depth']  # (batch_size, seq_len, img_height, img_width, 1)
        else:
            return None
        # One of them passed, so need to process it
        img_seq = tf.transpose(img_seq, (1, 0, 2, 3, 4))  # (seq_len, batch_size, img_height, img_width, N)
        img_embeddings = self._image_branch(img_seq)
        return img_embeddings

    def _internal_call(self, inputs):
        if self.config.use_position:
            imu_obs = inputs['imu']
        else:
            # always pass z
            imu_obs = inputs['imu'][:, :, 3:]
        if (not self.config.use_attitude):
            if self.config.use_position:
                print("ERROR: Do not use position without attitude!")
                return
            else:
                imu_obs = inputs['imu'][:, :, 12:] # velocity and optionally body rates
        imu_embeddings = self._imu_branch(imu_obs)
        img_embeddings = self._preprocess_frames(inputs)
        if img_embeddings is not None:
            total_embeddings = tf.concat((img_embeddings, imu_embeddings), axis=-1)  # [B, modes, MxM + 64]
        else:
            total_embeddings = imu_embeddings
        output = self._plan_branch(total_embeddings)
        return output
