from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense, Dropout,Lambda,ActivityRegularization
import tensorflow as tf
from keras.engine.topology import Layer
from keras import regularizers


class DenseLayerAutoencoder(Dense):
    def __init__(self, layer_sizes, l2_normalize=False, dropout=0.0, *args, **kwargs):
        self.layer_sizes = layer_sizes
        self.l2_normalize = l2_normalize
        self.dropout = dropout
        self.kernels = []
        self.biases = []
        self.biases2 = []
        self.uses_learning_phase = True
        self.shrink_thres = 1.5/layer_sizes[0]
        super().__init__(units=1, *args, **kwargs)  # 'units' not used

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        for i in range(len(self.layer_sizes)):

            self.kernels.append(
                self.add_weight(
                    shape=(
                        input_dim,
                        self.layer_sizes[i]),
                    initializer=self.kernel_initializer,
                    name='ae_kernel_{}'.format(i),
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(
                        shape=(
                            self.layer_sizes[i],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias_{}'.format(i),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))
            input_dim = self.layer_sizes[i]

        if self.use_bias:
            for n, i in enumerate(range(len(self.layer_sizes)-2, -1, -1)):
                self.biases2.append(
                    self.add_weight(
                        shape=(
                            self.layer_sizes[i],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias2_{}'.format(n),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))
            self.biases2.append(self.add_weight(
                        shape=(
                            input_shape[-1],
                        ),
                        initializer=self.bias_initializer,
                        name='ae_bias2_{}'.format(len(self.layer_sizes)),
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint))

        self.built = True

    def call(self, inputs):
        return self.decode(self.encode(inputs))

    def _apply_dropout(self, inputs):
        dropped =  K.dropout(inputs, self.dropout)
        return K.in_train_phase(dropped, inputs)

    def hard_shrink_relu(self,input, lambd=0., epsilon=1e-12):
        output = (K.relu(input - lambd) * input) / (K.abs(input - lambd) + epsilon)
        return output


    def encode(self, inputs):
        latent = inputs
        for i in range(len(self.layer_sizes)):
            if self.dropout > 0:
                latent = self._apply_dropout(latent)
            latent = K.dot(latent, self.kernels[i])
            if self.use_bias:
                latent = K.bias_add(latent, self.biases[i])
            if self.activation is not None:
                latent = self.activation(latent)
        if self.l2_normalize:
            latent = latent / K.l2_normalize(latent, axis=-1)
        att_weight0 = K.softmax(latent, axis=1)
        att_weight = self.hard_shrink_relu(att_weight0, lambd=self.shrink_thres)
        # <start> 进行l1范数归一化——>9月12日发现效果不好
        # att_weight_norm = tf.norm(att_weight0, ord=1, axis=1) # 是一个常数
        # print("att_weight_norm.shape", att_weight_norm.shape)
        # denominator_invert = Lambda(lambda x: K.pow(K.maximum(x, K.constant(1e-12)), -1))(att_weight_norm)
        # denominator_invert = K.expand_dims(denominator_invert, axis=1)
        # # denominator_invert = K.repeat_elements(denominator_invert, rep=self.layer_sizes[0], axis=1)
        # print("denominator_invert.shape",denominator_invert.shape)
        # att_weight_l1_0 = att_weight * denominator_invert
        # <end> 进行l1范数归一化——>9月12日发现效果不好
        def l1_reg(weight_matrix):
            return 0.0002 * K.sum(-weight_matrix * K.log(weight_matrix))
        att_weight=ActivityRegularizationtry(activity_regularizer=l1_reg)(att_weight)
        return att_weight

    def decode(self, latent):
        recon = latent
        for i in range(len(self.layer_sizes)):
            if self.dropout > 0:
                recon = self._apply_dropout(recon)
            recon = K.dot(recon, K.transpose(self.kernels[len(self.layer_sizes) - i - 1]))
            if self.use_bias:
                recon = K.bias_add(recon, self.biases2[i])
            if self.activation is not None:
                recon = self.activation(recon)
        return recon

    def get_config(self):
        config = {
            'layer_sizes': self.layer_sizes
        }
        base_config = super().get_config()
        base_config.pop('units', None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ActivityRegularizationtry(Layer):
    def __init__(self,activity_regularizer=None, **kwargs):
        super(ActivityRegularizationtry, self).__init__(**kwargs)
        self.supports_masking = True
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def get_config(self):
        config = {'activity_regularizer': regularizers.serialize(self.activity_regularizer)}
        base_config = super(ActivityRegularizationtry, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))