# @Time    : 18-12-26 下午12:07
# @Author  : Zhou-Lin-yong
# @File    : nn.py
# @SoftWare: PyCharm
import lasagne
import numpy as np
import theano.tensor as T
import theano as th
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
def l2normalize(layer, train_scale=True):
    W_param = layer.W
    s = W_param.get_value().shape
    if len(s) == 4:
        axes_to_sim = (1, 2, 3)
        dimshuffle_args = [0, 'x', 'x', 'x']
        k = s[0]
    else:
        axes_to_sim = 0
        dimshuffle_args = ['x', 0]
        k = s[1]

    layer.W_scale = layer.add_param(lasagne.init.Constant(1.), (k,), name='W_scale',
                                    trainable=train_scale, regularizable=False)
    layer.W = W_param * (layer.W_scale / T.sqrt(1e-6 + T.sum(T.square(W_param),
                                    axis=axes_to_sim))).dimshuffle(*dimshuffle_args)

    return layer

# GaussianNoiseLayer 该类用于在层上添加噪声，当sigma=0或则测试数据时不用添加噪声
class GaussianNoiseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, use_last_noise=False, **kwargs):
        if deterministic or self.sigma == 0:  # 当sigma=0或则测试数据时不用添加噪声
            return input
        else:
            if not use_last_noise:
                self.noise = self._srng.normal(input.shape, avg=0.0, std=self.sigma)
            return input + self.noise

def relu(x):
    return T.maximum(x, 0)

def lrelu(x, a=0.2):
    return T.maximum(x, a*x)

class DenseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, theta=lasagne.init.Normal(0.1), b=lasagne.init.Constant(0.),
                 weight_scale=lasagne.init.Constant(1.), train_scale=False, nonlinearity=relu, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.theta = self.add_param(theta, (num_inputs, num_units), name='theta')
        self.weight_scale = self.add_param(weight_scale, (num_units, ), name='weight_scale', trainable=train_scale)
        self.W = self.theta * (self.weight_scale / T.sqrt(T.sum(T.square(self.theta), axis=0))).dimshuffle('x', 0)
        self.b = self.add_param(b, (num_units, ), name='b')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, init=False, deterministic=False, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        activation = T.dot(input, self.W)

        if init:
            ma = T.mean(activation, axis=0)
            activation -= ma.dimshuffle('x', 0)
            stdv = T.sqrt(T.mean(T.square(activation), axis=0))
            activation /= stdv.dimshuffle('x', 0)
            self.init_update = [(self.weight_scale, self.weight_scale / stdv), (self.b, -ma/stdv)]
        else:
            activation += self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)


class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), nonlinearity=relu, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False)
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        self.avg_batch_var = self.add_param(lasagne.init.Constant(1.), (k,), name="avg_batch_var", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            norm_features = (input-self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)) / T.sqrt(1e-6 + self.avg_batch_var).dimshuffle(*self.dimshuffle_args)
        else:
            batch_mean = T.mean(input,axis=self.axes_to_sum).flatten()
            centered_input = input-batch_mean.dimshuffle(*self.dimshuffle_args)
            batch_var = T.mean(T.square(centered_input),axis=self.axes_to_sum).flatten()
            batch_stdv = T.sqrt(1e-6 + batch_var)
            norm_features = centered_input / batch_stdv.dimshuffle(*self.dimshuffle_args)

            # BN updates
            new_m = 0.9*self.avg_batch_mean + 0.1*batch_mean
            new_v = 0.9*self.avg_batch_var + T.cast((0.1*input.shape[0])/(input.shape[0]-1),th.config.floatX)*batch_var
            self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]

        if hasattr(self, 'g'):
            activation = norm_features*self.g.dimshuffle(*self.dimshuffle_args)
        else:
            activation = norm_features
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(activation)

class Deconv2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, target_shape, filter_size, stride=(2, 2),
                 W=lasagne.init.Normal(0.05), b=lasagne.init.Constant(0.), nonlinearity=relu, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.target_shape = target_shape
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.filter_size = lasagne.layers.dnn.as_tuple(filter_size, 2)
        self.stride = lasagne.layers.dnn.as_tuple(stride, 2)
        self.target_shape = target_shape

        self.W_shape = (incoming.output_shape[1], target_shape[1], filter_size[0], filter_size[1])
        self.W = self.add_param(W, self.W_shape, name="W")
        if b is not None:
            self.b = self.add_param(b, (target_shape[1],), name="b")
        else:
            self.b = None

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(imshp=self.target_shape, kshp=self.W_shape, subsample=self.stride, border_mode='half')
        activation = op(self.W, input, self.target_shape[2:])

        if self.b is not None:
            activation += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return self.target_shape

def batch_norm(layer, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), **kwargs):
    """
    adapted from https://gist.github.com/f0k/f1a6bd3c8585c400c190
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    else:
        nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, b, g, nonlinearity=nonlinearity, **kwargs)


class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), train_g=False, init_stdv=1., nonlinearity=relu, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.init_stdv = init_stdv
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=train_g)
        if len(self.input_shape) == 4:
            self.axes_to_sum = (0, 2, 3)
            self.dimshuffle_args = ['x', 0, 'x', 'x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x', 0]

        # scale weights in layer below
        incoming.W_param = incoming.W
        # incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim == 4:
            if isinstance(incoming, Deconv2DLayer):
                W_axes_to_sum = (0, 2, 3)
                W_dimshuffle_args = ['x', 0, 'x', 'x']
            else:
                W_axes_to_sum = (1, 2, 3)
                W_dimshuffle_args = [0, 'x', 'x', 'x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x', 0]
        if g is not None:
            incoming.W = incoming.W_param * (
                        self.g / T.sqrt(1e-6 + T.sum(T.square(incoming.W_param), axis=W_axes_to_sum))).dimshuffle(
                *W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(
                1e-6 + T.sum(T.square(incoming.W_param), axis=W_axes_to_sum, keepdims=True))

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            inv_stdv = self.init_stdv / T.sqrt(T.mean(T.square(input), self.axes_to_sum))
            input *= inv_stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m * inv_stdv), (self.g, self.g * inv_stdv)]
        elif hasattr(self, 'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)

        return self.nonlinearity(input)

def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)