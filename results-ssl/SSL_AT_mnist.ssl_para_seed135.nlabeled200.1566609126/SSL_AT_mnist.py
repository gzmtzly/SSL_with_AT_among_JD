import os, time, argparse, shutil

import numpy as np
import theano, lasagne
import theano.tensor as T
import lasagne.layers as ll
import lasagne.nonlinearities as ln
import nn
from lasagne.init import Normal
from theano.sandbox.rng_mrg import MRG_RandomStreams
from parmesan.datasets import load_mnist_realval
from shortcuts import MLPConcatLayer, convlayer
from components.objectives import categorical_crossentropy_ssl_separated, categorical_crossentropy
import utils.paramgraphics as paramgraphics
import shortcuts

'''
parameters
'''

parser = argparse.ArgumentParser()
parser.add_argument("-ssl_para_seed", type=int, default=135)
parser.add_argument("-nlabeled", type=int, default=200)
args = parser.parse_args()
args = vars(args).items()
cfg = {}
for name, val in args:
    cfg[name] = val

filename_script = os.path.basename(os.path.realpath(__file__))
outfolder = os.path.join("results-ssl", os.path.splitext(filename_script)[0])
outfolder += '.'
for item in cfg:
    if item is not 'oldmodel':
        outfolder += item + str(cfg[item]) + '.'
    else:
        outfolder += 'oldmodel.'
outfolder += str(int(time.time()))

if not os.path.exists(outfolder):
    os.makedirs(outfolder)
    sample_path = os.path.join(outfolder, 'sample')
    os.makedirs(sample_path)
logfile = os.path.join(outfolder, 'logfile.log')
shutil.copy(os.path.realpath(__file__), os.path.join(outfolder, filename_script))


num_labelled = cfg['nlabeled']
ssl_para_seed = cfg['ssl_para_seed']

print('ssl_para_seed %d, num_labelled %d' % (ssl_para_seed, num_labelled))

rng = np.random.RandomState(ssl_para_seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# dataset
data_dir = './data/mnist.pkl.gz'
# flags
valid_flag = False

# pre-train
pre_num_epoch = 0 if num_labelled >= 100 else 30
pre_alpha_unlabeled_entropy = .3
pre_alpha_average = .3
pre_lr = 3e-4
pre_batch_size_lc = min(100, num_labelled)
pre_batch_size_uc = 500
# C
alpha_decay = 1e-4
alpha_labeled = 1.
alpha_unlabeled_entropy = .3
alpha_average = 1e-3
alpha_cla = 1.
# G
n_z = 100
alpha_cla_g = .1
epoch_cla_g = 300
# D
noise_D_data = .3
noise_D = .5
# optimization
b1_g = .5  # mom1 in Adam
b1_d = .5
b1_c = .5

# threshold
alpha_stage1 = 10
alpha_stage2 = 300 if num_labelled >= 50 else 100
num_epochs = 700 if num_labelled >= 50 else 400
optim_flag = 1 if num_labelled >= 50 else 0

# adjust batch size for different number of labeled data
batch_size_g = 200
batch_size_l_c = min(100, num_labelled)
batch_size_u_c = max(100, int(10000 / num_labelled))
batch_size_u_d = 400
batch_size_l_d = max(int(num_labelled / 100), 1)
lr = 1e-3

anneal_lr_epoch = alpha_stage2
anneal_lr_every_epoch = 1
anneal_lr_factor = .995
# data dependent
gen_final_non = ln.sigmoid
num_classes = 10
dim_input = (28, 28)
in_channels = 1
colorImg = False
generation_scale = False
z_generated = num_classes
weight_d2 = 1e-4
weight_d3 = 1e-4
# evaluation
vis_epoch = 10
eval_epoch = 1
print('alpha_stage1:', alpha_stage1)
print('alpha_stage2:', alpha_stage2)
print('pre_num_epoch:', pre_num_epoch)
print('num_epochs:', num_epochs)
print('optim_flag:', optim_flag)

'''
data
'''
train_x, train_y, valid_x, valid_y, eval_x, eval_y = load_mnist_realval(data_dir)
if valid_flag:
    eval_x = valid_x
    eval_y = valid_y
else:
    train_x = np.concatenate([train_x, valid_x])
    train_y = np.hstack((train_y, valid_y))
train_y = np.int32(train_y)
eval_y = np.int32(eval_y)
train_x = train_x.astype('float32')
eval_x = eval_x.astype('float32')
x_unlabelled = train_x.copy()

rng_data = np.random.RandomState(1)
inds = rng_data.permutation(train_x.shape[0])
train_x = train_x[inds]
train_y = train_y[inds]
x_labelled = []
y_labelled = []
for j in range(num_classes):
    x_labelled.append(train_x[train_y == j][:int(num_labelled / num_classes)])
    y_labelled.append(train_y[train_y == j][:int(num_labelled / num_classes)])

x_labelled = np.concatenate(x_labelled, axis=0)
y_labelled = np.concatenate(y_labelled, axis=0)

print('x_labelled:', x_labelled.shape)
print('y_labelled:', y_labelled.shape)
del train_x

if True:
    print('Size of training data', x_labelled.shape[0], x_unlabelled.shape[0])
    y_order = np.argsort(y_labelled)
    _x_mean = x_labelled[y_order]
    image = paramgraphics.mat_to_img(_x_mean.T, dim_input, tile_shape=(num_classes, int(num_labelled / num_classes)),
                                     colorImg=colorImg, scale=generation_scale,
                                     save_path=os.path.join(outfolder, 'x_l_' + str(ssl_para_seed) + '_AT-JD.png'))

pretrain_batches_train_uc = int(x_unlabelled.shape[0] / pre_batch_size_uc)
pretrain_batches_train_lc = int(x_labelled.shape[0] / pre_batch_size_lc)
n_batches_train_u_c = int(x_unlabelled.shape[0] / batch_size_u_c)
n_batches_train_l_c = int(x_labelled.shape[0] / batch_size_l_c)
n_batches_train_u_d = int(x_unlabelled.shape[0] / batch_size_u_d)
n_batches_train_l_d = int(x_labelled.shape[0] / batch_size_l_d)
n_batches_train_g = int(x_unlabelled.shape[0] / batch_size_g)

'''
models
'''
# symbols
sym_z_image = T.tile(theano_rng.uniform((z_generated, n_z)), (num_classes, 1))
sym_z_rand = theano_rng.uniform(size=(batch_size_g, n_z))
sym_x_u = T.matrix()
sym_x_u_d = T.matrix()
sym_x_u_g = T.matrix()
sym_x_l = T.matrix()
sym_y = T.ivector()
sym_y_g = T.ivector()
sym_x_eval = T.matrix()
sym_lr = T.scalar()
sym_alpha_cla_g = T.scalar()
sym_alpha_unlabel_entropy = T.scalar()
sym_alpha_unlabel_average = T.scalar()

shared_unlabel = theano.shared(x_unlabelled, borrow=True)
slice_x_u_g = T.ivector()
slice_x_u_d = T.ivector()
slice_x_u_c = T.ivector()

"""
classifier x2y: p_c(x, y) = p(x) p_c(y | x)
ll: lasagne.layers
"""
cla_in_x = ll.InputLayer(shape=(None, 28 ** 2))
cla_layers = [cla_in_x]
cla_layers.append(ll.ReshapeLayer(cla_layers[-1], (-1, 1, 28, 28)))
cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0.5, ps=2, n_kerns=32, d_kerns=(5, 5),
                            pad='valid', stride=1, W=Normal(0.05), nonlinearity=ln.rectify,
                            name='cla-1'))
cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0, ps=1, n_kerns=64, d_kerns=(3, 3),
                            pad='same', stride=1, W=Normal(0.05), nonlinearity=ln.rectify,
                            name='cla-2'))
cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0.5, ps=2, n_kerns=64, d_kerns=(3, 3),
                            pad='valid', stride=1, W=Normal(0.05), nonlinearity=ln.rectify,
                            name='cla-3'))
cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0, ps=1, n_kerns=128, d_kerns=(3, 3),
                            pad='same', stride=1, W=Normal(0.05), nonlinearity=ln.rectify,
                            name='cla-4'))
cla_layers.append(convlayer(l=cla_layers[-1], bn=True, dr=0, ps=1, n_kerns=128, d_kerns=(3, 3),
                            pad='same', stride=1, W=Normal(0.05), nonlinearity=ln.rectify,
                            name='cla-5'))
cla_layers.append(ll.GlobalPoolLayer(cla_layers[-1]))
cla_layers.append(ll.DenseLayer(cla_layers[-1], num_units=num_classes, W=lasagne.init.Normal(1e-2, 0),
                                nonlinearity=ln.softmax, name='cla-6'))

classifier = cla_layers[-1]  #


gen_in_z = ll.InputLayer(shape=(None, n_z))
gen_in_y = ll.InputLayer(shape=(None,))
gen_layers = [gen_in_z]
gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-1'))

gen_layers.append(ll.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500,
                                              nonlinearity=ln.softplus, name='gen-2'), name='gen-3'))
gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-4'))

gen_layers.append(ll.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=500,
                                              nonlinearity=ln.softplus, name='gen-5'), name='gen-6'))
gen_layers.append(MLPConcatLayer([gen_layers[-1], gen_in_y], num_classes, name='gen-7'))
gen_layers.append(nn.l2normalize(ll.DenseLayer(gen_layers[-1], num_units=28 ** 2,
                                               nonlinearity=gen_final_non, name='gen-8')))

dis_in_x = ll.InputLayer(shape=(None, 28 ** 2))
dis_in_y = ll.InputLayer(shape=(None,))
dis_layers = [dis_in_x]

dis_layers.append(nn.GaussianNoiseLayer(dis_layers[-1], sigma=noise_D_data, name='dis-1'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-2'))
dis_layers.append(nn.DenseLayer(dis_layers[-1], num_units=1000, name='dis-3'))

dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-4'))

dis_layers.append(nn.GaussianNoiseLayer(dis_layers[-1], sigma=noise_D, name='dis-5'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-6'))
dis_layers.append(nn.DenseLayer(dis_layers[-1], num_units=500, name='dis-7'))

dis_layers.append(nn.GaussianNoiseLayer(dis_layers[-1], sigma=noise_D, name='dis-8'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-9'))
dis_layers.append(nn.DenseLayer(dis_layers[-1], num_units=250, name='dis-10'))

dis_layers.append(nn.GaussianNoiseLayer(dis_layers[-1], sigma=noise_D, name='dis-11'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-12'))
dis_layers.append(nn.DenseLayer(dis_layers[-1], num_units=250, name='dis-13'))

dis_layers.append(nn.GaussianNoiseLayer(dis_layers[-1], sigma=noise_D, name='dis-14'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-15'))
dis_layers.append(nn.DenseLayer(dis_layers[-1], num_units=250, name='dis-16'))

dis_layers.append(nn.GaussianNoiseLayer(dis_layers[-1], sigma=noise_D, name='dis-17'))
dis_layers.append(MLPConcatLayer([dis_layers[-1], dis_in_y], num_classes, name='dis-18'))

dis1 = [nn.DenseLayer(dis_layers[-1], num_units=1, nonlinearity=ln.sigmoid, name='dis-19')]
dis2 = [nn.DenseLayer(dis_layers[-1], num_units=1, nonlinearity=ln.sigmoid, name='dis-19')]
dis3 = [nn.DenseLayer(dis_layers[-1], num_units=1, nonlinearity=ln.sigmoid, name='dis-19')]

dis1_layers = dis_layers + dis1
dis2_layers = dis_layers + dis2
dis3_layers = dis_layers + dis3

'''
objectives
'''
gen_out_x = ll.get_output(layer_or_layers=gen_layers[-1], inputs={gen_in_y: sym_y_g, gen_in_z: sym_z_rand},
                          deterministic=False)

cla_out_y_l = ll.get_output(cla_layers[-1], sym_x_l, deterministic=False)
cla_out_y = ll.get_output(layer_or_layers=cla_layers[-1], inputs=sym_x_u, deterministic=False)
cla_out_y_g = ll.get_output(cla_layers[-1], {cla_in_x: gen_out_x}, deterministic=False)
cla_out_y_g_hard = cla_out_y_g.argmax(axis=1)

cla_out_y_d = ll.get_output(cla_layers[-1], {cla_in_x: sym_x_u_d}, deterministic=False)
cla_out_y_d_hard = cla_out_y_d.argmax(axis=1)


dis_out_p = ll.get_output(layer_or_layers=dis1_layers[-1],
                          inputs={dis_in_x: T.concatenate([sym_x_l, sym_x_u_d], axis=0),
                                  dis_in_y: T.concatenate([sym_y, cla_out_y_d_hard], axis=0)},
                          deterministic=False)


cla_out_y_hard = cla_out_y.argmax(axis=1)
dis_out_p_c = ll.get_output(layer_or_layers=dis1_layers[-1],
                            inputs={dis_in_x: sym_x_u, dis_in_y: cla_out_y_hard},
                            deterministic=False)


dis_out_p_g = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                            deterministic=False)


dis_out_p_g_c = ll.get_output(layer_or_layers=dis1_layers[-1], inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                              deterministic=False)


dis2_out_p_g_c = ll.get_output(layer_or_layers=dis2_layers[-1],
                               inputs={dis_in_x: gen_out_x, dis_in_y: cla_out_y_g_hard},
                               deterministic=False)


dis2_out_p_c = ll.get_output(layer_or_layers=dis2_layers[-1],
                             inputs={dis_in_x: sym_x_u, dis_in_y: cla_out_y_hard},
                             deterministic=False)


dis2_out_p_g = ll.get_output(layer_or_layers=dis2_layers[-1],
                             inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                             deterministic=False)


dis3_out_p_c = ll.get_output(layer_or_layers=dis3_layers[-1],
                             inputs={dis_in_x: sym_x_u, dis_in_y: cla_out_y_hard},
                             deterministic=False)


dis3_out_p_g = ll.get_output(layer_or_layers=dis3_layers[-1],
                             inputs={dis_in_x: gen_out_x, dis_in_y: sym_y_g},
                             deterministic=False)


image = ll.get_output(gen_layers[-1], {gen_in_y: sym_y_g, gen_in_z: sym_z_image}, deterministic=False)

# for testing
cla_out_y_eval = ll.get_output(cla_layers[-1], sym_x_eval, deterministic=True)
accurracy_eval = (lasagne.objectives.categorical_accuracy(cla_out_y_eval, sym_y))
accurracy_eval = accurracy_eval.mean()


bce = lasagne.objectives.binary_crossentropy
############
dis_cost_p = bce(dis_out_p, T.ones(dis_out_p.shape)).mean()

dis_cost_p_c = bce(dis_out_p_c, T.zeros(dis_out_p_c.shape)).mean()

dis_cost_p_g = bce(dis_out_p_g, T.zeros(dis_out_p_g.shape)).mean()


dis_cost_p_g_c = bce(dis_out_p_g_c, T.zeros(dis_out_p_g_c.shape)).mean()

############

dis2_cost_p_c = bce(dis2_out_p_c, T.zeros(dis2_out_p_c.shape)).mean()

dis2_cost_p_g = bce(dis2_out_p_g, T.zeros(dis2_out_p_g.shape)).mean()

dis2_cost_p_g_c = bce(dis2_out_p_g_c, T.ones(dis2_out_p_g_c.shape)).mean()

############
dis3_cost_p_c = bce(dis3_out_p_c, T.ones(dis3_out_p_c.shape)).mean()

dis3_cost_p_g = bce(dis3_out_p_g, T.zeros(dis3_out_p_g.shape)).mean()

############
gen_cost_p_g = bce(dis_out_p_g, T.ones(dis_out_p_g.shape)).mean()
gen2_cost_p_g = bce(dis2_out_p_g, T.ones(dis2_out_p_g.shape)).mean()
gen3_cost_p_g_c = bce(dis3_out_p_g, T.ones(dis3_out_p_g.shape)).mean()

############
cla_cost_p_c = bce(dis_out_p_c, T.ones(dis_out_p_c.shape))  
cla2_cost_p_c = bce(dis2_out_p_c, T.ones(dis2_out_p_c.shape))
cla3_cost_p_c = bce(dis3_out_p_c, T.zeros(dis3_out_p_c.shape))


p = cla_out_y.max(axis=1)
cla_cost_p_c = (cla_cost_p_c * p).mean()
cla2_cost_p_c = (cla2_cost_p_c * p).mean()
cla3_cost_p_c = (cla3_cost_p_c * p).mean()

weight_decay_classifier = lasagne.regularization.regularize_layer_params_weighted({cla_layers[-1]: 1},
                                                                                  lasagne.regularization.l2)

cla_cost_cla = categorical_crossentropy_ssl_separated(predictions_l=cla_out_y_l, targets=sym_y, predictions_u=cla_out_y,
                                                      weight_decay=weight_decay_classifier, alpha_labeled=alpha_labeled,
                                                      alpha_unlabeled=sym_alpha_unlabel_entropy,
                                                      alpha_average=sym_alpha_unlabel_average,
                                                      alpha_decay=alpha_decay) 

pretrain_cla_loss = categorical_crossentropy_ssl_separated(predictions_l=cla_out_y_l, targets=sym_y,
                                                           predictions_u=cla_out_y,
                                                           weight_decay=weight_decay_classifier,
                                                           alpha_labeled=alpha_labeled,
                                                           alpha_unlabeled=pre_alpha_unlabeled_entropy,
                                                           alpha_average=pre_alpha_average,
                                                           alpha_decay=alpha_decay)  
pretrain_cost = pretrain_cla_loss


cla_cost_cla_g = categorical_crossentropy(predictions=cla_out_y_g, targets=sym_y_g)

'''##################D1###################'''

dis_cost = dis_cost_p + .5 * dis_cost_p_g + .5 * dis_cost_p_c + .5 * dis_cost_p_g_c
dis_cost_list = [dis_cost]

'''##################D2###################'''

dis2_cost = .5 * dis2_cost_p_g + .5 * dis2_cost_p_c + .5 * dis2_cost_p_g_c
dis2_cost_list = [dis2_cost]

'''##################D3###################'''

dis3_cost = .5 * dis3_cost_p_g + .5 * dis3_cost_p_c
dis3_cost_list = [dis3_cost]

'''##################G###################'''
gen_cost = .5 * gen_cost_p_g + weight_d2 * gen2_cost_p_g + weight_d3 * gen3_cost_p_g_c
gen_cost_list = [gen_cost]

'''##################C###################'''
# first stage
cla_cost1 = cla_cost_cla
cla_cost_list1 = [cla_cost1]

# second stage
cla_cost2 = .5 * cla_cost_p_c + alpha_cla * cla_cost_cla + weight_d2 * cla2_cost_p_c + weight_d3 * cla3_cost_p_c
cla_cost_list2 = [cla_cost2]

# third stage
cla_cost3 = .5 * cla_cost_p_c + alpha_cla * (cla_cost_cla + sym_alpha_cla_g * cla_cost_cla_g) +\
            weight_d2 * cla2_cost_p_c + weight_d3 * cla3_cost_p_c
cla_cost_list3 = [cla_cost3]

dis_params = ll.get_all_params(dis1_layers, trainable=True)
dis_grads = T.grad(dis_cost, dis_params)
dis_updates = lasagne.updates.adam(loss_or_grads=dis_grads, params=dis_params, learning_rate=sym_lr, beta1=b1_d)

########## updates of D2
dis2_params = ll.get_all_params(dis2_layers, trainable=True)
dis2_grads = T.grad(dis2_cost, dis2_params)
dis2_updates = lasagne.updates.adam(loss_or_grads=dis2_grads, params=dis2_params, learning_rate=sym_lr, beta1=b1_d)

########## updates of D3
dis3_params = ll.get_all_params(dis3_layers, trainable=True)
dis3_grads = T.grad(dis3_cost, dis3_params)
dis3_updates = lasagne.updates.adam(loss_or_grads=dis3_grads, params=dis3_params, learning_rate=sym_lr, beta1=b1_d)

# updates of G
gen_params = ll.get_all_params(gen_layers, trainable=True)
gen_grads = T.grad(gen_cost, gen_params)
gen_updates = lasagne.updates.adam(gen_grads, gen_params, beta1=b1_g, learning_rate=sym_lr)

# updates of C
cla_params = ll.get_all_params(cla_layers, trainable=True)

# first stage
cla_grads1 = T.grad(cla_cost1, cla_params)
cla_updates1_ = lasagne.updates.adam(cla_grads1, cla_params, beta1=b1_c, learning_rate=sym_lr)

# second stage
cla_grads2 = T.grad(cla_cost2, cla_params)
cla_updates2_ = lasagne.updates.adam(cla_grads2, cla_params, beta1=b1_c, learning_rate=sym_lr)

# third stage
cla_grads3 = T.grad(cla_cost3, cla_params)
cla_updates3_ = lasagne.updates.adam(cla_grads3, cla_params, beta1=b1_c, learning_rate=sym_lr)

pre_cla_grad = T.grad(pretrain_cost, cla_params)
pretrain_updates_ = lasagne.updates.adam(pre_cla_grad, cla_params, beta1=0.9, beta2=0.999,
                                         epsilon=1e-8, learning_rate=pre_lr)

######## avg
avg_params = lasagne.layers.get_all_params(cla_layers)
cla_param_avg = []
for param in avg_params:
    value = param.get_value(borrow=True)
    cla_param_avg.append(theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable,
                                       name=param.name))


cla_avg_updates = [(a, a + 0.01 * (p - a)) for p, a in zip(avg_params, cla_param_avg)]
cla_avg_givens = [(p, a) for p, a in zip(avg_params, cla_param_avg)]
# first stage
cla_updates1 = list(cla_updates1_.items()) + cla_avg_updates
# second stage
cla_updates2 = list(cla_updates2_.items()) + cla_avg_updates
# third stage
cla_updates3 = list(cla_updates3_.items()) + cla_avg_updates

pretrain_updates = list(pretrain_updates_.items()) + cla_avg_updates

# functions
train_batch_dis = theano.function(inputs=[sym_x_l, sym_y, sym_y_g,
                                          slice_x_u_c, slice_x_u_d, sym_lr],
                                  outputs=dis_cost_list, updates=dis_updates,
                                  givens={sym_x_u: shared_unlabel[slice_x_u_c],
                                          sym_x_u_d: shared_unlabel[slice_x_u_d]})

train_batch_dis2 = theano.function(inputs=[sym_y_g, slice_x_u_c, sym_lr],
                                   outputs=dis2_cost_list, updates=dis2_updates,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_dis3 = theano.function(inputs=[sym_y_g, slice_x_u_c, sym_lr],
                                   outputs=dis3_cost_list, updates=dis3_updates,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_gen = theano.function(inputs=[sym_y_g, sym_lr],
                                  outputs=gen_cost_list, updates=gen_updates)

print('pass-1')

train_batch_cla1 = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_c,
                                           sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                   outputs=cla_cost_list1, updates=cla_updates1,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_cla2 = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_c,
                                           sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                   outputs=cla_cost_list2, updates=cla_updates2,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

train_batch_cla3 = theano.function(inputs=[sym_x_l, sym_y, sym_y_g, slice_x_u_c, sym_alpha_cla_g,
                                           sym_lr, sym_alpha_unlabel_entropy, sym_alpha_unlabel_average],
                                   outputs=cla_cost_list3, updates=cla_updates3,
                                   givens={sym_x_u: shared_unlabel[slice_x_u_c]})

print('pass-2')
sym_index = T.iscalar()
bslice = slice(sym_index * pre_batch_size_uc, (sym_index + 1) * pre_batch_size_uc)
pretrain_batch_cla = theano.function(inputs=[sym_x_l, sym_y, sym_index],
                                     outputs=[pretrain_cost], updates=pretrain_updates,
                                     givens={sym_x_u: shared_unlabel[bslice]})
# generating images
generate = theano.function(inputs=[sym_y_g], outputs=image)

# avg
evaluate = theano.function(inputs=[sym_x_eval, sym_y], outputs=[accurracy_eval], givens=cla_avg_givens)


# super_para = 'ssl_para_seed:' + str(ssl_para_seed) + '\nnum_labelled:' + str(num_labelled) +  \
#                 '\nalpha_stage1:' +str(alpha_stage1) + "\nalpha_stage2:" + str(alpha_stage2) + \
#                 '\npre_num_epoch:' +str(pre_num_epoch) + "\nnum_epochs:" + str(num_epochs)
#
# with open(logfile, 'a') as f:
#     f.write(super_para + '\n\n')


'''
Pretrain C
'''

for epoch in range(1, pre_num_epoch):  # pre_num_epoch
    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]

    for i in range(int(pretrain_batches_train_uc)):
        i_c = i % pretrain_batches_train_lc
        from_l_c = i_c * pre_batch_size_lc
        to_l_c = (i_c + 1) * pre_batch_size_lc

        pretrain_batch_cla(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c], i)
    acc = evaluate(eval_x, eval_y)
    print('acc:', acc)
    acc = acc[0]
    print(str(epoch) + ':Pretrain accuracy: ' + str(acc))

print('Done')

'''
train and evaluate
'''
our_model_mnist = []

for epoch in range(1, num_epochs):
    start = time.time()
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]

    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_d = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_g = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    g1 = [0.] * len(gen_cost_list)
    c1 = [0.] * len(cla_cost_list3)

    for i in range(n_batches_train_u_c):

        from_u_c = i * batch_size_u_c
        to_u_c = (i + 1) * batch_size_u_c

        i_c = i % n_batches_train_l_c
        from_l_c = i_c * batch_size_l_c
        to_l_c = (i_c + 1) * batch_size_l_c

        i_d = i % n_batches_train_l_d
        from_l_d = i_d * batch_size_l_d
        to_l_d = (i_d + 1) * batch_size_l_d

        i_d_ = i % n_batches_train_u_d
        from_u_d = i_d_ * batch_size_u_d
        to_u_d = (i_d_ + 1) * batch_size_u_d

        sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size_g / num_classes))

        if optim_flag == 1:
            d1_b = train_batch_dis(x_labelled[from_l_d:to_l_d], y_labelled[from_l_d:to_l_d], sample_y,
                                   p_u[from_u_c:to_u_c], p_u_d[from_u_d:to_u_d], lr)
            d2_b = train_batch_dis2(sample_y, p_u[from_u_c:to_u_c], lr)
            d3_b = train_batch_dis3(sample_y, p_u[from_u_c:to_u_c], lr)

        else:
            d1_b = train_batch_dis(x_labelled[from_l_d:to_l_d], y_labelled[from_l_d:to_l_d], sample_y,
                                   p_u[from_u_c:to_u_c], p_u_d[from_u_d:to_u_d], lr)

            for k in range(2):
                d2_b = train_batch_dis2(sample_y, p_u[from_u_c:to_u_c], lr)

            if epoch < alpha_stage1:
                d3_b = train_batch_dis3(sample_y, p_u[from_u_c:to_u_c], lr)

        g1_b = train_batch_gen(sample_y, lr)

        if epoch < alpha_stage1:
            c1_b = train_batch_cla1(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c],
                                    p_u[from_u_c:to_u_c], lr, alpha_unlabeled_entropy, alpha_average)

        elif epoch >= alpha_stage1 and epoch < alpha_stage2:

            c1_b = train_batch_cla2(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c],
                                    p_u[from_u_c:to_u_c], lr, alpha_unlabeled_entropy, alpha_average)

        else:

            c1_b = train_batch_cla3(x_labelled[from_l_c:to_l_c], y_labelled[from_l_c:to_l_c],
                                    sample_y, p_u[from_u_c:to_u_c], alpha_cla_g,
                                    lr, alpha_unlabeled_entropy, alpha_average)

        for j in range(len(g1)):
            g1[j] += g1_b[j]

        for j in range(len(c1)):
            c1[j] += c1_b[j]


    for i in range(len(g1)):
        g1[i] /= n_batches_train_u_c

    for i in range(len(c1)):
        c1[i] /= n_batches_train_u_c

    if (epoch > anneal_lr_epoch) and (epoch % anneal_lr_every_epoch) == 0:
        lr = lr * anneal_lr_factor

    t = time.time() - start
    print('time:', t)
    line = "*Epoch=%d LR=%.5f\n" % (epoch, lr) + \
           "GenLosses:" + str(g1) + "\nClaLosses:" + str(c1)
    print('pass-4')
    print(line)

    with open(logfile, 'a') as f:
        f.write(line + '\n')

    if epoch % eval_epoch == 0:
        acc = evaluate(eval_x, eval_y)
        acc = acc[0]
        print('ErrorEval=%.5f\n' % (1 - acc,))

        with open(logfile, 'a') as f:
            f.write(('ErrorEval=%.5f\n\n' % (1 - acc,)))

    # if epoch % eval_epoch == 0:
    #     print('Saving images')
    #     tail = '-' + str(epoch) + '.png'
    #     ran_y = np.int32(np.repeat(np.arange(num_classes), num_classes))
    #     # ran_y = np.int32(np.repeat(np.arange(num_classes), 1000))
    #     x_gen = generate(ran_y)
    #     x_gen = x_gen.reshape((z_generated * num_classes, -1))
    #
    #     np.save(sample_path + '/sample_' + str(epoch) + '.npy', x_gen)
    #     image = shortcuts.mat_to_img(x_gen.T, dim_input, colorImg=colorImg, scale=generation_scale,
    #                                  save_path=os.path.join(sample_path, 'sample' + tail))

    # if epoch % 10 == 0:
    #     print('Saving Model')
    #     from utils.checkpoints import save_weights
    #
    #     params = ll.get_all_params(dis_layers + cla_layers + gen_layers)
    #     save_weights(os.path.join(outfolder, 'model_epoch' + str(epoch) + '.npy'), params, None)

    print('#################################')

