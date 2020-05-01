import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import shutil
import glob
import math
import time
from functools import partial
import keras
import random
import socket
import subprocess
import numpy as np
import tensorflow as tf
import keras.backend as K
from collections import Counter
set_random_seed = tf.compat.v1.set_random_seed
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.utils import to_categorical
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import ipdb
import statistics

HOST = '127.0.0.1'
PORT = 12014

MAX_FILE_SIZE = 5000
MAX_BITMAP_SIZE = 1562
round_cnt = 0
# Choose a seed for random initilzation
#seed = int(time.time())
seed = 12
np.random.seed(seed)
random.seed(seed)
set_random_seed(seed)
seed_list = glob.glob('./seeds/*')
new_seeds = glob.glob('./seeds/id_*')
SPLIT_RATIO = len(seed_list)
# get binary argv
argvv = sys.argv[1:].copy()
seed_list.sort()
new_edges = []
beta = []
alpha = []
ec_num = 0
soft_num = 0
ctx_num = 0
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

import gc

def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

# process training data from afl raw data
def process_data():
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global SPLIT_RATIO
    global seed_list
    global new_seeds
    global new_edges
    global ec_num
    global ctx_num
    global soft_num


    argvv[0] = sys.argv[1] + '_ec'
    # process vari seeds

    vari_seeds = glob.glob('./vari_seeds/id_*_cov')
    for f in vari_seeds:
        if './seeds/'+f.split('/')[-1] in seed_list:
            continue
        try:
            subprocess.call(['./afl-tmin','-i',f,'-o', './seeds/'+f.split('/')[0], '-k', str(MAX_FILE_SIZE)]+argvv,stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            print("afl-tmin timeout" + f)

    # get training samples
    init_list = glob.glob('./seeds/id:*')

    init_list.sort()
    SPLIT_RATIO = len(seed_list)
    rand_index = np.arange(SPLIT_RATIO)
    #np.random.shuffle(seed_list)
    new_seeds = glob.glob('./seeds/id_*')
    new_seeds.sort(key=lambda x: int(x.split('_')[3]))
    seed_list = init_list + new_seeds
    call = subprocess.check_output

    # get MAX_FILE_SIZE
    cwd = os.getcwd()
    max_file_name = call(['ls', '-S', cwd + '/seeds/']).decode('utf8').split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize(cwd + '/seeds/' + max_file_name)

    # create directories to save label, spliced seeds, variant length seeds, crashes and mutated seeds.
    if os.path.isdir("./bitmaps_ec/") == False:
        os.makedirs('./bitmaps_ec')
    if os.path.isdir("./bitmaps_ctx/") == False:
        os.makedirs('./bitmaps_ctx')
    if os.path.isdir("./bitmaps_soft/") == False:
        os.makedirs('./bitmaps_soft')
    if os.path.isdir("./vari_seeds/") == False:
        os.makedirs('./vari_seeds')
    if os.path.isdir("./crashes/") == False:
        os.makedirs('./crashes')

    # ec process raw
    # obtain raw bitmaps
    raw_bitmap = {}
    tmp_cnt = []
    out = ''
    seed = np.zeros((len(seed_list),MAX_FILE_SIZE))
    bitmap_list = glob.glob('./bitmaps_ec/*')
    argvv[0] = sys.argv[1] + '_ec'
    for i,f in enumerate(seed_list):
        # read a input into a matrix
        tmp = open(f,'rb').read()
        ln = len(tmp)
        if ln < MAX_FILE_SIZE:
            tmp = tmp + (MAX_FILE_SIZE - ln) * b'\x00'
        seed[i] = np.array([j for j in list(tmp)]).astype('float32')/255

        # obtain bitmap
        tmp_list = []
        file_name = './bitmaps_ec/'+f.split('/')[-1]+'.npy'
        if file_name in bitmap_list:
            tmp_list = np.load(file_name)
            tmp_cnt = tmp_cnt + tmp_list.tolist()
        else:
            try:
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f])
            except subprocess.CalledProcessError:
                print("find a crash " + f)
                try:
                    out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '5000'] + argvv + [f])
                except subprocess.CalledProcessError:
                    print("crash again, skip")

            for line in out.splitlines():
                edge = int(line.split(b':')[0])
                tmp_cnt.append(edge)
                tmp_list.append(edge)
            #save afl-showmap results
            np.save(file_name, tmp_list)
        raw_bitmap[f] = tmp_list

    # process bitmaps for each input
    counter = Counter(tmp_cnt).most_common()
    ec_label = [int(f[0]) for f in counter]
    ec_label_dict = dict(zip(ec_label, range(len(ec_label))))
    bitmap = np.zeros((len(seed_list), len(ec_label)))
    for idx, i in enumerate(seed_list):
        tmp = raw_bitmap[i]
        for j in tmp:
            bitmap[idx][ec_label_dict[j]] = 1
    ec_bitmap = bitmap

    # ctx process raw
    # obtain raw bitmaps
    raw_bitmap = {}
    tmp_cnt = []
    out = ''

    bitmap_list = glob.glob('./bitmaps_ctx/*')
    argvv[0] = sys.argv[1] + '_ctx'

    for i,f in enumerate(seed_list):
        # obtain bitmap
        tmp_list = []
        file_name = './bitmaps_ctx/'+f.split('/')[-1]+'.npy'
        if file_name in bitmap_list:
            tmp_list = np.load(file_name)
            tmp_cnt = tmp_cnt + tmp_list.tolist()
        else:
            try:
                # append "-o tmp_file" to strip's arguments to avoid tampering tested binary.
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f])
            except subprocess.CalledProcessError:
                print("find a crash " + f)
                try:
                    out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '5000'] + argvv + [f])
                except subprocess.CalledProcessError:
                    print("crash again, skip")

            for line in out.splitlines():
                edge = int(line.split(b':')[0])
                tmp_cnt.append(edge)
                tmp_list.append(edge)
            #save afl-showmap results
            np.save(file_name, tmp_list)
        raw_bitmap[f] = tmp_list

    # process bitmaps for each input
    counter = Counter(tmp_cnt).most_common()
    ctx_label = ec_label + list(set([int(f[0]) for f in counter]) - set(ec_label))
    ctx_label_dict = dict(zip(ctx_label, range(len(ctx_label))))
    bitmap = np.zeros((len(seed_list), len(ctx_label)))
    bitmap[:,:ec_bitmap.shape[1]] = ec_bitmap
    for idx, i in enumerate(seed_list):
        tmp = raw_bitmap[i]
        for j in tmp:
            bitmap[idx][ctx_label_dict[j]] = 1
    ctx_bitmap = bitmap


    # process soft label
    bitmap_list = glob.glob('./bitmaps_soft/*')
    argvv[0] = sys.argv[1] + '_soft'
    soften_label = {}
    for i,f in enumerate(seed_list):
        tmp_list = []
        file_name = './bitmaps_soft/'+f.split('/')[-1]+'.npy'
        if file_name in bitmap_list:
            half_label = np.load(file_name)
        else:
            try:
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + argvv + [f])
            except subprocess.CalledProcessError:
                print("find a crash " + f)
                try:
                    out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '5000'] + argvv + [f])
                except subprocess.CalledProcessError:
                    print("crash again, skip")

            for line in out.splitlines():
                edge = int(line.split(b':')[0])
                tmp_list.append(edge)
            half_label =[ele for ele in tmp_list if ele not in raw_bitmap[f]]
            #save afl-showmap results
            np.save(file_name, half_label)

        for e in half_label:
            if e not in soften_label:
                soften_label[e] = [i]
            else:
                soften_label[e].append(i)

    #patch soft label to bitmap
    for edge_id,seed_id_list in soften_label.items():
        if edge_id not in ctx_label_dict:
            continue
        iidx = ctx_label_dict[edge_id]
        for seed_id in seed_id_list:
            # in case there is edge conflic and overwrite 1 with 0.25.
            if bitmap[seed_id][iidx] == 0:
                bitmap[seed_id][iidx] = 0.25
    soft_bitmap = bitmap

    # delete all 1 label
    all_1_idx = np.where(np.sum(soft_bitmap==1, axis=0) == soft_bitmap.shape[0])[0]
    soft_bitmap = np.delete(soft_bitmap, all_1_idx, 1)
    # debug
    # delete all 0 label
    all_0_idx = np.where(np.sum(soft_bitmap==0, axis=0) == soft_bitmap.shape[0])[0]
    soft_bitmap = np.delete(soft_bitmap, all_0_idx, 1)
    all_025_idx = np.where(np.sum(soft_bitmap<1, axis=0) == soft_bitmap.shape[0])[0]
    soft_bitmap = np.delete(soft_bitmap, all_025_idx, 1)

    # label dimension reduction
    fit_bitmap, indices = np.unique(soft_bitmap,axis=1, return_inverse=True)
    reconstruct_idx = list(set(indices.tolist()[:ec_bitmap.shape[1]]))
    ec_num = len(reconstruct_idx)
    reconstruct_idx = reconstruct_idx + list(set(indices.tolist()[ec_bitmap.shape[1]:ctx_bitmap.shape[1]]) - set(reconstruct_idx))
    ctx_num = len(reconstruct_idx) - ec_num
    soft_num = 0
    print(fit_bitmap[:, np.asarray(reconstruct_idx)].shape, ec_num, ctx_num, soft_num)

    fit_bitmap = fit_bitmap[:, np.asarray(reconstruct_idx)]
    print("#####data dimension############# " + str(fit_bitmap.shape))
    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    # select new edges
    if round_cnt >= 1:
        old_fitmap = np.load("prior_bitmap.npy")
        fit_bitmap_partial = fit_bitmap[:old_fitmap.shape[0]]
        new_edges=np.where(np.sum(fit_bitmap_partial,axis=0)==0)[0].tolist()
        print("####new_edge num################# : "+ str(len(new_edges)))
    else:
        new_edges = []
    np.save("prior_bitmap", fit_bitmap)


    # normalize seed
    mean_var = np.zeros((2,MAX_FILE_SIZE))
    for i in range(seed.shape[1]):
        mean_var[0,i] = np.mean(seed[:,i])
        mean_var[1,i] = np.std(seed[:,i])
        seed[:,i] = (seed[:,i]-mean_var[0,i])/mean_var[1,i] if mean_var[1,i]!=0 else seed[:,i]-mean_var[0,i]

    return seed,fit_bitmap


# learning rate decay
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.7
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print(step_decay(len(self.losses)))

# compute jaccard accuracy for multiple label
def accur_1(y_true, y_pred):
    y_true = tf.round(y_true)
    pred = tf.round(y_pred)
    summ = tf.constant(MAX_BITMAP_SIZE, dtype=tf.float32)
    wrong_num = tf.subtract(summ, tf.reduce_sum(tf.cast(tf.equal(y_true, pred), tf.float32), axis=-1))
    right_1_num = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_true, tf.bool), tf.cast(pred, tf.bool)), tf.float32), axis=-1)
    ret = K.mean(tf.divide(right_1_num, tf.add(right_1_num, wrong_num)))
    return ret

def weighted_cross_entropy(y_true, y_pred):
    """
    Penalizes miss predictions.

    Parameters
    ----------
    beta: float

    Notes
    -----
    WCE = - (beta * p * log(p_hat) + (1 - p) * log(1 - p_hat))
    - Setting beta < 1 will penalize false positives more than false negatives
    - Setting beta > 1 will penalize false negatives more than false positives
    """
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
    loss = alpha * loss

    return tf.reduce_mean(loss)

def gen_adv4(f, fl, model, layer_list, idxx, splice, seed):
    adv_list = []
    #loss = layer_list[-2][1].output[:, f]
    loss = layer_list[4][1].output[:,np.random.randint(512)]
    #loss = layer_list[3][1].output
    grads = K.gradients(loss, model.input)[0]
    iterate = K.function([model.input], [loss, grads])
    ll = len(fl)

    for index in range(ll):
        x = seed[fl[index]].reshape((1,seed.shape[1]))
        loss_value, grads_value = iterate([x])
        idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[:, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
        val = np.sign(grads_value[0][idx])
        adv_list.append((idx, val, seed_list[fl[index]]))
    return adv_list

# grenerate gradient information to guide furture muatation
def gen_mutate3(model, edge_num, sign, seed, label, weighted):
    tmp_list = []
    # select seeds
    rand_seed1 = []
    rand_seed2 = []

    # use rare edge selction or not
    #if (int(round_cnt/3)%2) == 1:
    if (int(round_cnt/2)%2) == 3:
        interested_indice = np.random.choice(range(label.shape[1]), edge_num, replace=True).tolist()
        rand_seed1 = np.random.choice(range(seed.shape[0]), edge_num, replace=True).tolist()
        weighted = False
    else:
        # select rare edges
        if(edge_num > len(new_edges)):
            interested_indice = new_edges

            if (round_cnt%2) == 1:
                rare_label = label[:,ec_num:ctx_num]
                rare_edge_list = [ele+ec_num for ele in np.argsort(np.sum(rare_label, axis=0)).tolist()]
                rare_label = label[:, :ec_num]
                rare_label = np.where(rare_label==0.25, 0, rare_label)
                rare_edge_list = rare_edge_list + np.argsort(np.sum(rare_label, axis=0)).tolist()

            else:
                rare_label = label[:,:ec_num]
                rare_label = np.where(rare_label==0.25, 0, rare_label)
                rare_edge_list = np.argsort(np.sum(rare_label, axis=0)).tolist()

            for rare_edge in rare_edge_list:
                if rare_edge in new_edges:
                    continue
                else:
                    interested_indice.append(rare_edge)
                if len(interested_indice) == edge_num:
                    break
        else:
            interested_indice = new_edges[:edge_num]

        # select inputs
        for edge in interested_indice:
            one_idx = np.where(label[:,edge]==1)[0]
            tmp_rand = np.random.choice(one_idx,1, replace=False)[0]
            rand_seed1.append(tmp_rand)

    print("### rare edge selection: " + str(weighted))

    fn = gen_adv4
    layer_list = [(layer.name, layer) for layer in model.layers]

    with open('gradient_info_p', 'w') as f:
        t0 = time.time()
        for idxx in range(len(interested_indice[:])):
            # kears's would stall after multiple gradient compuation. Release memory and reload model to fix it.
            if (idxx % 100 == 0):
                clear_session()
                model = build_model(label, True)
                model.load_weights('model.h5')
                layer_list = [(layer.name, layer) for layer in model.layers]
                print("number of feature " + str(idxx) + " " + str(time.time()-t0))
            index = int(interested_indice[idxx])
            fl = [rand_seed1[idxx]]
            adv_list = fn(index, fl, model, layer_list, idxx, 0, seed)
            tmp_list.append(adv_list)
            for ele in adv_list:
                ele0 = [str(el) for el in ele[0]]
                ele1 = [str(int(el)) for el in ele[1]]
                ele2 = ele[2]
                f.write(",".join(ele0) + '|' + ",".join(ele1) + '|' + ele2 + "\n")


def build_model(data, weighted_loss):
    global beta
    global alpha
    batch_size = 32
    num_classes = MAX_BITMAP_SIZE
    epochs = 50

    model = Sequential()
    model.add(Dense(2048, input_dim=MAX_FILE_SIZE))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.adam(lr=0.0001)
    pos_weight = (np.sum(data==0, axis=0)+ np.sum(data==0.25, axis=0)/4) / (np.sum(data==1, axis=0)+0.75*np.sum(data==0.25, axis=0))
    # pos_weight = ((data.shape[0] - np.sum(data,axis=0))/np.sum(data, axis=0))
    beta = pos_weight
    alpha = (1 + 1/beta)/2
    # loss = partial(weighted_cross_entropy, beta=pos_weight)
    model.summary()
    if weighted_loss:
        print("build weighted_loss model")
        model.compile(loss=weighted_cross_entropy, optimizer=opt, metrics=[accur_1])
    else:
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[accur_1])

    return model


def train(model, seed, label):

    loss_history = LossHistory()
    lrate = keras.callbacks.LearningRateScheduler(step_decay)
    save_best = keras.callbacks.ModelCheckpoint("model.h5", monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=1)

    callbacks_list = [loss_history, lrate, save_best]

    if seed.shape[0] > 5000:
        model.fit(seed,label,
                        batch_size=int(seed.shape[0]/50),
                        epochs=300,
                        verbose=1, callbacks=callbacks_list)
    else:
        model.fit(seed,label,
                        steps_per_epoch=50,
                        epochs=100,
                        verbose=1, callbacks=callbacks_list)

def gen_grad(data):
    global round_cnt
    t0 = time.time()
    seed, label = process_data()
    model = build_model(label,True)
    weighted = True
    train(model, seed, label)
    gen_mutate3(model,500, True, seed, label, weighted)
    round_cnt = round_cnt + 1
    print(time.time() - t0)


def setup_server():
    with open("mut_cnt", 'w') as f:
        f.write(str(0))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(5)

    conn, addr = sock.accept()
    print('@@@@@@@@@@@@@connected by mtfuzz execution moduel' + str(addr))
    gen_grad('train')
    print("@@@@@@@@@@@@@@gen_data done")
    conn.sendall(b"start")
    data = conn.recv(1024)
    print("@@@@@@@@@@@start gen_data")
    gen_grad(data)
    print("@@@@@@@@@@@@gen_data done")
    conn.sendall(b"close?")
    conn.recv(1024)
    conn.close()
    print("@@@@@@@@@@@@@@@close connection")

    while True:
        try:
            conn, addr = sock.accept()
            print('@@@@@@@@@@@@@connected by mtfuzz execution moduel' + str(addr))
            conn.sendall(b"start")
            data = conn.recv(1024)
            print("@@@@@@@@@@@start gen_data")
            gen_grad(data)
            print("@@@@@@@@@@@@@@gen_data done")
            conn.sendall(b"close?")
            conn.recv(1024)
        finally:
            conn.close()
            print("@@@@@@@@@@@@@@@close connection")
setup_server()

