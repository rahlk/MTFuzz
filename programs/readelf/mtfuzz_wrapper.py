import subprocess
import sys
import math
import shutil
import subprocess
import glob
import ipdb
import pickle
import os
import numpy as np
import struct
import time
FNULL = open(os.devnull, 'w')
mut_cnt = 0
'''
def train(x, y):
    model = Sequential()
    model.add(Dense(8, input_dim=x.shape[1]))
    #model.add(Dense(32, input_dim=x.shape[1]))
    model.add(Dense(1))
    opt = keras.optimizers.adam(lr=0.01)
    model.compile(loss='mse', optimizer=opt)
    save_best = keras.callbacks.ModelCheckpoint("best_w.h5", monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=1)
    model.fit(x, y, epochs=50, batch_size=int(x.shape[0]/32), verbose=0, callbacks=[save_best])
    model.load_weights("best_w.h5")
    layer_list = [(layer.name, layer) for layer in model.layers]
    loss = layer_list[-1][1].output[:, 0]
    grads = K.gradients(loss, model.input)[0]
    iterate = K.function([model.input], [loss, grads])

    loss_value, grads_value = iterate([x[0:1]])
    idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[:, -x.shape[1]:].reshape((x.shape[1],)), 0)[:1000]
    return idx
'''

check_out = subprocess.check_output
def find_unexplored_br(unexplored_1,unexplored_2, explored, seeds,tmp_argvv, argvv):
    global mut_cnt
    strcmp_cnt = 0
    tmp_argvv[6] = argvv[6] + '_br'
    for seed_id,seed in enumerate(seeds):
        out = ''
        try:
            # todo: add crach check for afl-showbr.
            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
        except subprocess.CalledProcessError:
            try:
                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
            except subprocess.CalledProcessError:
                print("### found a crash ")
                shutil.copyfile(seed, "./crashes/id_0_0_"+str(mut_cnt))
                mut_cnt = mut_cnt + 1
                continue
        for line in out.splitlines():
            tokens = line.split(b':')
            if len(tokens) == 2:
                edge = int(tokens[0])
                hit = int(tokens[1])
                if hit == 3:
                    if edge not in explored:
                        explored.append(edge)
                    if edge in unexplored_1:
                        del unexplored_1[edge]
                    if edge in unexplored_2:
                        del unexplored_2[edge]
                elif hit == 1:
                    # if edge is explored before, skip it.
                    if edge in explored:
                        continue
                    # if edge is not in explored_1, add to unexplored_1/ append
                    if edge not in unexplored_1:
                        unexplored_1[edge] = [seed_id]
                    else:
                        unexplored_1[edge].append(seed_id)
                    # if edge is in explored_2, set it explored
                    if edge in unexplored_2:
                        del unexplored_2[edge]
                        del unexplored_1[edge]
                        explored.append(edge)
                elif hit == 2:
                    # if edge is explored before, skip it.
                    if edge in explored:
                        continue
                    # if edge is not in explored_1, add to unexplored_1/ append
                    if edge not in unexplored_2:
                        unexplored_2[edge] = [seed_id]
                    else:
                        unexplored_2[edge].append(seed_id)
                    # if edge is in explored_2, set it explored
                    if edge in unexplored_1:
                        del unexplored_1[edge]
                        del unexplored_2[edge]
                        explored.append(edge)
            if len(tokens) == 3:
                edge = int(tokens[0])
                hit = int(tokens[1])
                lenn = int(tokens[2])
                if hit == 3:
                    if edge not in explored:
                        explored.append(edge)
                    if edge in unexplored_1:
                        del unexplored_1[edge]
                    if edge in unexplored_2:
                        del unexplored_2[edge]
                elif hit == 1:
                    # if edge is explored before, skip it.
                    if edge in explored:
                        continue
                    # if edge is not in explored_1, add to unexplored_1/ append
                    if edge not in unexplored_1:
                        unexplored_1[edge] = [(seed_id, lenn)]
                    else:
                        unexplored_1[edge].append((seed_id, lenn))
                    # if edge is in explored_2, set it explored
                    if edge in unexplored_2:
                        del unexplored_2[edge]
                        del unexplored_1[edge]
                        explored.append(edge)
                elif hit == 2:
                    # if edge is explored before, skip it.
                    if edge in explored:
                        continue
                    # if edge is not in explored_1, add to unexplored_1/ append
                    if edge not in unexplored_2:
                        unexplored_2[edge] = [(seed_id, lenn)]
                    else:
                        unexplored_2[edge].append((seed_id, lenn))
                    # if edge is in explored_2, set it explored
                    if edge in unexplored_1:
                        del unexplored_1[edge]
                        del unexplored_2[edge]
                        explored.append(edge)
    print(seed_id, len(unexplored_1) + len(unexplored_2), len(explored))
    # set the seed with proper size at the head of list
    for k,v in unexplored_1.items():
        if not isinstance(v[0], tuple):
            tmp_len = os.stat(seeds[v[0]]).st_size
            for ele in v:
                f_name = seeds[ele]
                f_len = os.stat(f_name).st_size
                if f_len > tmp_len:
                    v[0] = ele
        else:
            tmp_len = os.stat(seeds[v[0][0]]).st_size
            for ele in v:
                f_name = seeds[ele[0]]
                f_len = os.stat(f_name).st_size
                if f_len > tmp_len:
                    v[0] = ele

def crack(tmp_argvv, argvv):
    magic_dict = {}
    #possible_val = [1,3,7,15,31,63,127,255]
    #possible_val = [3,12,48,192]
    possible_val = [15,240]
    if os.path.isdir("./tmp_train/") == False:
        os.makedirs('./tmp_train')
    if os.path.isdir("./tmp_non_direct/") == False:
        os.makedirs('./tmp_non_direct')
    if os.path.exists('./br_log'):
        magic_dict = pickle.load(open("br_log", 'rb'))
    else:
        br_log_name = argvv[6] + '_br_log'
        with open(br_log_name, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                tokens = line.split(' ')
                br_id = int(tokens[2])
                br_type = int(tokens[4])
                constant_loc = int(tokens[6])
                constant_val = tokens[8]
                lenn = int(tokens[10])
                if br_id not in magic_dict:
                    magic_dict[br_id] = (br_type, constant_loc, constant_val, lenn)
        pickle.dump(magic_dict, open('br_log', 'wb'))

    # read mut_cnt counter
    global mut_cnt
    with open("mut_cnt", 'r') as f:
        mut_cnt = int(f.read())

    # obtain unexplored branches
    unexplored_1 = {}
    unexplored_2 = {}
    explored = []
    seeds = glob.glob("seeds/*")
    seeds.sort()
    find_unexplored_br(unexplored_1, unexplored_2, explored, seeds,tmp_argvv, argvv)
    if os.path.exists("crack_failed"):
        crack_failed_but_I_tried = pickle.load(open("crack_failed","rb"))
    else:
        crack_failed_but_I_tried = []

    # concatenate two dicts
    unexplored_1.update(unexplored_2)
    unexplored = unexplored_1
    del unexplored[0]
    #pickle.dump(unexplored, open('tmp_unexplored','wb'))
    #unexplored = pickle.load(open('tmp_unexplored','rb'))
    # k==br_id, v==seed_id
    for k,v in unexplored.items():
        if k in crack_failed_but_I_tried:
            continue
        crack_bool = False
        # parse branch information from magic_dict (from static analysis LLVM)
        (br_type, constant_loc, constant_magic, lenn) = magic_dict[k]
        #if br_type != 2 and br_type != 7 and br_type != 11:
        #if br_type != 10 and br_type != 12:# and br_type != 11:
        #    continue

        if br_type == 0 or br_type == 1:
            seed_id = v[0]
            init_seed = bytearray(open(seeds[seed_id],'rb').read())
            print("br id: " + str(k) + " br len: " + str(lenn) + " br type: " + str(br_type) + " magic: " +  constant_magic + " magic_loc: " + str(constant_loc) + " file len: " + str(len(init_seed)))

            # clean tmp dir
            for f in glob.glob("./tmp_train/*"):
                os.remove(f)
            # create baseline file
            with open("./tmp_train/"+str("121212"),'wb') as f:
                f.write(init_seed)
            # generate sample inputs
            for i in range(len(init_seed)):
                tmp_seed = init_seed.copy()
                for val in possible_val:
                    tmp_seed[i] = val
                    with open("./tmp_train/"+str(i)+"_"+str(val),'wb') as f:
                        f.write(tmp_seed)
            # parse variable values for each sample inputs
            tmp_argvv[6] = argvv[6] + '_br_fast'
            pro = subprocess.run(['./obtain_br','-i','tmp_train', '-o', './tmp_train', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors='ignore')
            line = pro.stdout
            lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
            tmp_dict = {}
            # parse result
            for line in lines:
                tokens = line.split(':')
                tokens2 = tokens[1].split(' ')
                tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

            if '121212' not in tmp_dict:
                continue
            init_op1 = tmp_dict['121212'][0]
            init_op2 = tmp_dict['121212'][1]
            init_distance = tmp_dict['121212'][0] - tmp_dict['121212'][1]
            hot_offsets = []

            min_dist = float('inf')
            file_name = ''
            # no magic constant case
            if constant_loc == 0:
                # parse hot bytes
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if ops[0] != init_op1 or ops[1] != init_op2:
                        # choose the optimal seed as starting point
                        if abs(distance) < min_dist:
                            min_dist = abs(distance)
                            file_name = offset

                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance > 0 and init_distance <= 0) or (distance <= 0 and init_distance > 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                # generate possible candidtate inputs to crack the branch
                init_seed = bytearray(open('./tmp_train/'+file_name,'rb').read())
                for f in glob.glob("./tmp_non_direct/*"):
                    os.remove(f)
                for hot_offset in hot_offsets[:64]:
                    tmp_seed = init_seed.copy()
                    for val in range(255):
                        tmp_seed[hot_offset] = val
                        with open("./tmp_non_direct/"+str(hot_offset)+"_"+str(val),'wb') as f:
                            f.write(tmp_seed)

                # check results using faster mode binary
                tmp_argvv[6] = argvv[6] + '_br_fast'
                pro = subprocess.run(['./obtain_br','-i','tmp_non_direct', '-o', './tmp_non_direct', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",errors='ignore')
                line = pro.stdout
                lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
                tmp_dict = {}
                # parse result
                for line in lines:
                    tokens = line.split(':')
                    tokens2 = tokens[1].split(' ')
                    tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if (distance > 0 and init_distance <= 0) or (distance <= 0 and init_distance > 0):
                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                        shutil.copyfile("./tmp_non_direct/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                        mut_cnt = mut_cnt + 1
                        crack_bool = True
                        break


            # magic constant case
            else:
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if distance != init_distance:
                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance > 0 and init_distance <= 0) or (distance <= 0 and init_distance > 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                if init_distance > 0:
                    # construct an equal case to satisfy <= case
                    magic_ori = struct.pack("@Q", int(constant_magic))
                else:
                    # construct an > case to safisfy > case
                    if constant_loc == 2:
                        magic_ori = struct.pack("@Q", int(constant_magic)+1)
                    elif constant_loc == 1:
                        if int(constant_magic) == 0:
                            continue
                        magic_ori = struct.pack("@Q", int(constant_magic)-1)
                    else:
                        print("error")
                        sys.exit(0)

                # llvm operand size
                magic_l = [magic_ori[:l] for l in [1,2,4,8]]

                # write magic bytes to input and check branch coverage
                for hot_offset in hot_offsets:
                    for magic in magic_l:
                        tmp_seed = init_seed.copy()
                        tmp_seed[hot_offset:hot_offset+len(magic)] = magic
                        with open("tmp_input",'wb') as f:
                            f.write(tmp_seed)

                        tmp_argvv[6] = argvv[6] + '_br'
                        out = ''
                        seed = './tmp_input'
                        try:
                            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                        except subprocess.CalledProcessError:
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                mut_cnt = mut_cnt + 1
                        for line in out.splitlines():
                            tokens = line.split(b':')
                            edge = int(tokens[0])
                            hit = int(tokens[1])
                            if edge == k:
                                if (init_distance > 0 and hit == 2) or (init_distance <= 0 and hit == 1):
                                    print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                    shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                                    crack_bool = True
                                    break

                        # crack success, early exit
                        if crack_bool:
                            break

                        if (hot_offset+1) >= len(magic):
                            tmp_seed = init_seed.copy()
                            tmp_seed[hot_offset-len(magic)+1 :hot_offset+1] = magic
                            with open("tmp_input",'wb') as f:
                                f.write(tmp_seed)

                            tmp_argvv[6] = argvv[6] + '_br'
                            out = ''
                            seed = './tmp_input'
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                try:
                                    out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                                except subprocess.CalledProcessError:
                                    print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                    shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                            for line in out.splitlines():
                                tokens = line.split(b':')
                                edge = int(tokens[0])
                                hit = int(tokens[1])
                                if edge == k:
                                    if (init_distance > 0 and hit == 2) or (init_distance <= 0 and hit == 1):
                                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                        shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                        mut_cnt = mut_cnt + 1
                                        crack_bool = True
                                        break

                        # crack success, early exit
                        if crack_bool:
                            break
                    # crack early exit
                    if crack_bool:
                        break

        if br_type == 2 or br_type == 7 or br_type == 11:

            t0 = time.time()

            seed_id = v[0]
            init_seed = bytearray(open(seeds[seed_id],'rb').read())
            print("br id: " + str(k) + " br len: " + str(lenn) + " br type: " + str(br_type) + " magic: " +  constant_magic + " magic_loc: " + str(constant_loc) + " file len: " + str(len(init_seed)))

            # clean tmp dir
            for f in glob.glob("./tmp_train/*"):
                os.remove(f)
            # create baseline file
            with open("./tmp_train/"+str("121212"),'wb') as f:
                f.write(init_seed)
            # generate sample inputs
            for i in range(len(init_seed)):
                tmp_seed = init_seed.copy()
                for val in possible_val:
                    tmp_seed[i] = val
                    with open("./tmp_train/"+str(i)+"_"+str(val),'wb') as f:
                        f.write(tmp_seed)
            # parse variable values for each sample inputs
            tmp_argvv[6] = argvv[6] + '_br_fast'
            pro = subprocess.run(['./obtain_br','-i','tmp_train', '-o', './tmp_train', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors='ignore')
            t1 = time.time()
            print("obtain_br time cost " + str(t1-t0))

            line = pro.stdout
            lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
            tmp_dict = {}
            # parse result
            for line in lines:
                tokens = line.split(':')
                tokens2 = tokens[1].split(' ')
                tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

            t2 = time.time()
            print("parse obtain_br result time cost " + str(t2-t1))


            if '121212' not in tmp_dict:
                continue
            init_op1 = tmp_dict['121212'][0]
            init_op2 = tmp_dict['121212'][1]
            init_distance = tmp_dict['121212'][0] - tmp_dict['121212'][1]
            hot_offsets = []

            min_dist = float('inf')
            file_name = ''
            # no magic constant case
            if constant_loc == 0:
                # parse hot bytes
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if ops[0] != init_op1 or ops[1] != init_op2:
                        # choose the optimal seed as starting point
                        if abs(distance) < min_dist:
                            min_dist = abs(distance)
                            file_name = offset

                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance == 0 and init_distance != 0) or (distance != 0 and init_distance == 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                # generate possible candidtate inputs to crack the branch
                init_seed = bytearray(open('./tmp_train/'+file_name,'rb').read())
                for f in glob.glob("./tmp_non_direct/*"):
                    os.remove(f)
                for hot_offset in hot_offsets[:64]:
                    tmp_seed = init_seed.copy()
                    for val in range(255):
                        tmp_seed[hot_offset] = val
                        with open("./tmp_non_direct/"+str(hot_offset)+"_"+str(val),'wb') as f:
                            f.write(tmp_seed)

                # check results using faster mode binary
                tmp_argvv[6] = argvv[6] + '_br_fast'
                pro = subprocess.run(['./obtain_br','-i','tmp_non_direct', '-o', './tmp_non_direct', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",errors='ignore')
                line = pro.stdout
                lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
                tmp_dict = {}
                # parse result
                for line in lines:
                    tokens = line.split(':')
                    tokens2 = tokens[1].split(' ')
                    tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if (distance == 0 and init_distance != 0) or (distance != 0 and init_distance == 0):
                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                        shutil.copyfile("./tmp_non_direct/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                        mut_cnt = mut_cnt + 1
                        crack_bool = True
                        break


            # magic constant case
            else:
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if distance != init_distance:
                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance == 0 and init_distance != 0) or (distance != 0 and init_distance == 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                t3 = time.time()
                print("parse distance result time cost " + str(t3-t2))
                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                if init_distance != 0:
                    # construct an equal case to satisfy == case
                    magic_ori = struct.pack("@Q", int(constant_magic))
                else:
                    # construct an inequality case to safisfy != case
                    magic_ori = struct.pack("@Q", int(constant_magic)+1)

                # llvm operand size
                magic_l = [magic_ori[:l] for l in [1,2,4,8]]

                # write magic bytes to input and check branch coverage
                for hot_offset in hot_offsets:
                    for magic in magic_l:
                        tmp_seed = init_seed.copy()
                        tmp_seed[hot_offset:hot_offset+len(magic)] = magic
                        with open("tmp_input",'wb') as f:
                            f.write(tmp_seed)

                        tmp_argvv[6] = argvv[6] + '_br'
                        out = ''
                        seed = './tmp_input'
                        try:
                            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                        except subprocess.CalledProcessError:
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                mut_cnt = mut_cnt + 1
                        for line in out.splitlines():
                            tokens = line.split(b':')
                            edge = int(tokens[0])
                            hit = int(tokens[1])
                            if edge == k:
                                if (init_distance == 0 and hit == 2) or (init_distance != 0 and hit == 1):
                                    print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                    shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                                    crack_bool = True
                                    break

                        # crack success, early exit
                        if crack_bool:
                            break

                        if (hot_offset+1) >= len(magic):
                            tmp_seed = init_seed.copy()
                            tmp_seed[hot_offset-len(magic)+1 :hot_offset+1] = magic
                            with open("tmp_input",'wb') as f:
                                f.write(tmp_seed)

                            tmp_argvv[6] = argvv[6] + '_br'
                            out = ''
                            seed = './tmp_input'
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                try:
                                    out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                                except subprocess.CalledProcessError:
                                    print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                    shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                            for line in out.splitlines():
                                tokens = line.split(b':')
                                edge = int(tokens[0])
                                hit = int(tokens[1])
                                if edge == k:
                                    if (init_distance == 0 and hit == 2) or (init_distance != 0 and hit == 1):
                                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                        shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                        mut_cnt = mut_cnt + 1
                                        crack_bool = True
                                        break

                        # crack success, early exit
                        if crack_bool:
                            break
                    # crack early exit
                    if crack_bool:
                        break
                    t4 = time.time()
                    print("crack time cost " + str(t4-t3))
        #'''


        if br_type == 3 or br_type == 4:
            seed_id = v[0]
            init_seed = bytearray(open(seeds[seed_id],'rb').read())
            print("br id: " + str(k) + " br len: " + str(lenn) + " br type: " + str(br_type) + " magic: " +  constant_magic + " magic_loc: " + str(constant_loc) + " file len: " + str(len(init_seed)))

            # clean tmp dir
            for f in glob.glob("./tmp_train/*"):
                os.remove(f)
            # create baseline file
            with open("./tmp_train/"+str("121212"),'wb') as f:
                f.write(init_seed)
            # generate sample inputs
            for i in range(len(init_seed)):
                tmp_seed = init_seed.copy()
                for val in possible_val:
                    tmp_seed[i] = val
                    with open("./tmp_train/"+str(i)+"_"+str(val),'wb') as f:
                        f.write(tmp_seed)
            # parse variable values for each sample inputs
            tmp_argvv[6] = argvv[6] + '_br_fast'
            pro = subprocess.run(['./obtain_br','-i','tmp_train', '-o', './tmp_train', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors='ignore')
            line = pro.stdout
            lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
            tmp_dict = {}
            # parse result
            for line in lines:
                tokens = line.split(':')
                tokens2 = tokens[1].split(' ')
                tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

            if '121212' not in tmp_dict:
                continue
            init_op1 = tmp_dict['121212'][0]
            init_op2 = tmp_dict['121212'][1]
            init_distance = tmp_dict['121212'][0] - tmp_dict['121212'][1]
            hot_offsets = []

            min_dist = float('inf')
            file_name = ''
            # no magic constant case
            if constant_loc == 0:
                # parse hot bytes
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if ops[0] != init_op1 or ops[1] != init_op2:
                        # choose the optimal seed as starting point
                        if abs(distance) < min_dist:
                            min_dist = abs(distance)
                            file_name = offset

                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance >= 0 and init_distance < 0) or (distance < 0 and init_distance >= 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                # generate possible candidtate inputs to crack the branch
                init_seed = bytearray(open('./tmp_train/'+file_name,'rb').read())
                for f in glob.glob("./tmp_non_direct/*"):
                    os.remove(f)
                for hot_offset in hot_offsets[:64]:
                    tmp_seed = init_seed.copy()
                    for val in range(255):
                        tmp_seed[hot_offset] = val
                        with open("./tmp_non_direct/"+str(hot_offset)+"_"+str(val),'wb') as f:
                            f.write(tmp_seed)

                # check results using faster mode binary
                tmp_argvv[6] = argvv[6] + '_br_fast'
                pro = subprocess.run(['./obtain_br','-i','tmp_non_direct', '-o', './tmp_non_direct', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",errors='ignore')
                line = pro.stdout
                lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
                tmp_dict = {}
                # parse result
                for line in lines:
                    tokens = line.split(':')
                    tokens2 = tokens[1].split(' ')
                    tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if (distance >= 0 and init_distance < 0) or (distance < 0 and init_distance >= 0):
                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                        shutil.copyfile("./tmp_non_direct/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                        mut_cnt = mut_cnt + 1
                        crack_bool = True
                        break


            # magic constant case
            else:
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if distance != init_distance:
                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance >= 0 and init_distance < 0) or (distance < 0 and init_distance >= 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                if init_distance < 0:
                    # construct a equal case to staisfy >=
                    magic_ori = struct.pack("@Q", int(constant_magic))
                else:
                    # construct an < case to safisfy < case
                    if constant_loc == 2:
                        if int(constant_magic) == 0:
                            continue
                        magic_ori = struct.pack("@Q", int(constant_magic)-1)
                    elif constant_loc == 1:
                        magic_ori = struct.pack("@Q", int(constant_magic)+1)
                    else:
                        print("error")
                        sys.exit(0)

                # llvm operand size
                magic_l = [magic_ori[:l] for l in [1,2,4,8]]

                # write magic bytes to input and check branch coverage
                for hot_offset in hot_offsets:
                    for magic in magic_l:
                        tmp_seed = init_seed.copy()
                        tmp_seed[hot_offset:hot_offset+len(magic)] = magic
                        with open("tmp_input",'wb') as f:
                            f.write(tmp_seed)

                        tmp_argvv[6] = argvv[6] + '_br'
                        out = ''
                        seed = './tmp_input'
                        try:
                            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                        except subprocess.CalledProcessError:
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                mut_cnt = mut_cnt + 1
                        for line in out.splitlines():
                            tokens = line.split(b':')
                            edge = int(tokens[0])
                            hit = int(tokens[1])
                            if edge == k:
                                if (init_distance >= 0 and hit == 2) or (init_distance < 0 and hit == 1):
                                    print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                    shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                                    crack_bool = True
                                    break

                        # crack success, early exit
                        if crack_bool:
                            break

                        if (hot_offset+1) >= len(magic):
                            tmp_seed = init_seed.copy()
                            tmp_seed[hot_offset-len(magic)+1 :hot_offset+1] = magic
                            with open("tmp_input",'wb') as f:
                                f.write(tmp_seed)

                            tmp_argvv[6] = argvv[6] + '_br'
                            out = ''
                            seed = './tmp_input'
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                try:
                                    out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                                except subprocess.CalledProcessError:
                                    print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                    shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                            for line in out.splitlines():
                                tokens = line.split(b':')
                                edge = int(tokens[0])
                                hit = int(tokens[1])
                                if edge == k:
                                    if (init_distance >= 0 and hit == 2) or (init_distance < 0 and hit == 1):
                                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                        shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                        mut_cnt = mut_cnt + 1
                                        crack_bool = True
                                        break

                        # crack success, early exit
                        if crack_bool:
                            break
                    # crack early exit
                    if crack_bool:
                        break

        if br_type == 5 or br_type == 6:
            seed_id = v[0]
            init_seed = bytearray(open(seeds[seed_id],'rb').read())
            print("br id: " + str(k) + " br len: " + str(lenn) + " br type: " + str(br_type) + " magic: " +  constant_magic + " magic_loc: " + str(constant_loc) + " file len: " + str(len(init_seed)))

            # clean tmp dir
            for f in glob.glob("./tmp_train/*"):
                os.remove(f)
            # create baseline file
            with open("./tmp_train/"+str("121212"),'wb') as f:
                f.write(init_seed)
            # generate sample inputs
            for i in range(len(init_seed)):
                tmp_seed = init_seed.copy()
                for val in possible_val:
                    tmp_seed[i] = val
                    with open("./tmp_train/"+str(i)+"_"+str(val),'wb') as f:
                        f.write(tmp_seed)
            # parse variable values for each sample inputs
            tmp_argvv[6] = argvv[6] + '_br_fast'
            pro = subprocess.run(['./obtain_br','-i','tmp_train', '-o', './tmp_train', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors='ignore')
            line = pro.stdout
            lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
            tmp_dict = {}
            # parse result
            for line in lines:
                tokens = line.split(':')
                tokens2 = tokens[1].split(' ')
                tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

            if '121212' not in tmp_dict:
                continue
            init_op1 = tmp_dict['121212'][0]
            init_op2 = tmp_dict['121212'][1]
            init_distance = tmp_dict['121212'][0] - tmp_dict['121212'][1]
            hot_offsets = []

            min_dist = float('inf')
            file_name = ''
            # no magic constant case
            if constant_loc == 0:
                # parse hot bytes
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if ops[0] != init_op1 or ops[1] != init_op2:
                        # choose the optimal seed as starting point
                        if abs(distance) < min_dist:
                            min_dist = abs(distance)
                            file_name = offset

                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance < 0 and init_distance >= 0) or (distance >= 0 and init_distance < 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                # generate possible candidtate inputs to crack the branch
                init_seed = bytearray(open('./tmp_train/'+file_name,'rb').read())
                for f in glob.glob("./tmp_non_direct/*"):
                    os.remove(f)
                for hot_offset in hot_offsets[:64]:
                    tmp_seed = init_seed.copy()
                    for val in range(255):
                        tmp_seed[hot_offset] = val
                        with open("./tmp_non_direct/"+str(hot_offset)+"_"+str(val),'wb') as f:
                            f.write(tmp_seed)

                # check results using faster mode binary
                tmp_argvv[6] = argvv[6] + '_br_fast'
                pro = subprocess.run(['./obtain_br','-i','tmp_non_direct', '-o', './tmp_non_direct', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",errors='ignore')
                line = pro.stdout
                lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
                tmp_dict = {}
                # parse result
                for line in lines:
                    tokens = line.split(':')
                    tokens2 = tokens[1].split(' ')
                    tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if (distance < 0 and init_distance >= 0) or (distance >= 0 and init_distance < 0):
                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                        shutil.copyfile("./tmp_non_direct/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                        mut_cnt = mut_cnt + 1
                        crack_bool = True
                        break


            # magic constant case
            else:
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if distance != init_distance:
                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance < 0 and init_distance >= 0) or (distance >= 0 and init_distance < 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                if init_distance < 0:
                    # construct an equal case to satisfy >= case
                    magic_ori = struct.pack("@Q", int(constant_magic))
                else:
                    # construct an < case to safisfy < case
                    if constant_loc == 2:
                        if int(constant_magic) == 0:
                            continue
                        magic_ori = struct.pack("@Q", int(constant_magic)-1)
                    elif constant_loc == 1:
                        magic_ori = struct.pack("@Q", int(constant_magic)+1)
                    else:
                        print("error")
                        sys.exit(0)

                # llvm operand size
                magic_l = [magic_ori[:l] for l in [1,2,4,8]]

                # write magic bytes to input and check branch coverage
                for hot_offset in hot_offsets:
                    for magic in magic_l:
                        tmp_seed = init_seed.copy()
                        tmp_seed[hot_offset:hot_offset+len(magic)] = magic
                        with open("tmp_input",'wb') as f:
                            f.write(tmp_seed)

                        tmp_argvv[6] = argvv[6] + '_br'
                        out = ''
                        seed = './tmp_input'
                        try:
                            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                        except subprocess.CalledProcessError:
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                mut_cnt = mut_cnt + 1
                        for line in out.splitlines():
                            tokens = line.split(b':')
                            edge = int(tokens[0])
                            hit = int(tokens[1])
                            if edge == k:
                                if (init_distance < 0 and hit == 2) or (init_distance >= 0 and hit == 1):
                                    print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                    shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                                    crack_bool = True
                                    break

                        # crack success, early exit
                        if crack_bool:
                            break

                        if (hot_offset+1) >= len(magic):
                            tmp_seed = init_seed.copy()
                            tmp_seed[hot_offset-len(magic)+1 :hot_offset+1] = magic
                            with open("tmp_input",'wb') as f:
                                f.write(tmp_seed)

                            tmp_argvv[6] = argvv[6] + '_br'
                            out = ''
                            seed = './tmp_input'
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                try:
                                    out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                                except subprocess.CalledProcessError:
                                    print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                    shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                            for line in out.splitlines():
                                tokens = line.split(b':')
                                edge = int(tokens[0])
                                hit = int(tokens[1])
                                if edge == k:
                                    if (init_distance < 0 and hit == 2) or (init_distance >= 0 and hit == 1):
                                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                        shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                        mut_cnt = mut_cnt + 1
                                        crack_bool = True
                                        break

                        # crack success, early exit
                        if crack_bool:
                            break
                    # crack early exit
                    if crack_bool:
                        break

        if br_type == 8 or br_type == 9:
            seed_id = v[0]
            init_seed = bytearray(open(seeds[seed_id],'rb').read())
            print("br id: " + str(k) + " br len: " + str(lenn) + " br type: " + str(br_type) + " magic: " +  constant_magic + " magic_loc: " + str(constant_loc) + " file len: " + str(len(init_seed)))

            # clean tmp dir
            for f in glob.glob("./tmp_train/*"):
                os.remove(f)
            # create baseline file
            with open("./tmp_train/"+str("121212"),'wb') as f:
                f.write(init_seed)
            # generate sample inputs
            for i in range(len(init_seed)):
                tmp_seed = init_seed.copy()
                for val in possible_val:
                    tmp_seed[i] = val
                    with open("./tmp_train/"+str(i)+"_"+str(val),'wb') as f:
                        f.write(tmp_seed)
            # parse variable values for each sample inputs
            tmp_argvv[6] = argvv[6] + '_br_fast'
            pro = subprocess.run(['./obtain_br','-i','tmp_train', '-o', './tmp_train', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors='ignore')
            line = pro.stdout
            lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
            tmp_dict = {}
            # parse result
            for line in lines:
                tokens = line.split(':')
                tokens2 = tokens[1].split(' ')
                tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]


            if '121212' not in tmp_dict:
                continue
            init_op1 = tmp_dict['121212'][0]
            init_op2 = tmp_dict['121212'][1]
            init_distance = tmp_dict['121212'][0] - tmp_dict['121212'][1]
            hot_offsets = []

            min_dist = float('inf')
            file_name = ''
            # no magic constant case
            if constant_loc == 0:
                # parse hot bytes
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if ops[0] != init_op1 or ops[1] != init_op2:
                        # choose the optimal seed as starting point
                        if abs(distance) < min_dist:
                            min_dist = abs(distance)
                            file_name = offset

                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance <= 0 and init_distance > 0) or (distance > 0 and init_distance <= 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                # generate possible candidtate inputs to crack the branch
                init_seed = bytearray(open('./tmp_train/'+file_name,'rb').read())
                for f in glob.glob("./tmp_non_direct/*"):
                    os.remove(f)
                for hot_offset in hot_offsets[:64]:
                    tmp_seed = init_seed.copy()
                    for val in range(255):
                        tmp_seed[hot_offset] = val
                        with open("./tmp_non_direct/"+str(hot_offset)+"_"+str(val),'wb') as f:
                            f.write(tmp_seed)

                # check results using faster mode binary
                tmp_argvv[6] = argvv[6] + '_br_fast'
                pro = subprocess.run(['./obtain_br','-i','tmp_non_direct', '-o', './tmp_non_direct', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",errors='ignore')
                line = pro.stdout
                lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
                tmp_dict = {}
                # parse result
                for line in lines:
                    tokens = line.split(':')
                    tokens2 = tokens[1].split(' ')
                    tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]

                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if (distance <= 0 and init_distance > 0) or (distance > 0 and init_distance <= 0):
                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                        shutil.copyfile("./tmp_non_direct/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                        mut_cnt = mut_cnt + 1
                        crack_bool = True
                        break


            # magic constant case
            else:
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if distance != init_distance:
                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance <= 0 and init_distance > 0) or (distance > 0 and init_distance <= 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue

                if init_distance > 0:
                    # construct an equal case to satisfy <= case
                    magic_ori = struct.pack("@Q", int(constant_magic))
                else:
                    # construct an > case to safisfy > case
                    if constant_loc == 2:
                        magic_ori = struct.pack("@Q", int(constant_magic)+1)
                    elif constant_loc == 1:
                        if int(constant_magic) == 0:
                            continue
                        magic_ori = struct.pack("@Q", int(constant_magic)-1)
                    else:
                        print("error")
                        sys.exit(0)

                # llvm operand size
                magic_l = [magic_ori[:l] for l in [1,2,4,8]]

                # write magic bytes to input and check branch coverage
                for hot_offset in hot_offsets:
                    for magic in magic_l:
                        tmp_seed = init_seed.copy()
                        tmp_seed[hot_offset:hot_offset+len(magic)] = magic
                        with open("tmp_input",'wb') as f:
                            f.write(tmp_seed)

                        tmp_argvv[6] = argvv[6] + '_br'
                        out = ''
                        seed = './tmp_input'
                        try:
                            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                        except subprocess.CalledProcessError:
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                mut_cnt = mut_cnt + 1
                        for line in out.splitlines():
                            tokens = line.split(b':')
                            edge = int(tokens[0])
                            hit = int(tokens[1])
                            if edge == k:
                                if (init_distance <= 0 and hit == 2) or (init_distance > 0 and hit == 1):
                                    print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                    shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                                    crack_bool = True
                                    break

                        # crack success, early exit
                        if crack_bool:
                            break

                        if (hot_offset+1) >= len(magic):
                            tmp_seed = init_seed.copy()
                            tmp_seed[hot_offset-len(magic)+1 :hot_offset+1] = magic
                            with open("tmp_input",'wb') as f:
                                f.write(tmp_seed)

                            tmp_argvv[6] = argvv[6] + '_br'
                            out = ''
                            seed = './tmp_input'
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                try:
                                    out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                                except subprocess.CalledProcessError:
                                    print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                    shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                            for line in out.splitlines():
                                tokens = line.split(b':')
                                edge = int(tokens[0])
                                hit = int(tokens[1])
                                if edge == k:
                                    if (init_distance <= 0 and hit == 2) or (init_distance > 0 and hit == 1):
                                        print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                        shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                        mut_cnt = mut_cnt + 1
                                        crack_bool = True
                                        break

                        # crack success, early exit
                        if crack_bool:
                            break
                    # crack early exit
                    if crack_bool:
                        break

        if br_type == 10 or br_type == 12:
            seed_id = v[0]
            if isinstance(v[0], tuple):
                seed_id = v[0][0]
            init_seed = bytearray(open(seeds[seed_id],'rb').read())
            print("br id: " + str(k) + " br len: " + str(lenn) + " br type: " + str(br_type) + " magic: " +  constant_magic + " magic_loc: " + str(constant_loc) + " file len: " + str(len(init_seed)))

            # clean tmp dir
            for f in glob.glob("./tmp_train/*"):
                os.remove(f)
            # create baseline file
            with open("./tmp_train/"+str("121212"),'wb') as f:
                f.write(init_seed)
            # generate sample inputs
            for i in range(len(init_seed)):
                tmp_seed = init_seed.copy()
                for val in possible_val:
                    tmp_seed[i] = val
                    with open("./tmp_train/"+str(i)+"_"+str(val),'wb') as f:
                        f.write(tmp_seed)
            # parse variable values for each sample inputs
            tmp_argvv[6] = argvv[6] + '_br_fast'
            pro = subprocess.run(['./obtain_br','-i','tmp_train', '-o', './tmp_train', '-l', str(len(init_seed)), '-t', str(k)] + tmp_argvv[6:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors='ignore')
            line = pro.stdout
            lines = line[line.find('###$$$ obtain br')+18:].split('\n')[:-1]
            tmp_dict = {}
            # parse result
            for line in lines:
                tokens = line.split(':')
                tokens2 = tokens[1].split(' ')
                tmp_dict[tokens[0]] = [int(tokens2[0]), int(tokens2[1])]


            if '121212' not in tmp_dict:
                continue
            init_op1 = tmp_dict['121212'][0]
            init_op2 = tmp_dict['121212'][1]
            init_distance = tmp_dict['121212'][0] - tmp_dict['121212'][1]
            hot_offsets = []

            min_dist = float('inf')
            file_name = ''
            # no magic constant case
            if constant_loc == 0:
                continue

            # magic constant case
            else:
                for offset, ops in tmp_dict.items():
                    distance = ops[0] - ops[1]
                    if distance != init_distance:
                        loc_offset = int(offset.split('_')[0])
                        if loc_offset not in hot_offsets:
                            hot_offsets.append(loc_offset)
                        if (distance == 0 and init_distance != 0) or (distance != 0 and init_distance == 0):
                            print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                            shutil.copyfile("./tmp_train/"+str(offset), "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                            crack_bool = True
                            break

                # crack success, skip to next branch
                if crack_bool:
                    continue
                # no hot byte candidates, skip
                if len(hot_offsets)==0:
                    continue
                # construct magic string
                magic = []
                for num in range(int(len(constant_magic)/2)):
                    magic.append(int('0x'+constant_magic[num*2:num*2+2],0))

                magic_rev = magic.copy()
                magic_rev.reverse()
                for hot_offset in hot_offsets:
                    tmp_seed = init_seed.copy()
                    tmp_seed[hot_offset:hot_offset+len(magic)] = magic
                    if br_type == 10:
                        tmp_seed[hot_offset+len(magic)] = 0
                    with open("tmp_input",'wb') as f:
                        f.write(tmp_seed)

                    # run inputs and check results
                    tmp_argvv[6] = argvv[6] + '_br'
                    out = ''
                    seed = './tmp_input'
                    try:
                        out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                    except subprocess.CalledProcessError:
                        try:
                            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                        except subprocess.CalledProcessError:
                            print("### found a crash " + str(k) + " br_tyte "+ br_type)
                            shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                            mut_cnt = mut_cnt + 1
                    for line in out.splitlines():
                        tokens = line.split(b':')
                        edge = int(tokens[0])
                        hit = int(tokens[1])
                        if edge == k:
                            if (init_distance == 0 and hit == 2) or (init_distance != 0 and hit == 1):
                                print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                mut_cnt = mut_cnt + 1
                                crack_bool = True
                                break

                    # crack success, early exit
                    if crack_bool:
                        break

                    if (hot_offset+1) >= len(magic_rev):
                        tmp_seed = init_seed.copy()
                        tmp_seed[hot_offset-len(magic_rev)+1 :hot_offset+1] = magic_rev
                        if br_type == 10:
                            if hot_offset >= len(magic_rev):
                                tmp_seed[hot_offset-len(magic_rev)] = 0
                        with open("tmp_input",'wb') as f:
                            f.write(tmp_seed)

                        tmp_argvv[6] = argvv[6] + '_br'
                        out = ''
                        seed = './tmp_input'
                        try:
                            out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '500'] + tmp_argvv[6:-1] + [seed])
                        except subprocess.CalledProcessError:
                            try:
                                out = check_out(['./afl-showbr', '-q', '-o', '/dev/stdout', '-m', '1024', '-t', '5000'] + tmp_argvv[6:-1] + [seed])
                            except subprocess.CalledProcessError:
                                print("### found a crash " + str(k) + " br_tyte "+ br_type)
                                shutil.copyfile("tmp_input", "./crash/id_0_"+str(k)+"_"+str(mut_cnt))
                                mut_cnt = mut_cnt + 1
                        for line in out.splitlines():
                            tokens = line.split(b':')
                            edge = int(tokens[0])
                            hit = int(tokens[1])
                            if edge == k:
                                if (init_distance == 0 and hit == 2) or (init_distance != 0 and hit == 1):
                                    print("###crack branch " + str(k) + " br_tyte "+ str(br_type) + " constant_loc " + str(constant_loc))
                                    shutil.copyfile("tmp_input", "./seeds/id_0_"+str(k)+"_"+str(mut_cnt))
                                    mut_cnt = mut_cnt + 1
                                    crack_bool = True
                                    break

                    # crack success, early exit
                    if crack_bool:
                        break

    crack_failed_but_I_tried = list(unexplored.keys())
    pickle.dump(crack_failed_but_I_tried, open("crack_failed",'wb'))
    with open("mut_cnt", 'w') as f:
        f.write(str(mut_cnt))

def main():
    argvv = sys.argv[1:]
    tmp_argvv = argvv.copy()
    while True:
        print("%%%%%%%%%%%%% run ec mode")
        tmp_argvv[6] = argvv[6]+"_ec"
        subprocess.run(['./mtfuzz']+tmp_argvv)
        print("%%%%%%%%%%%%% run ctx mode")
        tmp_argvv[6] = argvv[6]+"_ctx"
        subprocess.run(['./mtfuzz']+tmp_argvv)
        print("%%%%%%%%%%%% crack hard branch")
        crack(tmp_argvv, argvv)

if __name__== "__main__":
    main()
