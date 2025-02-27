# import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from scipy.sparse.linalg import svds


def random_sample(path, i_path, num_deltas=32136, num_samples=500, flag=False):
    print(path)
    num_samples = num_deltas = get_num_deltas(path)
    deltas_file = open(path, 'rb')
    print(i_path)
    if get_num_deltas(i_path) != num_samples:
        print("All data not available")
        return None
    i_deltas_file = open(i_path, 'rb')
    random_indices = np.array([_ for _ in range(num_deltas)])
    random_indices = np.random.choice(random_indices, size=num_samples, replace=False)

    deltas = []
    i_deltas = []
    idx = -1
    while len(deltas) < num_samples:
        idx += 1
        # print(idx)
        delta = np.load(deltas_file)
        i_delta = np.load(i_deltas_file)
        if num_deltas != num_samples and idx not in random_indices:
            continue

        if flag:
            deltas.append(delta)
            i_deltas.append(i_delta)
        else:
            deltas.append(delta.flatten())
            i_deltas.append(i_delta.flatten())

    return np.array(deltas), np.array(i_deltas), num_deltas


def svd(perturb_path, i_perturb_path, save_path):
    # num_deltas = 1000
    # u = np.random.rand(32 * 32 * 3).reshape((32 * 32 * 3, 1))
    # u = np.random.rand(75 * 75 * 3).reshape((75 * 75 * 3, 1))
    u = np.random.rand(224 * 224 * 3).reshape((224 * 224 * 3, 1))
    # u = np.random.rand(299 * 299 * 3).reshape((299 * 299 * 3, 1))
    u = u / np.linalg.norm(u) * 0.1
    path = save_path + 'APROX_UPI_PLUS_PCA_1500.npy'
    i = 0
    alpha = 0.0001
    # lamb = 1.2
    lamb = 1
    # x, v = random_sample(path=perturb_path, i_path=i_perturb_path, num_deltas=num_deltas, num_samples=num_deltas)
    x, v, num_deltas = random_sample(path=perturb_path, i_path=i_perturb_path)
    prev = v
    x, v = x / np.linalg.norm(x), v / np.linalg.norm(v)
    # print(x.shape, v.shape)
    # print(np.linalg.norm(x), np.linalg.norm(v))
    # print(np.sum(v) - np.sum(prev))
    while i < 5000:
        indices = np.random.choice(num_deltas, 500, replace=False)  # Select batch indices
        x_batch = x[indices]
        v_batch = v[indices]
        # x, v = random_sample(path=perturb_path, i_path=i_perturb_path, num_deltas=num_deltas, num_samples=num_deltas)
        # x, v = x / np.linalg.norm(x), v / np.linalg.norm(v)
        u = u + alpha * (np.dot(x_batch.transpose(), np.dot(x_batch, u)) -
                         lamb * np.dot(v_batch.transpose(), np.dot(v_batch, u)))
        # u = u + alpha * (np.dot(x.transpose(), np.dot(x, u)) - lamb * np.dot(v.transpose(), np.dot(v, u)))
        # u = u + alpha * (np.dot(x.transpose(), np.dot(x, u)))
        l2_norm = np.linalg.norm(u)
        u = u / l2_norm if l2_norm > 1 else u
        i += 1
    universal_file = open(path, 'wb')
    np.save(universal_file, u)


def trans_multi(A, filename):
    file = open(filename, 'wb')
    rows = A.shape[1]
    for r in range(rows):
        np.save(file, np.dot(A.T[r, :].reshape(1, -1), A))


def real_svd(perturb_path, i_perturb_path, save_path):
    # num_deltas = 17020
    # x, v = random_sample(path=perturb_path, i_path=i_perturb_path, num_deltas=num_deltas, num_samples=num_deltas)
    x, v, num_deltas = random_sample(path=perturb_path, i_path=i_perturb_path)
    print(x.shape[1], v.shape[1])
    num_k = 100
    print(0)
    trans_multi(x, save_path + 'arr1_' + args.epsilon + '.npy')
    print(1)
    trans_multi(v, save_path + 'arr2_' + args.epsilon + '.npy')
    print(2)

    x, v, num_deltas = random_sample(path=save_path + 'arr1_' + args.epsilon + '.npy',
                         i_path=save_path + 'arr2_' + args.epsilon + '.npy',
                         num_deltas=150528, num_samples=150528, flag=True)
    u, s, vT = svds(x - v, k=num_k)
    print(3)
    path = save_path + 'REAL_UPI_PCA_' + args.epsilon + '.npy'
    universal_file = open(path, 'wb')
    np.save(universal_file, vT[-1].reshape((224*224*3, 1)))


def new_svd(perturb_path, i_perturb_path, save_path):
    # num_deltas = 16096
    # num_deltas = 19382
    print("NEW")
    # num_deltas = 17020
    # num_deltas = 19076
    # num_deltas = 17109

    # num_deltas = get_num_deltas(path)
    x, v, num_deltas = random_sample(path=perturb_path, i_path=i_perturb_path)
    num_k = 1000
    u, s, vT = svds(x, k=num_k)
    # u, s, vT = np.linalg.svd(x)

    path = save_path + 'UPI_PCA_' + args.epsilon + '_without_power.npy'
    # path = save_path
    # num_pos = np.sum(np.where(s > 0, 1, 0))
    # print(min(s), max(s), num_pos, len(s))
    # file = open('singular_values_before.npy', 'wb')
    # num_pos = np.sum(np.where(s > 0, 1, 0))
    # print(min(s), max(s), num_pos, len(s))
    # np.save(file, s)
    # # print(x.shape)
    #
    # rank = np.linalg.matrix_rank(x)
    # print(rank)
    # after: 1024, before: 1024
    # plt.plot(s)
    # plt.savefig('s.png')

    # bef: 0.0 2.1821625 777 1000
    # aft: 0.0 2.1671224 765 1000
    universal_file = open(path, 'wb')
    np.save(universal_file, vT[-1].reshape((32*32*3, 1)))
    # np.save(universal_file, vT[-1].reshape((75*75*3, 1)))
    # np.save(universal_file, vT[-1].reshape((224*224*3, 1)))
    # np.save(universal_file, vT[-1].reshape((299*299*3, 1)))
    # random


def get_num_deltas(path):
    ###
    # total number of deltas
    # ###

    deltas_file = open(path, 'rb')

    num_deltas = 0
    while True:
        try:
            np.load(deltas_file)
            # print(np.linalg.norm(vec))
            num_deltas += 1
        except:
            break

    print(num_deltas)
    return num_deltas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=str, default='1500')
    args = parser.parse_args()

    path = '/data/ImageNet/ResNet50_'
    # get_num_deltas(path + 'simple_deltas_' + args.epsilon + '_without_power.npy')
    # get_num_deltas(path + 'sgd_deltas_' + args.epsilon + '.npy')
    svd(path + 'simple_deltas_' + args.epsilon + '_without_power.npy', path + 'sgd_deltas_' + args.epsilon + '.npy',
        path + 'simple_')
    # new_svd(path + 'simple_deltas_' + args.epsilon + '_without_power.npy', path + 'sgd_deltas_' + args.epsilon + '.npy',
    #          path + 'simple_')
    # new_svd('VGG16_CIFAR10_interp_bef.npy', 'VGG16_CIFAR10_interp_bef.npy',
    #         'before.npy')
    # real_svd(path + 'simple_deltas_' + args.epsilon + '_without_power.npy', path + 'sgd_deltas_' + args.epsilon + '.npy',
    #          path + 'simple_')
    # path += 'simple_NEW_UPI_PCA_' + args.epsilon + '_without_power.npy'
    # universal_file = open(path, 'rb')
    # delta = np.load(universal_file)
    # delta = delta * 10000
    # print(np.mean(delta))