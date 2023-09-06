import numpy as np
import random
import argparse
from scipy.sparse.linalg import svds


def random_sample(path, i_path, num_deltas=32136, num_samples=500, flag=False):
    deltas_file = open(path, 'rb')
    i_deltas_file = open(i_path, 'rb')
    random_indices = []
    while len(random_indices) < num_samples:
        rand = random.randint(0, num_deltas - 1)
        if rand in random_indices:
            continue
        random_indices.append(rand)

    deltas = []
    i_deltas = []
    idx = -1
    while len(deltas) < num_samples:
        idx += 1
        delta = np.load(deltas_file)
        i_delta = np.load(i_deltas_file)
        if idx not in random_indices:
            continue

        if flag:
            deltas.append(delta)
            i_deltas.append(i_delta)
        else:
            deltas.append(delta.flatten())
            i_deltas.append(i_delta.flatten())

    return np.array(deltas), np.array(i_deltas)


def svd(perturb_path, i_perturb_path, save_path):
    num_deltas = 16096
    u = np.random.rand(224 * 224 * 3).reshape((224 * 224 * 3, 1))
    u = u / np.linalg.norm(u) * 0.1
    path = save_path + 'NEW_UPI_PCA_' + args.epsilon + '.npy'
    i = 0
    alpha = 0.0001
    x, v = random_sample(path=perturb_path, i_path=i_perturb_path, num_deltas=num_deltas, num_samples=num_deltas)
    x, v = x * 1000, v * 1000 * 1000
    while i < 1000:
        # u = u + alpha * (np.dot(x.transpose(), np.dot(x, u)) - np.dot(v.transpose(), np.dot(v, u)))
        u = u + alpha * (np.dot(x.transpose(), np.dot(x, u)))
        l2_norm = np.linalg.norm(u)
        u = u / l2_norm if l2_norm > 1 else u
        i += 1
        universal_file = open(path, 'wb')
        np.save(universal_file, u)


def trans_multi(A, filename):
    with open(filename, "wb") as f:
        rows = A.shape[1]
        for r in range(rows):
            np.save(filename, np.dot(A.T[r, :].reshape(1, A.shape[0]), A))


def real_svd(perturb_path, i_perturb_path, save_path):
    num_deltas = 16096
    x, v = random_sample(path=perturb_path, i_path=i_perturb_path, num_deltas=num_deltas, num_samples=num_deltas)
    num_k = 100
    print(0)
    trans_multi(x, save_path + 'arr1_' + args.epsilon + '.npy')
    print(1)
    trans_multi(v, save_path + 'arr2_' + args.epsilon + '.npy')
    print(2)
    x, v = random_sample(path=save_path + 'arr1_' + args.epsilon + '.npy',
                         i_path=save_path + 'arr2_' + args.epsilon + '.npy',
                         num_deltas=num_deltas, num_samples=num_deltas, flag=True)
    u, s, vT = svds(x - v, k=num_k)
    print(3)
    path = save_path + 'NEW_UPI_PCA_' + args.epsilon + '.npy'
    universal_file = open(path, 'wb')
    np.save(universal_file, vT[-1].reshape((224*224*3, 1)))


def new_svd(perturb_path, i_perturb_path, save_path):
    num_deltas = 16096
    x, v = random_sample(path=perturb_path, i_path=i_perturb_path, num_deltas=num_deltas, num_samples=num_deltas)
    num_k = 100
    u, s, vT = svds(x, k=num_k)
    path = save_path + 'UPI_PCA_' + args.epsilon + '_without_power.npy'
    universal_file = open(path, 'wb')
    np.save(universal_file, vT[-1].reshape((224*224*3, 1)))


def get_num_deltas(path):
    ###
    # total number of deltas
    # ###

    deltas_file = open(path, 'rb')

    num_deltas = 0
    while True:
        try:
            np.load(deltas_file)
            num_deltas += 1
        except:
            break

    print(num_deltas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=str)
    args = parser.parse_args()

    path = ''
    # get_num_deltas(path + 'deltas_' + args.epsilon + '.npy')
    # svd(path + 'simple/deltas_' + args.epsilon + '.npy', path + 'deltas_' + args.epsilon + '.npy',
    #     path + 'simple/')
    new_svd(path + 'simple/deltas_' + args.epsilon + '.npy', path + 'deltas_' + args.epsilon + '.npy',
             path + 'simple/')
