from fgm import *
import cv2


def evaluate(model, test_generator, grad_method, path, perturbation_type=None, M=10, alpha=500, dataset='imagenet',
             plus=True, ord=2):
    if dataset == 'tiny' or dataset == 'imagenet':
        shape = (224, 224, 3)
        # shape = (299, 299, 3)
    elif dataset == 'cifar10':
        # shape = (75, 75, 3)
        shape = (32, 32, 3)
    elif dataset == 'mnist':
        shape = (32, 32, 3)

    if grad_method == 'integrated':
        explainer = ModifiedIntegratedGradients()
        explainer.set_num_steps(M)
    elif grad_method == 'simple':
        explainer = SimpleGradients()

    if dataset == 'cifar10':
        mean, std = 120.70748, 64.150024
    else:
        mean, std = 0, 1

    if perturbation_type is None:
        return 0
        delta = 0
    elif perturbation_type == 'universal_random':
        delta = (np.random.rand(np.product(shape)).reshape((np.product(shape), 1)) - 0.5) * 2
        delta /= np.linalg.norm(delta.flatten(), ord)
        delta = delta.reshape(shape)
        delta = delta * alpha
        delta /= std
    elif perturbation_type == 'upi_pca_pgd':
        universal_file = open(path + grad_method + '/UPI_PCA_PGD_' + str(alpha) + '.npy', 'rb')
        delta = np.load(universal_file)
        delta /= np.linalg.norm(delta.flatten(), ord)
        delta = delta.reshape(shape)
        delta = delta * alpha
        delta /= std
    elif perturbation_type == 'upi_pca_fgm':
        print(path)
        # universal_file = open(path + '' + grad_method + '_UPI_PCA_500_without_power.npy', 'rb')
        universal_file = open(path + '' + grad_method + '_APROX_UPI_PLUS_PCA_1500.npy', 'rb') #1.2 1.1 0.9 0.8
        # universal_file = open(path + '' + grad_method + '_UPI_PLUS_PCA_500_lamb1.npy', 'rb') #0.8 1
        delta = np.load(universal_file)
        # print(delta)
        # print(delta.shape, delta)
        # delta = delta.reshape((32, 32, 3))
        # delta = cv2.resize(delta, (75, 75))
        # delta = img_to_array(array_to_img(delta).resize((75, 75)))
        # delta = img_to_array(delta.resize((75, 75)))

        print(delta.shape, delta)
        delta /= np.linalg.norm(delta.flatten(), ord)
        delta = delta.reshape(shape)
        delta = delta * alpha
        delta /= std
    elif perturbation_type == 'uap':
        add = str(alpha)
        if args.times != np.inf:
            add += '_' + str(args.times)
        print("PATHHHHHHHHH =", 'UAP_' + add + '.npy')
        universal_file = open(path + 'UAP_' + add + '.npy', 'rb')
        delta = np.load(universal_file)
        # delta /= np.linalg.norm(delta.flatten(), ord)
        delta = delta.reshape(shape)
        delta = delta * alpha
        delta /= std
    elif perturbation_type == 'upi_grad':
        universal_file = open(path + grad_method + '/UPI_Grad_' + str(alpha) + '.npy', 'rb')
        delta = np.load(universal_file)
        delta /= np.linalg.norm(delta.flatten(), ord)
        delta = delta.reshape(shape)
        delta = delta * alpha
        delta /= std
    else:
        deltas_file = open(path + grad_method + '/deltas_' + str(alpha) + '.npy', 'rb')

    dissimilarity_sum_p,  dissimilarity_sum = 0, 0
    num = 0

    mean_image = np.zeros((224, 224, 3))
    mean_image[:, :, 0] = 103.939
    mean_image[:, :, 1] = 116.779
    mean_image[:, :, 2] = 123.68
    fooled_p, fooled = 0, 0

    # interpretability_before = open('VGG16_CIFAR10_interp_bef.npy', 'wb')
    # interpretability_after = open('VGG16_CIFAR10_interp_af.npy', 'wb')

    for batch_idx in range(test_generator.__len__()):
        # if batch_idx >= args.times:
        # # if batch_idx >= 100:
        #     break
        batch = test_generator.__getitem__(batch_idx)
        Xs = batch[0]
        ys = batch[1]
        batch_predict = model.predict(Xs)

        for idx in range(len(Xs)):
            X, y = Xs[idx], ys[idx]
            gt_class = np.where(y == 1)[0][0]
            p_class = np.argmax(batch_predict[idx])
            if gt_class != p_class:
                continue

            # print(batch_idx * 64 + idx)

            if perturbation_type == 'per_image':
                delta = np.load(deltas_file)
                delta /= np.linalg.norm(delta.reshape((np.product(shape), 1)))
                delta = delta * alpha
                delta /= std
            elif perturbation_type == 'random':
                delta = (np.random.rand(np.product(shape)).reshape((np.product(shape), 1)) - 0.5) * 2
                delta /= np.linalg.norm(delta)
                delta = delta.reshape(shape)
                delta = delta * alpha
                delta /= std

            perturbed = X + delta
            perturbed_tensor = tf.convert_to_tensor([perturbed], dtype=tf.float32)
            p_class_perturbed = np.argmax(model(perturbed_tensor))

            ############
            X_tensor = tf.convert_to_tensor([X], dtype=tf.float32)
            I = explainer.explain(X_tensor, model, gt_class)
            I_p = explainer.explain(perturbed_tensor, model, gt_class)
            # np.save(interpretability_before, I)
            # np.save(interpretability_after, I_p)

            dissimilarity = tf.norm(I[0] - I_p[0], ord=2) / tf.norm(I[0], ord=2)
            dissimilarity_sum_p += dissimilarity

            ###########

            if p_class_perturbed != gt_class:
                fooled_p += 1

            perturbed = X - delta
            perturbed_tensor = tf.convert_to_tensor([perturbed], dtype=tf.float32)
            p_class_perturbed = np.argmax(model(perturbed_tensor))

            ############
            I_p = explainer.explain(perturbed_tensor, model, gt_class)

            dissimilarity = tf.norm(I[0] - I_p[0], ord=2) / tf.norm(I[0], ord=2)
            dissimilarity_sum += dissimilarity

            ###########

            if p_class_perturbed != gt_class:
                fooled += 1

            num += 1

    print(num)
    fooled = fooled_p if dissimilarity_sum_p > dissimilarity_sum else fooled
    dissimilarity_sum = dissimilarity_sum_p if dissimilarity_sum_p > dissimilarity_sum else dissimilarity_sum
    return fooled / num, (dissimilarity_sum / num).numpy()
    # return fooled / num, 0


def visualize(epsilon):
    path = '../perturbations/VGG16_ImageNet_NEW/simple_'
    universal_file = open(path + 'UPI_PCA_' + str(epsilon) + '_without_power.npy', 'rb')
    u = np.load(universal_file)
    u -= np.mean(u)
    u = u / np.max(np.abs(u)) * 255.0
    u = u.reshape((224, 224, 3))
    img = cv2.cvtColor(u.astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(path + 'NEW_UPI_PCA_' + str(epsilon) + '.jpg', img)


# def visualize(epsilon):
#     path = '../perturbations/VGG16_ImageNet/'
#     universal_file = open(path + 'UAP_' + str(epsilon) + '.npy', 'rb')
#     uap = np.load(universal_file)
#     uap = uap.reshape((224, 224, 3))
#     uap -= np.mean(uap)
#     uap_max = np.max(np.abs(uap))
#     arr = ((np.transpose((uap / uap_max), (1, 0, 2))) * 255.0)
#     print(np.max(arr))
#     plt.imshow(arr.astype(np.uint8))
#     plt.savefig(path + 'UAP_' + str(epsilon) + '.jpg')


def main(args):
    epsilon = args.epsilon
    tf.keras.utils.disable_interactive_logging()
    grad_method = 'simple'

    model = get_model(model_name=args.model, dataset=args.dataset)
    # model = get_model(model_name="Xception")
    # model = get_model(model_name="ResNet50")
    # model = get_model(model_name="InceptionV3")
    # model = get_model(model_name="MobileNet")

    if args.dataset == 'ImageNet':
        train_generator, val_generator, test_generator = load_ImageNet()
    elif args.dataset == 'CIFAR10':
        train_generator, val_generator, test_generator = load_CIFAR10()

    # if args.times != np.inf:
    #     test_generator = valid_generator
    #     print('with shift')

    path = '/data/' + args.dataset + '/' + args.model + '_'
    # path = '/data/' + args.dataset + '/' + 'ResNet50' + '_'

    # iters = 3
    # mean_d, fooled = 0, 0
    # for i in range(iters):
    #     fooled_p, mean_d_p = evaluate(model, test_generator,
    #                                   grad_method=grad_method,
    #                                   alpha=epsilon,
    #                                   path=path,
    #                                   perturbation_type='universal_random')
    #     mean_d += mean_d_p
    #     fooled += fooled_p
    # print('Universal random:', '\nMean dissimilarity:', mean_d / iters, '\nFooling rate:', fooled / iters)
    #
    # fooled_p, mean_d_p = evaluate(model, test_generator,
    #                               grad_method=grad_method,
    #                               alpha=epsilon,
    #                               path=path,
    #                               perturbation_type='upi_pca_fgm')
    fooled, mean_d = evaluate(model, test_generator,
                              grad_method=grad_method,
                              alpha=epsilon,
                              path=path,
                              plus=False,
                              perturbation_type='upi_pca_fgm')
    # fooled = fooled_p if mean_d_p > mean_d else fooled
    # mean_d = mean_d_p if mean_d_p > mean_d else mean_d
    # fooled = fooled_p if fooled > fooled_p else fooled
    # mean_d = mean_d_p if fooled > fooled_p else mean_d
    print('UPI-PCA-FGM:', '\nMean dissimilarity:', mean_d, '\nFooling rate:', fooled)

    # fooled_p, mean_d_p = evaluate(model, test_generator,
    #                               grad_method=grad_method,
    #                               alpha=epsilon,
    #                               path=path,
    #                               perturbation_type='uap')
    # fooled, mean_d = evaluate(model, test_generator,
    #                           grad_method=grad_method,
    #                           alpha=epsilon,
    #                           path=path,
    #                           plus=False,
    #                           perturbation_type='uap')
    # fooled = fooled_p if fooled_p > fooled else fooled
    # # mean_d = mean_d_p if mean_d_p > mean_d else mean_d
    # print('UAP-SGD:', '\nMean dissimilarity:', mean_d, '\nFooling rate:', fooled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=int, default=10000)
    parser.add_argument('--times', type=int, default=np.inf)
    parser.add_argument('--model', type=str, default='MobileNet')
    parser.add_argument('--dataset', type=str, default='ImageNet')

# Xception, InceptionV3, ResNet50

    args = parser.parse_args()

    main(args)
    # visualize(args.epsilon)
# python fgm.py --epsilon 1500; python svd.py --epsilon 1500; python evaluate.py --epsilon 1500
