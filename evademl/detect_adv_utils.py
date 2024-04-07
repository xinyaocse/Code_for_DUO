

import numpy as np

def get_kmeans_random_batch(model, X, X_noisy, X_adv, dataset, k=10, batch_size=100, pca=False):
    """
    Get the mean distance of each Xi in X_adv to its k nearest neighbors.

    :param model:
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images
    :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :param pca: using pca or not, if True, apply pca to the referenced sample and a
            minibatch of normal samples, then compute the knn mean distance of the referenced sample.
    :return: kms_normal: kmean of normal images (num_examples, 1)
            kms_noisy: kmean of normal images (num_examples, 1)
            kms_adv: kmean of adv images (num_examples, 1)
    """
    # get deep representations

    km_dim = 1
    print("Number of layers to use: ", km_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        km_batch = np.zeros(shape=(n_feed, km_dim))
        km_batch_adv = np.zeros(shape=(n_feed, km_dim))
        km_batch_noisy = np.zeros(shape=(n_feed, km_dim))
        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_act: ", X_act.shape)

            X_adv_act = func([X_adv[start:end], 0])[0]
            X_adv_act = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_adv_act: ", X_adv_act.shape)

            X_noisy_act = func([X_noisy[start:end], 0])[0]
            X_noisy_act = np.asarray(X_noisy_act, dtype=np.float32).reshape((n_feed, -1))
            # print("X_noisy_act: ", X_noisy_act.shape)

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            if pca:
                km_batch[:, i] = kmean_pca_batch(X_act, X_act, k=k)
            else:
                km_batch[:, i] = kmean_batch(X_act, X_act, k=k)
            # print("lid_batch: ", lid_batch.shape)
            if pca:
                km_batch_adv[:, i] = kmean_pca_batch(X_act, X_adv_act, k=k)
            else:
                km_batch_adv[:, i] = kmean_batch(X_act, X_adv_act, k=k)
            # print("lid_batch_adv: ", lid_batch_adv.shape)
            if pca:
                km_batch_noisy[:, i] = kmean_pca_batch(X_act, X_noisy_act, k=k)
            else:
                km_batch_noisy[:, i] = kmean_batch(X_act, X_noisy_act, k=k)
                # print("lid_batch_noisy: ", lid_batch_noisy.shape)
        return km_batch, km_batch_noisy, km_batch_adv

    kms = []
    kms_adv = []
    kms_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        km_batch, km_batch_noisy, km_batch_adv = estimate(i_batch)
        kms.extend(km_batch)
        kms_adv.extend(km_batch_adv)
        kms_noisy.extend(km_batch_noisy)
        # print("kms: ", kms.shape)
        # print("kms_adv: ", kms_noisy.shape)
        # print("kms_noisy: ", kms_noisy.shape)

    kms = np.asarray(kms, dtype=np.float32)
    kms_noisy = np.asarray(kms_noisy, dtype=np.float32)
    kms_adv = np.asarray(kms_adv, dtype=np.float32)

    return kms, kms_noisy, kms_adv