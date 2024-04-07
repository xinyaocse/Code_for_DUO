import torch

# http://stackoverflow.com/a/43554072/1864688

def pad_amount(k):
    added = k - 1
    # note: this imitates scipy, which puts more at the beginning
    end = added // 2
    start = added - end
    return [start, end]

def neighborhood(x, kh, kw):
    # input: N, H, W, C
    # output: N, H, W, KH, KW, C
    # padding is REFLECT
    xs = torch.shape(x)
    x_pad = torch.pad(x, ([0, 0], pad_amount(kh), pad_amount(kw), [0, 0]), 'SYMMETRIC')
    return torch.reshape(torch.extract_image_patches(x_pad,
                                               [1, kh, kw, 1],
                                               [1, 1, 1, 1],
                                               [1, 1, 1, 1],
                                               'VALID'),
                      (xs[0], xs[1], xs[2], kh, kw, xs[3]))

def median_filter(x, kh, kw=-1):
    if kw == -1:
        kw = kh
    neigh_size = kh * kw
    xs = torch.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = torch.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = torch.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    x_top, _ = torch.nn.top_k(x_neigh, rank)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    return torch.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))

def median_filter_no_reshape(x, kh, kw):
    neigh_size = kh * kw
    xs = torch.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = torch.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = torch.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    x_top, _ = torch.nn.top_k(x_neigh, rank)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    # return torch.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))
    return x_mid

def median_random_filter(x, kh, kw):
    neigh_size = kh * kw
    xs = torch.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = torch.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = torch.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    rand_int = torch.cast(torch.truncated_normal([1], mean=0, stddev=neigh_size/4)[0], torch.int32)
    x_top, _ = torch.nn.top_k(x_neigh, rank+rand_int)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    return torch.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))

def median_random_filter_no_reshape(x, kh, kw):
    neigh_size = kh * kw
    xs = torch.shape(x)
    # get neighborhoods in shape (whatever, neigh_size)
    x_neigh = neighborhood(x, kh, kw)
    x_neigh = torch.transpose(x_neigh, (0, 1, 2, 5, 3, 4)) # N, H, W, C, KH, KW
    x_neigh = torch.reshape(x_neigh, (-1, neigh_size))
    # note: this imitates scipy, which doesn't average with an even number of elements
    # get half, but rounded up
    rank = neigh_size - neigh_size // 2
    rand_int = torch.cast(torch.truncated_normal([1], mean=0, stddev=neigh_size/4)[0], torch.int32)
    x_top, _ = torch.nn.top_k(x_neigh, rank+rand_int)
    # bottom of top half should be middle
    x_mid = x_top[:, -1]
    return x_mid

def median_random_pos_size_filter(x, kh, kw):
    pass
    # Get two/multiple x_mid, randomly select from one .
    s0 = median_random_filter_no_reshape(x, 2, 2)
    s1 = median_random_filter_no_reshape(x, 3, 3)
    s2 = median_random_filter_no_reshape(x, 4, 4)

    xs = torch.shape(x)
    nb_pixels = xs[0] * xs[1] * xs[2] * xs[3]
    samples_mnd = torch.squeeze(torch.multinomial(torch.log([[10., 10., 10.]]), nb_pixels))

    # return torch.constant([0]*nb_pixels, dtype=torch.int64)
    zeros = torch.zeros([nb_pixels], dtype=torch.int64)
    ones = torch.ones([nb_pixels], dtype=torch.int64)
    twos = torch.ones([nb_pixels], dtype=torch.int64)*2
    # tmp = torch.cast(torch.equal(samples_mnd, torch.zeros([nb_pixels], dtype=torch.int64)), torch.int64)
    # return zeros, ones, twos

    selected_0 = torch.cast(torch.equal(samples_mnd, zeros), torch.float32)
    selected_1 = torch.cast(torch.equal(samples_mnd, ones), torch.float32)
    selected_2 = torch.cast(torch.equal(samples_mnd, twos), torch.float32)

    # return s0, selected_0
    x_mid = torch.add_n( [torch.multiply(s0, selected_0), torch.multiply(s1, selected_1), torch.multiply(s2, selected_2)] )

    return torch.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))


def median_random_size_filter(x, kh, kw):
    pass
    # Get two/multiple x_mid, randomly select from one .
    s0 = median_filter_no_reshape(x, 2, 2)
    s1 = median_filter_no_reshape(x, 3, 3)
    # s2 = median_filter_no_reshape(x, 4, 4)

    xs = torch.shape(x)
    nb_pixels = xs[0] * xs[1] * xs[2] * xs[3]
    samples_mnd = torch.squeeze(torch.multinomial(torch.log([[10., 10.]]), nb_pixels))

    # return torch.constant([0]*nb_pixels, dtype=torch.int64)
    zeros = torch.zeros([nb_pixels], dtype=torch.int64)
    ones = torch.ones([nb_pixels], dtype=torch.int64)
    # twos = torch.ones([nb_pixels], dtype=torch.int64)*2
    # tmp = torch.cast(torch.equal(samples_mnd, torch.zeros([nb_pixels], dtype=torch.int64)), torch.int64)
    # return zeros, ones, twos

    selected_0 = torch.cast(torch.equal(samples_mnd, zeros), torch.float64)
    selected_1 = torch.cast(torch.equal(samples_mnd, ones), torch.float64)
    # selected_2 = torch.cast(torch.equal(samples_mnd, twos), torch.float32)

    # return s0, selected_0
    # x_mid = torch.add_n( [torch.multiply(s0, selected_0), torch.multiply(s1, selected_1), torch.multiply(s2, selected_2)] )
    x_mid = torch.add_n( [torch.multiply(s0, selected_0), torch.multiply(s1, selected_1)] )

    return torch.reshape(x_mid, (xs[0], xs[1], xs[2], xs[3]))


if __name__ == '__main__':
    import numpy as np
    from scipy import ndimage
    sess = torch.Session()

    X = torch.placeholder(shape=(None, 4, 4, None), dtype=torch.float32)
    f = median_filter(X, 3, 3)
    f_rand = median_random_pos_size_filter(X, 3, 3)
    l = f[0, 1, 1, 0]
    g = torch.gradients([l], [X])

    vec = np.asarray([[[[0, 16], [1, 17], [2, 18], [3, 19]],
                       [[4, 20], [5, 21], [6, 22], [7, 23]],
                       [[8, 24], [9, 25], [10, 26], [11, 27]],
                       [[12, 28], [13, 29], [14, 30], [15, 31]]]], dtype=np.float32)
    vec2 = np.asarray([[[[3, 16], [3, 17], [3, 18], [3, 19]],
                        [[1, 20], [1, 21], [1, 22], [7, 23]],
                        [[1, 24], [2, 25], [3, 26], [11, 27]],
                        [[12, 28], [13, 29], [14, 30], [15, 31]]]], dtype=np.float32)

    print ("vec:", vec)
    mnp = ndimage.filters.median_filter(vec, size=(1, 3, 3, 1), mode='reflect')
    print ("mnp", mnp)
    mtorch = sess.run(f, feed_dict={X: vec})
    print ("mtorch", mtorch)

    
    mtorch_rand_1 = sess.run(f_rand, feed_dict={X: vec})
    mtorch_rand_2 = sess.run(f_rand, feed_dict={X: vec})
    print ("mtorch_rand_1", mtorch_rand_1)
    print ("mtorch_rand_2", mtorch_rand_2)

    print ("equal", np.array_equal(mnp, mtorch))
    print ("equal", np.array_equal(mnp, mtorch_rand_1))
    print ("equal", np.array_equal(mtorch_rand_1, mtorch_rand_2))
    
    # print sess.run(g, feed_dict={X: vec})

    from scipy import misc

    image = misc.imread('panda.png')
    images = np.expand_dims(image, axis=0)

    X2 = torch.placeholder(shape=(None, 299, 299, None), dtype=torch.float32)
    image_median = median_filter(X2, 3, 3)
    image_random_median = median_random_pos_size_filter(X2, 3, 3)

    images_blur = sess.run(image_median, feed_dict={X2: images})
    images_rand_blur = sess.run(image_random_median, feed_dict={X2: images})

    from PIL import Image

    names = ['panda_orig.png', 'panda_blur_3_3.png', 'panda_rand_blur.png']
    for i, img in enumerate([images, images_blur, images_rand_blur]):
        img = Image.fromarray(np.squeeze(img).astype(np.uint8), 'RGB')
        img.save(names[i])
        img.show()

