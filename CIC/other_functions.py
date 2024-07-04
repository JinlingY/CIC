
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.spatial.distance import cdist
from scipy.special import psi


def cal_mi_from_knn(x, y, n_neighbors=3, norm=True, norm_mode='min'):
    """Compute mutual information between two continuous variables.
    Parameters
    ----------
    x, y : ndarray, shape (n_samples, n_features)
        Samples of two continuous random variables, must have an identical
        shape.
    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    n_samples = len(x)
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric='chebyshev')
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric='chebyshev')
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    psi_sp = psi(n_samples)
    psi_nb = psi(n_neighbors)
    psi_x = np.mean(psi(nx + 1))
    psi_y = np.mean(psi(ny + 1))

    # MI(X, Y) = psi(n_neighbors) + psi(n_samples) - < psi(nX + 1) + psi(nY + 1) >
    mi = max(0, psi_nb + psi_sp - psi_x - psi_y)
    if norm:
        return _norm_mi(mi, psi_sp, psi_x, psi_y, norm_mode)
    else:
        return mi


def cal_cmi_from_knn(x, y, z, n_neighbors=3, norm=True, norm_mode='min'):
    """Compute conditional mutual information between two continuous variables on
    the third variable.

    Parameters
    ----------
    x, y, z : ndarray, shape (n_samples, n_features)
        Samples of two continuous random variables, must have an identical
        shape.
    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information. If it turned out to be negative
        it is replace by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    yz = np.hstack((y, z))
    xz = np.hstack((x, z))
    xyz = np.hstack((x, y, z))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xyz)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(z, metric='chebyshev')
    nz = kd.query_radius(z, radius, count_only=True, return_distance=False)
    nz = np.array(nz) - 1.0

    kd = KDTree(xz, metric='chebyshev')
    nxz = kd.query_radius(xz, radius, count_only=True, return_distance=False)
    nxz = np.array(nxz) - 1.0

    kd = KDTree(yz, metric='chebyshev')
    nyz = kd.query_radius(yz, radius, count_only=True, return_distance=False)
    nyz = np.array(nyz) - 1.0

    psi_nb = psi(n_neighbors)
    psi_xz = np.mean(psi(nxz + 1))
    psi_yz = np.mean(psi(nyz + 1))
    psi_z = np.mean(psi(nz + 1))
    # CMI(X, Y | Z) = psi(n_neighbors) - < psi(nXZ + 1) + psi(nYZ + 1) - psi(nZ + 1) >
    # cmi = (psi(n_neighbors) + np.mean(psi(nz + 1)) -
    #        np.mean(psi(nxz + 1)) - np.mean(psi(nyz + 1)))
    cmi = max(0, psi_nb + psi_z - psi_xz - psi_yz)

    if norm:
        kd = KDTree(x, metric='chebyshev')
        nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
        nx = np.array(nx) - 1.0

        kd = KDTree(y, metric='chebyshev')
        ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
        ny = np.array(ny) - 1.0

        psi_sp = psi(len(x))
        psi_x = np.mean(psi(nx + 1))
        psi_y = np.mean(psi(ny + 1))
        return _norm_mi(cmi, psi_sp, psi_x, psi_y, norm_mode)
    else:
        return cmi

def _norm_mi(mi, psi_sp, psi_x, psi_y, norm_mode='min'):
    Hx = psi_sp - psi_x
    Hy = psi_sp - psi_y
    if norm_mode == 'min':
        entropy = min(Hx, Hy)
    elif norm_mode == 'target':
        entropy = Hy
    elif norm_mode == 'source':
        entropy = Hx
    else:
        return mi

    if entropy <= 0:
        return 0
    else:
        return mi / entropy

def cal_mi_from_knn_old(X, Y, n_neighbor=3, n_excluded=0):
    """
    estimate the mutual information based on the following paper.
    "Kraskov et., Estimating Mutual Information, 2004"

    Parameters
    ----------
    X: 2d array
        manifold X
    Y: 2d array
        manifold Y
    n_neighbor: int
        the number of neighbors
    n_excluded: int
        the number of excluded neighbors

    Returns
    -------
    mi_estimated: float
        estimated mutual information value
    """

    assert X.ndim == 2 and Y.ndim == 2, 'X and Y must be 2d array.'
    assert X.shape[0] == Y.shape[0], 'the numbers of row of X and Y must be same.'

    n_row = X.shape[0]

    XY = np.hstack((X, Y))
    n_X = np.zeros(n_row)
    n_Y = np.zeros(n_row)
    for i in range(n_row):
        # Theiler correction. Points around i are excluded.
        excluded_idx = utils.exclude_range(n_row, i, n_excluded)

        X_tmp = X[excluded_idx, :]
        Y_tmp = Y[excluded_idx, :]
        XY_tmp = XY[excluded_idx, :]
        neigh = NearestNeighbors(n_neighbors=n_neighbor, metric='chebyshev')
        neigh.fit(XY_tmp)

        target_XY = XY[i:i + 1, :]
        dist_k, _ = neigh.kneighbors(target_XY)
        half_epsilon = dist_k[0, -1]
        n_X[i] = np.sum(cdist(X_tmp, X[i:i + 1], metric='chebyshev') < half_epsilon)
        n_Y[i] = np.sum(cdist(Y_tmp, Y[i:i + 1], metric='chebyshev') < half_epsilon)

    idx = np.where((n_X != 0) & (n_Y != 0))
    mi_estimated = psi(n_neighbor) - np.mean(psi(n_X[idx] + 1)) - \
                   np.mean(psi(n_Y[idx] + 1)) + psi(n_row)

    return max(0, mi_estimated)

def cal_cmi_from_knn_old(X, Y, Z, n_neighbor=3, n_excluded=0, metric='chebyshev'):
    """
    This function aims to estimate the MI of two variables condition on the third
    one, namely, Mx and My condition on Mz.

    Parameters
    ----------
    X: 2d array
        manifold X
    Y: 2d array
        manifold Y
    Z: 2d array
        manifold Z
    n_neighbor: int
        the number of neighbors
    n_excluded: int
        the number of excluded neighbors
    metric: str
        distance metric

    Returns
    -------
    cmi_estimated: float
        estimated conditional mutual information value
    """
    assert X.ndim == 2 and Y.ndim == 2 and Z.ndim == 2, 'X, Y and Z must be 2d array.'
    assert X.shape[0] == Y.shape[0] and X.shape[0] == Z.shape[0], \
        'the numbers of row of X and Y must be same.'

    n_row = X.shape[0]

    # n_XY = np.zeros(n_row)
    n_YZ = np.zeros(n_row)
    n_XZ = np.zeros(n_row)
    n_Z = np.zeros(n_row)

    # XY = np.hstack((X, Y))
    YZ = np.hstack((Y, Z))
    XZ = np.hstack((X, Z))
    XYZ = np.hstack((X, Y, Z))
    for i in range(n_row):
        excluded_idx = utils.exclude_range(n_row, i, n_excluded)
        Z_tmp = Z[excluded_idx, :]
        # XY_tmp = XY[excluded_idx, :]
        YZ_tmp = YZ[excluded_idx, :]
        XZ_tmp = XZ[excluded_idx, :]
        XYZ_tmp = XYZ[excluded_idx, :]
        neigh = NearestNeighbors(n_neighbors=n_neighbor, metric=metric)
        neigh.fit(XYZ_tmp)

        target_XYZ = XYZ[i:i + 1, :]
        dist_k, _ = neigh.kneighbors(target_XYZ)
        half_epsilon_XYZ = dist_k[0, -1]
        
        # n_XY[i] = np.sum(cdist(XY_tmp, XY[i:i + 1], metric=metric) < half_epsilon_XYZ)
        n_YZ[i] = np.sum(cdist(YZ_tmp, YZ[i:i + 1], metric=metric) < half_epsilon_XYZ)
        n_XZ[i] = np.sum(cdist(XZ_tmp, XZ[i:i + 1], metric=metric) < half_epsilon_XYZ)
        n_Z[i] = np.sum(cdist(Z_tmp, Z[i:i + 1], metric=metric) < half_epsilon_XYZ)

    idx = np.where((n_YZ != 0) & (n_XZ != 0) & (n_Z != 0))

    # CMI(X, Y | Z) = H(X | Z) - H(X | Y, Z) = psi(k) - < psi(nXZ + 1) + psi(nYZ + 1) - psi(nZ + 1) >
    cmi_estimated = psi(n_neighbor) - np.mean(psi(n_YZ[idx] + 1)) - \
                    np.mean(psi(n_XZ[idx] + 1)) + np.mean(psi(n_Z[idx] + 1))
    return max(0, cmi_estimated)

def embed_vector(vec, embed_dim, lag=1, slide=1):
    """
    delay embedding a vector or 1d array.

    Parameters
    ----------
    vec: list or 1d array
        input vector
    embed_dim: int
        delay embedding dimension
    lag: int
        delay time
    slide: int
        slide step

    Returns
    -------
    delay_mat: 2d array
        matrix after delay embedding
    """
    vec = np.array(vec)
    n_ele = vec.shape[0]
    n_dim = n_ele - (lag * (embed_dim - 1))
    row_idx = np.arange(0, n_dim, slide)

    delay_mat = np.zeros((len(row_idx), embed_dim))
    for i, idx in enumerate(row_idx):
        end_val = idx + lag * (embed_dim - 1) + 1
        part = vec[idx: end_val]
        delay_mat[i, :] = part[::lag]

    return delay_mat

def embed_data(data, embed_dim, **kwargs):
    return np.apply_along_axis(embed_vector, 1, data.T, embed_dim, **kwargs)

def _norm_distance(x, y, order=2):
    return np.linalg.norm(x - y, ord=order)

def _score_list(x, ys, order=2):
    ref_dis = _norm_distance(x, ys[0], order=order)
    score_list = [np.exp(-_norm_distance(x, y, order=order) / ref_dis) for y in ys]
    return score_list

def cal_distance_weight(x, ys, order=2):
    score_list = _score_list(x, ys, order=order)
    sum_score = np.sum(score_list)
    weight_list = [score / sum_score for score in score_list]
    return weight_list

def predict_with_weights(cm_points, weights):
    return np.sum([p * w for p, w in zip(cm_points, weights)], axis=0)

def weights_from_neighbors(x, x_NN):
    """
    calculate the weight of the distance between each neighbor (of x) and x.

    Parameters
    ----------
    x: 2d array
        manifold of x
    x_NN: 2d array
        the nearest neighbors of x

    Returns
    -------
    weights: 1d array
        weights of x_NN
    """
    n_NN = x_NN.shape[0]
    ref_dis = np.linalg.norm(x - x_NN[0])
    scores = np.zeros(n_NN)
    for i in range(n_NN):
        scores[i] = np.exp(-np.linalg.norm(x - x_NN[i]) / ref_dis)
    sum_score = np.sum(scores)
    weights = np.zeros(n_NN)
    for i in range(n_NN):
        weights[i] = scores[i] / sum_score
    return weights

def weights_from_distances(distances):
    n = len(distances)
    ref_dis = distances[0]
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = np.exp(-distances[i] / ref_dis)
    sum_score = np.sum(scores)
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = scores[i] / sum_score
    return weights

def exclude_range(end, element, n_excluded=0, start=0):
    """
    range excluded the interval with width n_exclude at element

    Parameters
    ----------
    end: int
        end of the range
    element: int
        element at the center of the interval
    n_excluded: int
        width of the interval
    start: int
        start of the range

    Returns
    -------
        excluded range (1d array)

    Example
    -------
    >>> exclude_range(10, 4, 2)
    array([0, 1, 7, 8, 9])
    """
    if n_excluded >= 0:
        return np.hstack((np.arange(start, element - n_excluded),
                          np.arange(element + n_excluded + 1, end)))
    else:
        return np.arange(start, end)

def partial_correlation(x, y, z):
    """
    calculate the partial correlation coefficient of x, y and z.

    Parameters
    ----------
    x: vector or 1d array
        time series x
    y: vector or 1d array
        time series y
    z: vector or 1d array
        time series z

    Returns
    -------
    partial_cor: float
        partial correlation coefficient
    """
    pcc_mat = np.corrcoef([x, y, z])
    r_xy = pcc_mat[0, 1]
    r_xz = pcc_mat[0, 2]
    r_yz = pcc_mat[1, 2]
    partial_cor = (r_xy - r_xz * r_yz) / (((1 - r_xz ** 2) ** .5) * ((1 - r_yz ** 2) ** .5))
    return partial_cor

def random_masked_array(nrow, ncol):
    arr = np.random.random((nrow, ncol))
    arr[arr <= .5] = 0
    arr[arr > .5] = 1
    return arr

def resort_masked_array(arr):
    col_sums = np.sum(arr, axis=0)
    row_sums = np.sum(arr, axis=1)
    col_new_idx = np.argsort(col_sums)
    row_new_idx = np.argsort(row_sums)
    new_arr = arr[row_new_idx][:, col_new_idx]
    return new_arr

def compute_squared_distance_loop(X):
    m, n = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.linalg.norm(X[:, i] - X[:, j]) ** 2
            D[j, i] = D[i, j]
    return D

def compute_squared_distance_vec(X):
    m, n = X.shape
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    return H + H.T - 2 * G

def compute_distance_loop(X):
    m, n = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.linalg.norm(X[:, i] - X[:, j])
            D[j, i] = D[i, j]
    return D

def compute_distance_vec(X):
    return np.sqrt(compute_squared_distance_vec(X))

def sorted_distance_index(x, embed_dim, lag=1):
    Mx = embed_vector(x, embed_dim=embed_dim, lag=lag)
    dis_x = compute_squared_distance_vec(Mx.T)
    idx = np.argsort(dis_x)
    return idx

def idx_intersec_count_tot(src_idx, tgt_idx):
    n_row = src_idx.shape[0]
    counts = (len(set(src_idx[i]) & set(tgt_idx[i])) for i in range(n_row))
    return sum(counts)

def idx_intersec_count_ind(src_idx, mid_idx, tgt_idx):
    n_row = src_idx.shape[0]
    counts = (len(set(src_idx[i]) & set(mid_idx[i]) & set(tgt_idx[i])) for i in range(n_row))
    return sum(counts)

def idx_intersec_count_dir(src_idx, mid_idx, tgt_idx):
    return idx_intersec_count_tot(src_idx, tgt_idx) - \
           idx_intersec_count_ind(src_idx, mid_idx, tgt_idx)

def check_dis_mat(dis_mat, n_neighbor):
    non_zeros = np.count_nonzero(dis_mat, axis=1)
    if len(np.where(non_zeros < n_neighbor)[0]) > 0:
        raise RuntimeError("Error: too many 0 in distance matrix.")
    return non_zeros

def skip_diag_masking(A):
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)

def skip_diag_broadcasting(A):
    m = A.shape[0]
    idx = (np.arange(1, m + 1) + (m + 1) * np.arange(m - 1)[:, None]).reshape(m, -1)
    return A.ravel()[idx]

def skip_diag_strided(A):
    m, n = A.shape
    assert m == n, 'input must be a NxN matrix.'
    strided = np.lib.stride_tricks.as_strided
    s0, s1 = A.strides
    return strided(A.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

def skip_diag_tri(mat, k=0):
    return np.tril(mat, -1 - k)[:, :-1 - k] + np.triu(mat, 1 + k)[:, 1 + k:]

def two_array_to_edges_set(array1, array2):
    set1 = {tuple(x) for x in array1}
    set2 = {tuple(x) for x in array2}
    set1and2 = set1 & set2
    set1or2 = set1 | set2
    set1not2 = set1 - set2
    set2not1 = set2 - set1
    return set1and2, set1or2, set1not2, set2not1

def cal_fpr_tpr(scores, truths, thresholds=np.arange(0, 1.01, 0.01)):
    p_idx = np.where(truths == 1)
    n_idx = np.where(truths == 0)
    n_threshold = len(thresholds)
    tpr = np.zeros(n_threshold)
    fpr = np.zeros(n_threshold)
    for i, t in enumerate(thresholds[::-1]):
        n_tp = np.count_nonzero(scores[p_idx] >= t)
        n_fp = np.count_nonzero(scores[n_idx] >= t)
        n_tn = np.count_nonzero(scores[n_idx] < t)
        n_fn = np.count_nonzero(scores[p_idx] < t)
        tpr[i] = n_tp / (n_tp + n_fn)
        fpr[i] = n_fp / (n_fp + n_tn)
    return fpr, tpr

def find_optimal_threshold(fpr, tpr, thresholds=np.arange(0, 1.01, 0.01)):
    Youden_idx = np.argmax(tpr - fpr)
    optim_threshold = thresholds[Youden_idx]
    point = (fpr[Youden_idx], tpr[Youden_idx])
    return optim_threshold, point

def exclude_vec_mat(vec, n_excluded=0):
    return skip_diag_tri(np.tile(vec, (len(vec), 1)), n_excluded)

def series_to_embed_dismat(x, embed_dim, **kwargs):
    Mx = embed_vector(x, embed_dim, **kwargs)
    dis_mat = compute_distance_vec(Mx.T)
    return skip_diag_tri(dis_mat)

def embed_to_dismat(Mx, k=0):
    return skip_diag_tri(compute_distance_vec(Mx.T), k)

def dismat_to_idx(dis_mat):
    idx = np.argsort(dis_mat)
    return idx

def counts_zeros_of_dismat(dis_mat):
    non_zeros = np.count_nonzero(dis_mat, axis=1)
    zeros = dis_mat.shape[1] - non_zeros
    return zeros

def series_to_idx(x, embed_dim, n_neighbor, **kwargs):
    return dismat_to_idx(series_to_embed_dismat(x, embed_dim, **kwargs), n_neighbor)

def idx_to_mapping(idx, idy, zeros_y, n_neighbor):
    n_row = idx.shape[0]
    map_x2y = np.asarray([
        np.where(idx[t] == idy[t][zeros_y[t]: n_neighbor + zeros_y[t], None])[-1]
        for t in range(n_row)
    ])
    return map_x2y

def edges_to_mat(edges, n_nold=None):
    assert edges.ndim == 2 and edges.shape[1] >= 2
    nolds = np.max(edges) + 1
    if n_nold is None or n_nold < nolds:
        n_nold = nolds
    mat = np.zeros((n_nold, n_nold))
    mat[edges[:, 0], edges[:, 1]] = 1
    return mat

def mat_to_edges(mat):
    return np.asarray(np.where(mat == 1)).T

def score_to_accuracy(scores, truths, thresholds=np.arange(0, 1.01, 0.01)):
    assert scores.shape == truths.shape
    assert scores.ndim in [1, 2]
    if scores.ndim == 2:
        scores = skip_diag_tri(scores)
        truths = skip_diag_tri(truths)
    accs = []
    for t in thresholds:
        pos = truths[scores >= t]
        true_pos = np.count_nonzero(pos)
        neg = truths[scores < t]
        true_neg = len(neg) - np.count_nonzero(neg)
        accs.append((true_pos + true_neg) / truths.size)
    return accs

def to_positive(arr):
    arr_tmp = arr.copy()
    arr_tmp[np.where(arr < 0)] = 0.
    return arr_tmp

def nan_to_val(arr, val=0.):
    arr_tmp = arr.copy()
    arr_tmp[np.where(np.isnan(arr_tmp))] = val
    return arr_tmp

def revise_strength(arr):
    arr[np.where(np.isnan(arr))] = 0.
    arr[np.where(arr < 0)] = 0.

def normalize_by_maximum(arr):
    arr_tmp = arr.copy()
    return arr_tmp / np.nanmax(arr_tmp)

def cross_sum_row_and_col(arr):
    n = arr.shape[0]
    out = np.empty((n, n, n))
    for i, row in enumerate(arr):
        for j, col in enumerate(arr.T):
            out[i, j] = row + col
    return out

def exclude_range_mat(n_row, n_excluded):
    return skip_diag_tri(np.tile(np.arange(n_row), (n_row, 1)), n_excluded)

def score_seq_to_matrix(score_list):
    n_score = len(score_list)
    n_var = int(np.sqrt(n_score)) + 1
    rev_list = list(score_list)[::-1]
    mat = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(n_var):
            if i != j:
                mat[i, j] = rev_list.pop()
    return mat

def discretize_score(score, v=0.5):
    out = score.copy()
    out[out >= v] = 1.
    out[out < v] = 0.
    return out
