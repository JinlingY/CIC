#import crossmapy as cmp

# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
#from ._base import _EmbeddingCausality, _DirectCausality
import itertools
import other_functions as OF
from itertools import tee
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.spatial.distance import cdist
from scipy.special import psi
from sklearn.linear_model import LinearRegression
import os
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import torch
#from other_functions import cal_mi_from_knn,cal_cmi_from_knn, embed_data,embed_to_dismat,dismat_to_idx,counts_zeros_of_dismat, exclude_vec_mat, revise_strength, weights_from_distances

class _ConventionCausality(object):
    def __init__(self, embed_dim):
        assert isinstance(embed_dim, int) and embed_dim > 1, \
            'Embedding dimension must be integer (> 1).'
        self.embed_dim = embed_dim

    def fit(self):
        self._fit_data()

    def _preprocess(self, data):
        assert data.ndim == 2, 'Input data should be 2D array of shape (n_points, n_variables).'
        assert np.all(~np.isnan(data)) and np.all(~np.isinf(data)), \
            'Unsupported data that contains nan or inf.'

        self.data = data
        self.n_var = data.shape[1]
        cut = self.data.shape[0] - self.embed_dim
        self.embeddings = OF.embed_data(data, self.embed_dim)[:, :cut]

    def _fit_data(self):
        raise NotImplementedError

class _EmbeddingCausality(object):
    def __init__(self, embed_dim, lag=1, n_neighbor=None, n_excluded=0):
        assert isinstance(embed_dim, int) and embed_dim > 1, \
            'Embedding dimension must be integer (> 1).'

        if n_neighbor is None:
            n_neighbor = embed_dim + 1
        assert isinstance(n_neighbor, int) and n_neighbor > 2, \
            'Number of neighbors must be integer (> 2).'

        assert isinstance(lag, int) and lag >= 1, \
            'Delay time constant must be integer (>= 1).'

        assert isinstance(n_excluded, int) and n_neighbor >= 0, \
            'Number of excluded neighbors must be integer (>= 1).'

        self.embed_dim = embed_dim
        self.n_neighbor = n_neighbor
        self.lag = lag
        self.n_excluded = n_excluded

    def fit(self):
        self._fit_data()

    def _preprocess(self, data):
        assert data.ndim == 2, 'Input data should be 2D array of shape (n_points, n_variables).'
        assert np.all(~np.isnan(data)) and np.all(~np.isinf(data)), \
            'Unsupported data that contains nan or inf.'

        self.data = data
        self.n_var = data.shape[1]

        self.embeddings = OF.embed_data(data, self.embed_dim, lag=self.lag)
        self.dismats = [OF.embed_to_dismat(e, self.n_excluded) for e in self.embeddings]
        self.ids = [OF.dismat_to_idx(dismat) for dismat in self.dismats]
        self.zeros = [OF.counts_zeros_of_dismat(dismat) for dismat in self.dismats]
        self.n_embed = self.embeddings[0].shape[0]
        n_val_neighbors = self.dismats[0].shape[1]

        self.skip_var_idx = []
        for i, zeros in enumerate(self.zeros):
            if not np.all(n_val_neighbors - zeros >= self.n_neighbor):
                print(f"Warning: does not have enough neighbors for variable {i}, "
                      "set causal strength to 0. by default.")
                self.skip_var_idx.append(i)

    def _fit_data(self):
        raise NotImplementedError

class _DirectCausality(object):
    def _fit_data_dir(self):
        raise NotImplementedError

    def _check_max_conditions(self, max_conditions):
        if isinstance(max_conditions, int):
            max_c = max_conditions
            if max_c <= 0:
                max_c = 1
            if max_c > self.n_var - 2:
                max_c = self.n_var - 2
        elif isinstance(max_conditions, str):
            if max_conditions == 'auto':
                if self.n_var > 5:
                    max_c = 3
                else:
                    max_c = self.n_var - 2
            elif max_conditions == 'full':
                max_c = self.n_var - 2
            elif max_conditions == 'fast':
                max_c = 1
            else:
                print('Warning: unknown max_conditions, set to 1 by default.')
                max_c = 1
        else:
            print('Warning: unknown max_conditions, set to 1 by default.')
            max_c = 1
        return max_c
#CCM  
class ConvergeCrossMapping(_EmbeddingCausality):
    def fit(self, data):
        self._preprocess(data)
        self._fit_data()

    def _fit_data(self):
        n_row = self.ids[0].shape[0]
        self.weights = []
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                tmp_weight = np.asarray([OF.weights_from_distances(
                    self.dismats[i][j][self.ids[i][j, self.zeros[i][j]:self.n_neighbor + self.zeros[i][j]]])
                               for j in range(n_row)])
            else:
                tmp_weight = None
            self.weights.append(tmp_weight)

        self.real = [OF.exclude_vec_mat(v, self.n_excluded) for v in self.data[-n_row:].T]
        self.scores = np.zeros((self.n_var, self.n_var))
        self.predicts = []
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                predicts_ = []
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        neighbor_real = np.asarray(
                            [self.real[i][k][self.ids[j][k, self.zeros[j][k]:self.n_neighbor + self.zeros[j][k]]]
                             for k in range(n_row)])
                        predict = np.average(neighbor_real, axis=1, weights=self.weights[j])
                        self.scores[i, j] = abs(stats.pearsonr(predict, self.data[-n_row:, i])[0])
                        predicts_.append(predict)
                    else:
                        predicts_.append(None)
            else:
                predicts_ = None
            self.predicts.append(predicts_)
        OF.revise_strength(self.scores)

    def _parse_embedding(self, embedding):
        dismat = OF.embed_to_dismat(embedding, self.n_excluded)
        idx = OF.dismat_to_idx(dismat)
        zero = OF.counts_zeros_of_dismat(dismat)
        return dismat, idx, zero

    def _eval_weight(self, dismat, idx, zero):
        weight = np.asarray([OF.weights_from_distances(
            dismat[j][idx[j, zero[j]:self.n_neighbor + zero[j]]]) for j in range(self.n_embed)])
        return weight
#PCM
class PartialCrossMapping(ConvergeCrossMapping, _DirectCausality):
    def fit(self, data, max_conditions='auto'):
        self._preprocess(data)
        self._fit_data()
        self._fit_data_dir(max_conditions)

    def _fit_data_dir(self, max_conditions):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = OF.cross_sum_row_and_col(self.scores)

        n_row = self.ids[0].shape[0]
        self.use_index = OF.exclude_range_mat(n_row, self.n_excluded)
        dir_scores = np.zeros((self.n_var, self.n_var, max_c))
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        predict = self.predicts[i][j]
                        n_condition = 0
                        tmp_idx = np.argsort(cross_sum[i, j])
                        tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                        for k in range(self.n_var):
                            if k != i and k != j and k in tmp_ex_idx[:max_c]:
                                try:
                                    neighbor_con = self._embedding_transform(k, j)
                                    dismat, idx, zero = self._parse_embedding(neighbor_con)
                                    weight_con = self._eval_weight(dismat, idx, zero)
                                    neighbor_real = np.asarray(
                                        [self.real[i][t][idx[t, zero[t]:self.n_neighbor + zero[t]]]
                                         for t in range(self.n_embed)])
                                    predict_con = np.average(neighbor_real, weights=weight_con, axis=1)
                                    dir_scores[i, j, n_condition] = abs(OF.partial_correlation(
                                        predict, self.data[-n_row:, i], predict_con))
                                except:
                                    dir_scores[i, j, n_condition] = self.scores[i, j]
                                n_condition += 1
        OF.revise_strength(dir_scores)
        self.scores = np.min(dir_scores, axis=-1)

    def _embedding_transform(self, i, j):
        embedding = np.asarray(
            [np.average(self.embeddings[i][self.use_index[k]][
                            self.ids[j][k, self.zeros[j][k]: self.zeros[j][k] + self.n_neighbor]],
                        weights=self.weights[j][k], axis=0) for k in range(self.n_embed)])
        return embedding

class _EmbeddingCausality2(object):
    def __init__(self, embed_dim, lag=1, n_neighbor=None, n_excluded=0):
        assert isinstance(embed_dim, int) and embed_dim > 1, \
            'Embedding dimension must be integer (> 1).'

        if n_neighbor is None:
            n_neighbor = embed_dim + 1
        assert isinstance(n_neighbor, int) and n_neighbor > 2, \
            'Number of neighbors must be integer (> 2).'

        assert isinstance(lag, int) and lag >= 1, \
            'Delay time constant must be integer (>= 1).'

        assert isinstance(n_excluded, int) and n_neighbor >= 0, \
            'Number of excluded neighbors must be integer (>= 1).'

        self.embed_dim = embed_dim
        self.n_neighbor = n_neighbor
        self.lag = lag
        self.n_excluded = n_excluded

    def fit(self):
        self._fit_data()

    def _preprocess(self, data):
        assert data.ndim == 2, 'Input data should be 2D array of shape (n_points, n_variables).'
        assert not (np.any(np.isnan(data)) and np.any(np.isinf(data))), \
            'Unsupported data that contains nan or inf.'

        self.n_var = data.shape[1]

        dismats = self._dismats(data)
        dismats1, dismats2 = tee(dismats, 2)
        self.ids = (OF.dismat_to_idx(dismat) for dismat in dismats1)
        self.zeros = [OF.counts_zeros_of_dismat(dismat) for dismat in dismats2]

        self.n_embed = data.shape[0] - self.lag * (self.embed_dim - 1)
        n_val_neighbors = self.n_embed - 1

        self.skip_var_idx = []
        for i, zeros in enumerate(self.zeros):
            if not np.all(n_val_neighbors - zeros >= self.n_neighbor):
                print(f"Warning: does not have enough neighbors for variable {i}, "
                      "set causal strength to 0. by default.")
                self.skip_var_idx.append(i)

    def _fit_data(self):
        raise NotImplementedError

    def _dismats(self, data):
        for i in range(self.n_var):
            e = OF.embed_vector(data[:, i], self.embed_dim, lag=self.lag)
            yield OF.embed_to_dismat(e, self.n_excluded)
#CMC
class CrossMappingCardinality(_EmbeddingCausality2):
    def fit(self, data):
        assert data.shape[1] > 1, 'data must have more than 1 column (variable).'
        self._preprocess(data)
        self._fit_data()

    def _fit_data(self):
        self.scores = np.zeros((self.n_var, self.n_var))
        maps = iter(self._idx_to_mapping())
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        ratio = self.mapping_to_ratio(next(maps))
                        self.scores[i, j] = self.ratio_to_score(ratio)
        OF.revise_strength(self.scores)

    def _idx_to_mapping(self):
        self.ids, ids_copy = tee(self.ids, 2)
        for i, idx in enumerate(ids_copy):
            if i not in self.skip_var_idx:
                self.ids, ids_copy = tee(self.ids, 2)
                for j, idy in enumerate(ids_copy):
                    if (i != j) and (j not in self.skip_var_idx):
                        yield OF.idx_to_mapping(idx, idy, self.zeros[j], self.n_neighbor)

    @staticmethod
    def count_mapping(map_x2y, tgt_neighbor):
        return len(np.where(map_x2y < tgt_neighbor)[-1])

    def mapping_to_ratio(self, map_x2y):
        n_row = map_x2y.shape[0]
        n_ele = map_x2y.size
        ratios_x2y = np.asarray([self.count_mapping(map_x2y, i) for i in range(n_row)]) / n_ele
        return ratios_x2y

    @staticmethod
    def ratio_to_auc(ratios):
        n_ele = len(ratios)
        neighbor_ratios = np.arange(n_ele) / (n_ele - 1)
        auc = metrics.auc(neighbor_ratios, ratios)
        return auc

    @staticmethod
    def auc_to_score(aucs):
        dat = np.array(aucs)
        dat[dat < 0.5] = 0.5
        scores = 2 * (dat - .5)
        return scores

    def ratio_to_score(self, ratios):
        return self.auc_to_score(self.ratio_to_auc(ratios))
#DCMC
class DirectCrossMappingCardinality(CrossMappingCardinality, _DirectCausality):
    def fit(self, data, max_conditions='auto', CMC_scores=None):
        assert data.shape[1] > 2, 'data must have more than 2 columns (variables).'
        self._preprocess(data)
        if CMC_scores is None:
            self._fit_data()
        else:
            self.scores = CMC_scores
        self._fit_data_dir(max_conditions)

    def _fit_data_dir(self, max_conditions):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = OF.cross_sum_row_and_col(self.scores)
        maps = self._idx_to_mapping_list()

        dir_scores = np.zeros((self.n_var, self.n_var, max_c))
        for i in range(self.n_var):
            if i not in self.skip_var_idx:
                for j in range(self.n_var):
                    if (i != j) and (j not in self.skip_var_idx):
                        n_condition = 0
                        tmp_idx = np.argsort(cross_sum[i, j])
                        tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                        for k in range(self.n_var):
                            if k != i and k != j and k in tmp_ex_idx[:max_c]:
                                try:
                                    ratios_i2j = self.mapping_to_ratio(maps[i][j])
                                    ratios_k2j = self.mapping_to_ratio(maps[k][j])
                                    map_i2k = maps[i][k]
                                    ratio = self.ratio_to_dir_ratio(
                                        ratios_i2j, ratios_k2j, map_i2k, self.n_neighbor)
                                    dir_scores[i, j, n_condition] = self.ratio_to_score(ratio)
                                except:
                                    dir_scores[i, j, n_condition] = self.scores[i, j]
                                n_condition += 1
        OF.revise_strength(dir_scores)
        self.scores = np.min(dir_scores, axis=-1)

    def _idx_to_mapping_list(self):
        maps = []
        self.ids, ids_copy = tee(self.ids, 2)
        for i, idx in enumerate(ids_copy):
            maps_ = []
            self.ids, ids_copy = tee(self.ids, 2)
            for j, idy in enumerate(ids_copy):
                if (i == j) or (i in self.skip_var_idx) or (j in self.skip_var_idx):
                    maps_.append(None)
                else:
                    maps_.append(OF.idx_to_mapping(idx, idy, self.zeros[j], self.n_neighbor))
            maps.append(maps_)
        return maps

    def ratio_to_dir_ratio(self, ratios_x2y, ratios_z2y, map_x2z, n_neighbor):
        dir_ratios_x2y = ratios_x2y - self.count_mapping(map_x2z, n_neighbor) / map_x2z.size * ratios_z2y
        return dir_ratios_x2y
#CME
class CrossMappingEntropy(_EmbeddingCausality):
    def fit(self, data, mi_kwargs=None):
        assert data.shape[1] > 1, 'data must have more than 1 column (variable).'
        self._preprocess(data)
        self._fit_data(mi_kwargs)

    def _fit_data(self, mi_kwargs):
        n_row = self.ids[0].shape[0]
        mi_kwargs = {} if mi_kwargs is None else mi_kwargs.copy()
        self.use_index = OF.exclude_range_mat(n_row, self.n_excluded)
        self.scores = np.zeros((self.n_var, self.n_var))
        for i in range(self.n_var):
            neighbor_tgt = self._neighbor(i, i)
            for j in range(self.n_var):
                if i != j:
                    neighbor_src = self._neighbor(i, j)
                    self.scores[i, j] = OF.cal_mi_from_knn(neighbor_src, neighbor_tgt, **mi_kwargs)
        OF.revise_strength(self.scores)

    def _neighbor(self, i, j):
        return np.asarray(
                [self.embeddings[i][self.use_index[k]][
                     self.ids[j][k, self.zeros[j][k]: self.zeros[j][k] + self.n_neighbor]].ravel()
                 for k in range(self.n_embed)])
#DCME
class DirectCrossMappingEntropy(CrossMappingEntropy, _DirectCausality):
    def fit(self, data, mi_kwargs=None, max_conditions='auto'):
        assert data.shape[1] > 2, 'data must have more than 2 columns (variables).'
        self._preprocess(data)
        self._fit_data(mi_kwargs)
        self._fit_data_dir(max_conditions, mi_kwargs)

    def _fit_data_dir(self, max_conditions, mi_kwargs):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = OF.cross_sum_row_and_col(self.scores)
        dir_scores = np.zeros((self.n_var, self.n_var, max_c))

        n_row = self.ids[0].shape[0]
        mi_kwargs = {} if mi_kwargs is None else mi_kwargs.copy()
        self.use_index = OF.exclude_range_mat(n_row, self.n_excluded)
        for i in range(self.n_var):
            neighbor_tgt = self._neighbor(i, i)
            for j in range(self.n_var):
                if i != j:
                    neighbor_src = self._neighbor(i, j)
                    n_condition = 0
                    tmp_idx = np.argsort(cross_sum[i, j])
                    tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                    for k in range(self.n_var):
                        if k != i and k != j and k in tmp_ex_idx[:max_c]:
                            neighbor_con = self._neighbor(i, k)
                            dir_scores[i, j, n_condition] = OF.cal_cmi_from_knn(
                                neighbor_src, neighbor_tgt, neighbor_con, **mi_kwargs)
                            n_condition += 1
        self.scores = np.min(dir_scores, axis=-1)
        OF.revise_strength(self.scores)

class DirectCrossMappingEntropySimple(CrossMappingEntropy, _DirectCausality):
    def fit(self, data, mi_kwargs=None, max_conditions='auto'):
        assert data.shape[1] > 2, 'data must have more than 2 columns (variables).'
        self._preprocess(data)
        self._fit_data(mi_kwargs)
        self._fit_data_dir(max_conditions)

    def _fit_data_dir(self, max_conditions):
        max_c = self._check_max_conditions(max_conditions)
        cross_sum = OF.cross_sum_row_and_col(self.scores)
        dir_scores = np.zeros((self.n_var, self.n_var, max_c))
        for i in range(self.n_var):
            for j in range(self.n_var):
                if i != j:
                    n_condition = 0
                    tmp_idx = np.argsort(cross_sum[i, j])
                    tmp_ex_idx = tmp_idx[~np.isin(tmp_idx, [i, j])]
                    for k in range(self.n_var):
                        if k != i and k != j and k in tmp_ex_idx[:max_c]:
                            indir_score = self._indiret_score(i, j, k)
                            dir_scores[i, j, n_condition] = self.scores[i, j] - indir_score
                            n_condition += 1
        self.scores = np.min(dir_scores, axis=-1)
        OF.revise_strength(self.scores)

    def _indiret_score(self, i, j, k):
        return self.scores[i, k] * self.scores[k, j]
#GC
class GrangerCausality(_ConventionCausality):
    def fit(self, data, ddof=1, **kwargs):
        self._preprocess(data)
        self._fit_data(ddof=1, **kwargs)

    def _fit_data(self, ddof=1, **kwargs):
        self.scores = np.zeros((self.n_var, self.n_var))
        for i in range(self.n_var):
            for j in range(self.n_var):
                if i != j:
                    self.scores[i, j] = self._estimate_gc_index(
                        self.embeddings[j],
                        np.hstack((self.embeddings[i], self.embeddings[j])),
                        self.data[self.embed_dim:, j],
                        ddof,
                        **kwargs
                    )
        OF.revise_strength(self.scores)

    @staticmethod
    def _estimate_gc_index(M1, M2, Y_real, ddof=1, **kwargs):
        """
        estimate the GC index based on single and composite manifolds.

        Parameters
        ----------
        M1: 2d array
            single manifold
        M2: 2d array
            composite manifold
        Y_real: 1d array
            real time series
        ddof: int
            “Delta Degrees of Freedom”
        kwargs:
            other keyword arguments are passed through to "LinearRegression"

        Returns
        -------
        gc_estimated: float
            estimated Granger causality
        """
        # calculate the error between predicted y (based on M1) and real y
        LR1 = LinearRegression(**kwargs).fit(M1, Y_real)
        Y_pred_LR1 = LR1.predict(M1)
        error1 = Y_pred_LR1 - Y_real

        # calculate the error between predicted y (based on M2) and real y
        LR2 = LinearRegression(**kwargs).fit(M2, Y_real)
        Y_pred_LR2 = LR2.predict(M2)
        error2 = Y_pred_LR2 - Y_real

        # estimate the gc index based on two errors
        gc_estimated = -np.log(
            np.var(error2, ddof=ddof) / np.var(error1, ddof=ddof))
        return gc_estimated
#TE
class TransferEntropy(_ConventionCausality):
    def fit(self, data, mi_kwargs=None):
        self._preprocess(data)
        self._fit_data(mi_kwargs)

    def _fit_data(self, mi_kwargs=None):
        mi_kwargs = {} if mi_kwargs is None else mi_kwargs.copy()
        self.scores = np.zeros((self.n_var, self.n_var))
        for i in range(self.n_var):
            for j in range(self.n_var):
                if i != j:
                    self.scores[i, j] = OF.cal_cmi_from_knn(
                        self.data[self.embed_dim:, j:j + 1],
                        self.embeddings[i],
                        self.embeddings[j],
                        **mi_kwargs
                    )
        OF.revise_strength(self.scores)

def main_methods(data,res_dir,embed_dim,n_neighbor,n_excluded):
    #embed_dim = 20
    #n_neighbor = 3
    GC = GrangerCausality(embed_dim=embed_dim)
    TE = TransferEntropy(embed_dim=embed_dim)
    CCM = ConvergeCrossMapping(embed_dim=embed_dim, n_neighbor=n_neighbor, n_excluded=n_excluded)
    PCM = PartialCrossMapping(embed_dim=embed_dim, n_neighbor=n_neighbor, n_excluded=n_excluded)
    CME = CrossMappingEntropy(embed_dim=embed_dim, n_neighbor=n_neighbor, n_excluded=n_excluded)
    DCME = DirectCrossMappingEntropy(embed_dim=embed_dim, n_neighbor=n_neighbor, n_excluded=n_excluded)
    CMC = CrossMappingCardinality(embed_dim=embed_dim, n_neighbor=n_neighbor, n_excluded=n_excluded)
    DCMC = DirectCrossMappingCardinality(embed_dim=embed_dim, n_neighbor=n_neighbor, n_excluded=n_excluded)
    
    methods = [GC, TE, CCM, PCM, CME, DCME, CMC, DCMC]
    labels = ['GC', 'TE', 'CCM', 'PCM', 'CME', 'DCME', 'CMC', 'DCMC']
    
  
    #res_dir = f'results/foodchain/'
    method_Net_scores = []
    for i, method in enumerate(methods):
        method.fit(data)
        method_Net_scores.append(method.scores)
        np.save(f'{res_dir}method{labels[i]}.npy', method.scores)
        print(f'method {labels[i]} complete!')
    return method_Net_scores
    

def threshold_methods(Net_ground,Net_causal,quantile):
    row_indices, col_indices = np.where(Net_ground == 1)
    element = torch.tensor(Net_causal[row_indices, col_indices]).float()
    element_sorted, _ = torch.sort(element, descending=True)
    threshold = element_sorted[int(len(element_sorted) * quantile)-1]
    i=1
    while True:
        if threshold <= 0:
            threshold = element_sorted[int(len(element_sorted) * quantile) - (1 + i)]
            i += 1
        if threshold != 0:
            break
    threshold =threshold - 0.001
    return  threshold

def threshold_methods1(Net_ground,Net_causal,quantile):
    #row_indices, col_indices = np.where(Net_ground == 1)
    #element = torch.tensor(Net_causal[row_indices, col_indices]).float()
  
    diagonal_mask = torch.eye(torch.tensor(Net_causal).float().size(0), dtype=torch.bool)
    non_diagonal_elements = torch.masked_select(torch.tensor(Net_causal).float(), ~diagonal_mask)
    element_sorted, _ = torch.sort(non_diagonal_elements, descending=False)
    threshold = element_sorted[int(len(element_sorted) * quantile)-1]
    i=1
    while True:
        if threshold <= 0:
            if i<=int(len(element_sorted)):
                threshold = element_sorted[int(len(element_sorted) * quantile) - (1 + i)]
                i += 1
            else:
                break
        if threshold != 0:
            break
    threshold =threshold - 0.0001
    return  threshold
def diag2(matrix,value):
    diag_indices = torch.arange(min(matrix.size(0), matrix.size(1)))
    matrix[diag_indices, diag_indices] = value
    return  matrix
def compare_methods(res_dir,Net_ground,causal_index, Net_causal,num):
    methods = ['CIC', 'GC', 'TE', 'CCM', 'CME', 'CMC', 'PCM', 'DCME', 'DCMC']
    N=Net_ground.shape[1]
    #mat_diff8 = {}
    Causal_diff8 = {};CauNet_diff8 = {}
    #Nega=np.count_nonzero((Net_ground != 1))
    thrs=np.zeros((len(methods), num));Causal_thrs=np.zeros((len(methods), 1))

    TP=np.zeros((len(methods), num));FP=np.zeros((len(methods), num));TN=np.zeros((len(methods), num));FN=np.zeros((len(methods), num))
    tpr=np.zeros((len(methods), num));fpr=np.zeros((len(methods), num));roc_auc=np.zeros((len(methods), 1))
    precision0=np.zeros((len(methods), num));precision1=np.zeros((len(methods), num));recall0=np.zeros((len(methods), num));recall1=np.zeros((len(methods), num))
    accuracy=np.zeros((len(methods), num)); f1=np.zeros((len(methods), num))
    TPR={};FPR={};Precision0={};Precision1={};Recall0={};Recall1={};Accuracy={}; F1={}
    Net_groud=Net_ground.clone()
    Net_groud=diag2(Net_groud, 2)
    for k, method in enumerate(methods):
        Causal_Net=torch.zeros((N, N))
        #1 plot
        if k > 0:
            mat= np.load(f'{res_dir}method{method}.npy')#_diff8[f"X{k}"] 
            Causal_thrs[k]=threshold_methods1(Net_groud,mat,0.75)
            for i in range(N):
                for j in range(N):
                    if i != j:         
                        if mat[i,j]>=Causal_thrs[k]:
                            Causal_Net[i,j]=1
                        elif mat[i,j]<Causal_thrs[k]:
                            Causal_Net[i,j]=0
        else:
            Causal_Net=Net_causal    
        CauNet_diff8[f"X{k}"]=Causal_Net
        
        #2 AUC
        if k > 0:             
            Net=torch.from_numpy(mat)
            Net=Net.masked_fill(torch.isinf(Net).to(torch.bool), 0)
            thrs[k,] = np.linspace(torch.max(Net)+0.01, 0, num)
            #thrs[k,] = ([1,0.75,0.5,0.25,0])
        else:
            Net=causal_index
            thrs[k,] = np.linspace(torch.max(Net)+0.01, 0, num)
            #thrs[k,] = ([0, 1/3, 2/3, 1, 3])

        for l, threshold in enumerate(thrs[k,]):
            if k > 0:
                y_pred = torch.where(Net >= threshold, 1, 0)
                y_pred=diag2(y_pred, 2)
            else:
                y_pred = torch.where(Net >= threshold, 1, 0)
                #y_pred = torch.where(torch.logical_and(Net >= threshold, Net > 0), 1, 0)
            TP[k,l] = np.count_nonzero((Net_groud.reshape(-1) == 1) & (y_pred.reshape(-1) == 1))#预测为正，实际为正
            FN[k,l] = np.count_nonzero((Net_groud.reshape(-1) == 1) & (y_pred.reshape(-1) == 0))#预测为负，实际为正
            TN[k,l] = np.count_nonzero((Net_groud.reshape(-1) == 0) & (y_pred.reshape(-1) == 0))#预测为负，实际为负#+np.count_nonzero((Net_groud == 2) & (Net == 2))
            FP[k,l] = np.count_nonzero((Net_groud.reshape(-1) == 0) & (y_pred.reshape(-1) == 1))#Nega-TN[k,l]#预测为正，实际为负##Nega-TN[k,]
            tpr[k,l]=(TP[k,l] / (TP[k,l] + FN[k,l]))
            fpr[k,l]=(FP[k,l] / (FP[k,l] + TN[k,l]))
            
            precision0[k,l]=(TN[k,l] / (TN[k,l] + FP[k,l]+0.000001))
            #precision1[k,l]=(TP[k,l] / (TP[k,l] + FP[k,l]+0.000001))
            recall0[k,l]=(TN[k,l] / (TN[k,l] + FN[k,l]+0.000001))
            #recall1[k,l]=(TP[k,l] / (TP[k,l] + FN[k,l]+0.000001))
            accuracy[k,l]=(TP[k,l]+TN[k,l]) / (TP[k,l] + TN[k,l] + FP[k,l]+ FN[k,l])
            #f1[k,l]=(2*precision0[k,l]+recall1[k,l])/(precision1[k,l]+recall1[k,l]+0.000001)
              
            #fpr[k,], tpr[k,], thresholds = roc_curve(Net_groud.view(-1), Net.view(-1))
        FPR[f"X{k}"]=fpr[k,]; TPR[f"X{k}"]=tpr[k,]
        precision1, recall1, thresholds = precision_recall_curve(Net_ground.reshape(-1), Net.reshape(-1))
        Precision0[f"X{k}"]=precision0[k,]; Precision1[f"X{k}"]=precision1;Recall0[f"X{k}"]=recall0[k,];Recall1[f"X{k}"]=recall1
        Accuracy[f"X{k}"]=accuracy[k,]; F1[f"X{k}"]=f1[k,]
        roc_auc[k] = round(auc(FPR[f"X{k}"], TPR[f"X{k}"]),3)
        Causal_diff8[f"X{k}"]=Net
    #return TP, FP, TN, FN, FPR, TPR, roc_auc,methods,Causal_thrs,Causal_diff8,CauNet_diff8,thrs
    return TP, FP, TN, FN, FPR, TPR, Precision0, Precision1, Recall0, Recall1, Accuracy, roc_auc,methods,CauNet_diff8,Causal_diff8,Causal_thrs

def compare_methods1(res_dir,Net_ground,causal_index, Net0,num):
    methods = ['CIC', 'GC', 'TE', 'CCM', 'CME', 'CMC', 'PCM', 'DCME', 'DCMC']
    N=Net_ground.shape[1]
    #mat_diff8 = {}
    Causal_diff8 = {};CauNet_diff8 = {}
    num=3
    Nega=np.count_nonzero((Net_ground != 1))
    thrs=np.zeros((len(methods), 1))
    TP=np.zeros((len(methods), num));FP=np.zeros((len(methods), num));TN=np.zeros((len(methods), num));FN=np.zeros((len(methods), num))
    tpr=np.zeros((len(methods), num));fpr=np.zeros((len(methods), num));roc_auc=np.zeros((len(methods), 1))
    precision0=np.zeros((len(methods), num));
    #precision1=np.zeros((len(methods), num));
    recall0=np.zeros((len(methods), num));
    #recall1=np.zeros((len(methods), num))
    accuracy=np.zeros((len(methods), num)); f1=np.zeros((len(methods), num))
    TPR={};FPR={};Precision0={};Precision1={};Recall0={};Recall1={};Accuracy={}; F1={}
    for k, method in enumerate(methods):
        if k >0:
            mat= np.load(f'{res_dir}method{method}.npy')#_diff8[f"X{k}"] 
            mat[np.isinf(mat)] = 0
            thrs[k]=threshold_methods1(Net_ground,mat,0.65)
            Net=torch.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if i != j:         
                        if mat[i,j]>=thrs[k]:
                            Net[i,j]=1
                        elif mat[i,j]<thrs[k]:
                            Net[i,j]=0
            Causal_diff8[f"X{k}"]=mat
            CauNet_diff8[f"X{k}"]=Net
        else:
            Causal_diff8[f"X{k}"]=causal_index
            Net=Net0
            CauNet_diff8[f"X{k}"]=Net
        thresholds = np.linspace(2, 0, num)
        for l, threshold in enumerate(thresholds):
            y_pred = torch.where(Net >= threshold, 1, 0)
            y_pred=diag2(y_pred, 2)
            TP[k,l] = np.count_nonzero((y_pred.reshape(-1) == 1) & (Net_ground.reshape(-1) == 1))#
            FN[k,l] = np.count_nonzero((y_pred.reshape(-1) == 0) & (Net_ground.reshape(-1) == 1))#
            TN[k,l] = np.count_nonzero((y_pred.reshape(-1) == 0) & (Net_ground.reshape(-1) == 0))#
            FP[k,l] = np.count_nonzero((y_pred.reshape(-1) == 1) & (Net_ground.reshape(-1) == 0))#
            #tpr[k,l]=(TP[k,l] / (TP[k,l] + FN[k,l]))
            #fpr[k,l]=(FP[k,l] / (FP[k,l] + TN[k,l]))
            
            precision0[k,l]=(TN[k,l] / (TN[k,l] + FP[k,l]+0.000001))
            #precision1[k,l]=(TP[k,l] / (TP[k,l] + FP[k,l]+0.000001))
            recall0[k,l]=(TN[k,l] / (TN[k,l] + FN[k,l]+0.000001))
            #[k,l]=(TP[k,l] / (TP[k,l] + FN[k,l]+0.000001))
            accuracy[k,l]=(TP[k,l]+TN[k,l]) / (TP[k,l] + TN[k,l] + FP[k,l]+ FN[k,l])
            #f1[k,l]=(2*precision0[k,l]+recall1[k,l])/(precision1[k,l]+recall1[k,l]+0.000001)
              
            #fpr[k,], tpr[k,], thresholds = roc_curve(Net_groud.view(-1), Net.view(-1))
        
        precision1, recall1, thresholds = precision_recall_curve(Net_ground[~torch.eye(Net_ground.shape[0], dtype=bool)].view(-1), Net[~torch.eye(Net.shape[0], dtype=bool)].view(-1))
        fpr, tpr, thresholds = roc_curve(Net_ground[~torch.eye(Net_ground.shape[0], dtype=bool)].reshape(-1), Net[~torch.eye(Net.shape[0], dtype=bool)].reshape(-1))
        #f1=(2*precision0+recall1)/(precision1+recall1+0.000001)
        FPR[f"X{k}"]=fpr ; TPR[f"X{k}"]=tpr 
        Precision0[f"X{k}"]=precision0[k,]; Precision1[f"X{k}"]=precision1 ;Recall0[f"X{k}"]=recall0[k,];Recall1[f"X{k}"]=recall1 
        Accuracy[f"X{k}"]=accuracy[k,]; #F1[f"X{k}"]=f1[k,]
        roc_auc[k] = round(auc(FPR[f"X{k}"], TPR[f"X{k}"]),3)
        #roc_auc[k] = auc(fpr[k,], tpr[k,])
    return TP, FP, TN, FN, FPR, TPR, Precision0, Precision1, Recall0, Recall1, Accuracy, roc_auc,methods,CauNet_diff8,Causal_diff8,thrs
    #return TP, FP, TN, FN, FPR, TPR, roc_auc,methods,Causal_thrs,Causal_diff8,CauNet_diff8,thrs

def confounder_index(Net,Net_ground):
    #y_pred=Net
    TP = np.count_nonzero((Net.reshape(-1) == 1) & (Net_ground.reshape(-1) == 1))#
    FN = np.count_nonzero((Net.reshape(-1) == 0) & (Net_ground.reshape(-1) == 1))#
    TN = np.count_nonzero((Net.reshape(-1) == 0) & (Net_ground.reshape(-1) == 0))#
    FP = np.count_nonzero((Net.reshape(-1) == 1) & (Net_ground.reshape(-1) == 0))#
    precision1, recall1, thresholds = precision_recall_curve(Net_ground[~torch.eye(Net_ground.shape[0], dtype=bool)].reshape(-1), Net[~torch.eye(Net.shape[0], dtype=bool)].reshape(-1))
    precision0=(TN / (TN + FP+0.000001))
    #precision1=(TP / (TP + FP+0.000001))
    recall0=(TN / (TN + FN+0.000001))
    #recall1=(TP / (TP + FN+0.000001))
    accuracy=(TP+TN) / (TP + TN + FP+ FN)
    f1=(2*precision0+recall1)/(precision1+recall1+0.000001)  
        #fpr[k,], tpr[k,], thresholds = roc_curve(Net_groud.view(-1), Net.view(-1))
        #roc_auc[k] = auc(fpr[k,], tpr[k,])
    return precision0, precision1, recall0, recall1, accuracy, f1