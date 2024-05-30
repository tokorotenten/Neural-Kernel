from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.spatial.distance import cdist



def wasserstein(true_points, sample_points):
  WAS=np.zeros((true_points.shape[0], 1))
  if true_points.shape[2]==1:
    for i in range(len(true_points)):
      WAS[i]=scipy.stats.wasserstein_distance(true_points[i][:,0], sample_points[i][:,0])
  else:
    for i in range(len(true_points)):
      d=cdist(true_points[i], sample_points[i])
      assignment= scipy.optimize.linear_sum_assignment(d)
      WAS[i]=d[assignment].sum()/true_points.shape[1]
  return WAS

def MSE(test_data, sample_points):
  prediction = sample_points.mean(axis=1)
  mse = ((test_data - prediction)**2).mean()
  return mse


"""
The codes here are taken from the Github repository of the CARD (Han et al., 2022)
https://github.com/XzwHan/CARD/blob/main/regression/card_regression.py
"""
def compute_QI(n_bins, all_true_y, all_generated_y, verbose=False):
  quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
  # compute generated y quantiles
  y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
  y_true = all_true_y.T
  quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
  y_true_quantile_membership = quantile_membership_array.sum(axis=0)
  # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
  y_true_quantile_bin_count = np.array([(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])
  # combine true y falls outside of 0-100 gen y quantile to the first and last interval
  y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
  y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
  y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
  # compute true y coverage ratio for each gen y quantile interval
  y_true_ratio_by_bin = y_true_quantile_bin_count_ / len(all_true_y)#dataset_object.test_n_samples
  #print(y_true_ratio_by_bin)
  assert np.abs(np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
  qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
  return y_true_ratio_by_bin, qice_coverage_ratio, y_true

