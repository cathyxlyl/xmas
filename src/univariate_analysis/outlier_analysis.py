"""Many methods for outlier analysis."""

import numpy as np
import pandas as pd

from .continuous_variable_analysis import ContinuousVariableAnalysis
from .discrete_variable_analysis import DiscreteVariableAnalysis


class OutlierAnalysis (object):

    # The variable analyzer
    analyzer = None

    def __init__(self, analyzer=None):

        self.analyzer = analyzer


#######################################################################################################################
class ContinuousOutlierAnalysis (OutlierAnalysis):
    def __init__(self, analyzer=None):

        super(ContinuousOutlierAnalysis, self).__init__(analyzer)
        if not isinstance(self.analyzer, ContinuousVariableAnalysis):
            raise TypeError("The analyzer should by in type of 'ContinuousVariableAnalysis'.")

    def outlier_by_quantile(self, lower=0.05, upper=0.95):
        """Calculate outliers by quantile, missing values are ignored for outliers.
        
        Parameters
        ----------
        lower: double, optimal, (default=0.05)
            The lower percentile bound for judging outliers.
        upper: double, optimal, (default=0.95)
            The upper percentile bound for judging outliers.
        
        Returns
        -------
        outlier_info: dict
            The outlier_info has five keys, which are: 
            a) lower_bound: double, lower bound of normal values.
            b) upper_bound: double, upper bound of normal values.
            c) outlier_mask: numpy.ndarray, masks for outliers, True for outliers.
            d) outlier_count: int, counts for outliers.
            e) outlier_percentage: double, percentage of outliers out of all samples (including missing values).
        """

        data_df = pd.Series(self.analyzer.data)
        lower_bound = data_df.quantile(lower)
        upper_bound = data_df.quantile(upper)
        outlier_mask = data_df.apply(lambda x: True if x < lower_bound or x > upper_bound else False)
        outlier_count = outlier_mask[outlier_mask].count()
        outlier_info = {'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'outlier_mask': outlier_mask.as_matrix(),
                        'outlier_count': outlier_count,
                        'outlier_percentage': 1.0 * outlier_count / self.analyzer.count
                        }
        return outlier_info

    def outlier_by_std(self, num=2):
        """Calculate outliers by std, missing values are ignored for outliers.

        Parameters
        ----------
        num: int, optimal, (default=2)
            Numbers of std to judging outliers.

        Returns
        -------
        outlier_info: dict
            The outlier_info has five keys, which are: 
            a) lower_bound: double, lower bound of normal values.
            b) upper_bound: double, upper bound of normal values.
            c) outlier_mask: numpy.ndarray, masks for outliers, True for outliers.
            d) outlier_count: int, counts for outliers.
            e) outlier_percentage: double, percentage of outliers out of all samples (including missing values).
        """

        mean = self.analyzer.mean
        std = self.analyzer.std

        data_df = pd.Series(self.analyzer.data)
        lower_bound = mean - num * std
        upper_bound = mean + num * std
        outlier_mask = data_df.apply(lambda x: True if x < lower_bound or x > upper_bound else False)
        outlier_count = outlier_mask[outlier_mask].count()
        outlier_info = {'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'outlier_mask': outlier_mask.as_matrix(),
                        'outlier_count': outlier_count,
                        'outlier_percentage': 1.0 * outlier_count / self.analyzer.count
                        }
        return outlier_info

    def outlier_by_boxplot(self):
        """Calculate outliers by std, missing values are ignored for outliers.

        Returns
        -------
        outlier_info: dict
            The outlier_info has five keys, which are: 
            a) lower_bound: double, lower bound of normal values.
            b) upper_bound: double, upper bound of normal values.
            c) outlier_mask: numpy.ndarray, masks for outliers, True for outliers.
            d) outlier_count: int, counts for outliers.
            e) outlier_percentage: double, percentage of outliers out of all samples (including missing values).
        """

        data_df = pd.Series(self.analyzer.data)
        quantile_25 = data_df.quantile(0.25)
        quantile_75 = data_df.quantile(0.75)
        lower_bound = quantile_25 - (quantile_75 - quantile_25) * 1.5
        upper_bound = quantile_75 + (quantile_75 - quantile_25) * 1.5
        outlier_mask = data_df.apply(lambda x: True if x < lower_bound or x > upper_bound else False)
        outlier_count = outlier_mask[outlier_mask].count()
        outlier_info = {'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'outlier_mask': outlier_mask.as_matrix(),
                        'outlier_count': outlier_count,
                        'outlier_percentage': 1.0 * outlier_count / self.analyzer.count
                        }
        return outlier_info

    def outlier_integration(self, masks, num=None):
        """Calculate outliers by multi-masks.
        
        Parameters
        ----------
        masks: list[numpy.ndarray]
            Multi-masks for identify outliers. The element of the list should be value of key 'outlier_mask' from
             results of outlier identifying methods.
        num: int, optimal, (default=None)
            For a single sample, the minimum number that different outlier identifying methods judge it as a outlier
             to decide whether it is an outlier. When its values is None, it requires that all methods in masks list
             to be true to regard it as an outlier.
        
        Returns
        -------
        outlier_info: dict
            The outlier_info has five keys, which are: 
            a) lower_bound: double, lower bound of normal values.
            b) upper_bound: double, upper bound of normal values.
            c) outlier_mask: numpy.ndarray, masks for outliers, True for outliers.
            d) outlier_count: int, counts for outliers.
            e) outlier_percentage: double, percentage of outliers out of all samples (including missing values).
        """

        mask_count = len(masks)
        if num is None:
            num = mask_count
        if mask_count < 1:
            raise ValueError("There should be at least one outlier_mask.")
        if mask_count > num:
            raise ValueError("The length of outlier masks should not be larger than the num.")

        judge_count = np.array(masks, dtype=np.int32).sum(axis=0)
        outlier_mask = np.array([True if x >= num else False for x in judge_count])
        outlier_count = outlier_mask[outlier_mask].count()

        normal_data = self.analyzer.data[(~self.analyzer.mask) & (~outlier_mask)]

        outlier_info = {'lower_bound': np.min(normal_data),
                        'upper_bound': np.max(normal_data),
                        'outlier_mask': outlier_mask,
                        'outlier_count': outlier_count,
                        'outlier_percentage': 1.0 * outlier_count / self.analyzer.count
                        }
        return outlier_info


#######################################################################################################################
class DiscreteOutlierAnalysis (OutlierAnalysis):
    def __init__(self, analyzer=None):

        super(DiscreteOutlierAnalysis, self).__init__(analyzer)
        if not isinstance(self.analyzer, DiscreteVariableAnalysis):
            raise TypeError("The analyzer should by in type of 'DiscreteVariableAnalysis'.")
