"""Many methods for continuous univariate analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ContinuousVariableAnalysis(object):

    # Data name
    name = None
    # Data to be analyzed
    data = None
    # Missing values
    missing = None

    # Whether to show the plotted fig, reserved.
    is_show = True

    def __init__(self, data, name, missing, is_show=True):

        self.missing = list(set(missing))
        self.name = name.lower().capitalize()
        self.data = np.array([np.nan if x in self.missing else x for x in list(data)])
        self.is_show = is_show

        # Mask to find out missing values.
        self.mask = np.isnan(self.data)

    @property
    def count(self):
        """Calculate data count."""

        return len(self.data)

    @property
    def _count_of_missing(self):
        """Calculate data count of missing values."""

        return len(self.data[self.mask])

    @property
    def _count_without_missing(self):
        """Calculate data count without missing values."""

        return self.count - self._count_of_missing

    @property
    def coverage(self):
        """Calculate data coverage."""

        return 1.0 * self._count_without_missing / self.count

    @property
    def mean(self):
        """Calculate variable mean without missing values."""

        return np.mean(self.data[~self.mask])

    @property
    def median(self):
        """Calculate variable median without missing values."""

        return np.median(self.data[~self.mask])

    @property
    def modes(self):
        """Calculate variable modes without missing values."""

        return pd.Series(self.data).mode().values

    @property
    def max(self):
        """Calculate variable maximum without missing values."""

        return np.max(self.data[~self.mask])

    @property
    def min(self):
        """Calculate variable minimum without missing values."""

        return np.min(self.data[~self.mask])

    @property
    def std(self):
        """Calculate variable std without missing values."""

        return np.std(self.data[~self.mask])

    @property
    def _value_count(self):
        """Calculate count for every value without missing values."""

        data_df = pd.Series(self.data)
        return data_df.groupby(by=data_df.values).count()

    @property
    def percentile(self):
        """Calculate percentiles from 0.00 to 1.00 with 0.05 interval."""

        quantile = 0.05

        data_df = pd.Series(self.data)
        index = pd.Series(np.arange(0, 1.0001, quantile))
        value = index.apply(lambda x: data_df.quantile(x))
        return {'index': index.as_matrix(), 'value': value.as_matrix()}

    @property
    def percentile_line_chart(self):
        """Draw percentile line chart for the variable."""

        plt.close()
        plt.plot(self.percentile['index'], self.percentile['value'])
        plt.xlabel("percentile")
        plt.ylabel("value")
        plt.title("Percentile Line Chart For continuous Variable %s" % self.name)
        self._assert_show()

        return [self.percentile['index'], self.percentile['value']]

    @property
    def segmentation_count_distribution(self):
        """Divide the variable range to 10 equilong segmentation and calculate their data counts."""

        # it should equal to the 'segs' in 'segmentation_percentage_distribution'
        segs = 10

        data_df = pd.Series(self.data)
        range_min = self.min * 0.99
        range_max = self.max * 1.01
        step = 1.0 / segs * (range_max - range_min)
        start = np.round(np.arange(range_min, range_max, step), 2)
        end = np.round(np.concatenate([start[1:], [range_max]]), 2)
        count = np.array([data_df[(data_df >= start[i]) & (data_df < end[i])].count() for i in range(segs)])
        return {'index': np.concatenate([[start], [end]]).T, 'value': count}

    @property
    def segmentation_percentage_distribution(self):
        """Divide the variable range to 10 equilong segmentation and calculate their percentage."""

        return {'index': self.segmentation_count_distribution['index'],
                'value': 1.0 * self.segmentation_count_distribution['value'] / self.count}

    @property
    def entropy(self):
        """Calculate variable entropy without missing values."""

        prob = 1.0 * self._value_count.values / self._count_without_missing
        return -1.0 * pd.Series(prob * np.log(prob) / np.log(2)).sum()

    @property
    def gini(self):
        """Calculate variable gini without missing values."""

        return 1 - 1.0 * np.square(self._value_count.values).sum() / self._count_without_missing ** 2

    @property
    def segmentation_distribution_map(self):
        """Draw segmentation distribution without missing values."""

        start = self.segmentation_percentage_distribution['index'].T[0]
        end = self.segmentation_percentage_distribution['index'].T[1]
        percentage = self.segmentation_percentage_distribution['value']

        plt.close()
        plt.bar(start, percentage, width=(end - start), tick_label=start, align='edge', alpha=0.5)
        plt.xlabel("variable")
        plt.ylabel("percentage")
        plt.title("Segmentation Distribution Map For Continuous Variable %s" % self.name)
        self._assert_show()

        return [self.segmentation_percentage_distribution['index'], percentage]

    @property
    def probability_distribution_map(self):
        """Draw probability distribution map without missing values."""

        plt.close()
        sns.distplot(self.data[~self.mask], hist=True, kde_kws={"shade": False})
        plt.xlabel("variable")
        plt.ylabel("percentage")
        plt.title("Distribution Distribution Map For Continuous Variable %s" % self.name)
        self._assert_show()

        return self.data[~self.mask]

    @property
    def boxplot(self):
        """Draw boxplot without missing values."""

        plt.close()
        plt.boxplot(self.data[~self.mask], meanline=True, labels=[self.name])
        plt.title("Boxplot For Continuous Variable %s" % self.name)
        self._assert_show()

        return self.data[~self.mask]

    def _assert_show(self):
        """Assert for whether to show the chart."""

        if self.is_show:
            try:
                plt.show()
            except:
                raise Exception("Could not plot the chart.")
