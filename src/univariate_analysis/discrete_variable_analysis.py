"""Many methods for continuous univariate analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DiscreteVariableAnalysis(object):

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
    def categories(self):
        """Get unique categories with missing values."""

        return np.unique(self.data[~self.mask])

    @property
    def category_num(self):
        """Calculate category num with missing values."""

        return len(self.categories)

    @property
    def modes(self):
        """Calculate modes for the variable."""

        return pd.Series(self.data).mode().values

    @property
    def category_count_distribution(self):
        """Calculate count for each category."""

        data_df = pd.Series(self.data)
        category_count_distribution = dict(data_df.groupby(by=data_df.values).count())
        category_count_distribution[np.nan] = self._count_of_missing

        index = sorted(category_count_distribution.keys())
        value = [category_count_distribution[x] for x in index]
        return {'index': np.array(index), 'value': np.array(value)}

    @property
    def category_percentage_distribution(self):
        """Calculate percentage for each category."""

        return {'index': self.category_count_distribution['index'],
                'value': 1.0 * self.category_count_distribution['value'] / self.count}

    @property
    def distribution_histogram(self):
        """Draw distribution histogram for the data."""

        cate_label = self.category_percentage_distribution['index']
        x_axis = range(1, len(cate_label) + 1)
        y_axis = self.category_percentage_distribution['value']

        plt.close()
        plt.bar(x_axis, y_axis, width=0.5, tick_label=cate_label)
        plt.xlabel("category")
        plt.ylabel("percentage")
        plt.title("Distribution Histogram For Discrete Variable %s" % self.name)
        self._assert_show()

        return [x_axis, y_axis, cate_label]

    def _assert_show(self):
        """Assert for whether to show the chart."""

        if self.is_show:
            try:
                plt.show()
            except:
                raise Exception("Could not plot the chart.")
