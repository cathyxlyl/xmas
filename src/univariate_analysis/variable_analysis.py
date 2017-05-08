"""Many methods for univariate analysis"""

import numpy as np
import pandas as pd

from .continuous_variable_analysis import ContinuousVariableAnalysis
from .discrete_variable_analysis import DiscreteVariableAnalysis
from .outlier_analysis import ContinuousOutlierAnalysis, DiscreteOutlierAnalysis


class VariableAnalysis(object):

    # Data name
    name = None
    # Data to be analysis
    data = None
    # Missing values
    missing = [None, np.nan]
    # Whether the data is discrete or continuous
    data_type = None
    # Define the analyzer
    analyzer = None
    # Define the outlier analyzer
    outlier = None

    def __init__(self, data, name, data_type='discrete', missing=None):
        """Initialize variable analysis class.
        
        Parameters
        ----------
        data: pandas.core.series.Series or numpy.ndarray or list
            The variable data that we want to analyze.
        name: str
            The name of this variable.
        data_type: {'discrete', 'continuous'}, optimal, (default='discrete')
            Whether the data is discrete data or continuous data.
        missing: list, optimal, (default='None')
            Missing values of the variable.
        """

        if isinstance(data, pd.Series):
            self.data = data.as_matrix()
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            raise TypeError("The label data must have its type as 'pandas.core.series.Series' or 'numpy.ndarray' "
                            "or 'list'.")

        if missing is not None:
            if not isinstance(data, list):
                raise TypeError("The parameter 'missing' must have its type as 'list'.")
            self.missing += missing

        self.name = name
        if data_type in ['discrete', 'continuous']:
            self.data_type = data_type
            if self.data_type == 'continuous':
                self.analyzer = ContinuousVariableAnalysis(self.data, self.name, self.missing)
                self.outlier = ContinuousOutlierAnalysis(self.analyzer)
            if self.data_type == 'discrete':
                self.analyzer = DiscreteVariableAnalysis(self.data, self.name, self.missing)
                self.outlier = DiscreteOutlierAnalysis(self.analyzer)
        else:
            raise ValueError("The label data type must either be 'discrete' or 'continuous'.")

    def _clear(self):

        self.name = None
        self.data = None
        self.data_type = None
        self.analyzer = None

    def get_analyzer(self):

        return self.analyzer

    def get_outlier(self):

        return self.outlier
