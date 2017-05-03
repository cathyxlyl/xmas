"""Base classes and methods for most kinds of models analysis, mostly universal."""

import copy
import matplotlib.pyplot as plt
import numpy as np


class BaseModelAnalysis(object):
    """Base class for model analysis"""
    
    @classmethod    
    def factor_analysis_plotter(cls, import_module, import_model, model, x_sample, i, feature_name, change_range,
                                line_format='k', dot_format='bo'):

        try:
            exec("from %s import %s" % (import_module, import_model))
        except:
            raise ImportError("Module import error.")
        if not eval("isinstance(%s, %s)" % (model, import_model)):
            raise TypeError("Model is not an instance of the imported estimator.")
            
        y_origin = model.predict([x_sample])
        x = copy.deepcopy(x_sample)
        y = []
        change_range = sorted(change_range)
        for item in change_range:
            x[i] = item
            y.append(model.predict([x]))
        
        chart = plt.plot(change_range, y, line_format, [x_sample[i]], [y_origin], dot_format)
        plt.xlabel(feature_name)
        plt.ylabel('label')
        plt.title('Factor Analysis For %s' % (feature_name.capitalize()))
        return np.array(y), chart

    @classmethod
    def feature_removal_predict(cls, import_module, import_model, model, x_samples, remove_features, features_default):

        try:
            exec("from %s import %s" % (import_module, import_model))
        except:
            raise ImportError("Module import error.")
        if not eval("isinstance(%s, %s)" % (model, import_model)):
            raise TypeError("Model is not an instance of the imported estimator.")

        x_samples = np.array(x_samples)
        x_copy = copy.deepcopy(x_samples)
        for i, index in enumerate(remove_features):
            x_copy[:, index] = features_default[i]
        return model.predict(x_copy)
    

#######################################################################################################################
class RegressionModelAnalysis(BaseModelAnalysis):
    """Base class for regression model analysis"""
            

#######################################################################################################################
class ClassificationModelAnalysis(BaseModelAnalysis):
    """Base class for classification model analysis"""
