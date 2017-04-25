"""Judgement that which class to use."""

from .gbdt import *
from .random_forest import *
from .xg_boost import *


class ModelAnalysisWhiteBox(object):
    """Interface"""

    def __init__(self, m_type='regression', m_model='gbdt'):

        self.m_type = m_type
        self.model_analysis = BaseModelAnalysis

        if m_type == 'regression':
            self.model_analysis = RegressionModelAnalysis
            if m_model == 'tree':
                self.model_analysis = RegressionTreeModelAnalysis
            elif m_model == 'gbdt':
                self.model_analysis = RegressionGBDTModelAnalysis
            elif m_model == 'xgboost':
                self.model_analysis = RegressionXGBoostModelAnalysis
            elif m_model == 'rf':
                self.model_analysis = RegressionRFModelAnalysis
        elif m_type == 'classification':
            self.model_analysis = ClassificationModelAnalysis
            if m_model == 'tree':
                self.model_analysis = ClassificationTreeModelAnalysis
            elif m_model == 'gbdt':
                self.model_analysis = ClassificationGBDTModelAnalysis
            elif m_model == 'xgboost':
                self.model_analysis = ClassificationXGBoostModelAnalysis
            elif m_model == 'rf':
                self.model_analysis = ClassificationRFModelAnalysis
        else:
            if m_model == 'tree':
                self.model_analysis = TreeModelAnalysis
            elif m_model == 'gbdt':
                self.model_analysis = GBDTModelAnalysis
            elif m_model == 'xgboost':
                self.model_analysis = XGBoostModelAnalysis
            elif m_model == 'rf':
                self.model_analysis = RFModelAnalysis

    def factor_analysis_plotter(self, import_module, import_model, model, x_sample, i, feature_name, change_range,
                                line_format='k', dot_format='bo'):
        """Factor analysis plotter. Change one feature while fix others, and observe the change of model prediction.
            General methods for all models.
        
        Parameters
        ----------
        import_module: string
            The module which the model belongs to, e.g. 'sklearn.ensemble'.
        import_model: string
            The estimator which the model comes from, e.g. 'GradientBoostingRegressor'.
            The estimator mush have a 'predict' method.
        model: a model which is an instance of the 'import_model'
            The model that we want to analyze.
        x_sample: ndarray, shape = (n_features)
            The sample feature vector.
        i: int
            The index the chosen feature from the model.
        feature_name: string
            The name of the feature.
        change_range: ndarray
            The values that the chosen feature to take.
        line_format: string, optimal, (default='k')
            The color and format of the plotted line.
        dot_format: string, optimal, (default='bo')
            The color and format of the plotted dot.
        
        Returns
        -------
        predictions: ndarray
            Returns a tuple of an array of prediction results.
        chart: matplotlib.lines.Line2D
            You can use plt.show() to show the chart.
        """

        return self.model_analysis.factor_analysis_plotter(import_module, import_model, model, x_sample, i,
                                                           feature_name, change_range, line_format, dot_format)

    def feature_removal_predict(self, import_module, import_model, model, x_samples,
                                remove_features, features_default):
        """Calculate model predictions while remove some feature.
            General methods for all models.
        
        Parameters
        ----------
        import_module: string
            The module which the model belongs to, e.g. 'sklearn.ensemble'.
        import_model: string
            The estimator which the model comes from, e.g. 'GradientBoostingRegressor'.
            The estimator mush have a 'predict' method.
        model: a model which is an instance of the 'import_model'
            The model that we want to analyze.
        x_samples: ndarray, shape = (n_samples, n_features)
            The feature matrix of samples.
        remove_features: ndarray
            Index of those features to be removed.
        features_default: ndarray
            Default value of those features to be removed.
        
        Returns
        -------
        predictions: ndarray, shape = (n_samples)
            Returns the prediction results after feature removal.
        """

        return self.model_analysis.feature_removal_predict(import_module, import_model, model, x_samples,
                                                           remove_features, features_default)

    def feature_importances_train_stage(self, model):
        """Get feature importances during training stage.
            General methods for tree models.
        
        Parameters
        ----------
        model: a model which is an instance of the 'import_model'
            The model that we want to analyze. 
            It Should have the attributes 'feature_importances_'.
        
        Returns
        -------
        feature_importances_train: ndarray, shape = (n_features)
            The feature importances of the model during training stage.
        """

        return self.model_analysis.feature_importances_train_stage(model)

    def feature_importances_predict_stage(self, model, x_sample, change='abs'):
        """Get feature importances during predicting stage.
            General methods for tree models, only implemented for gbdt with 'ls' loss.
        
        Parameters
        ----------
        model: Some kind of model/estimator
            The model that we want to analyze.
        x_sample: ndarray, shape = (n_features)
            The sample feature vector.
        change: string, optimal, (default='abs')
            Whether to get the absolute change value or the original change. 
            Can be 'abs' or None.
            
        Returns
        -------
        feature_importances_predict: ndarray, shape = (n_features)
            The feature importances of the model during predicting stage.
        """

        return self.model_analysis.feature_importances_predict_stage(model, x_sample, change)

    def inter_feature_importances(self, model):
        """Compute current feature importances after every stage.
            Imitate the implementation of method 'feature_importances_' in gradient_boosting.
            Only implemented for gbdt models (or say boosting methods).
        
        Parameters
        ----------
        model: The gbdt model.
            The model that we want to analyze.
        
        Returns
        -------
        feature_importances_inter: ndarray, shape = (n_estimators, n_features)
            The inter feature importances after every stage.
        """

        return self.model_analysis.inter_feature_importances(model)

    def feature_importance_trend_plotter(self, inter_fi, i, feature_name, line_format='g'):
        """Visualize feature importance trend with training stages for the chosen feature.
            Only implemented for gbdt models.
        
        Parameters
        ----------
        inter_fi: ndarray, shape = (n_estimators, n_features)
            The inter feature importances after every stage.
            The output of method 'inter_feature_importances'.
        i: int
            The index the chosen feature from the model.
        feature_name: string
            The name of the feature.
        line_format: string, optimal, (default='g')
            The color of the plotted line.
        
        Returns
        -------
        line_chart: matplotlib.lines.Line2D
            The line chart which represent the feature importance trend with 
            the change of training stages.
        """

        return self.model_analysis.feature_importance_trend_plotter(inter_fi, i, feature_name, line_format)

    def tree_splits_points(self, model):
        """Get all split points of all features from the model.
            General methods for tree models, only implemented for gbdt.

        Parameters
        ----------
        model: The model based on trees.
            The model that we want to analyze.

        Returns
        -------
        splits_points: ndarray, shape = (n_features, None)
            Sorted split points for all features.
        """

        return self.model_analysis.tree_splits_points(model)

    def tree_split_points_plotter(self, splits_points, feature_name, fs_min=None, fs_max=None, bins=5, value='abs',
                                  bar_format='b', line_format='k--', dot_format='mo'):
        """Visualize the distribution of the chosen feature.
            General methods for tree models.

        Parameters
        ----------
        splits_points: ndarray, shape = (k)
            All split points of the chosen feature, k is the number of split points for the feature and a row from 
            the result of method 'tree_splits_points'.
        feature_name: string
            The name of the feature.
        fs_min: double
            Minimum value of the feature.
        fs_max: double
            Maximum value of the feature.
        bins: int, optimal, (default=5)
            Number of segmentation.
        value: string, optimal, (default='abs')
            Whether to plot abs count value or relative count value of split points.
            Can be 'abs' or 'rel'.
        bar_format: string, optimal, (default='b')
            The color of the plotted bar.
        line_format: string, optimal, (default='k--')
            The color and format of the plotted line.
        dot_format: string, optimal, (default='mo')
            The color and format of the plotted dot.

        Returns
        -------
        chart: matplotlib.lines.Line2D
            You can use plt.show() to show the chart.
        """

        return self.model_analysis.tree_split_points_plotter(splits_points, feature_name, fs_min, fs_max, bins, value,
                                                             bar_format, line_format, dot_format)
