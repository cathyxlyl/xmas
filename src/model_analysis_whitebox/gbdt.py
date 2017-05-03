"""Classes and methods for gbdt model analysis."""

from sklearn.ensemble.gradient_boosting import BaseGradientBoosting
from sklearn.ensemble.gradient_boosting import LeastSquaresError
from .tree import *


class GBDTModelAnalysis(TreeModelAnalysis):
    """Class for gbdt model analysis"""

    @classmethod
    def _feature_importances_predict_stage_ls(cls, model, x_sample, change='abs'):

        estimators = model.estimators_

        total_sum = [0] * model.n_features
        for stage in estimators:
            estimator = stage[0]
            e_tree = estimator.tree_
            # get the decision path
            decision_path = estimator.decision_path([x_sample]).toarray()[0]

            node = 0
            while e_tree.children_left[node] != TREE_LEAF:
                feature = e_tree.feature[node]
                value_pre = e_tree.value[node][0][0]
                if decision_path[e_tree.children_left[node]] == 1:
                    node = e_tree.children_left[node]
                else:
                    node = e_tree.children_right[node]
                value_aft = e_tree.value[node][0][0]
                if change == 'abs':
                    total_sum[feature] += abs(value_aft - value_pre)
                else:
                    total_sum[feature] += (value_aft - value_pre)

        # The sum of total_sum may be much large, it is caused by two reasons.
        # One is that before every stage, MeanEstimator predict the init prediction.
        # Two is that total_sum should be multiplied by learning rate.
        return np.array(total_sum) / np.array(total_sum).sum()

    @classmethod
    def feature_importances_predict_stage(cls, model, x_sample, change='abs'):
        """Only available for gbdt models with 'ls' loss."""

        if not isinstance(model, BaseGradientBoosting):
            raise TypeError("Model type error. Please make sure it is a GBDT model.")
        if not isinstance(model.loss_, LeastSquaresError):
            raise TypeError("Loss type error. The loss function should be LeastSquaresError")

        if isinstance(model.loss_, LeastSquaresError):
            return cls._feature_importances_predict_stage_ls(model, x_sample, change)

    @classmethod
    def inter_feature_importances(cls, model):

        if not isinstance(model, BaseGradientBoosting):
            raise TypeError("Model type error. Please make sure it is a GBDT model.")

        feature_importances_inter = []
        estimators = model.estimators_

        # compute feature importances after every stage
        total_sum = np.zeros((model.n_features,), dtype=np.float64)
        for i, stage in enumerate(estimators):
            stage_sum = sum(dtree.feature_importances_ for dtree in stage) / len(stage)
            total_sum += stage_sum
            inter_importances = list(total_sum / (i + 1))
            feature_importances_inter.append(inter_importances)
        return np.array(feature_importances_inter)

    @classmethod
    def feature_importance_trend_plotter(cls, inter_fi, i, feature_name, line_format='g'):

        # get the importances of the chosen feature
        fii = inter_fi[:, i]
        n_stages = len(fii)
        x = range(1, n_stages + 1)

        line_chart = plt.plot(x, fii, line_format, alpha=0.7)
        plt.xlabel(feature_name)
        plt.ylabel('feature importance')
        plt.title('Feature Importance Trend For %s With Training Stages' % (feature_name.capitalize()))
        return line_chart

    @classmethod 
    def tree_splits_points(cls, model):

        if not isinstance(model, BaseGradientBoosting):
            raise TypeError("Model type error. Please make sure it is a GBDT model.")
        
        estimators = model.estimators_
        n_features = model.n_features
        splits_points = []
        for i in range(n_features):
            splits_points.append([])
        
        # obtain splits points
        for stage in estimators:
            for est in stage:
                e_tree = est.tree_
                for i, fi in enumerate(e_tree.feature):
                    if e_tree.children_left[i] != TREE_LEAF:
                        splits_points[fi].append(e_tree.threshold[i])
        
        for split_points in splits_points:
            split_points.sort()
        return np.array(splits_points)


################################################################################
class RegressionGBDTModelAnalysis(GBDTModelAnalysis, 
                                  RegressionTreeModelAnalysis):
    """Class for gradient boosting regression model analysis"""
    pass
    
    
################################################################################
class ClassificationGBDTModelAnalysis(GBDTModelAnalysis,
                                      ClassificationTreeModelAnalysis):
    """Class for gradient boosting classification model analysis"""
    pass
