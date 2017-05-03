"""Classes and methods for xgboost model analysis."""

from xgboost.core import Booster
from xgboost.sklearn import XGBModel

from .tree import *


class XGBoostModelAnalysis(TreeModelAnalysis):
    """Class for xgboost model analysis"""

    @classmethod
    def feature_importances_train_stage(cls, model):

        return cls.inter_feature_importances(model)[-1]

    @classmethod
    def _tree_feature_importances(cls, tree, feature_names):

        n_features = len(feature_names)
        tree_feature_importances = np.zeros([n_features], dtype=np.float64)

        tree = tree.split('\n')
        for item in tree:
            i1 = item.split('[')
            if len(i1) > 1:
                i2 = i1[1].split(']')
                feature = i2[0].split('<')[0]
                gain = i2[1].split('gain=')[1].split(',')[0]
                if feature in feature_names:
                    index = feature_names.index(feature)
                    tree_feature_importances[index] += float(gain)
        return tree_feature_importances / tree_feature_importances.sum()

    @classmethod
    def inter_feature_importances(cls, model):

        if (not isinstance(model, XGBModel)) and (not isinstance(model, Booster)):
            raise TypeError("Model type error. Please make sure it is a xgboost model.")
        bst = model
        if isinstance(model, XGBModel):
            bst = model.booster()

        feature_importances_inter = []
        trees = bst.get_dump(with_stats=True)
        feature_names = bst.feature_names
        n_features = len(feature_names)

        # compute feature importances after every stage
        total_sum = np.zeros((n_features,), dtype=np.float64)
        for i, tree in enumerate(trees):
            tree_fi = cls._tree_feature_importances(tree, feature_names)
            total_sum += tree_fi
            inter_importances = list(total_sum / (i + 1))
            feature_importances_inter.append(inter_importances)
        return np.array(feature_importances_inter)

    @classmethod
    def tree_splits_points(cls, model):

        if (not isinstance(model, XGBModel)) and (not isinstance(model, Booster)):
            raise TypeError("Model type error. Please make sure it is a xgboost model.")
        bst = model
        if isinstance(model, XGBModel):
            bst = model.booster()

        trees = bst.get_dump(with_stats=True)
        feature_names = bst.feature_names
        n_features = len(feature_names)
        splits_points = []
        for i in range(n_features):
            splits_points.append([])

        # obtain splits points
        for tree in trees:
            tree = tree.split('\n')
            for item in tree:
                i1 = item.split('[')
                if len(i1) > 1:
                    i2 = i1[1].split(']')[0]
                    feature, value = i2.split('<')
                    if feature in feature_names:
                        index = feature_names.index(feature)
                        splits_points[index].append(float(value))

        for split_points in splits_points:
            split_points.sort()
        return np.array(splits_points)


#######################################################################################################################
class RegressionXGBoostModelAnalysis(XGBoostModelAnalysis, RegressionTreeModelAnalysis):
    """Class for xgboost regression model analysis"""
    pass


#######################################################################################################################
class ClassificationXGBoostModelAnalysis(XGBoostModelAnalysis, ClassificationTreeModelAnalysis):
    """Class for xgboost classification model analysis"""
    pass
