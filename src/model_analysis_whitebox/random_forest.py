"""Classes and methods for random forest model analysis."""

from sklearn.ensemble.forest import BaseForest
from sklearn.ensemble.forest import ForestClassifier
from sklearn.ensemble.forest import ForestRegressor

from .tree import *


class RFModelAnalysis(TreeModelAnalysis):
    """Class for random forest model analysis"""

    @classmethod
    def feature_importances_predict_stage(cls, model, x_sample, change='abs'):

        if not isinstance(model, BaseForest):
            raise TypeError("Model type error. Please make sure it is a RF model.")

        if isinstance(model, ForestRegressor):
            return RegressionRFModelAnalysis.feature_importances_predict_stage(model, x_sample, change)
        if isinstance(model, ForestClassifier):
            return ClassificationRFModelAnalysis.feature_importances_predict_stage(model, x_sample, change)

        raise TypeError("Model type error. Please make sure it is a RF model.")

    @classmethod
    def inter_feature_importances(cls, model):

        raise ValueError("Inter feature importances cannot be calculated for random forest.")

    @classmethod 
    def tree_splits_points(cls, model):
        if not isinstance(model, BaseForest):
            raise TypeError("Model type error. Please make sure it is a RF model.")
        
        estimators = model.estimators_
        n_features = model.n_features_
        splits_points = []
        for i in range(n_features):
            splits_points.append([])
        
        # obtain splits points
        for est in estimators:
            e_tree = est.tree_
            for i, fi in enumerate(e_tree.feature):
                if e_tree.children_left[i] != TREE_LEAF:
                    splits_points[fi].append(e_tree.threshold[i])
        
        for split_points in splits_points:
            split_points.sort()
        return np.array(splits_points)


#######################################################################################################################
class RegressionRFModelAnalysis(RFModelAnalysis, RegressionTreeModelAnalysis):
    """Class for random forest regression model analysis"""
    
    @classmethod
    def feature_importances_predict_stage(cls, model, x_sample, change='abs'):

        estimators = model.estimators_
        
        total_sum = [0] * model.n_features_
        for estimator in estimators:
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
                
        return np.array(total_sum) / np.array(total_sum).sum()
    
    
#######################################################################################################################
class ClassificationRFModelAnalysis(RFModelAnalysis, ClassificationTreeModelAnalysis):
    """Class for random forest classification model analysis"""
    
    @classmethod
    def feature_importances_predict_stage(cls, model, x_sample, change='abs'):
        """Not suitable for multi-label problems"""

        if model.n_outputs_ > 1:
            raise NotImplementedError("Not Implemented for multi-label problems.")
        n_classes = model.n_classes_
        n_features = model.n_features_
        estimators = model.estimators_
        
        total_sum = np.zeros((n_features, n_classes), dtype=np.float64)
        for estimator in estimators:
            e_tree = estimator.tree_
            # get the decision path
            decision_path = estimator.decision_path([x_sample]).toarray()[0]
            
            node = 0
            while e_tree.children_left[node] != TREE_LEAF:
                feature = e_tree.feature[node]
                value_pre = e_tree.value[node][0]
                prob_pre = value_pre / value_pre.sum()
                if decision_path[e_tree.children_left[node]] == 1:
                    node = e_tree.children_left[node]
                else:
                    node = e_tree.children_right[node]
                value_aft = e_tree.value[node][0]
                prob_aft = value_aft / value_aft.sum()
                if change == 'abs':
                    total_sum[feature] += np.abs(prob_aft - prob_pre)
                else:
                    total_sum[feature] += (prob_aft - prob_pre)
        
        return (total_sum / sum(total_sum)).T
