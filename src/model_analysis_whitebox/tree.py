"""Classes and methods for tree model analysis."""

import pandas as pd
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.tree.tree import DecisionTreeRegressor

from .base import *


# from sklearn.tree._tree import TREE_LEAF
TREE_LEAF = -1


class TreeModelAnalysis(BaseModelAnalysis):
    """Class for tree model analysis"""

    @classmethod
    def feature_importances_train_stage(cls, model):

        return model.feature_importances_

    @classmethod
    def feature_importances_predict_stage(cls, model, x_sample, change='abs'):

        if not isinstance(model, BaseDecisionTree):
            raise TypeError("Model type error. Please make sure it is a tree model.")

        if isinstance(model, DecisionTreeRegressor):
            return RegressionTreeModelAnalysis.feature_importances_predict_stage(model, x_sample, change)
        if isinstance(model, DecisionTreeClassifier):
            return ClassificationTreeModelAnalysis.feature_importances_predict_stage(model, x_sample, change)

        raise NotImplementedError("Not Implemented. Please use this method in subclasses.")

    @classmethod 
    def tree_splits_points(cls, model):

        if not isinstance(model, BaseDecisionTree):
            raise TypeError("Model type error. Please make sure it is a decision tree model.")
        
        n_features = model.n_features_
        splits_points = []
        for i in range(n_features):
            splits_points.append([])
        
        # obtain splits points
        e_tree = model.tree_
        for i, fi in enumerate(e_tree.feature):
            if e_tree.children_left[i] != TREE_LEAF:
                splits_points[fi].append(e_tree.threshold[i])
        
        for split_points in splits_points:
            split_points.sort()
        return np.array(splits_points)

    @classmethod
    def tree_splits_points_plotter(cls, split_points, feature_name, fs_min=None, fs_max=None, bins=5, value='abs',
                                   bar_format='b', line_format='k--', dot_format='mo'):

        if len(split_points) < 1:
            return None
        if (fs_min is None) or (min(split_points) < fs_min):
            fs_min = min(split_points)
        if (fs_max is None) or (max(split_points) > fs_max):
            fs_max = max(split_points)
        domain_origin = float(fs_max) - float(fs_min)
        fs_min = fs_min - 0.05 * domain_origin
        fs_max = fs_max + 0.05 * domain_origin
        split_points = np.sort(split_points)
        
        x = []
        y = []
        step = 1.0 * (fs_max - fs_min) / bins
        for i in range(bins):
            xi = fs_min + i * step
            datai = split_points[(split_points >= xi) & (split_points < xi + step)]
            yi = datai.size
            x.append(round(xi, 2))
            y.append(yi)
        x.append(round(fs_max, 2))
        y.append(0)
        
        if value == 'abs':
            y = np.array(y)
        elif value == 'rel':
            y = 1.0 * np.array(y) / np.sum(y)
        else:
            raise ValueError("Parameter 'value' must be 'abs' or 'rel'.")
        
        # draw the histogram
        chart = plt.bar(x, y, width=[step * 1] * bins + [0], align='edge', tick_label=x, color=bar_format, alpha=0.3,
                        edgecolor=bar_format, linewidth=3)
        for i in range(len(x) - 1):
            height = chart[i].get_height()
            if value == 'abs':
                height = int(height)
            else:
                height = round(height, 3)
            plt.text(chart[i].get_x() + chart[i].get_width() / 4.0, 1.01 * height, '%s' % height)
        
        # draw the dots
        points = pd.DataFrame(zip(split_points, split_points), columns=['s1', 's2']).groupby('s1').count()
        if value == 'abs':
            point_counts = points.values
        elif value == 'rel':
            point_counts = 1.0 * points.values / np.sum(points.values)
        else:
            raise ValueError("Parameter 'value' must be 'abs' or 'rel'.")
        dot = plt.plot(points.index, point_counts, line_format, points.index, point_counts, dot_format, alpha=0.4)
        plt.xlabel(feature_name)
        plt.ylabel('split-point count')
        plt.title('Split Point Distribution For %s' % (feature_name.capitalize())) 
        return [chart, dot]
        

################################################################################
class RegressionTreeModelAnalysis(TreeModelAnalysis, RegressionModelAnalysis):
    """Class for regression tree model analysis"""

    @classmethod
    def feature_importances_predict_stage(cls, model, x_sample, change='abs'):

        total_sum = [0] * model.n_features_
        e_tree = model.tree_
        # get the decision path
        decision_path = model.decision_path([x_sample]).toarray()[0]
            
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
    
    
################################################################################
class ClassificationTreeModelAnalysis(TreeModelAnalysis, 
                                      ClassificationModelAnalysis):
    """Class for classification tree model analysis"""

    @classmethod
    def feature_importances_predict_stage(cls, model, x_sample, change='abs'):
        """Not suitable for multi-label problems"""

        if model.n_outputs_ > 1:
            raise NotImplementedError("Not Implemented for multi-label problems.")
        n_classes = model.n_classes_
        n_features = model.n_features_

        total_sum = np.zeros((n_features, n_classes), dtype=np.float64)
        e_tree = model.tree_
        # get the decision path
        decision_path = model.decision_path([x_sample]).toarray()[0]
            
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
