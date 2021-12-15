import numpy as np
from .utils import RMSE
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None,
                 feature_subsample_size=None,
                 random_state=0,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree.
            If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree.
            If None then use one-third of all features.

        random_state : int
            The seed for random generator initialization.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.random_seed = random_state
        self.models = []

    def fit(self, X, y, X_val=None, y_val=None,
            return_train_loss=None,
            return_val_loss=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects

        return_train_loss : bool
            Determines whether to return the list of
            losses on train for all iterations

        return_val_loss : bool
            Determines whether to return the list of
            losses on validation for all iterations
            (X_val and y_val should be provided)
        
        Returns 
        -------
        tuple(losses)
            losses : list of
                train_loss : numpy ndarray
                    Array of size n_estimators that consists of RMSE
                    on train set on each count of decision trees in 
                    random forest: from 1 to n_estimators
                    (exists only if 'return_train_loss' is true)

                val_loss : numpy ndarray
                    Array of size n_estimators that consists of RMSE
                    on validation set on each count of decision trees in
                    random forest: from 1 to n_estimators
                    (exists only if 'return_val_loss' is true)
        """
        # number of features to choose splitting: n // 3 for regression
        if self.max_features is None:
            max_features = max(1, X.shape[1] // 3)
        else:
            max_features = self.max_features

        # n_estimators decision trees on the corresponding bootstrap samples
        rng = np.random.default_rng(self.random_seed)
        y_len_ = len(y)
        for i in range(self.n_estimators):
            bootstrap_ind = rng.choice(y_len_, y_len_)
            tree_seed = rng.integers(1e5) # big interval for best independence
            model = DecisionTreeRegressor(max_depth=self.max_depth,
                                          max_features=max_features,
                                          random_state=tree_seed,
                                          **self.trees_parameters)
            model.fit(X[bootstrap_ind], y[bootstrap_ind])
            self.models.append(model)
        
        # returns
        losses = []
        if return_train_loss:
            losses.append(self.get_loss(X, y))
        if return_val_loss:
            losses.append(self.get_loss(X_val, y_val))
        if losses:
            return tuple(losses)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        # each 'model' in 'models' list is a decision tree in this ensemble
        return np.mean([model.predict(X) for model in self.models], axis=0)

    def get_loss(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        Returns
        -------
        RMSE : numpy ndarray
            Array of size n_estimators. RMSE[i] is the RMSE of the RandomForest
            combined of the first (i + 1) models from 'models' list
            (RMSE on X set with y target)
        """
        # single predictions on n_estimators : list with len=n_estimators
        # consisting of ndarrays with shape=(n_objects,)
        pred_arr = [model.predict(X) for model in self.models]
        # estimators count in ensemble : ndarray with shape=(n_estimators, 1)
        estimators_arr = (np.arange(self.n_estimators) + 1).reshape(-1,1)
        # ensemble predictions on n_estimators : ndarray with
        # shape=(n_estimators, n_objects), y_pred_matrix[i][j] is a prediction
        # on j-object on ensemble combined of first (i + 1) models
        y_pred_matrix = np.cumsum(pred_arr, axis=0) / estimators_arr
        return RMSE(y, y_pred_matrix)



class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
