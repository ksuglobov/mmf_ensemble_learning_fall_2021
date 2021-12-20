import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ensembles import RandomForestMSE, GradientBoostingMSE


class Ensemble:
    __models = {
        'RF': RandomForestMSE,
        'GBM': GradientBoostingMSE,
    }

    __rus_types = {
        'RF': 'Случайный лес',
        'GBM': 'Градиентный бустинг'
    }

    def __init__(self, name, ens_type, form):
        """
        initializing
        """
        self.name = name
        hyparams = form.data
        d = [('Тип ансамбля', self.__rus_types[ens_type])]
        d += [(form[param].label.text, hyparams[param]) for param in hyparams]
        self.description = pd.DataFrame(d, columns=['Параметр', 'Значение'])
        trees_parameters = hyparams.pop('trees_parameters')
        hyparams = {**hyparams, **trees_parameters}
        self.model = self.__models[ens_type](**hyparams)
        self.train_loss = None
        self.val_loss = None
        self.target_name = None

    def fit(self, data_train, data_val=None):
        """
        fitting
        """
        X_train = data_train.features
        y_train = data_train.target
        self.target_name = data_train.target_name
        if data_val is not None:
            self.train_loss, self.val_loss = self.model.fit(
                X_train, y_train, data_val.features,
                data_val.target, True, True)
        else:
            self.train_loss = self.model.fit(X_train, y_train,
                                             return_train_loss=True)[0]

    @property
    def is_fitted(self):
        """
        when model is already fitted
        """
        return self.train_loss is not None

    def predict(self, data_test):
        """
        prediction
        """
        y_pred = self.model.predict(data_test.features)
        return pd.DataFrame(
            y_pred,
            index=data_test.data.index,
            columns=[self.target_name]
        )

    def plot(self):
        """
        for plotting loss
        """
        # plot settings
        plt.rc('axes', axisbelow=True, grid=True)
        plt.rc('grid', c='grey', ls=':')
        plt.rc('savefig', facecolor='white')
        opt_color = 'mediumspringgreen'

        fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
        ax.set_title('Loss')
        lim = self.model.n_estimators
        ax.plot(np.arange(1, lim + 1), self.train_loss, label='train')
        if self.val_loss is not None:
            ax.plot(np.arange(1, lim + 1), self.val_loss, label='validation')
            opt_arg = self.val_loss.argmin()
            ax.scatter(opt_arg + 1, self.val_loss[opt_arg], c=opt_color,
                       zorder=3, label='optimal')
        ax.set_xlabel('n_estimators')
        ax.set_ylabel('RMSE')
        ax.legend()
        fig.tight_layout()
        return fig


class Dataset:
    def __init__(self, name, data, target_name):
        """
        initializing
        """
        self.name = name
        self.data = data
        self.target_name = target_name
        self.has_target = target_name != ''

    @property
    def features(self):
        """
        what should we do with data features
        """
        if self.has_target:
            return self.data.drop(columns=self.target_name).to_numpy()
        return self.data.to_numpy()

    @property
    def target(self):
        """
        what should we do with data target
        """
        if self.has_target:
            return self.data[self.target_name].to_numpy()
        raise ValueError(f'The target if {self.name} is unknown!')
