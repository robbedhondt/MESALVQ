import os
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sksurv.util import Surv
from .slvq import SurvivalLVQ

class MultiEventSurvivalLVQ(torch.nn.Module, BaseEstimator):
    def __init__(self, n_prototypes=2, n_omega_rows=None, batch_size=128,
                 init='kmeans', device=torch.device("cpu"), lr=1e-3, epochs=50, verbose=True):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.n_omega_rows = n_omega_rows
        self.init = init
        self.verbose = verbose
        self.batch_size = batch_size

    def fit(self, X, y):
        """
        Fit this multi-event SurvivalLVQ object to the given data. 
        n = number of instances, p = number of features, k = number of events

        @param X: n x p matrix of input features
        @param y: structured array of shape n x k, with as first field the
            censoring status (0 is censored, 1 is observed) and as second field
            the time to event.
        """
        X = torch.tensor(X, dtype=torch.float32)
        # // CHANGE START =====
        # D, T = map(np.array, zip(*y)) # this trick doesn't work anymore with matrices
        D = torch.tensor(np.copy(y[y.dtype.names[0]]), dtype=torch.float32)
        T = torch.tensor(np.copy(y[y.dtype.names[1]]), dtype=torch.float32)
        # ^ copy is necessary because the array needs to occupy contiguous memory for Torch

        # Initialize the models
        self.n_events   = y.shape[1]
        self.n_features = X.shape[1]
        self.models = [None]*self.n_events
        for k in range(self.n_events):
            init = [self.init, "random"][k>0] # avoid doing k-means init for every model
            self.models[k] = SurvivalLVQ(init=init, n_prototypes=self.n_prototypes, 
                n_omega_rows=self.n_omega_rows, batch_size=self.batch_size, device=self.device, 
                lr=self.lr, epochs=self.epochs, verbose=self.verbose)
            self.models[k]._init_model(X, D[:,k], T[:,k])
            self.models[k].w     = self.models[0].w     # Harmonize models
            self.models[k].omega = self.models[0].omega # Harmonize models
        self.w     = self.models[0].w     # Convenience attribute
        self.omega = self.models[0].omega # Convenience attribute
        # ===== CHANGE END //

        dataset = torch.utils.data.TensorDataset(X, T, D)

        random_sampler = torch.utils.data.RandomSampler(dataset, replacement=False,
                                                        num_samples=self.batch_size - (X.size(0) % self.batch_size))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer_w = torch.optim.Adam([
            {'params': self.w},
            {'params': self.omega, 'lr': self.lr * 0.1}], lr=self.lr)

        # Training loop
        for epoch in range(self.epochs):
            # // CHANGE START =====
            for model in self.models:
                model.fit_labels()
            # ===== CHANGE END //
            minibatch_loss = []
            for batch in dataloader:
                x, t, d = batch

                # fill up the batch if not full:
                if x.size(0) < self.batch_size:
                    ids = list(random_sampler)
                    x = torch.cat([x, X[ids]])
                    t = torch.cat([t, T[ids]])
                    d = torch.cat([d, D[ids]])

                optimizer_w.zero_grad()  # zero-out gradients
                # // CHANGE START =====
                batch_loss = 0.0
                for k, model in enumerate(self.models):
                    batch_loss += model.loss_brier(x, t[:,k], d[:,k])
                # batch_loss /= len(self.models)
                # ===== CHANGE END //
                minibatch_loss.append(batch_loss)
                batch_loss.backward()
                # Take a gradient descent step
                optimizer_w.step()
                # // CHANGE START =====
                # make the total feature relevance sum to 1
                # (note that all models internally use the same w / omega matrix
                #  so this automatically also normalizes self.models[k])
                self.models[0].normalize_trace()
                # ===== CHANGE END //

            if self.verbose:
                epoch_loss = torch.tensor(minibatch_loss).mean()
                print(f"Epoch: {epoch} Loss: {epoch_loss}")
        return self

    def predict(self, X, closest=False, squeeze=True):
        y_pred = np.stack([model.predict(X, closest=closest) 
                           for model in self.models], axis=1)
        if closest: # then all columns should be equal
            assert np.all(y_pred == y_pred[:,[0]])
            if squeeze:
                y_pred = y_pred[:,0] # so just take the first one
        return y_pred
    
    def predict_survival_function(self, X):
        return np.stack([model.predict_survival_function(X) 
                        for model in self.models], axis=1)

    def lambda_mat(self):
        return self.models[0].lambda_mat() # or self.omega.T @ self.omega

class LocalMultiEventSurvivalLVQ(BaseEstimator):
    """
    Simple wrapper class to handle a list of SurvivalLVQ objects. Written to
    be compatible with older code that still had a list of objects
    """
    def __init__(self, models):
        """models = simple list of SurvivalLVQ objects"""
        super().__init__()
        assert isinstance(models, list), "accidentally passed a list of LocalMultiEventSurvivalLVQ objects?"
        self.models = models

    def set_hyperparameters(self, hyperparams):
        for i in range(len(hyperparameters)):
            self.models[i] = SurvivalLVQ(**hyperparameters[i])

    def fit(self, X, y):
        self.n_events   = y.shape[1]
        self.n_features = X.shape[1]
        for i in range(y.shape[1]):
            self.models[i].fit(X, y[:,i])
    
    def predict(self, X, closest=False):
        y_pred = []
        for i in range(len(self.models)):
            y_pred.append(self.models[i].predict(X, closest=closest))
        y_pred = np.stack(y_pred, axis=1)
        return y_pred
    
    def predict_survival_function(self, X):
        y_pred = []
        for i in range(len(self.models)):
            y_pred.append(self.models[i].predict_survival_function(X))
        y_pred = np.stack(y_pred, axis=1)
        return y_pred
    
    def __getitem__(self, i):
        return self.models[i]
    
    def __iter__(self):
        for model in self.models:
            yield model
    
    def __len__(self):
        return len(self.models)

