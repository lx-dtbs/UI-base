###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.best_weight = np.empty([0, self.dim])
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, weight, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        print("高斯核超参数权重为{}".format(weights))
        print("融入的时空权重为{}".format(weight))
        new_weights = np.multiply(weight, weights)
        print("新权重为{}".format(new_weights))
        new_weights = new_weights / new_weights.mean()  # This will make the next line more stable
        new_weights = new_weights / np.prod(np.power(new_weights, 1.0 / len(new_weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - new_weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + new_weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            if len(self._fX) > 0 and self.verbose:
                n_evals, fbest = self.n_evals, self._fX.min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            # Initialize parameters
            self._restart()

            # Generate and evalute initial design points
            #X_init = latin_hypercube(self.n_init, self.dim)
            #X_init = from_unit_cube(X_init, self.lb, self.ub)
            fX_init = np.empty([0,1])
            weight_init = np.empty([0,self.dim])

            X_init = np.array([
            [19.1512902,  30.66999372, 33.85319697, 18.86152453, 18.25533214, 11.17938789,
             10.68834258, 31.34137937,  5.96554352, 34.80400402,  6.73433447, 32.37329027,
             31.85395915, 23.25921838, 28.18165874, 30.4265282,  16.96362894,  6.53539927,
             11.38382367, 30.16079996, 34.90904051,  6.71203666, 11.99139555, 24.21643866,
             10.51195945, 14.23215485, 31.84674244],
            [ 7.39484249, 10.63687559,  5.1069129,  27.41286891, 23.77709681, 19.32551571,
             24.6522096,   6.84525384, 21.59807794, 19.07438694, 12.45945066,  9.2240206,
             23.49520556, 22.13841779,  8.63759773, 22.25014471, 30.52197153,  8.48023622,
             23.83094733, 28.84500944, 24.45388896, 12.74425894, 26.1676655,  31.45411957,
             25.83166999, 31.66315423, 20.05879179],
            [10.6651647,  25.80779481, 22.74780636, 15.57285446, 21.46817746, 25.69186811,
             29.87084864, 21.77474566, 23.83065821,  9.24907276, 19.68347124, 20.40416124,
             11.46159443, 29.90303373, 25.35690075, 25.93825967, 24.07797206, 24.10699166,
              6.35901921, 33.87849813, 12.80058202, 21.06788945, 24.53334475, 20.76771864,
             13.68014588, 21.11622521, 24.30305205],
            [24.05948144, 33.17387082, 27.34327235,  9.17096526, 33.0369055,  33.91122144,
             33.87352942, 13.10362384, 34.51309587,  7.54628224, 15.83027532, 17.29521066,
             16.82397675, 16.47076813, 33.37984507, 12.97212126, 28.49652062, 22.94940423,
             14.75565888, 19.86337842, 28.69101337, 18.88440276, 18.36645447, 33.14253687,
             27.72225536, 18.36746382, 32.06770133],
            [27.32326222, 17.26335887, 18.09912825, 30.71338534, 15.0191844,  29.71026095,
             15.71845041, 26.13258823, 29.62477729, 11.84991683, 31.92875068, 15.99287084,
             26.55593012, 18.9109692,  12.51937757, 27.18832443, 21.54255071, 11.14824819,
              9.96560908,  7.89580507, 19.90281272, 31.99847782, 21.28184002, 16.64458679,
             34.1417351,  13.67016852, 26.05116852]
            ])
            #fX_init = np.array([[self.f(x)] for x in X_init])
            for x in X_init:
                re1, re2 = self.f(x)
                fX_init = np.append(fX_init, np.array([[re1]]), axis=0)
                weight_init = np.append(weight_init, re2.reshape(-1,self.dim), axis=0)
            
            print(fX_init)

            # Update budget and set as initial data for this TR
            self.n_evals += self.n_init
            self._X = deepcopy(X_init)
            self._fX = deepcopy(fX_init)
            self.best_weight = np.append(self.best_weight, weight_init[np.argmin(fX_init)].reshape(-1,self.dim), axis=0)

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_init)))
            self.fX = np.vstack((self.fX, deepcopy(fX_init)))

            if self.verbose:
                fbest = self._fX.min()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)
                print(X.shape[0])

                # Standardize values
                fX = deepcopy(self._fX).ravel()
                print(fX.shape[0])
                
                ### 确定当前次迭代的权重
                if self.n_evals == self.n_init:
                    weight = self.best_weight[0]

                # Create th next batch
                X_cand, y_cand, _ = self._create_candidates(
                    X, fX, weight=weight, length=self.length, n_training_steps=self.n_training_steps, hypers={}
                )
                X_next = self._select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)

                # Evaluate batch
                fX_next = np.empty([0, 1])
                next_weight = np.empty([0, self.dim])
                for x in X_next:
                    re1, re2 = self.f(x)
                    fX_next = np.append(fX_next, np.array([[re1]]), axis=0)
                    next_weight = np.append(next_weight, re2.reshape(-1,self.dim), axis=0)
                ### fX_next = np.array([[self.f(x)] for x in X_next])

                # Update trust region
                self._adjust_length(fX_next)

                # Update budget and append data
                self.n_evals += self.batch_size
                self._X = np.vstack((self._X, X_next))
                self._fX = np.vstack((self._fX, fX_next))

                if self.verbose and fX_next.min() < self.fX.min():
                    weight = next_weight[np.argmin(fX_init)]
                    self.best_weight = np.append(self.best_weight, weight.reshape(-1,self.dim), axis=0)
                    n_evals, fbest = self.n_evals, fX_next.min()
                    print(f"{n_evals}) New best: {fbest:.4}")
                    sys.stdout.flush()
                
                ### 如果无法找到最好值则更新为1或者之前最好
                if fX_next.min() >= self.fX.min():
                    weight = self.best_weight[-1]

                # Append data to the global history
                self.X = np.vstack((self.X, deepcopy(X_next)))
                self.fX = np.vstack((self.fX, deepcopy(fX_next)))
