from dataclasses import dataclass, field
import numpy as np
from functools import cached_property
from typing import Generic, TypeVar
import scipy.stats
T = TypeVar('T', bound=np.ndarray)


@dataclass
class MultiVarGauss(Generic[T]):
    """A class for using Gaussians"""
    mean: T
    cov: np.ndarray

    time: float = field(default=None)

    @property
    def ndim(self) -> int:
        return self.mean.shape[0]

    def mahalanobis_distance(self, x: T) -> float:
        """Calculate the mahalanobis distance between self and x.

        This is also known as the quadratic form of the Gaussian.
        See (3.2) in the SF book.
        """
        err = x.reshape(-1, 1) - self.mean.reshape(-1, 1)
        mahalanobis_distance = float(err.T @ np.linalg.solve(self.cov, err))
        return mahalanobis_distance

    def mahal_dist(self, x: T) -> float:
        return self.mahalanobis_distance(x)

    def pdf(self, x: np.ndarray) -> float:
        """Calculate the likelihood of x given the Gaussian"""
        return scipy.stats.multivariate_normal(self.mean, self.cov).pdf(x)

    def logpdf(self, x: np.ndarray) -> float:
        """Calculate the log likelihood of x given the Gaussian"""
        return scipy.stats.multivariate_normal(self.mean, self.cov).logpdf(x)

    def sample(self) -> T:
        """Sample from the Gaussian"""
        noise = np.random.multivariate_normal(
            np.zeros_like(self.mean), self.cov, 1).reshape(-1)
        return self.mean + noise

    def get_marginalized(self, indices):
        i_idx, j_idx = np.meshgrid(indices, indices,
                                   sparse=True, indexing='ij')
        mean = self.mean[i_idx.ravel()]
        cov = self.cov[i_idx, j_idx]
        return MultiVarGauss(mean, cov)

    @cached_property
    def cholesky(self):
        return np.linalg.cholesky(self.cov)

    @property
    def meta(self) -> T:
        return self.mean

    def __iter__(self):
        """Enable iteration over the mean and covariance.
        i.e.
            est = MultiVarGauss2d(mean=[1, 2], cov=[[1, 0], [0, 1]])
            mean, cov = est
        """
        return iter((self.mean, self.cov))

    def __repr__(self) -> str:
        """Used for pretty printing"""
        def sci(x): return f'{x: .23}'
        out = '\n'
        for i in range(self.mean.shape[0]):
            mline = sci(self.mean[i])
            cline = ' |'.join(sci(self.cov[i, j])
                              for j in range(self.cov.shape[1]))
            out += f"|{mline} |      |{cline} |\n"
        return out

    def __str__(self) -> str:
        return str(self)

