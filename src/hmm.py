"""Gaussian Hidden Markov Model for dividing daily returns into regimes (plain NumPy).

fit_gaussian_hmm  - Baum-Welch (EM) on the training returns
filter_probs      - P(regime_t | returns up to and including t), parameters fixed:
                    this is what we may use on day t without looking into the future
to_continuous     - daily estimates -> yearly drift, volatility and generator Q
"""
import numpy as np
from scipy.stats import norm

DT = 1 / 252


def _emissions(x, mu, sd):
    return norm.pdf(x[:, None], mu[None, :], sd[None, :]) + 1e-300


def _forward_backward(x, pi, A, mu, sd):
    T, m = len(x), len(pi)
    B = _emissions(x, mu, sd)
    alpha = np.zeros((T, m)); c = np.zeros(T)
    alpha[0] = pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    beta = np.ones((T, m))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
    return gamma, xi, alpha, np.log(c).sum()


def fit_gaussian_hmm(x, m=2, n_iter=300, tol=1e-7, seed=0):
    """Returns dict: mu, sd (per day), A (daily transition matrix), pi (start probs),
    gamma (smoothed probs, uses the whole sample - for plots only), filtered (alpha),
    loglik. States are sorted by volatility: 0 = calm, m-1 = most volatile."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    qs = np.quantile(np.abs(x), np.linspace(0.3, 0.9, m))
    mu = np.zeros(m) + x.mean(); sd = np.maximum(qs, 1e-4) * (1 + 0.1 * rng.random(m))
    A = np.full((m, m), 0.02 / max(m - 1, 1)); np.fill_diagonal(A, 0.98)
    pi = np.full(m, 1 / m)
    ll_old = -np.inf
    for _ in range(n_iter):
        gamma, xi, alpha, ll = _forward_backward(x, pi, A, mu, sd)
        pi = gamma[0]
        A = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
        w = gamma / gamma.sum(0)
        mu = (w * x[:, None]).sum(0)
        sd = np.sqrt((w * (x[:, None] - mu) ** 2).sum(0))
        if ll - ll_old < tol:
            break
        ll_old = ll
    o = np.argsort(sd)
    return dict(mu=mu[o], sd=sd[o], A=A[np.ix_(o, o)], pi=pi[o], gamma=gamma[:, o],
                filtered=alpha[:, o], loglik=ll, m=m)


def filter_probs(x, fit, p0=None):
    """Forward filter with FIXED parameters: row t = P(regime_t | x_0..x_t).
    p0 = regime probabilities just before x_0 (e.g. the last filtered row of the training set)."""
    x = np.asarray(x, dtype=float)
    A, mu, sd = fit["A"], fit["mu"], fit["sd"]
    p = fit["pi"] if p0 is None else p0
    B = _emissions(x, mu, sd)
    out = np.zeros((len(x), len(mu)))
    for t in range(len(x)):
        p = (p @ A) * B[t]
        p /= p.sum()
        out[t] = p
    return out


def to_continuous(fit, dt=DT):
    """Daily HMM estimates -> continuous-time model (per year):
    drift of the LOG-price m_i, volatility sigma_i, generator Q."""
    from scipy.linalg import logm
    sigma = fit["sd"] / np.sqrt(dt)
    drift = fit["mu"] / dt
    Q = np.real(logm(fit["A"])) / dt
    if (Q - np.diag(np.diag(Q)) < 0).any():          # logm not a valid generator -> fallback
        Q = (fit["A"] - np.eye(len(Q))) / dt
    Q -= np.diag(Q.sum(1))                           # rows sum exactly to 0
    return drift, sigma, Q
