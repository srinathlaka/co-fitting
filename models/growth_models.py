import numpy as np

def exponential_model(t, B, X0, mu0, mu1):
    mu = mu0 - mu1 * B
    return X0 * np.exp(mu * t)

def logistic_model(t, B, X0, mu0, mu1, K):
    mu = mu0 - mu1 * B
    return K / (1 + ((K - X0) / X0) * np.exp(-mu * t))

def linear_model(t, B, X0, mu0, mu1):
    return X0 * np.exp(mu0 - mu1 * B) * t

def baranyi_model(t, B, X0, mu0, mu1, K, h0):
    mu = mu0 - mu1 * B
    A = t + (1 / mu) * np.log(np.exp(-mu * t) + np.exp(-h0) - np.exp(-mu * t - h0))
    return K / (1 + ((K - X0) / X0) * np.exp(-mu * A))

def drug_effect_model(t, B, X0, mu0, K1, K2):
    X0_safe = max(X0, 1e-6)
    K2_safe = max(K2, 1e-6)
    mu = mu0 + (K1 * B) / (K2_safe + B)
    exp_term = np.clip(mu * t, -50, 50)
    return X0_safe * np.exp(exp_term)
