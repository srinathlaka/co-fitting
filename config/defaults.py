def get_default_params(model_type):
    if model_type == "Logistic":
        return ["X0", "mu0", "mu1", "K"], [0.05, 0.3, 0.1, 1.0]
    elif model_type == "Baranyi":
        return ["X0", "mu0", "mu1", "K", "h0"], [0.05, 0.3, 0.1, 1.0, 0.5]
    elif model_type == "Drug_Effect":
        return ["X0", "mu0", "K1", "K2"], [0.05, 0.3, 0.1, 0.5]
    elif model_type in ["Exponential", "Linear"]:
        return ["X0", "mu0", "mu1"], [0.05, 0.3, 0.1]
    else:
        return [], []
