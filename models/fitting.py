import numpy as np
from .growth_models import *

def residuals(params, time, y_data_list, B_list, model_type):
    res = []
    try:
        # Ensure parameters are positive to prevent model errors
        if any(p <= 0 for p in params):
            return np.ones(len(time) * len(y_data_list)) * 1e6
            
        for i, y in enumerate(y_data_list):
            B = B_list[i]
            
            if model_type == "Logistic":
                X0, mu0, mu1, K = params
                # Prevent division by zero and ensure sensible parameters
                if mu0 <= mu1 * B or K <= 0 or X0 <= 0:
                    return np.ones(len(time) * len(y_data_list)) * 1e6
                y_pred = logistic_model(time, B, X0, mu0, mu1, K)
                
            elif model_type == "Exponential":
                X0, mu0, mu1 = params
                # Prevent negative growth rates
                if mu0 <= mu1 * B:
                    return np.ones(len(time) * len(y_data_list)) * 1e6
                y_pred = exponential_model(time, B, X0, mu0, mu1)
                
            elif model_type == "Linear":
                X0, mu0, mu1 = params
                y_pred = linear_model(time, B, X0, mu0, mu1)
                
            elif model_type == "Baranyi":
                X0, mu0, mu1, K, h0 = params
                # Prevent division by zero
                if mu0 <= mu1 * B or K <= 0 or X0 <= 0:
                    return np.ones(len(time) * len(y_data_list)) * 1e6
                y_pred = baranyi_model(time, B, X0, mu0, mu1, K, h0)
                
            elif model_type == "Drug_Effect":
                X0, mu0, K1, K2 = params
                if X0 <= 0 or mu0 <= 0:
                    return np.ones(len(time) * len(y_data_list)) * 1e6
                y_pred = drug_effect_model(time, B, X0, mu0, K1, K2)
                
            # Check for NaN values in predictions
            if np.any(np.isnan(y_pred)):
                return np.ones(len(time) * len(y_data_list)) * 1e6
                
            res.extend(y - y_pred)
    except Exception as e:
        print(f"Error in residuals calculation: {e}")
        return np.ones(len(time) * len(y_data_list)) * 1e6
    return res
