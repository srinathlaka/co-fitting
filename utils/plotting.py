import plotly.graph_objects as go
import numpy as np
from models.growth_models import *

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

def plot_fits(time, y_data_list, params, B_list, model_type, confidence_interval=None):
    # Create a default params structure based on model type if None
    if params is None:
        if model_type == "Logistic":
            params = [0.1, 0.1, 0.1, 1.0]  # X0, mu0, mu1, K
        elif model_type == "Exponential":
            params = [0.1, 0.1, 0.1]  # X0, mu0, mu1
        elif model_type == "Linear":
            params = [0.1, 0.1, 0.1]  # X0, mu0, mu1
        elif model_type == "Baranyi":
            params = [0.1, 0.1, 0.1, 1.0, 0.1]  # X0, mu0, mu1, K, h0
        elif model_type == "Drug_Effect":
            params = [0.1, 0.1, 0.1, 0.1]  # X0, mu0, K1, K2
        else:
            params = [0.1, 0.1, 0.1]  # Default fallback

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Special case: If B_list is empty, just plot sample data without fitting
    if len(B_list) == 0:
        for i, y_obs in enumerate(y_data_list):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=time, y=y_obs, mode='markers', 
                                     name=f'Sample {i+1}', 
                                     marker=dict(color=color)))
    else:
        # Normal co-fitting case
        for i, y_obs in enumerate(y_data_list):
            color = colors[i % len(colors)]
            B = B_list[i]
            if model_type == "Logistic":
                X0, mu0, mu1, K = params
                y_pred = logistic_model(time, B, X0, mu0, mu1, K)
            elif model_type == "Exponential":
                X0, mu0, mu1 = params
                y_pred = exponential_model(time, B, X0, mu0, mu1)
            elif model_type == "Linear":
                X0, mu0, mu1 = params
                y_pred = linear_model(time, B, X0, mu0, mu1)
            elif model_type == "Baranyi":
                X0, mu0, mu1, K, h0 = params
                y_pred = baranyi_model(time, B, X0, mu0, mu1, K, h0)
            elif model_type == "Drug_Effect":
                X0, mu0, K1, K2 = params
                y_pred = drug_effect_model(time, B, X0, mu0, K1, K2)

            fig.add_trace(go.Scatter(x=time, y=y_obs, mode='markers', 
                                     name=f'Data B={B}', 
                                     marker=dict(color=color)))
            fig.add_trace(go.Scatter(x=time, y=y_pred, mode='lines', 
                                     name=f'Fit B={B}', 
                                     line=dict(color=color)))

            if confidence_interval is not None and i in confidence_interval:
                lower, upper = confidence_interval[i]
                fig.add_trace(go.Scatter(
                    x=np.concatenate([time, time[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=hex_to_rgba(color, 0.2),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))

    fig.update_layout(
        title='Sample Data' if len(B_list) == 0 else f'{model_type} Co-Fitting Results',
        xaxis_title='Time',
        yaxis_title='OD',
        template='plotly_white',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
