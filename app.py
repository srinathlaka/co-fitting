import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from models.fitting import residuals
from utils.layout import generate_labels, create_stable_button_layout
from utils.plotting import plot_fits
from config.defaults import get_default_params
import plotly.graph_objects as go


def main():
    st.set_page_config(page_title="Co-Fitting Models App", page_icon="🧪", layout="wide")
    st.title("🧪 Multi-Model Co-Fitting Application")
    st.markdown("""<h4 style='color: teal;'>Upload your data, select the growth model, and visualize your co-fitting results beautifully.</h4>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload CSV or Excel File", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df.columns = ["Time"] + [str(c) for c in df.columns[1:]]
        st.session_state.df = df

        st.dataframe(df.head())

        rows = st.number_input("Number of Rows", 1, 16, 8)
        columns = st.number_input("Number of Columns", 1, 24, 12)
        num_groups = st.number_input("Number of Groups", 1, 5, 1)

        labels = generate_labels(rows, columns)
        mapped_df = pd.DataFrame({"Time": df["Time"]})
        for i, col in enumerate(df.columns[1:]):
            if i < len(labels):
                mapped_df[labels[i]] = df[col]

        sample_wells_by_group = {}
        blank_wells_by_group = {}
        bg_subtracted_dfs = {}

        # Initialize the bg_subtracted_dfs with default values for all groups
        for group_num in range(1, num_groups + 1):
            bg_subtracted_dfs[group_num] = mapped_df  # Default to original data

        group_tabs = st.tabs([f"Group {i+1}" for i in range(num_groups)])

        for i, tab in enumerate(group_tabs):
            group_num = i + 1
            with tab:
                # Get sample wells and store them in session state
                sample_wells, plot_clicked = create_stable_button_layout(rows, columns, labels, f"group_{group_num}_sample", mapped_df)
                
                # Ensure sample wells are stored and retrieved properly
                if sample_wells:
                    st.session_state[f"sample_wells_group_{group_num}"] = sample_wells
                elif f"sample_wells_group_{group_num}" in st.session_state:
                    sample_wells = st.session_state[f"sample_wells_group_{group_num}"]
                    
                # Update the dictionary for later use
                sample_wells_by_group[group_num] = sample_wells
                
                # Show selected wells if available
                if sample_wells:
                    st.success(f"✅ Selected Sample Wells for Group {group_num}: {', '.join(sample_wells)}")
                    
                    # Toggle plot visibility when the plot button is clicked
                    if f"show_plot_{group_num}" not in st.session_state:
                        st.session_state[f"show_plot_{group_num}"] = False
                    
                    if plot_clicked:
                        st.session_state[f"show_plot_{group_num}"] = not st.session_state[f"show_plot_{group_num}"]
                    
                    # Show or hide plot based on state
                    if st.session_state[f"show_plot_{group_num}"]:
                        with st.expander("Sample Plot", expanded=True):
                            try:
                                st.info("Plotting sample wells data...")
                                fig = plot_fits(mapped_df["Time"].values, 
                                             [mapped_df[well] for well in sample_wells],
                                             None, 
                                             [], 
                                             "Exponential")
                                st.plotly_chart(fig)
                            except Exception as e:
                                st.error(f"Error plotting sample data: {str(e)}")
                
                # Background subtraction section
                do_bg = st.checkbox("Perform Background Subtraction", key=f"bg_{group_num}")
                
                if do_bg and sample_wells:
                    blank_wells, _ = create_stable_button_layout(rows, columns, labels, f"group_{group_num}_blank", mapped_df)
                    
                    # Ensure blank wells are stored and retrieved properly
                    if blank_wells:
                        st.session_state[f"blank_wells_group_{group_num}"] = blank_wells
                    elif f"blank_wells_group_{group_num}" in st.session_state:
                        blank_wells = st.session_state[f"blank_wells_group_{group_num}"]
                        
                    blank_wells_by_group[group_num] = blank_wells
                    
                    if blank_wells:
                        st.success(f"✅ Selected Blank Wells for Group {group_num}: {', '.join(blank_wells)}")
                        
                        # Calculate background-subtracted data
                        blank_avg = mapped_df[blank_wells].mean(axis=1)
                        group_df = mapped_df.copy()
                        for well in sample_wells:
                            group_df[well] = mapped_df[well] - blank_avg
                        bg_subtracted_dfs[group_num] = group_df
                        
                        # Initialize session state for background plot
                        if f"plot_bg_subtracted_{group_num}" not in st.session_state:
                            st.session_state[f"plot_bg_subtracted_{group_num}"] = False
                        
                        # Create two columns for better layout - just like the sample plot
                        bg_col1, bg_col2 = st.columns([1, 3])
                        
                        # Toggle button for background-subtracted plot
                        if bg_col1.button("📊 Plot BG-Subtracted", key=f"plot_bg_btn_{group_num}"):
                            st.session_state[f"plot_bg_subtracted_{group_num}"] = not st.session_state[f"plot_bg_subtracted_{group_num}"]
                        
                        # Show/hide status in second column
                        bg_display_status = "🟢 Plot Visible" if st.session_state[f"plot_bg_subtracted_{group_num}"] else "⚪ Plot Hidden"
                        bg_col2.markdown(f"**Status:** {bg_display_status}")
                        
                        # Use an expander for the plot - just like the sample plot
                        if st.session_state[f"plot_bg_subtracted_{group_num}"]:
                            with st.expander("Background-Subtracted Plot", expanded=True):
                                try:
                                    st.info("Plotting background-subtracted data...")
                                    fig = plot_fits(group_df["Time"].values, 
                                                 [group_df[well] for well in sample_wells],
                                                 None, 
                                                 [], 
                                                 "Exponential")
                                    st.plotly_chart(fig)
                                except Exception as e:
                                    st.error(f"Error plotting background-subtracted data: {str(e)}")
                else:
                    if sample_wells:
                        bg_subtracted_dfs[group_num] = mapped_df

        st.markdown("---")
        st.subheader("📊 Combined Analysis")
        time = mapped_df["Time"].values
        y_data_list = []
        B_values = []
        combined_fig = None

        for group_num in range(1, num_groups + 1):
            if sample_wells_by_group.get(group_num):
                df_group = bg_subtracted_dfs[group_num]
                sample_data = [df_group[well].values for well in sample_wells_by_group[group_num]]
                avg = np.mean(sample_data, axis=0)
                y_data_list.append(avg)
        
        # Add this section to show the group averages plot
        if y_data_list:
            st.subheader("Group Averages")
            try:
                avg_fig = plot_fits(time, y_data_list, None, [], "Averages")
                # Add group labels to the legend
                for i, trace in enumerate(avg_fig.data):
                    trace.name = f"Group {i+1} Average" 
                st.plotly_chart(avg_fig)
            except Exception as e:
                st.error(f"Error plotting group averages: {str(e)}")

        model_type = st.selectbox("Select Growth Model", ["Logistic", "Exponential", "Linear", "Baranyi", "Drug_Effect"])
        
        # Display the equation for the selected model in LaTeX
        eq_col1, eq_col2 = st.columns([1, 3])
        with eq_col1:
            st.markdown("### Model Equation:")
        with eq_col2:
            if model_type == "Logistic":
                st.latex(r"X(t) = \frac{K}{1 + \left(\frac{K - X_0}{X_0}\right) e^{-(\mu_0 - \mu_1 B)t}}")
            elif model_type == "Exponential":
                st.latex(r"X(t) = X_0 \cdot e^{(\mu_0 - \mu_1 B)t}")
            elif model_type == "Linear":
                st.latex(r"X(t) = X_0 \cdot e^{\mu_0 - \mu_1 B} \cdot t")
            elif model_type == "Baranyi":
                st.latex(r"X(t) = X_0 + (\mu_0 - \mu_1 B) \cdot A(t) - \ln\left(1 + \frac{e^{(\mu_0 - \mu_1 B) \cdot A(t)} - 1}{e^{K - X_0}}\right)")
                st.latex(r"A(t) = t + \frac{1}{\mu} \ln\left(e^{-\mu t} + e^{-h_0} - e^{-\mu t - h_0}\right)")
            elif model_type == "Drug_Effect":
                st.latex(r"X(t) = X_0 \cdot e^{\mu_0 t} \cdot e^{-K_1 B t} \cdot e^{-K_2 B^2 t}")
        
        # Get default parameters
        param_names, default_vals = get_default_params(model_type)

        # Add this before setting up the parameter inputs
        # Generate better initial estimates based on the data
        estimated_defaults = []
        
        if y_data_list:
            # For all models, estimate X0 from the data
            X0_est = np.mean([data[0] for data in y_data_list])
            
            if model_type == "Exponential":
                # Estimate growth rate from first and last points
                time_range = time[-1] - time[0]
                growth_rates = []
                for data in y_data_list:
                    if data[-1] > data[0] and data[0] > 0:  # Only use positive growth
                        rate = np.log(data[-1]/data[0]) / time_range
                        growth_rates.append(rate)
                
                mu0_est = np.mean(growth_rates) if growth_rates else 0.3
                estimated_defaults = [X0_est, mu0_est, 0.1]
            
            elif model_type == "Logistic":
                # Estimate K from max values
                K_est = np.max([np.max(data) for data in y_data_list]) * 1.1
                estimated_defaults = [X0_est, 0.3, 0.1, K_est]
        
        # If we have estimated values, use them; otherwise use defaults
        if estimated_defaults and all(v > 0 for v in estimated_defaults):
            default_vals = estimated_defaults

        # After selecting the model and before B values input, add time range selection
        st.subheader("⏱️ Time Range Selection")
        
        # Get min and max time values from the data
        min_time = float(time[0])
        max_time = float(time[-1])
        
        # Create two columns for side-by-side time range inputs
        time_col1, time_col2 = st.columns(2)
        
        with time_col1:
            start_time = st.number_input(
                "Start Time", 
                min_value=min_time,
                max_value=max_time, 
                value=min_time,
                step=0.1,
                format="%.1f"
            )
        
        with time_col2:
            end_time = st.number_input(
                "End Time", 
                min_value=min_time,
                max_value=max_time, 
                value=max_time,
                step=0.1,
                format="%.1f"
            )
        
        # Add a visual indicator of selected range
        time_range_fig = go.Figure()
        time_range_fig.add_trace(go.Scatter(
            x=time, 
            y=[np.mean(data) for data in zip(*y_data_list)],
            mode='lines',
            name='Average of all groups'
        ))
        
        # Add vertical lines indicating selected range
        time_range_fig.add_vline(x=start_time, line_width=2, line_dash="dash", line_color="green")
        time_range_fig.add_vline(x=end_time, line_width=2, line_dash="dash", line_color="red")
        
        time_range_fig.update_layout(
            title="Selected Time Range for Fitting",
            xaxis_title="Time",
            yaxis_title="Average OD",
            height=300
        )
        
        st.plotly_chart(time_range_fig)

        # REMOVED: B_scale slider
        B_input = st.text_input("Enter B values (comma-separated)", value=", ".join([str(i * 0.1) for i in range(1, num_groups + 1)]))
        try:
            # Use B values directly without scaling
            B_values = [float(b.strip()) for b in B_input.split(",")]
        except:
            # Default values without scaling
            B_values = [0.1] * num_groups

        st.subheader("Initial Parameters")
        param_inputs = []
        cols = st.columns(len(param_names))
        for i, (name, default) in enumerate(zip(param_names, default_vals)):
            with cols[i]:
                val = st.number_input(name, value=default, key=f"param_{name}")
                param_inputs.append(val)

        if st.button("🚀 Run Co-Fitting"):
            try:
                # Filter time and data based on selected range
                time_mask = (time >= start_time) & (time <= end_time)
                filtered_time = time[time_mask]
                filtered_y_data_list = [y_data[time_mask] for y_data in y_data_list]
                
                # Add constraints to prevent negative or near-zero parameter values
                param_bounds = ([1e-3] * len(param_inputs), [np.inf] * len(param_inputs))
                
                # Use FILTERED data for fitting
                result = least_squares(
                    residuals, 
                    param_inputs, 
                    args=(filtered_time, filtered_y_data_list, B_values, model_type),
                    bounds=param_bounds,
                    method='trf',
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                    max_nfev=1000,
                    verbose=1
                )
                
                if result.success:
                    st.success("Fitting successful!")
                    for name, val in zip(param_names, result.x):
                        st.write(f"**{name}**: {val:.5f}")
                    
                    # Create tabs for showing different views
                    fit_tab1, fit_tab2 = st.tabs(["Fitted Time Range", "Full Time Range"])
                    
                    with fit_tab1:
                        # Show only the filtered time range (what was actually fitted)
                        st.subheader("Fitted Data (Selected Range Only)")
                        fitted_range_fig = plot_fits(filtered_time, 
                                                    [y_data[time_mask] for y_data in y_data_list], 
                                                    result.x, 
                                                    B_values, 
                                                    model_type)
                        st.plotly_chart(fitted_range_fig)
                    
                    with fit_tab2:
                        # Show extrapolation over the entire dataset
                        st.subheader("Model Prediction (Full Range)")
                        full_range_fig = plot_fits(time, 
                                                  y_data_list, 
                                                  result.x, 
                                                  B_values, 
                                                  model_type)
                        st.plotly_chart(full_range_fig)
                
                # Add diagnostic information
                st.subheader("Fitting Diagnostics")
                st.write(f"**Optimization Status:** {result.status}")
                st.write(f"**Function Evaluations:** {result.nfev}")
                st.write(f"**Optimality (lower is better):** {result.optimality:.6f}")
                
                # Plot residuals
                residual_fig = go.Figure()
                residual_array = np.array(result.fun).reshape(len(y_data_list), -1)
                for i, res in enumerate(residual_array):
                    residual_fig.add_trace(go.Scatter(
                        x=time, y=res, mode='lines+markers', 
                        name=f'Group {i+1} Residuals'
                    ))
                residual_fig.update_layout(title='Fitting Residuals', xaxis_title='Time', yaxis_title='Residual')
                st.plotly_chart(residual_fig)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
