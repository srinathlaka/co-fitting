import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from models.fitting import residuals
from utils.layout import generate_labels, create_stable_button_layout
from utils.plotting import plot_fits
from config.defaults import get_default_params


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
        param_names, default_vals = get_default_params(model_type)
        B_input = st.text_input("Enter B values (comma-separated)", value=", ".join([str(i * 0.1) for i in range(1, num_groups + 1)]))
        try:
            B_values = [float(b.strip()) for b in B_input.split(",")]
        except:
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
                result = least_squares(residuals, param_inputs, args=(time, y_data_list, B_values, model_type))
                if result.success:
                    st.success("Fitting successful!")
                    for name, val in zip(param_names, result.x):
                        st.write(f"**{name}**: {val:.5f}")
                    st.subheader("Fitted Curves")
                    st.plotly_chart(plot_fits(time, y_data_list, result.x, B_values, model_type))
                else:
                    st.error("Fitting failed. Adjust initial parameters.")
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
