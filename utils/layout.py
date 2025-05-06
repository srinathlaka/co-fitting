import streamlit as st
import string

def generate_labels(rows, columns):
    """Generate well labels (e.g., A1, A2, ..., H12)."""
    labels = []
    for r in range(rows):
        for c in range(columns):
            labels.append(f"{string.ascii_uppercase[r]}{c+1}")
    return labels

def create_stable_button_layout(rows, columns, labels, key_prefix, df=None):
    if f"{key_prefix}_selection" not in st.session_state:
        st.session_state[f"{key_prefix}_selection"] = []

    st.write("### Select Wells")
    col_headers = st.columns([0.5] + [1] * columns + [2])
    with col_headers[0]:
        st.write("##")
    for c in range(columns):
        with col_headers[c+1]:
            st.write(f"**{c+1}**")
    with col_headers[-1]:
        select_cols = st.columns(2)
        with select_cols[0]:
            if st.button("Select All", key=f"{key_prefix}_select_all"):
                st.session_state[f"{key_prefix}_selection"] = labels.copy()
                st.rerun()
        with select_cols[1]:
            if st.button("Clear All", key=f"{key_prefix}_clear_all"):
                st.session_state[f"{key_prefix}_selection"] = []
                st.rerun()

    for r in range(rows):
        cols = st.columns([0.5] + [1] * columns + [2])
        with cols[0]:
            st.write(f"**{string.ascii_uppercase[r]}**")
        for c in range(columns):
            idx = r * columns + c
            if idx < len(labels):
                well = labels[idx]
                with cols[c+1]:
                    is_selected = well in st.session_state[f"{key_prefix}_selection"]
                    disabled = not (df is None or well in df.columns)
                    style = "primary" if is_selected else "secondary"
                    if st.button(
                        well, 
                        key=f"{key_prefix}_{well}",
                        type=style,
                        disabled=disabled,
                        use_container_width=True
                    ):
                        if well in st.session_state[f"{key_prefix}_selection"]:
                            st.session_state[f"{key_prefix}_selection"].remove(well)
                        else:
                            st.session_state[f"{key_prefix}_selection"].append(well)
                        st.rerun()

    st.write("")
    plot_col1, plot_col2, plot_col3 = st.columns([1, 1, 1])
    with plot_col2:
        plot_clicked = st.button("📊 Plot Selected Wells", key=f"{key_prefix}_plot", use_container_width=True)

    return st.session_state[f"{key_prefix}_selection"], plot_clicked
