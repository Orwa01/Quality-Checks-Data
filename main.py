from WFP_SUDAN_CFSVA import run_cfsa
from WFP_SUDAN_FSMS import run_fsms
import streamlit as st

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

# Initialize session state for first load
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.run_cfsa = True
    st.session_state.run_fsms = False

# Create a sidebar for navigation buttons
st.sidebar.title("Navigation")

# Add buttons to the sidebar
if st.sidebar.button("View CFSA"):
    st.session_state.run_cfsa = True
    st.session_state.run_fsms = False

if st.sidebar.button("View FSMS"):
    st.session_state.run_fsms = True
    st.session_state.run_cfsa = False

# Display content based on session state
if st.session_state.run_cfsa:
    run_cfsa()
    st.session_state.run_cfsa = False

if st.session_state.run_fsms:
    run_fsms()
    st.session_state.run_fsms = False