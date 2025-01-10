from WFP_SUDAN_CFSVA import run_cfsa
from WFP_SUDAN_FSMS import run_fsms
import streamlit as st

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

# Credentials
USERNAME = "wfp2025"
PASSWORD = "wfp2025"

# Login function
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
        else:
            st.error("Invalid username or password. Please try again.")

            # Initialize session state for login
            if "logged_in" not in st.session_state:
                st.session_state.logged_in = False

            # If not logged in, show login page
            if not st.session_state.logged_in:
                login()
            else:

                # Initialize session state for first load
                if 'initialized' not in st.session_state:
                    st.session_state.initialized = True
                    st.session_state.active_module = "cfsa"

                # Create a sidebar for navigation buttons
                st.sidebar.title("Navigation")

                # Add buttons to the sidebar
                if st.sidebar.button("View CFSA"):
                    st.session_state.active_module = "cfsa"

                if st.sidebar.button("View FSMS"):
                    st.session_state.active_module = "fsms"

                # Display content based on session state
                if st.session_state.active_module == "cfsa":
                    run_cfsa()

                if st.session_state.active_module == "fsms":
                    run_fsms()