import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from scipy.stats import pearsonr, spearmanr

from WFP_SUDAN_CFSVA import load_logo


def display_fsms_data(df):
    # Combined CSS for full-width layout and styled tabs
    st.markdown("""
        <style>
            /* Full-width container for the main content */
            .main {
                max-width: 100%;
                padding: 0;
            }
            /* Center align all headers for consistency */
            h1, h2, h3, h4, h5, h6 {
                text-align: center;
                font-family: Arial, sans-serif;
            }
            /* Style the sidebar to maintain a professional look */
            [data-testid="stSidebar"] {
                min-width: 15%;
                max-width: 20%;
            }
            /* Reduce padding around the container to maximize space usage */
            .block-container {
                padding: 1rem 2rem;
            }
            /* Styling tabs for better visibility and appeal */
            .stTabs [role="tablist"] button {
                padding: 16px 20px; /* Increase padding for larger clickable areas */
                font-size: 18px; /* Larger font size for better readability */
                font-weight: bold;
                color: white;
            }
            /* Color code tabs */
            .stTabs [role="tablist"] button:nth-child(2) {
                background-color: #007BFF; /* Blue for Dashboard */
            }
            .stTabs [role="tablist"] button:nth-child(3) {
                background-color: #28A745; /* Green for Data Issues */
            }
            .stTabs [role="tablist"] button:nth-child(1) {
                background-color: #FFC107; /* Yellow for Progress Summary */
            }
            /* Highlight the active tab */
            .stTabs [role="tablist"] button[aria-selected="true"] {
                background-color: #6C757D !important; /* Grey for selected tab */
            }
            /* Add border to focused tab for clarity */
            .stTabs [role="tablist"] button:focus {
                border: 2px solid #000; /* Black border for focused tab */
            }
            /* Ensure tables and plots scale appropriately */
            .stDataFrame, .plotly-graph-div {
                max-width: 100%;
                margin: 0 auto; /* Center align tables and plots */
            }
        </style>
    """, unsafe_allow_html=True)

    # Function to load and encode the local logo file

    # Path to your local logo
    logo_path = "logo/wfp_logo.jpg"

    # Load and encode the logo
    encoded_logo = load_logo(logo_path)

    # Add a logo and styled header with background color
    st.markdown(f"""
        <style>
            .header {{
                background-color: #f0f8ff; /* Light blue background */
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            }}
            .header h1 {{
                text-align: center;
                color: #0047ab;
                font-family: Arial, sans-serif;
                font-size: 32px;
                margin-bottom: 10px;
            }}
            .header img {{
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 120px; /* Adjust logo size */
            }}
        </style>
        <div class="header">
            <img src="data:image/jpg;base64,{encoded_logo}" alt="WFP Logo">
            <h1>Food Security Monitoring System (FSMS) Survey - WFP Sudan</h1>
        </div>
    """, unsafe_allow_html=True)

    # Add tabs for navigation
    tab1, tab2, tab3 = st.tabs(["Progress Summary", "Dashboard", "Data Issues"])

    # Tab 2: Dashboard
    with tab2:
        # Writeup about Outcome Indicators in a styled colored box
        st.markdown("""
                    <style>
                        .info-box {
                            background-color: #f0f8ff; /* Light blue background */
                            padding: 15px;
                            border-radius: 10px;
                            border-left: 5px solid #007BFF; /* Blue border for emphasis */
                            font-family: Arial, sans-serif;
                            font-size: 16px;
                            margin-bottom: 20px;
                            color: #004085; /* Darker text for readability */
                        }
                    </style>
                       <div class="info-box">
                           <h3>Overview of Outcome Indicators</h3>
                           <p>The outcome indicators on food security are essential tools for assessing the severity of food insecurity among populations. The assessed indicators include:</p>
                           <ul>
                               <li><b>Food Consumption Score (FCS):</b> A composite score that measures food frequency and dietary diversity. It is used to compare food consumption across geography and time, and to target households in need of food assistance.</li>
                               <li><b>Livelihood Coping Strategies – Food Security (LCS-FS):</b> An indicator used to understand households' medium and longer-term coping capacity in response to lack of food or money to buy food and their ability to overcome challenges in the future.</li>
                               <li><b>Reduced Coping Strategies Index (rCSI):</b> An indicator used to compare the hardship faced by households due to a shortage of food. It measures the frequency and severity of food consumption behaviors that households had to engage in due to food shortage in the 7 days prior to the survey.</li>
                           </ul>
                       </div>
                """, unsafe_allow_html=True)
        # Filter options
        states = df['QState'].unique()
        state_filter = st.sidebar.multiselect(
            "Filter by State",
            options=["All"] + list(states),
            default="All"
        )

        # Count occurrences of each label (normalized to percentages)
        lcs_counts = df['LCS_labels'].value_counts(normalize=True) * 100
        hhs_ipc_counts = df['HHS_IPC_labels'].value_counts(normalize=True) * 100
        hhs_std_counts = df['HHSCat_labels'].value_counts(normalize=True) * 100
        rcsi_ipc_counts = df['rCSI_IPC_Label'].value_counts(normalize=True) * 100
        rcsi_wfp_counts = df['rCSI_WFP_Label'].value_counts(normalize=True) * 100
        fcs_categories_counts = df['fcs_categories_labels'].value_counts(normalize=True) * 100

        # Define a global color mapping
        category_colors = {
            'Minimal': 'rgb(205, 250, 205)',  # Light Green
            'Phase 1': 'rgb(205, 250, 205)',  # Light Green
            'No or little hunger': 'rgb(300, 250, 205)',
            'Low (<6)': 'rgb(205, 250, 250)',
            'Stressed': 'rgb(250, 230, 030)',  # Light Yellow
            'Phase 2': 'rgb(250, 230, 030)',  # Light Yellow
            'Moderate hunger': 'rgb(000, 300, 010)',
            'Medium (6-11)': 'rgb(255, 230, 100)',
            'Crisis': 'rgb(230, 120, 000)',  # Orange
            'Crisis-Emergency': 'rgb(230, 120, 000)',  # Orange
            'Phase 3': 'rgb(230, 120, 000)',  # Orange
            'Emergency': 'rgb(200, 000, 000)',  # Red
            'Phase 4': 'rgb(200, 000, 000)',  # Red
            'Catastrophe': 'rgb(128, 000, 000)',  # Dark Red
            'Severe hunger': 'rgb(128, 000, 000)',
            'High (>11)': 'rgb(255, 255, 205)',
            'Phase 5': 'rgb(128, 000, 000)',  # Dark Red
            'Acceptable': 'rgb(205, 250, 205)',  # Light Green
            'Borderline': 'rgb(230, 120, 000)',  # Orange
            'Poor': 'rgb(200, 000, 000)'  # Bright Red
        }

        # Function to get colors for a given label set
        def get_colors(labels, color_mapping):
            return [color_mapping.get(label, 'rgb(200, 200, 200)') for label in labels]

        # Create a 2x3 subplot layout for all charts
        fig = sp.make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}],
                   [{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=(
                'FCS Categories', 'rCSI_IPC Categories',
                'rCSI_WFP Categories', 'HHS_IPC Categories',
                'HHS STD Categories', 'LCS Categories'
            )
        )

        # Add pie charts
        pie_data = [
            (fcs_categories_counts, 1, 1),
            (rcsi_ipc_counts, 1, 2),
            (rcsi_wfp_counts, 1, 3),
            (hhs_ipc_counts, 2, 1),
            (hhs_std_counts, 2, 2),
            (lcs_counts, 2, 3),
        ]

        for counts, row, col in pie_data:
            fig.add_trace(
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    textinfo='label+percent',
                    hoverinfo='label+percent',
                    marker=dict(colors=get_colors(counts.index, category_colors)),
                    showlegend=False,
                    textfont_size=14  # Increase font size for labels
                ),
                row=row, col=col
            )

        # Update layout for balanced charts and larger visuals
        fig.update_layout(
            title={
                'text': 'Distribution of Food Security Categories',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}  # Increase title font size
            },
            height=1000,  # Increase height for better spacing
            width=1200,  # Adjust width for better balance
            margin=dict(l=20, r=20, t=80, b=20),  # Adjust margins for optimal spacing
        )

        # Display the plot with full-width scaling
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Data Issues
    with tab3:
        st.markdown("<h2>Data Issues</h2>", unsafe_allow_html=True)

        # Bullet 1: Filter records that actualy have zero expenditure for food items
        expenditure_food_items_columns = ["Q4_1a", "Q4_1b", "Q4_1c", "Q4_2a", "Q4_2b", "Q4_2c",
                                          "Q4_3a", "Q4_3b", "Q4_3c", "Q4_4a", "Q4_4b", "Q4_4c",
                                          "Q4_5a", "Q4_5b", "Q4_5c", "Q4_6a", "Q4_6b", "Q4_6c",
                                          "Q4_7a", "Q4_7b", "Q4_7c", "Q4_8a", "Q4_8b", "Q4_8c",
                                          "Q4_9a", "Q4_9b", "Q4_9c", "Q4_10a", "Q4_10b", "Q4_10c"]

        df['expenditure_food_items'] = df[expenditure_food_items_columns].sum(axis=1)
        expenditure_food_items_too_low_zero = df[df['expenditure_food_items'] == 0]

        st.markdown("1. **Records indicating zero spending on food items:**")
        st.write(f"There are {len(expenditure_food_items_too_low_zero)} such records.")

        if not expenditure_food_items_too_low_zero.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_food_exp = expenditure_food_items_too_low_zero.to_csv(index=False).encode()
            b64_food_exp = base64.b64encode(csv_food_exp).decode()  # Encode as Base64 and decode to string
            href_food_exp = f'<a href="data:file/csv;base64,{b64_food_exp}" download="expenditure_food_items_too_low_zero.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Zero spending on food) as CSV</a>'
            st.markdown(href_food_exp, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # ***FLAG EXPENDITURE ON EDUCATION BUT NO CHILD***
        expenditure_education_columns = ["Q4_14a", "Q4_14b"]

        # Get sum of expenditure on education and store in a variable called 'expenditure_education'
        df['expenditure_education'] = df[expenditure_education_columns].sum(axis=1)

        children_24months_17_years_columns = ['Q2_4_2a', 'Q2_4_2b', 'Q2_4_3a', 'Q2_4_3b', 'Q2_4_4a', 'Q2_4_4b']

        ##Create a column that holds the total number of children aged
        df['children_24months_17_years_sum'] = df[children_24months_17_years_columns].sum(axis=1)
        expenditure_education_gt_0_no_child = df[
            (df['expenditure_education'] > 0) & (df['children_24months_17_years_sum'] == 0)]

        st.markdown(
            "2. **Records indicating expenditure on education greater than 0 but no children of age 24 months to 17 years:**")
        st.write(f"There are {len(expenditure_education_gt_0_no_child)} such records.")

        if not expenditure_education_gt_0_no_child.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_edu_exp_gt_0_no_child = expenditure_education_gt_0_no_child.to_csv(index=False).encode()
            b64_edu_exp_gt_0_no_child = base64.b64encode(
                csv_edu_exp_gt_0_no_child).decode()  # Encode as Base64 and decode to string
            href_edu_exp_gt_0_no_child = f'<a href="data:file/csv;base64,{b64_edu_exp_gt_0_no_child}" download="expenditure_education_gt_0_no_child.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (expenditure on education is greater than 0 but no child between 2 to 17 years) as CSV</a>'
            st.markdown(href_edu_exp_gt_0_no_child, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 1: Filter records based on the first condition
        current_livelihood = ['liv_activ_crops',
                              'liv_activ_livestock',
                              'liv_activ_donation_gift',
                              'liv_activ_business',
                              'liv_activ_agric_wage_labour',
                              'liv_activ_non_agric_wage_labour',
                              'liv_activ_sale _aid_Food',
                              'liv_activ_sale_firewood_charcoal',
                              'liv_activ_traditional_mining',
                              'liv_activ_salaried_work',
                              'liv_activ_begging',
                              'liv_activ_remittances',
                              'liv_activ_pension']

        df['current_live_Income_Total'] = df[current_livelihood].sum(axis=1)
        invalid_current_live_Income_Total = df[df['current_live_Income_Total'] != 100]

        st.markdown(
            "3. **Records indicating invalid total income total proportions from current livelihood activities:**")
        st.write(f"There are {len(invalid_current_live_Income_Total)} such records.")
        if not invalid_current_live_Income_Total.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_liv_1 = invalid_current_live_Income_Total.to_csv(index=False).encode()
            b64_liv_1 = base64.b64encode(csv_liv_1).decode()  # Encode as Base64 and decode to string
            href_liv_1 = f'<a href="data:file/csv;base64,{b64_liv_1}" download="invalid_current_live_Income_Total.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (invalid income totals) as CSV</a>'
            st.markdown(href_liv_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        food_con_7days_columns = ["Q5_1a", "Q5_2a", "Q5_3a", "Q5_4a", "Q5_4_1a", "Q5_4_2a", "Q5_4_3a", "Q5_4_4a",
                                  "Q5_5a", "Q5_5_1a", "Q5_5_2a", "Q5_6a", "Q5_6_1a", "Q5_7a", "Q5_8a", "Q5_9a"]

        # Create new columns with the desired names and copy the df
        df['FCSStap'] = df['Q5_1a']
        df['FCSPulse'] = df['Q5_2a']
        df['FCSDairy'] = df['Q5_3a']
        df['FCSPr'] = df['Q5_4a']
        df['FCSVeg'] = df['Q5_5a']
        df['FCSFruit'] = df['Q5_6a']
        df['FCSFat'] = df['Q5_7a']
        df['FCSSugar'] = df['Q5_8a']
        df['FCSCond'] = df['Q5_9a']

        df['food_con_7days_sum'] = df[food_con_7days_columns].sum(axis=1)

        food_con_7days_sum_zero = df[df['food_con_7days_sum'] == 0]

        st.markdown("4. **Records indicating no consumption of any food item in the last 7 days:**")
        st.write(f"There are {len(food_con_7days_sum_zero)} such records.")

        if not food_con_7days_sum_zero.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_con_7days_sum_zero = food_con_7days_sum_zero.to_csv(index=False).encode()
            b64_con_7days_sum_zero = base64.b64encode(
                csv_con_7days_sum_zero).decode()  # Encode as Base64 and decode to string
            href_con_7days_sum_zero = f'<a href="data:file/csv;base64,{b64_con_7days_sum_zero}" download="food_con_7days_sum_zero.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (HHs with no consumption of any food item in the last 7 days) as CSV</a>'
            st.markdown(href_con_7days_sum_zero, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 4: Filter records based on the first condition
        filtered_data_q5_1 = df[
            (df['Q5_1c'] == 1) &
            ((df['Q5_1a'] == 0) | (df['Q5_1a'].isnull()))
            ]
        st.markdown(
            "5. **Records indicating no consumption of cereals in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_q5_1)} such records.")
        if not filtered_data_q5_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_q5_1 = filtered_data_q5_1.to_csv(index=False).encode()
            b64_q5_1 = base64.b64encode(csv_q5_1).decode()  # Encode as Base64 and decode to string
            href_q5_1 = f'<a href="data:file/csv;base64,{b64_q5_1}" download="filtered_data_q5_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Cereals) as CSV</a>'
            st.markdown(href_q5_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where q5_1 cereals 24 hours is 0 and 7 days is blank or 7
        filtered_data_q5_1 = df[
            (df['Q5_1c'] == 0) & ((df['Q5_1a'] == 7))]

        st.markdown(
            "6. **Records indicating no consumption of cereals in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_q5_1)} such records.")
        if not filtered_data_q5_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_q5_1 = filtered_data_q5_1.to_csv(index=False).encode()
            b64_q5_1 = base64.b64encode(csv_q5_1).decode()  # Encode as Base64 and decode to string
            href_q5_1 = f'<a href="data:file/csv;base64,{b64_q5_1}" download="filtered_data_q5_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Cereals consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_q5_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 7: Filter records based on the first condition
        filtered_data_Q5_2 = df[
            (df['Q5_2c'] == 1) &
            ((df['Q5_2a'] == 0) | (df['Q5_2a'].isnull()))
            ]
        st.markdown(
            "7. **Records indicating no consumption of pulses in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_2)} such records.")
        if not filtered_data_Q5_2.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_2 = filtered_data_Q5_2.to_csv(index=False).encode()
            b64_Q5_2 = base64.b64encode(csv_Q5_2).decode()  # Encode as Base64 and decode to string
            href_Q5_2 = f'<a href="data:file/csv;base64,{b64_Q5_2}" download="filtered_data_Q5_2.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (pulses) as CSV</a>'
            st.markdown(href_Q5_2, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_2 pulses 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_2 = df[
            (df['Q5_2c'] == 0) & ((df['Q5_2a'] == 7))]

        st.markdown(
            "8. **Records indicating no consumption of pulses in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_2)} such records.")
        if not filtered_data_Q5_2.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_2 = filtered_data_Q5_2.to_csv(index=False).encode()
            b64_Q5_2 = base64.b64encode(csv_Q5_2).decode()  # Encode as Base64 and decode to string
            href_Q5_2 = f'<a href="data:file/csv;base64,{b64_Q5_2}" download="filtered_data_Q5_2.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (pulses consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_2, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 9: Filter records based on the first condition
        filtered_data_Q5_3 = df[
            (df['Q5_3c'] == 1) &
            ((df['Q5_3a'] == 0) | (df['Q5_3a'].isnull()))
            ]
        st.markdown(
            "9. **Records indicating no consumption of Milk in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_3)} such records.")
        if not filtered_data_Q5_3.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_3 = filtered_data_Q5_3.to_csv(index=False).encode()
            b64_Q5_3 = base64.b64encode(csv_Q5_3).decode()  # Encode as Base64 and decode to string
            href_Q5_3 = f'<a href="data:file/csv;base64,{b64_Q5_3}" download="filtered_data_Q5_3.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Milk) as CSV</a>'
            st.markdown(href_Q5_3, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_3 Milk 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_3 = df[
            (df['Q5_3c'] == 0) & ((df['Q5_3a'] == 7))]

        st.markdown(
            "10. **Records indicating no consumption of Milk in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_3)} such records.")
        if not filtered_data_Q5_3.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_3 = filtered_data_Q5_3.to_csv(index=False).encode()
            b64_Q5_3 = base64.b64encode(csv_Q5_3).decode()  # Encode as Base64 and decode to string
            href_Q5_3 = f'<a href="data:file/csv;base64,{b64_Q5_3}" download="filtered_data_Q5_3.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Milk consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_3, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

            # Bullet 11: Filter records based on the first condition
        filtered_data_Q5_4 = df[
            (df['Q5_4c'] == 1) &
            ((df['Q5_4a'] == 0) | (df['Q5_4a'].isnull()))
            ]
        st.markdown(
            "11. **Records indicating no consumption of Meat, fish and eggs in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_4)} such records.")
        if not filtered_data_Q5_4.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4 = filtered_data_Q5_4.to_csv(index=False).encode()
            b64_Q5_4 = base64.b64encode(csv_Q5_4).decode()  # Encode as Base64 and decode to string
            href_Q5_4 = f'<a href="data:file/csv;base64,{b64_Q5_4}" download="filtered_data_Q5_4.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Meat, fish and eggs) as CSV</a>'
            st.markdown(href_Q5_4, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_4 Meat, fish and eggs 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_4 = df[
            (df['Q5_4c'] == 0) & ((df['Q5_4a'] == 7))]

        st.markdown(
            "12. **Records indicating no consumption of Meat, fish and eggs in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_4)} such records.")
        if not filtered_data_Q5_4.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4 = filtered_data_Q5_4.to_csv(index=False).encode()
            b64_Q5_4 = base64.b64encode(csv_Q5_4).decode()  # Encode as Base64 and decode to string
            href_Q5_4 = f'<a href="data:file/csv;base64,{b64_Q5_4}" download="filtered_data_Q5_4.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Meat, fish and eggs consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_4, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 13: Filter records based on the first condition
        filtered_data_Q5_4_1 = df[
            (df['Q5_4_1c'] == 1) &
            ((df['Q5_4_1a'] == 0) | (df['Q5_4_1a'].isnull()))
            ]
        st.markdown(
            "13. **Records indicating no consumption of Flesh meat in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_4_1)} such records.")
        if not filtered_data_Q5_4_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_1 = filtered_data_Q5_4_1.to_csv(index=False).encode()
            b64_Q5_4_1 = base64.b64encode(csv_Q5_4_1).decode()  # Encode as Base64 and decode to string
            href_Q5_4_1 = f'<a href="data:file/csv;base64,{b64_Q5_4_1}" download="filtered_data_Q5_4_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Flesh meat) as CSV</a>'
            st.markdown(href_Q5_4_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_4_1 Flesh meat 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_4_1 = df[
            (df['Q5_4_1c'] == 0) & ((df['Q5_4_1a'] == 7))]

        st.markdown(
            "15. **Records indicating no consumption of Flesh meat in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_4_1)} such records.")
        if not filtered_data_Q5_4_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_1 = filtered_data_Q5_4_1.to_csv(index=False).encode()
            b64_Q5_4_1 = base64.b64encode(csv_Q5_4_1).decode()  # Encode as Base64 and decode to string
            href_Q5_4_1 = f'<a href="data:file/csv;base64,{b64_Q5_4_1}" download="filtered_data_Q5_4_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Flesh meat consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_4_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 16: Filter records based on the first condition
        filtered_data_Q5_4_2 = df[
            (df['Q5_4_2c'] == 1) &
            ((df['Q5_4_2a'] == 0) | (df['Q5_4_2a'].isnull()))
            ]
        st.markdown(
            "16. **Records indicating no consumption of Organ meat in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_4_2)} such records.")
        if not filtered_data_Q5_4_2.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_2 = filtered_data_Q5_4_2.to_csv(index=False).encode()
            b64_Q5_4_2 = base64.b64encode(csv_Q5_4_2).decode()  # Encode as Base64 and decode to string
            href_Q5_4_2 = f'<a href="data:file/csv;base64,{b64_Q5_4_2}" download="filtered_data_Q5_4_2.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Organ meat) as CSV</a>'
            st.markdown(href_Q5_4_2, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_4_2 Organ meat 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_4_2 = df[
            (df['Q5_4_2c'] == 0) & ((df['Q5_4_2a'] == 7))]

        st.markdown(
            "17. **Records indicating no consumption of Organ meat in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_4_2)} such records.")
        if not filtered_data_Q5_4_2.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_2 = filtered_data_Q5_4_2.to_csv(index=False).encode()
            b64_Q5_4_2 = base64.b64encode(csv_Q5_4_2).decode()  # Encode as Base64 and decode to string
            href_Q5_4_2 = f'<a href="data:file/csv;base64,{b64_Q5_4_2}" download="filtered_data_Q5_4_2.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Organ meat consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_4_2, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 18: Filter records based on the first condition
        filtered_data_Q5_4_3 = df[
            (df['Q5_4_3c'] == 1) &
            ((df['Q5_4_3a'] == 0) | (df['Q5_4_3a'].isnull()))
            ]
        st.markdown(
            "18. **Records indicating no consumption of Fish/shellfish in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_4_3)} such records.")
        if not filtered_data_Q5_4_3.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_3 = filtered_data_Q5_4_3.to_csv(index=False).encode()
            b64_Q5_4_3 = base64.b64encode(csv_Q5_4_3).decode()  # Encode as Base64 and decode to string
            href_Q5_4_3 = f'<a href="data:file/csv;base64,{b64_Q5_4_3}" download="filtered_data_Q5_4_3.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Fish/shellfish) as CSV</a>'
            st.markdown(href_Q5_4_3, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_4_3 Fish/shellfish 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_4_3 = df[
            (df['Q5_4_3c'] == 0) & ((df['Q5_4_3a'] == 7))]

        st.markdown(
            "19. **Records indicating no consumption of Fish/shellfish in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_4_3)} such records.")
        if not filtered_data_Q5_4_3.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_3 = filtered_data_Q5_4_3.to_csv(index=False).encode()
            b64_Q5_4_3 = base64.b64encode(csv_Q5_4_3).decode()  # Encode as Base64 and decode to string
            href_Q5_4_3 = f'<a href="data:file/csv;base64,{b64_Q5_4_3}" download="filtered_data_Q5_4_3.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Fish/shellfish consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_4_3, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##20-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Bullet 20: Filter records based on the first condition
        filtered_data_Q5_4_4 = df[
            (df['Q5_4_4c'] == 1) &
            ((df['Q5_4_4a'] == 0) | (df['Q5_4_4a'].isnull()))
            ]
        st.markdown(
            "20. **Records indicating no consumption of Eggs in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_4_4)} such records.")
        if not filtered_data_Q5_4_4.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_4 = filtered_data_Q5_4_4.to_csv(index=False).encode()
            b64_Q5_4_4 = base64.b64encode(csv_Q5_4_4).decode()  # Encode as Base64 and decode to string
            href_Q5_4_4 = f'<a href="data:file/csv;base64,{b64_Q5_4_4}" download="filtered_data_Q5_4_4.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Eggs) as CSV</a>'
            st.markdown(href_Q5_4_4, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_4_4 Eggs 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_4_4 = df[
            (df['Q5_4_4c'] == 0) & ((df['Q5_4_4a'] == 7))]

        st.markdown(
            "21. **Records indicating no consumption of Eggs in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_4_4)} such records.")
        if not filtered_data_Q5_4_4.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_4_4 = filtered_data_Q5_4_4.to_csv(index=False).encode()
            b64_Q5_4_4 = base64.b64encode(csv_Q5_4_4).decode()  # Encode as Base64 and decode to string
            href_Q5_4_4 = f'<a href="data:file/csv;base64,{b64_Q5_4_4}" download="filtered_data_Q5_4_4.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Eggs consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_4_4, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##22-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Bullet 22: Filter records based on the first condition
        filtered_data_Q5_5 = df[
            (df['Q5_5c'] == 1) &
            ((df['Q5_5a'] == 0) | (df['Q5_5a'].isnull()))
            ]
        st.markdown(
            "22. **Records indicating no consumption of Vegetables and leaves in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_5)} such records.")
        if not filtered_data_Q5_5.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_5 = filtered_data_Q5_5.to_csv(index=False).encode()
            b64_Q5_5 = base64.b64encode(csv_Q5_5).decode()  # Encode as Base64 and decode to string
            href_Q5_5 = f'<a href="data:file/csv;base64,{b64_Q5_5}" download="filtered_data_Q5_5.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Vegetables and leaves) as CSV</a>'
            st.markdown(href_Q5_5, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_5 Vegetables and leaves 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_5 = df[
            (df['Q5_5c'] == 0) & ((df['Q5_5a'] == 7))]

        st.markdown(
            "23. **Records indicating no consumption of Vegetables and leaves in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_5)} such records.")
        if not filtered_data_Q5_5.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_5 = filtered_data_Q5_5.to_csv(index=False).encode()
            b64_Q5_5 = base64.b64encode(csv_Q5_5).decode()  # Encode as Base64 and decode to string
            href_Q5_5 = f'<a href="data:file/csv;base64,{b64_Q5_5}" download="filtered_data_Q5_5.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Vegetables and leaves consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_5, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##24-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Bullet 24: Filter records based on the first condition
        filtered_data_Q5_5_1 = df[
            (df['Q5_5_1c'] == 1) &
            ((df['Q5_5_1a'] == 0) | (df['Q5_5_1a'].isnull()))
            ]
        st.markdown(
            "24. **Records indicating no consumption of Orange vegetables in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_5_1)} such records.")
        if not filtered_data_Q5_5_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_5_1 = filtered_data_Q5_5_1.to_csv(index=False).encode()
            b64_Q5_5_1 = base64.b64encode(csv_Q5_5_1).decode()  # Encode as Base64 and decode to string
            href_Q5_5_1 = f'<a href="data:file/csv;base64,{b64_Q5_5_1}" download="filtered_data_Q5_5_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Orange vegetables) as CSV</a>'
            st.markdown(href_Q5_5_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_5_1 Orange vegetables 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_5_1 = df[
            (df['Q5_5_1c'] == 0) & ((df['Q5_5_1a'] == 7))]

        st.markdown(
            "25. **Records indicating no consumption of Orange vegetables in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_5_1)} such records.")
        if not filtered_data_Q5_5_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_5_1 = filtered_data_Q5_5_1.to_csv(index=False).encode()
            b64_Q5_5_1 = base64.b64encode(csv_Q5_5_1).decode()  # Encode as Base64 and decode to string
            href_Q5_5_1 = f'<a href="data:file/csv;base64,{b64_Q5_5_1}" download="filtered_data_Q5_5_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Orange vegetables consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_5_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##26-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Bullet 26: Filter records based on the first condition
        filtered_data_Q5_5_2 = df[
            (df['Q5_5_2c'] == 1) &
            ((df['Q5_5_2a'] == 0) | (df['Q5_5_2a'].isnull()))
            ]
        st.markdown(
            "26. **Records indicating no consumption of Green leafy vegetables in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_5_2)} such records.")
        if not filtered_data_Q5_5_2.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_5_2 = filtered_data_Q5_5_2.to_csv(index=False).encode()
            b64_Q5_5_2 = base64.b64encode(csv_Q5_5_2).decode()  # Encode as Base64 and decode to string
            href_Q5_5_2 = f'<a href="data:file/csv;base64,{b64_Q5_5_2}" download="filtered_data_Q5_5_2.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Green leafy vegetables) as CSV</a>'
            st.markdown(href_Q5_5_2, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_5_2 Green leafy vegetables 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_5_2 = df[
            (df['Q5_5_2c'] == 0) & ((df['Q5_5_2a'] == 7))]

        st.markdown(
            "27. **Records indicating no consumption of Green leafy vegetables in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_5_2)} such records.")
        if not filtered_data_Q5_5_2.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_5_2 = filtered_data_Q5_5_2.to_csv(index=False).encode()
            b64_Q5_5_2 = base64.b64encode(csv_Q5_5_2).decode()  # Encode as Base64 and decode to string
            href_Q5_5_2 = f'<a href="data:file/csv;base64,{b64_Q5_5_2}" download="filtered_data_Q5_5_2.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Green leafy vegetables consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_5_2, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##28-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Bullet 28: Filter records based on the first condition
        filtered_data_Q5_6_1 = df[
            (df['Q5_6_1c'] == 1) &
            ((df['Q5_6_1a'] == 0) | (df['Q5_6_1a'].isnull()))
            ]
        st.markdown(
            "28. **Records indicating no consumption of Orange fruits in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_6_1)} such records.")
        if not filtered_data_Q5_6_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_6_1 = filtered_data_Q5_6_1.to_csv(index=False).encode()
            b64_Q5_6_1 = base64.b64encode(csv_Q5_6_1).decode()  # Encode as Base64 and decode to string
            href_Q5_6_1 = f'<a href="data:file/csv;base64,{b64_Q5_6_1}" download="filtered_data_Q5_6_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Orange fruits) as CSV</a>'
            st.markdown(href_Q5_6_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_6_1 Orange fruits 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_6_1 = df[
            (df['Q5_6_1c'] == 0) & ((df['Q5_6_1a'] == 7))]

        st.markdown(
            "29. **Records indicating no consumption of Orange fruits in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_6_1)} such records.")
        if not filtered_data_Q5_6_1.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_6_1 = filtered_data_Q5_6_1.to_csv(index=False).encode()
            b64_Q5_6_1 = base64.b64encode(csv_Q5_6_1).decode()  # Encode as Base64 and decode to string
            href_Q5_6_1 = f'<a href="data:file/csv;base64,{b64_Q5_6_1}" download="filtered_data_Q5_6_1.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (Orange fruits consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_6_1, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##30-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Bullet 30: Filter records based on the first condition
        filtered_data_Q5_6 = df[
            (df['Q5_6c'] == 1) &
            ((df['Q5_6a'] == 0) | (df['Q5_6a'].isnull()))
            ]
        st.markdown(
            "30. **Records indicating no consumption of fruits in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_6)} such records.")
        if not filtered_data_Q5_6.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_6 = filtered_data_Q5_6.to_csv(index=False).encode()
            b64_Q5_6 = base64.b64encode(csv_Q5_6).decode()  # Encode as Base64 and decode to string
            href_Q5_6 = f'<a href="data:file/csv;base64,{b64_Q5_6}" download="filtered_data_Q5_6.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (fruits) as CSV</a>'
            st.markdown(href_Q5_6, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_6 fruits 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_6 = df[
            (df['Q5_6c'] == 0) & ((df['Q5_6a'] == 7))]

        st.markdown(
            "31. **Records indicating no consumption of fruits in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_6)} such records.")
        if not filtered_data_Q5_6.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_6 = filtered_data_Q5_6.to_csv(index=False).encode()
            b64_Q5_6 = base64.b64encode(csv_Q5_6).decode()  # Encode as Base64 and decode to string
            href_Q5_6 = f'<a href="data:file/csv;base64,{b64_Q5_6}" download="filtered_data_Q5_6.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (fruits consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_6, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##32------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Bullet 32: Filter records based on the first condition
        filtered_data_Q5_7 = df[
            (df['Q5_7c'] == 1) &
            ((df['Q5_7a'] == 0) | (df['Q5_7a'].isnull()))
            ]
        st.markdown(
            "32. **Records indicating no consumption of oil-fats in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_7)} such records.")
        if not filtered_data_Q5_7.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_7 = filtered_data_Q5_7.to_csv(index=False).encode()
            b64_Q5_7 = base64.b64encode(csv_Q5_7).decode()  # Encode as Base64 and decode to string
            href_Q5_7 = f'<a href="data:file/csv;base64,{b64_Q5_7}" download="filtered_data_Q5_7.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (oil-fats) as CSV</a>'
            st.markdown(href_Q5_7, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 6: Filter records based on the first condition
        # Select records where Q5_7 oil-fats 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_7 = df[
            (df['Q5_7c'] == 0) & ((df['Q5_7a'] == 7))]

        st.markdown(
            "33. **Records indicating no consumption of oil-fats in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_7)} such records.")
        if not filtered_data_Q5_7.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_7 = filtered_data_Q5_7.to_csv(index=False).encode()
            b64_Q5_7 = base64.b64encode(csv_Q5_7).decode()  # Encode as Base64 and decode to string
            href_Q5_7 = f'<a href="data:file/csv;base64,{b64_Q5_7}" download="filtered_data_Q5_7.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (oil-fats consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_7, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 34: Filter records based on the first condition
        filtered_data_Q5_8 = df[
            (df['Q5_8c'] == 1) &
            ((df['Q5_8a'] == 0) | (df['Q5_8a'].isnull()))
            ]
        st.markdown(
            "34. **Records indicating no consumption of sugar in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_8)} such records.")
        if not filtered_data_Q5_8.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_8 = filtered_data_Q5_8.to_csv(index=False).encode()
            b64_Q5_8 = base64.b64encode(csv_Q5_8).decode()  # Encode as Base64 and decode to string
            href_Q5_8 = f'<a href="data:file/csv;base64,{b64_Q5_8}" download="filtered_data_Q5_8.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (sugar) as CSV</a>'
            st.markdown(href_Q5_8, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 35: Filter records based on the first condition
        # Select records where Q5_8 sugar 24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_8 = df[
            (df['Q5_8c'] == 0) & ((df['Q5_8a'] == 7))]

        st.markdown(
            "35. **Records indicating no consumption of sugar in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_8)} such records.")
        if not filtered_data_Q5_8.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_8 = filtered_data_Q5_8.to_csv(index=False).encode()
            b64_Q5_8 = base64.b64encode(csv_Q5_8).decode()  # Encode as Base64 and decode to string
            href_Q5_8 = f'<a href="data:file/csv;base64,{b64_Q5_8}" download="filtered_data_Q5_8.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (sugar consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_8, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ##34------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Bullet 34: Filter records based on the first condition
        filtered_data_Q5_9 = df[
            (df['Q5_9c'] == 1) &
            ((df['Q5_9a'] == 0) | (df['Q5_9a'].isnull()))
            ]
        st.markdown(
            "34. **Records indicating no consumption of condiments  in the last 7 days but consumed in the last 24hrs:**")
        st.write(f"There are {len(filtered_data_Q5_9)} such records.")
        if not filtered_data_Q5_9.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_9 = filtered_data_Q5_9.to_csv(index=False).encode()
            b64_Q5_9 = base64.b64encode(csv_Q5_9).decode()  # Encode as Base64 and decode to string
            href_Q5_9 = f'<a href="data:file/csv;base64,{b64_Q5_9}" download="filtered_data_Q5_9.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (condiments ) as CSV</a>'
            st.markdown(href_Q5_9, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        # Bullet 35: Filter records based on the first condition
        # Select records where Q5_9 condiments  24 hours is 0 and 7 days is blank or 7
        filtered_data_Q5_9 = df[
            (df['Q5_9c'] == 0) & ((df['Q5_9a'] == 7))]

        st.markdown(
            "35. **Records indicating no consumption of condiments  in the last 24 hours but consumed all day (7 days) in the last one week:**")
        st.write(f"There are {len(filtered_data_Q5_9)} such records.")
        if not filtered_data_Q5_9.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_Q5_9 = filtered_data_Q5_9.to_csv(index=False).encode()
            b64_Q5_9 = base64.b64encode(csv_Q5_9).decode()  # Encode as Base64 and decode to string
            href_Q5_9 = f'<a href="data:file/csv;base64,{b64_Q5_9}" download="filtered_data_Q5_9.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (condiments  consumed all days last 7 days but not consumed in the last 24hrs) as CSV</a>'
            st.markdown(href_Q5_9, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        ####FCS Computation-----------------------------------------------------------------------------------------------------------
        df['fcs'] = df["Q5_1a"] * 2 + df["Q5_2a"] * 3 + df["Q5_3a"] * 4 + df["Q5_4a"] * 4 + df["Q5_5a"] * 1 + df[
            "Q5_6a"] * 1 + df["Q5_7a"] * 0.5 + df["Q5_8a"] * 0.5 + df["Q5_9a"] * 0

        ##Except if in EXTREME cases, it will be very rare for many/any HHs to have such low FCS scores
        very_low_fcs = df[df['fcs'] < 10]
        st.markdown(
            "36. **Records indicating very low FCS (less than 10 which is considered very rare):**")
        st.write(f"There are {len(very_low_fcs)} such records.")
        if not very_low_fcs.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_low_fcs = very_low_fcs.to_csv(index=False).encode()
            b64_low_fcs = base64.b64encode(csv_low_fcs).decode()  # Encode as Base64 and decode to string
            href_low_fcs = f'<a href="data:file/csv;base64,{b64_low_fcs}" download="very_low_fcs.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (of very low FCS - less than 10) as CSV</a>'
            st.markdown(href_low_fcs, unsafe_allow_html=True)
            st.write(
                f"check the {len(very_low_fcs)} records across other columns such as expenditure, main livelihoods, HH size, etc. Do they make sense?")
        else:
            st.write("No records found for this condition.")

        # RUN CORRELATION TEST BETWEEN FCS & EXPENDITURE ON FOOD

        # We expect a positive correlation

        # H0:ρ=0

        "37. **Correlation between fcs & expenditure on food items:- We expect a positive correlation between fcs & expenditure on food**"

        # Ensure the columns exist
        if 'fcs' in df.columns and 'expenditure_food_items' in df.columns:

            pearson_corr, pearson_p = pearsonr(df['fcs'], df['expenditure_food_items'])
            spearman_corr, spearman_p = spearmanr(df['fcs'], df['expenditure_food_items'])

            st.write(f"Pearson Correlation: {pearson_corr}")
            st.write(f"Pearson p-value: {pearson_p}")
            st.write(f"Spearman Correlation: {spearman_corr}")
            st.write(f"Spearman p-value: {spearman_p}")
        else:
            st.write("The required columns are missing.")

        ##*****************************************************************CONVERTING EXPENDITURE TO usd*********************************************************************
        "38. **This is the summary of total expenditure on food items. The task is to find out whether or not the summary is realistic based on context, e.g. do minimum and maximum figures make sense?**"

        df['expenditure_food_items_offi_usd'] = df['expenditure_food_items'] / 1987
        df['expenditure_food_items_oth_market_usd'] = df['expenditure_food_items'] / 2350

        # Descriptive statistics side by side
        description_offi_usd = df['expenditure_food_items_offi_usd'].describe()
        description_oth_market_usd = df['expenditure_food_items_oth_market_usd'].describe()

        combined_descriptions = pd.DataFrame({
            'Official Rate (USD)': description_offi_usd,
            'Other Market Rate (USD)': description_oth_market_usd
        })

        # Display the table in Streamlit
        st.header("Descriptive Statistics")
        st.table(combined_descriptions)

        ###************************************COMPARE THE EXPENDITURE PATTERN ACROSS FCS CATEGORIES**********************************************
        grouped_description = df.groupby('fcs_categories_labels', observed=True)[
            'expenditure_food_items_oth_market_usd'].describe()

        "39. **We expect higher expenditure among those who have acceptable FCS compared to those having poor and borderline FCS. i.e. increase in expenditure from poor FCS to acceptable FCS, please check**"

        # Display the table in Streamlit
        st.header("Expenditure on food items across FCS categories")
        st.table(grouped_description)

        ##*********************************************FLAG RECORDS HAVING HIGHER THAN MEAN EXPENDITURE ON FOOD BUT STILL HAVE POOR FCS*************************************
        # Define threshold for high expenditure (e.g., 75th percentile)
        threshold = df.loc[df['fcs_categories_labels'] == 'Poor', 'expenditure_food_items_oth_market_usd'].quantile(
            0.75)

        # Flag records with 'Poor' FCS and expenditure above the threshold
        df['high_spending_poor'] = (
                (df['fcs_categories_labels'] == 'Poor') &
                (df['expenditure_food_items_oth_market_usd'] > threshold)
        )

        # Display flagged records
        flagged_records = df[df['high_spending_poor']]

        st.markdown(
            "40. **We do not expect households spending very high income on food to still have poor to bordeline FCS, We therefore need to flag such cases**")
        st.write(f"There are {len(flagged_records)} such records.")
        if not flagged_records.empty:
            # Convert DataFrame to CSV and encode to Base64
            csv_high_exp_poor_fcs = flagged_records.to_csv(index=False).encode()
            b64_high_exp_poor_fcs = base64.b64encode(
                csv_high_exp_poor_fcs).decode()  # Encode as Base64 and decode to string
            href_high_exp_poor_fcs = f'<a href="data:file/csv;base64,{b64_high_exp_poor_fcs}" download="flagged_records.csv" style="color: blue; text-decoration: underline;">Download Filtered Data (of poor-borderline FCS - but high spending) as CSV</a>'
            st.markdown(href_high_exp_poor_fcs, unsafe_allow_html=True)
        else:
            st.write("No records found for this condition.")

        "41. **We expect correlation coefficient between FCS & rCSI to be negative. We therefore run correlation test to confirm this**"
        # RUN CORRELATION TEST BETWEEN FCS & rCSI

        # We expect a positive correlation

        # H0:ρ=0

        # Ensure the columns exist

        pearson_corr, pearson_p = pearsonr(df['fcs'], df['rCSI'])
        spearman_corr, spearman_p = spearmanr(df['fcs'], df['rCSI'])

        st.write(f"Pearson Correlation: {pearson_corr}")
        st.write(f"Pearson p-value: {pearson_p}")
        st.write(f"Spearman Correlation: {spearman_corr}")
        st.write(f"Spearman p-value: {spearman_p}")

    # Define the total target
    TARGET = 12000  # Set your target number of samples here

    # Add Progress Summary section
    with tab1:
        st.markdown("<h2>Progress Summary</h2>", unsafe_allow_html=True)

        # Key Metrics Calculation
        total_samples = len(df)  # Replace 'df' with your actual dataframe variable
        avg_household_size = round(df['hh_size'].mean(), 2)  # Ensure 'hh_size' exists in your dataframe
        progress = round((total_samples / TARGET) * 100, 2)

        # Display Key Metrics with Enhanced Styling
        st.markdown(f"""
            <div style="padding: 20px; background-color: #f7f7f7; border-radius: 10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
                <h2 style="color: #2b6cb0; text-align: center; font-family: Arial, sans-serif;">Key Metrics</h2>
                <div style="margin-top: 15px;">
                    <h3 style="color: #1a202c; font-family: Arial, sans-serif;">Total Samples Collected:</h3>
                    <p style="font-size: 48px; color: #4caf50; font-weight: bold; text-align: center;">{total_samples}</p>
                </div>
                <div style="margin-top: 15px;">
                    <h3 style="color: #1a202c; font-family: Arial, sans-serif;">Progress:</h3>
                    <p style="font-size: 48px; color: #ff5722; font-weight: bold; text-align: center;">{progress}% of {TARGET}</p>
                </div>
                <div style="margin-top: 15px;">
                    <h3 style="color: #1a202c; font-family: Arial, sans-serif;">Average Household Size:</h3>
                    <p style="font-size: 48px; color: #2196f3; font-weight: bold; text-align: center;">{avg_household_size}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        # Pie Chart: Gender Distribution
        with col1:
            gender_summary = df['Q2_2a'].value_counts().reset_index()
            gender_summary.columns = ['Gender', 'Count']
            gender_pie_chart = px.pie(
                gender_summary, names='Gender', values='Count',
                title='Distribution by Gender',
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.3
            )
            st.plotly_chart(gender_pie_chart, use_container_width=True)

        # Bar Chart: Residence Status
        with col2:
            residence_summary = df['Q2_1'].value_counts().reset_index()
            residence_summary.columns = ['Residence Status', 'Count']
            residence_bar_chart = px.bar(
                residence_summary, x='Residence Status', y='Count',
                text='Count', title='Distribution by Residence Status',
                color='Residence Status', color_discrete_sequence=px.colors.qualitative.Set2
            )
            residence_bar_chart.update_layout(
                xaxis_title="Residence Status",
                yaxis_title="Number of Samples",
                showlegend=False
            )
            st.plotly_chart(residence_bar_chart, use_container_width=True)

        # Create Progress Table

        st.header("Records Collected per State per Enumerator")

        try:
            # --- Filter options for QState and Enumerator ---
            qstate_options = sorted(df["QState"].unique())
            enumerator_options = sorted(df["Enumerator"].unique())

            # Sidebar filters for QState and Enumerator
            selected_qstates = st.sidebar.multiselect("Filter by State (QState):", options=qstate_options,
                                                      default=qstate_options)
            selected_enumerators = st.sidebar.multiselect("Filter by Enumerator:", options=enumerator_options,
                                                          default=enumerator_options)

            # Apply the filters to the dataframe
            filtered_df = df[df["QState"].isin(selected_qstates) & df["Enumerator"].isin(selected_enumerators)]

            # --- 1) Create a pivot table ---
            data_collection_summary = filtered_df.pivot_table(
                index=["QState", "Enumerator"],  # Rows
                columns="Day",  # Columns become unique Day values
                aggfunc="size",  # Count the number of records
                fill_value=0  # Fill missing Day/Enumerator combos with 0
            )

            # --- 2) Add a 'Total' column ---
            data_collection_summary["Total"] = data_collection_summary.sum(axis=1)

            # --- 3) Convert multi-index pivot table to a normal DataFrame ---
            data_collection_summary = data_collection_summary.reset_index()

            # --- 4) Rename Day columns to 'Day 1', 'Day 2', etc. ---
            new_columns = []
            for col in data_collection_summary.columns:
                if isinstance(col, (int, float)):  # If the column is a number (Day values)
                    new_columns.append(f"Day {int(col)}")
                else:  # For non-Day columns
                    new_columns.append(col)
            data_collection_summary.columns = new_columns

            # --- 5) Optional: Sort day columns for clarity ---
            day_cols = [c for c in data_collection_summary.columns if c.startswith('Day ')]
            day_cols_sorted = sorted(day_cols, key=lambda x: int(x.split(' ')[1]))  # Sort numerically by day number
            final_cols = ["QState", "Enumerator"] + day_cols_sorted + ["Total"]
            data_collection_summary = data_collection_summary[final_cols]

            # --- Display the table in Streamlit ---
            st.table(data_collection_summary)

        except KeyError as e:
            st.error(f"Missing expected column in the dataset: {e}")
        except Exception as e:
            st.error(f"An error occurred while creating the progress table: {e}")


@st.cache_data
def preprocess_fsms_data(df, residence_mapping):
    df = df.rename(columns={"QState": "QState_orig",
                            "Q2_4": "hh_size",
                            "Q6_2_1": "Lcs_stress_DomAsset",
                            "Q6_2_2": "Lcs_crisis_Health",
                            "Q6_2_3": "Lcs_crisis_con_stock",
                            "Q6_2_4": "Lcs_stress_Saving",
                            "Q6_2_5": "Lcs_stress_accum_debt",
                            "Q6_2_6": "Lcs_em_ResAsset",
                            "Q6_2_7": "Lcs_stress_red_farm_liv_input",
                            "Q6_2_8": "Lcs_em_last_female",
                            "Q6_2_9": "Lcs_em_Begged",
                            "Q6_2_10": "Lcs_crisis_wild_food",
                            "Q3_1_1": "liv_activ_crops",
                            "Q3_1_2": "liv_activ_livestock",
                            "Q3_1_3": "liv_activ_donation_gift",
                            "Q3_1_4": "liv_activ_business",
                            "Q3_1_5": "liv_activ_agric_wage_labour",
                            "Q3_1_6": "liv_activ_non_agric_wage_labour",
                            "Q3_1_7": "liv_activ_sale _aid_Food",
                            "Q3_1_8": "liv_activ_sale_firewood_charcoal",
                            "Q3_1_9": "liv_activ_traditional_mining",
                            "Q3_1_10": "liv_activ_salaried_work",
                            "Q3_1_11": "liv_activ_begging",
                            "Q3_1_12": "liv_activ_remittances",
                            "Q3_1_13": "liv_activ_pension",
                            "Q7_2_1": "Q7_2_1_HH_own_cattle",
                            "Q7_2_2": "Q7_2_2_HH_own_donkey",
                            "Q7_2_3": "Q7_2_3_HH_own_camel",
                            "Q7_2_4": "Q7_2_4_HH_own_goats_sheep",
                            "Q7_2_5": "Q7_2_5_HH_own_poultry",
                            "Q6_6": "Q6_6_HHSNoFood",
                            "Q6_7": "Q6_7_HHSNoFood_FR",
                            "Q6_8": "Q6_8_HHSBedHung",
                            "Q6_9": "Q6_9_HHSBedHung_FR",
                            "Q6_10": "Q6_10_HHSNotEat",
                            "Q6_11": "Q6_11_HHSNotEat_FR"

                            })

    state_mapping = {
        1: "North Darfur",
        2: "South Darfur",
        3: "West Darfur",
        4: "Central Darfur",
        5: "East Darfur",
        6: "Kassala",
        7: "Red Sea",
        8: "Blue Nile",
        9: "White nile",
        10: "North Kordofan",
        11: "West Kordofan",
        12: "South Kordofan",
        13: "Gadarif",
        14: "Khartoum",
        15: "Sinnar",
        16: "Northern State",
        17: "AL Gazira",
        18: "River Nile"
    }

    # CREATING A COLUMN OF State with labels -
    df['QState'] = df['QState_orig'].map(state_mapping)

    # Map numeric values to descriptive labels
    gender_mapping = {1: 'Male', 2: 'Female'}
    df['Q2_2a'] = df['Q2_2a'].map(gender_mapping)

    # Map numeric values to descriptive labels
    df['Q2_1'] = df['Q2_1'].map(residence_mapping)

    ###**********************CREATE ENUMERATOR AND DAY COLUMNS****************************************************************
    import pandas as pd
    import numpy as np

    # Let's assume your data has a 'State' column.
    # If it's named differently, rename the code references accordingly.

    # Define the enumerators
    enumerator_names = ["A", "B", "C", "D", "F", "G", "H"]

    def assign_enumerators_and_days(group):
        """
        This function will be applied to each state group:
        1. Assign enumerators in a repeating sequence (A, B, C, D).
        2. Assign Day values in a manner that ensures each enumerator
        has at least 5 different days of data collection.
        """
        n = len(group)

        # --- 1) Assign enumerators in a round-robin manner ---
        enumerators = np.tile(enumerator_names, (n // 4) + 1)[:n]
        group["Enumerator"] = enumerators

        # --- 2) Assign Day values so each enumerator has at least 5 days ---
        # For simplicity, we'll assign days in a repeating pattern [1..5]
        # for each enumerator, enough times to cover all rows for that enumerator.

        # Create an empty list to store subgroups with Days assigned
        output_subgroups = []

        for enum in enumerator_names:
            # Filter the group for current enumerator
            enum_sub = group[group["Enumerator"] == enum].copy()
            # Number of rows for this enumerator
            sub_count = len(enum_sub)

            # Create day values from 1..5 repeated enough times
            days = np.tile(np.arange(1, 6), (sub_count // 5) + 1)[:sub_count]
            enum_sub["Day"] = days

            output_subgroups.append(enum_sub)

        # Concatenate the enumerator subgroups back together
        group_with_days = pd.concat(output_subgroups).sort_index()

        return group_with_days

    # 2. Group by State and apply the enumerator/day assignment
    df = df.groupby("QState", group_keys=False).apply(assign_enumerators_and_days)

    food_con_7days_columns = ["Q5_1a", "Q5_2a", "Q5_3a", "Q5_4a", "Q5_4_1a", "Q5_4_2a", "Q5_4_3a", "Q5_4_4a",
                              "Q5_5a", "Q5_5_1a", "Q5_5_2a", "Q5_6a", "Q5_6_1a", "Q5_7a", "Q5_8a", "Q5_9a"]

    df['food_con_7days_sum'] = df[food_con_7days_columns].sum(axis=1)

    df['fcs'] = df["Q5_1a"] * 2 + df["Q5_2a"] * 3 + df["Q5_3a"] * 4 + df["Q5_4a"] * 4 + df["Q5_5a"] * 1 + df[
        "Q5_6a"] * 1 + df["Q5_7a"] * 0.5 + df["Q5_8a"] * 0.5 + df["Q5_9a"] * 0

    # Assuming `df` is your DataFrame and `fcs` is a column in it
    # Define the new column 'fcs_categories' based on conditions
    conditions = [
        (df['fcs'] <= 28),  # Low range
        (df['fcs'] > 28) & (df['fcs'] <= 42),  # Mid range
        (df['fcs'] > 42)  # High range
    ]

    # Assign categories based on conditions
    categories = [3, 2, 1]  # Poor, Borderline, Acceptable
    df['fcs_categories'] = pd.cut(df['fcs'], bins=[-float('inf'), 28, 42, float('inf')], labels=[3, 2, 1])

    # Optional: Add human-readable labels
    value_labels = {1: 'Acceptable', 2: 'Borderline', 3: 'Poor'}
    df['fcs_categories_labels'] = df['fcs_categories'].map(value_labels)

    df['rCSILessQlty'] = df['Q6_1_1']
    df['rCSIBorrow'] = df['Q6_1_2']
    df['rCSIMealNb'] = df['Q6_1_5']
    df['rCSIMealSize'] = df['Q6_1_3']
    df['rCSIMealAdult'] = df['Q6_1_4']

    # Assign labels to variables (in comments for documentation)
    variable_labels = {
        'rCSILessQlty': 'Rely on less preferred and less expensive food in the past 7 days',
        'rCSIBorrow': 'Borrow food or rely on help from a relative or friend in the past 7 days',
        'rCSIMealNb': 'Reduce number of meals eaten in a day in the past 7 days',
        'rCSIMealSize': 'Limit portion size of meals at meal times in the past 7 days',
        'rCSIMealAdult': 'Restrict consumption by adults in order for small children to eat in the past 7 days',
        'rCSI': 'Reduced coping strategies index (rCSI)'
    }

    # Compute rCSI
    df['rCSI'] = (df['rCSILessQlty'] * 1 +
                  df['rCSIBorrow'] * 2 +
                  df['rCSIMealNb'] * 1 +
                  df['rCSIMealSize'] * 1 +
                  df['rCSIMealAdult'] * 3)

    # Recoding G_rCSI into categories based on thresholds
    def recode_rCSI(value):
        if value <= 3:
            return 1
        elif 4 <= value <= 18:
            return 2
        else:
            return 3

    df['rCSI_IPC'] = df['rCSI'].apply(recode_rCSI)

    # Assigning value labels for G_rCSI_IPC
    value_labels = {
        1: 'Minimal',
        2: 'Stressed',
        3: 'Crisis-Emergency'
    }

    df['rCSI_IPC_Label'] = df['rCSI_IPC'].map(value_labels)

    # Recoding G_rCSI into categories based on thresholds for WFP Sudan
    def recode_rCSI(value):
        if value <= 5:
            return 1
        elif 6 <= value <= 11:
            return 2
        else:
            return 3

    df['rCSI_WFP'] = df['rCSI'].apply(recode_rCSI)

    # Assigning value labels for G_rCSI_IPC
    value_labels = {
        1: 'Low (<6)',
        2: 'Medium (6-11)',
        3: 'High (>11)'
    }

    df['rCSI_WFP_Label'] = df['rCSI_WFP'].map(value_labels)

    # Cleaning of HHS variables
    # HHSNoFood and HHSNoFood_FR
    df['Q6_6_HHSNoFood'] = df['Q6_7_HHSNoFood_FR'].apply(lambda x: 1 if x > 0 else 0)

    # HHSBedHung and HHSBedHung_FR
    df['Q6_9_HHSBedHung_FR'] = df['Q6_8_HHSBedHung'].apply(lambda x: 1 if x > 0 else 0)

    # HHSNotEat and HHSNotEat_FR
    df['Q6_10_HHSNotEat'] = df['Q6_11_HHSNotEat_FR'].apply(lambda x: 1 if x > 0 else 0)

    # Define a function to recode the frequency categories
    def recode_frequency(value):
        if value == 1 or value == 2:  # "rarely" or "sometimes"
            return 1
        elif value == 3:  # "often"
            return 2
        else:  # Any other value
            return 0

    # Apply the recoding to the relevant variables
    df['HHSQ1'] = df['Q6_7_HHSNoFood_FR'].apply(recode_frequency)
    df['HHSQ2'] = df['Q6_8_HHSBedHung'].apply(recode_frequency)
    df['HHSQ3'] = df['Q6_11_HHSNotEat_FR'].apply(recode_frequency)

    # Adding variable labels (can be added as comments or metadata in Python)
    # 'HHSQ1': 'Was there ever no food to eat in HH?'
    # 'HHSQ2': 'Did any HH member go sleep hungry?'
    # 'HHSQ3': 'Did any HH member go whole day without food?'

    # Compute the HHS score by summing HHSQ1, HHSQ2, and HHSQ3
    df['HHS'] = df['HHSQ1'] + df['HHSQ2'] + df['HHSQ3']

    # Recoding the HHS variable into categorical scores
    def recode_hhs_cat(value):
        if 0 <= value <= 1:
            return 1  # No or little hunger in the household
        elif 2 <= value <= 3:
            return 2  # Moderate hunger in the household
        elif value >= 4:
            return 3  # Severe hunger in the household
        else:
            return None  # In case of unexpected values

    df['HHSCat'] = df['HHS'].apply(recode_hhs_cat)

    # Optional: Add human-readable labels
    value_labels = {1: 'No or little hunger', 2: 'Moderate hunger', 3: 'Severe hunger'}
    df['HHSCat_labels'] = df['HHSCat'].map(value_labels)

    ##Calculate with IPC threshold

    # Recoding HHS into HHS_IPC
    def recode_hhs_ipc(value):
        if value == 0:
            return 1  # Minimal
        elif value == 1:
            return 2  # Stressed
        elif 2 <= value <= 3:
            return 3  # Crisis
        elif value == 4:
            return 4  # Emergency
        elif 5 <= value <= 6:
            return 5  # Catastrophe
        else:
            return None  # For unexpected values

    df['HHS_IPC'] = df['HHS'].apply(recode_hhs_ipc)

    # Optional: Add human-readable labels
    value_labels = {1: 'Minimal', 2: 'Stressed', 3: 'Crisis', 4: 'Emergency', 5: 'Catastrophe'}
    df['HHS_IPC_labels'] = df['HHS_IPC'].map(value_labels)

    # Apply the logic to compute the `emergency_coping_FS` variable
    df['emergency_coping_FS'] = df.apply(
        lambda row: 4 if (
                row['Lcs_em_ResAsset'] in [2, 4] or
                row['Lcs_em_Begged'] in [2, 4] or
                row['Lcs_em_last_female'] in [2, 4]
        ) else 1, axis=1
    )

    # Apply the logic to compute the `crisis_coping_FS` variable
    df['crisis_coping_FS'] = df.apply(
        lambda row: 3 if (
                row['Lcs_crisis_Health'] in [2, 4] or
                row['Lcs_crisis_con_stock'] in [2, 4] or
                row['Lcs_crisis_wild_food'] in [2, 4]
        ) else 1, axis=1
    )

    # Apply the logic to compute the `stress_coping_FS` variable
    df['stress_coping_FS'] = df.apply(
        lambda row: 2 if (
                row['Lcs_stress_Saving'] in [2, 4] or
                row['Lcs_stress_accum_debt'] in [2, 4] or
                row['Lcs_stress_red_farm_liv_input'] in [2, 4] or
                row['Lcs_stress_DomAsset'] in [2, 4]
        ) else 1, axis=1
    )

    ##Create column "LCS" that returns maximum of the other three created columns
    df['LCS'] = df[['emergency_coping_FS', 'crisis_coping_FS', 'stress_coping_FS']].max(axis=1)

    value_labels = {1: 'Minimal', 2: 'Stressed', 3: 'Crisis', 4: 'Emergency'}

    # Optional: Add human-readable labels
    df['LCS_labels'] = df['LCS'].map(value_labels)
    return df


def run_fsms():
    # Set working directory and load the dataset
    df = pd.read_csv('data/FSMS_Dec_2024.csv', low_memory=False, encoding="ISO-8859-1")
    residence_mapping = {
        2: 'IDP in Camp',
        3: 'IDP outside camps',
        4: 'Refugees in Camp',
        5: 'Refugees outside Camps',
        6: 'Returnees IDPs',
        7: 'Returnees Refugees',
        8: 'IDPs in Gathering points'
    }
    # df = preprocess_fsms_data(df, residence_mapping)
    df = preprocess_fsms_data(df, residence_mapping)
    display_fsms_data(df)
