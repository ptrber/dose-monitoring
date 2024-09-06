import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#import sys
#import datetime

# import webbrowser
#import subprocess

############################
# Deep dose equivalent (DDE) is the dose equivalent at a tissue depth of 1 cm (i.e., the minimum
# depth of internal organs) and is used to compute your whole-body dose.
# The whole-body dose should be compared to the deep dose equivalent or effective dose equivalent limit (5,000 mrem/year).
# • Lens dose equivalent (LDE) is the dose equivalent at a tissue depth of 0.3 cm (i.e., to the lens of the
# eye) and should be compared to the lens dose limit (15,000 mrem/year).
# • Shallow dose equivalent (SDE) is the dose equivalent at a tissue depth of 0.007 cm (i.e., to the
# dermis) and should be compared to the skin and extremity dose limits (50,000 mrem/year).
# YTD DDE stands for "Year-to-Date Diagnostic Dose Equivalent." It is a term commonly used in dose monitoring systems in the medical field. YTD DDE refers to the cumulative diagnostic dose equivalent received by an individual or a group of individuals over the course of the current calendar year up to the present date.
#############################
# load dose data
# file_path = '/home/peter/Documents/research/dose-monitoring/python_/data/DoseReport_20180101_20230301.xlsx'
# df = pd.read_excel(file_path, sheet_name='DoseReport')

#####################
""" 
# Load the data from the ODS file
file_path = './dose-monitoring/python_/data/DoseReport_20180101_20230301.ods'
#df = pd.read_excel(file_path, engine='odf')
df = pd.read_excel(file_path, sheet_name='DoseReport')

# Convert dose columns to numeric (assuming they are not already)
dose_columns = ['Current DDE', 'Current LDE', 'Current SDE']
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Period Begin Date' and 'Period End Date' to datetime objects
df['Period Begin Date'] = pd.to_datetime(df['Period Begin Date'], errors='coerce')
df['Period End Date'] = pd.to_datetime(df['Period End Date'], errors='coerce')

# Filter data for the specified period (2021-07 to 2022-06)
filtered_data = df[(df['Period Begin Date'] >= '2021-07-01') & (df['Period End Date'] <= '2022-06-30')]

# Filter data for each type of monitor
chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']
lens_monitors = filtered_data[filtered_data['Use'] == 'LENS']
finger_monitors = filtered_data[filtered_data['Use'].isin(['RFINGER', 'LFINGER'])]

# Group by 'Participant Name' and sum the doses for each monitor type
chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum().reset_index()
lens_doses = lens_monitors.groupby('Participant Name')['Current LDE'].sum().reset_index()
finger_doses = finger_monitors.groupby('Participant Name')['Current SDE'].sum().reset_index()

# Merge the results on 'Participant Name'
total_doses = chest_doses.merge(lens_doses, on='Participant Name', how='outer')
total_doses = total_doses.merge(finger_doses, on='Participant Name', how='outer')

# Fill NaN values with 0 (since we are summing doses)
total_doses = total_doses.fillna(0)

# Calculate the total sum of current doses
total_doses['Total Sum Current'] = total_doses[['Current DDE', 'Current LDE', 'Current SDE']].sum(axis=1)

# Filter out participants with non-zero total doses and above 0.2 mSv
non_zero_total_sum = total_doses[total_doses['Total Sum Current'] > 0]

# Sort the data by 'Total Sum Current' in descending order
sorted_total_doses = non_zero_total_sum.sort_values(by='Total Sum Current', ascending=False)
#sorted_total_doses = total_doses.sort_values(by='Total Sum Current', ascending=False)

# Create a bar chart with Plotly
fig = px.bar(sorted_total_doses, x='Participant Name',
             y=['Current DDE', 'Current LDE', 'Current SDE'],
             labels={'value': 'Dose (mSv)'},
             color_discrete_map={'Current DDE': 'red', 'Current LDE': 'blue', 'Current SDE': 'green'},
             title='Current Doses for Participants with Specific Monitors')

# Set x-axis labels explicitly
fig.update_xaxes(title_text='Participant Name', tickangle=45, tickfont=dict(size=10))

# Customize layout for better readability
fig.update_layout(
    font=dict(size=14),
    xaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=10)  # Keep x-axis tickfont size small for better readability
    ),
    yaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=16)
    ),
    legend=dict(
        title_font=dict(size=16),
        font=dict(size=14)
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=40, b=80)
)

# Show the interactive plot
fig.show()

#Define the base dir for saving plots ########
base_dir = '/home/peter/Documents/research/dose-monitoring/python_/results/'

# Save the plot as a high-resolution PNG
fig_file_path = base_dir + 'sum_all_dose_equiv_histogram.png'
fig.write_image(fig_file_path, width=1200, height=800, scale=3)  # Increase scale for higher resolution
####################################
### CURRENT DDE ### (2021-07 to 2022-06)
####################################
# Filter out participants with non-zero total doses and above N mSv
current_dde_sum = total_doses[total_doses['Current DDE'] > 0]

# Sort the data by 'Current DDE' in descending order
sorted_dde_sum = current_dde_sum.sort_values(by='Current DDE', ascending=False)

# Create a bar chart with Plotly
fig = px.bar(sorted_dde_sum, x='Participant Name',
             y='Current DDE',
             labels={'value': 'Dose (mSv)'},
             color_discrete_sequence=['red'],
             title='Current DDE for Participants with CHEST Monitors')

# Set x-axis labels explicitly
fig.update_xaxes(title_text='Participant Name', tickangle=45, tickfont=dict(size=10))

# Customize layout for better readability
fig.update_layout(
    font=dict(size=14),
    xaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=10)  # Keep x-axis tickfont size small for better readability
    ),
    yaxis=dict(
        title_font=dict(size=18),
        tickfont=dict(size=16)
    ),
    legend=dict(
        title_font=dict(size=16),
        font=dict(size=14)
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=40, b=80)
)

# Show the interactive plot
fig.show()

# Save the plot as a high-resolution PNG
fig_file_path_2 = base_dir + 'current_dde_hist.png'
fig.write_image(fig_file_path_2, width=1200, height=800, scale=3)  # Increase scale for higher resolution

"""
################
# Load the data from the ODS file
# file_path = '/home/peter/Documents/research/dose-monitoring/python_/data/DoseReport_20180101_20230301.csv'
#file_path = "/home/peter/Documents/research/dose-monitoring/python_/data/DoseReport_20180101_20240607.csv"
file_path = "./data/DoseReport_20180101_20240607.csv"
df = pd.read_csv(file_path)

# Convert dose columns to numeric
dose_columns = ["Current DDE", "Current LDE", "Current SDE"]
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors="coerce")

# Convert 'Period Begin Date' and 'Period End Date' to datetime objects
df["Period Begin Date"] = pd.to_datetime(df["Period Begin Date"], errors="coerce")
df["Period End Date"] = pd.to_datetime(df["Period End Date"], errors="coerce")

# Define the base directory for saving plots
base_dir = "./results/"
'''
# Define the years to analyze
years_to_analyze = [2018, 2019, 2020, 2021, 2022, 2023]

for year in years_to_analyze:
    # Filter data for the specified period (July of the current year to July of the next year)
    start_date = f"{year}-07-01"
    end_date = f"{year + 1}-06-30"
    filtered_data = df[
        (df["Period Begin Date"] >= start_date) & (df["Period End Date"] <= end_date)
    ]

    # Filter data for CHEST monitors only
    chest_monitors = filtered_data[filtered_data["Use"] == "CHEST"]

    # Group by 'Participant Name' and sum the DDE
    chest_doses = (
        chest_monitors.groupby("Participant Name")["Current DDE"].sum().reset_index()
    )

    # Filter out participants with non-zero DDE
    non_zero_dde = chest_doses[chest_doses["Current DDE"] > 0]

    # Sort the data by 'Current DDE' in descending order
    sorted_dde_sum = non_zero_dde.sort_values(by="Current DDE", ascending=False)

    # Create a bar chart with Plotly
    fig = px.bar(
        sorted_dde_sum,
        x="Participant Name",
        y="Current DDE",
        labels={"value": "Dose (mSv)"},
        color_discrete_sequence=["red"],
        title=f"Current DDE for Participants with CHEST Monitors ({year}-{year + 1})",
    )

    # Set x-axis labels explicitly
    fig.update_xaxes(
        title_text="Participant Name", tickangle=45, tickfont=dict(size=10)
    )

    # Customize layout for better readability
    fig.update_layout(
        font=dict(size=14),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(
                size=10
            ),  # Keep x-axis tickfont size small for better readability
        ),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=16)),
        legend=dict(title_font=dict(size=16), font=dict(size=14)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=80),
    )

#    # Show the interactive plot
    fig.show()



#
#    # Save the plot as a high-resolution PNG
#    fig_file_path = f'{base_dir}current_dde_hist_{year}_{year + 1}.png'
#    fig.write_image(fig_file_path, width=1200, height=800, scale=3)  # Increase scale for higher resolution
#
'''
### combined stacked histogram #######
import pandas as pd
import plotly.express as px
import sys

# Load the data from the CSV file
file_path = './data/DoseReport_20180101_20240607.csv'
df = pd.read_csv(file_path)

# Convert dose columns to numeric
dose_columns = ['Current DDE', 'Current LDE', 'Current SDE']
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Period Begin Date', 'Period End Date', and 'Scan Date' to datetime objects
df['Period Begin Date'] = pd.to_datetime(df['Period Begin Date'], errors='coerce')
df['Period End Date'] = pd.to_datetime(df['Period End Date'], errors='coerce')
df['Scan Date'] = pd.to_datetime(df['Scan Date'], errors='coerce')

# Define the base dir for saving plots
base_dir = './results/'

# Define the years to analyze
years_to_analyze = [2018, 2019, 2020, 2021, 2022]

# Create an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Create dictionaries to store counts per year
zero_dose_counts = {}
below_threshold_counts = {}
around_threshold_counts = {}
above_threshold_counts = {}

for year in years_to_analyze:
    # Filter data for the specified period (July of the current year to July of the next year)
    start_date = f'{year}-07-01'
    end_date = f'{year + 1}-06-30'  # Ending on June 30th of the next year
    filtered_data = df[(df['Period Begin Date'] >= start_date) & (df['Period End Date'] <= end_date)]
    print(f"\nYear {year}-{year + 1}:--------------")

    # Filter data for CHEST monitors only
    chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']

    # Count participants with 'Unused' NoteCode within CHEST monitors
    unused_in_chest = chest_monitors[chest_monitors['NoteCode'] == 'Unused']
    unused_count = len(unused_in_chest)
    cost_unused = unused_count * 4 * 10.70

    # Group by 'Participant Name' and sum the DDE
    chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum().reset_index()
    chest_doses.columns = ['Participant Name', 'Current DDE']

    # Calculate total number of participants
    total_participants = len(chest_doses)

    # Filter out 'Unused' participants
    used_chest_doses = chest_doses[~chest_doses['Participant Name'].isin(unused_in_chest['Participant Name'])].copy()
    used_count = len(used_chest_doses)
    #cost_used = used_count * 4 * 10.70

    prct_used = used_count / total_participants if total_participants > 0 else 0
    #prct_unused = unused_count / total_participants if total_participants > 0 else 0

    # Print names of participants who have used their monitors
    print(f"\nParticipants who have used their monitors for {year}-{year + 1}:\n", used_chest_doses['Participant Name'].tolist())

    # Append the year column
    used_chest_doses.loc[:, 'Year'] = f'{year}-{year + 1}'

    # Append to combined data
    combined_data = pd.concat([combined_data, used_chest_doses], ignore_index=True)

    # 0.8-1.2mSv Categorize participants based on their dose for the current year
   # zero_dose = used_chest_doses[used_chest_doses['Current DDE'] == 0]
   # below_threshold = used_chest_doses[(used_chest_doses['Current DDE'] > 0) & (used_chest_doses['Current DDE'] < 0.8)]
   # around_threshold = used_chest_doses[(used_chest_doses['Current DDE'] >= 0.8) & (used_chest_doses['Current DDE'] <= 1.2)]
   # above_threshold = used_chest_doses[used_chest_doses['Current DDE'] > 1.2]

    # 0.5-1 Categorize participants based on their dose for the current year
    zero_dose = used_chest_doses[used_chest_doses['Current DDE'] == 0]
    below_threshold = used_chest_doses[(used_chest_doses['Current DDE'] > 0) & (used_chest_doses['Current DDE'] < 0.5)]
    around_threshold = used_chest_doses[(used_chest_doses['Current DDE'] >= 0.5) & (used_chest_doses['Current DDE'] <= 1.0)]
    above_threshold = used_chest_doses[used_chest_doses['Current DDE'] > 1.0]


    zero_dose_counts[year] = len(zero_dose)
    below_threshold_counts[year] = len(below_threshold)
    around_threshold_counts[year] = len(around_threshold)
    above_threshold_counts[year] = len(above_threshold)

    #print(f"Number of participants with 'Unused' NoteCode (CHEST monitors): {unused_count}, % unused: {prct_unused}") #not true?
    #print(f"Cost for 'Unused' CHEST monitors (a $10.70/quarter): ${cost_unused}")#not true?
    #print(f"Cost for 'used' CHEST monitors (a $10.70/quarter): ${cost_used}")#not true?
    print(f"Total number of participants: {total_participants}")
    print(f"Number of used dose monitors: {used_count}, % used: {prct_used}")
    print(f"Number of participants with zero dose: {zero_dose_counts[year]}")
   # ### 0.8 -1.2
   # print(f"Number of participants with dose > 0 but < 0.8 mSv: {below_threshold_counts[year]}")
   # print(f"Number of participants with dose between 0.8 mSv and 1.2 mSv: {around_threshold_counts[year]}")
   # print(f"Number of participants with dose > 1.2 mSv: {above_threshold_counts[year]}")

   # print("\nParticipants with zero dose:\n", zero_dose['Participant Name'].tolist())
   # print("\nParticipants with dose > 0 but < 0.8 mSv:\n", below_threshold['Participant Name'].tolist())
   # print("\nParticipants with dose between 0.8 mSv and 1.2 mSv:\n", around_threshold['Participant Name'].tolist())
   # print("\nParticipants with dose > 1.2 mSv:\n", above_threshold['Participant Name'].tolist())
   # ### 0.5-1

    print(f"Number of participants with dose > 0 but < 0.5 mSv: {below_threshold_counts[year]}")
    print(f"Number of participants with dose between 0.5 mSv and 1.0 mSv: {around_threshold_counts[year]}")
    print(f"Number of participants with dose > 1.0 mSv: {above_threshold_counts[year]}")

    print("\nParticipants with zero dose:\n", zero_dose['Participant Name'].tolist())
    print("\nParticipants with dose > 0 but < 0.5 mSv:\n", below_threshold['Participant Name'].tolist())
    print("\nParticipants with dose between 0.5 mSv and 1.0 mSv:\n", around_threshold['Participant Name'].tolist())
    print("\nParticipants with dose > 1.0 mSv:\n", above_threshold['Participant Name'].tolist())
    


# Plotting the combined stacked histogram
#dose_categories = ['0 mSv', '> 0 - < 0.8 mSv', '0.8 - 1.2 mSv', '> 1.2 mSv']
dose_categories = ['0 mSv', '> 0 - < 0.5 mSv', '0.5 - 1.0 mSv', '> 1.0 mSv']
combined_data['Dose Category'] = pd.cut(
    combined_data['Current DDE'],
    bins=[-float('inf'), 0, 0.5, 1.0, float('inf')],
    labels=dose_categories,
    ordered=True
)

fig = px.histogram(
    combined_data,
    x='Year',
    color='Dose Category',
    title='Dose Distribution by Year',
    labels={'count': 'Number of Participants'},
    category_orders={'Dose Category': dose_categories}  # Ensure legend is ordered
)
fig.update_layout(
    barmode='stack',
    title_font=dict(size=18),
    xaxis_title_font=dict(size=18),
    yaxis_title_font=dict(size=18),
    legend_title_font=dict(size=18),
    legend=dict(title=dict(text='Dose Category'), font=dict(size=16)),
    font=dict(size=16)  # Adjusts tick font size
)
fig.show()

sys.exit()

##############
import pandas as pd
import plotly.express as px
import sys

# Load the data from the CSV file
file_path = './dose-monitoring/python_/data/DoseReport_20180101_20240607.csv'
df = pd.read_csv(file_path)

# Convert dose columns to numeric
dose_columns = ['Current DDE', 'Current LDE', 'Current SDE']
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Period Begin Date', 'Period End Date', and 'Scan Date' to datetime objects
df['Period Begin Date'] = pd.to_datetime(df['Period Begin Date'], errors='coerce')
df['Period End Date'] = pd.to_datetime(df['Period End Date'], errors='coerce')
df['Scan Date'] = pd.to_datetime(df['Scan Date'], errors='coerce')

# Define the base dir for saving plots
base_dir = './results/'

# Define the years to analyze
years_to_analyze = [2018, 2019, 2020, 2021, 2022]

# Create an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Create dictionaries to store counts per year
zero_dose_counts = {}
below_threshold_counts = {}
around_threshold_counts = {}
above_threshold_counts = {}

for year in years_to_analyze:
    # Filter data for the specified period (July of the current year to July of the next year)
    start_date = f'{year}-07-01'
    end_date = f'{year + 1}-06-30'  # Ending on June 30th of the next year
    filtered_data = df[(df['Period Begin Date'] >= start_date) & (df['Period End Date'] <= end_date)]
    print(f"\nYear {year}-{year + 1}:--------------")

    # Filter data for CHEST monitors only
    chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']
    nr_monitors = len(chest_monitors)

    # Count participants with 'Unused' NoteCode within CHEST monitors
    unused_in_chest = chest_monitors[chest_monitors['NoteCode'] == 'Unused']
    unused_count = len(unused_in_chest)
    cost_unused = unused_count * 4 * 10.70

    # Group by 'Participant Name' and sum the DDE
    chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum().reset_index()
    chest_doses.columns = ['Participant Name', 'Current DDE']

    # Filter out 'Unused' participants
    used_chest_doses = chest_doses[~chest_doses['Participant Name'].isin(unused_in_chest['Participant Name'])]
    used_count = len(used_chest_doses)
    cost_used = used_count * 4 * 10.70

    prct_used = used_count / nr_monitors
    prct_unused = unused_count / nr_monitors

    # Print names of participants who have used their monitors
    print(f"\nParticipants who have used their monitors for {year}-{year + 1}:\n", used_chest_doses['Participant Name'].tolist())

    # Append the year column
    used_chest_doses['Year'] = f'{year}-{year + 1}'

    # Append to combined data
    combined_data = pd.concat([combined_data, used_chest_doses], ignore_index=True)

    # Categorize participants based on their dose for the current year
    zero_dose = used_chest_doses[used_chest_doses['Current DDE'] == 0]
    below_threshold = used_chest_doses[(used_chest_doses['Current DDE'] > 0) & (used_chest_doses['Current DDE'] < 0.8)]
    around_threshold = used_chest_doses[(used_chest_doses['Current DDE'] >= 0.8) & (used_chest_doses['Current DDE'] <= 1.2)]
    above_threshold = used_chest_doses[used_chest_doses['Current DDE'] > 1.2]

    zero_dose_counts[year] = len(zero_dose)
    below_threshold_counts[year] = len(below_threshold)
    around_threshold_counts[year] = len(around_threshold)
    above_threshold_counts[year] = len(above_threshold)

    print(f"Number of monitors: {nr_monitors}")
    print(f"Number of participants with 'Unused' NoteCode (CHEST monitors): {unused_count}, % unused: {prct_unused}")
    print(f"Cost for 'Unused' CHEST monitors (a $10.70/quarter): ${cost_unused}")
    print(f"Cost for 'used' CHEST monitors (a $10.70/quarter): ${cost_used}")
    print(f"Number of used dose monitors: {used_count}, % used: {prct_used}")
    print(f"Number of participants with zero dose: {zero_dose_counts[year]}")
    print(f"Number of participants with dose > 0 but < 0.8 mSv: {below_threshold_counts[year]}")
    print(f"Number of participants with dose between 0.8 mSv and 1.2 mSv: {around_threshold_counts[year]}")
    print(f"Number of participants with dose > 1.2 mSv: {above_threshold_counts[year]}")

    print("\nParticipants with zero dose:\n", zero_dose['Participant Name'].tolist())
    print("\nParticipants with dose > 0 but < 0.8 mSv:\n", below_threshold['Participant Name'].tolist())
    print("\nParticipants with dose between 0.8 mSv and 1.2 mSv:\n", around_threshold['Participant Name'].tolist())
    print("\nParticipants with dose > 1.2 mSv:\n", above_threshold['Participant Name'].tolist())

sys.exit()

import pandas as pd
import plotly.express as px
import sys

# Load the data from the CSV file
file_path = './dose-monitoring/python_/data/DoseReport_20180101_20240607.csv'
df = pd.read_csv(file_path)

# Convert dose columns to numeric
dose_columns = ['Current DDE', 'Current LDE', 'Current SDE']
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Period Begin Date', 'Period End Date', and 'Scan Date' to datetime objects
df['Period Begin Date'] = pd.to_datetime(df['Period Begin Date'], errors='coerce')
df['Period End Date'] = pd.to_datetime(df['Period End Date'], errors='coerce')
df['Scan Date'] = pd.to_datetime(df['Scan Date'], errors='coerce')

# Define the base dir for saving plots
base_dir = '/home/peter/Documents/research/dose-monitoring/python_/results/'

# Define the years to analyze
years_to_analyze = [2018, 2019, 2020, 2021, 2022]

# Create an empty DataFrame to store combined data
combined_data = pd.DataFrame()

# Create dictionaries to store counts per year
zero_dose_counts = {}
below_threshold_counts = {}
around_threshold_counts = {}
above_threshold_counts = {}

# Calculate the total number of participants and find late entries for CHEST monitors
for year in years_to_analyze:
    start_date = f'{year}-07-01'
    end_date = f'{year + 1}-06-30'
    print(f"\n-----------Year {year}-{year + 1}:")
    filtered_data = df[(df['Period Begin Date'] >= start_date) & (df['Period End Date'] <= end_date)]
    chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']
    print(f"\nNumber of CHEST monitors for {year}-{year + 1}: {len(chest_monitors)}")
    
     # Display the number of participants with 'Unused' in NoteCode
    print(f"\nNumber of (CHEST) participants with 'Unused' in NoteCode for {year}-{year + 1}: {len(chest_monitors[chest_monitors['NoteCode'] == 'Unused'])}")


    # Group by 'Participant Name' and sum the DDE
    chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum().reset_index()
    chest_doses.columns = ['Participant Name', 'Current DDE']
    print(f"\nNumber of CHEST - Name - DDE groupby for {year}-{year + 1}: {len(chest_doses)}")
    
    # Filter out 'Unused' participants
    chest_doses = chest_doses[~chest_doses['Participant Name'].isin(chest_monitors[chest_monitors['NoteCode'] == 'Unused']['Participant Name'])]
    print(f"\nNumber of CHEST - Name - DDE after Unused filter {year}-{year + 1}: {len(chest_doses)}")

    # Display the number of participants with zero dose
    zero_dose = chest_doses[chest_doses['Current DDE'] == 0]
    zero_dose_counts[year] = len(zero_dose)
    print(f"Number of participants with zero dose: {zero_dose_counts[year]}")
    print("\nParticipants with zero dose:\n", zero_dose['Participant Name'].tolist())

    # Categorize participants based on their dose for the current year
    below_threshold = chest_doses[(chest_doses['Current DDE'] > 0) & (chest_doses['Current DDE'] < 0.8)]
    around_threshold = chest_doses[(chest_doses['Current DDE'] >= 0.8) & (chest_doses['Current DDE'] <= 1.2)]
    above_threshold = chest_doses[chest_doses['Current DDE'] > 1.2]

    below_threshold_counts[year] = len(below_threshold)
    around_threshold_counts[year] = len(around_threshold)
    above_threshold_counts[year] = len(above_threshold)

    print(f"Number of participants with dose > 0 but < 0.8 mSv: {below_threshold_counts[year]}")
    print(f"Number of participants with dose between 0.8 mSv and 1.2 mSv: {around_threshold_counts[year]}")
    print(f"Number of participants with dose > 1.2 mSv: {above_threshold_counts[year]}")
    print("\nParticipants with dose > 0 but < 0.8 mSv:\n", below_threshold['Participant Name'].tolist())
    print("\nParticipants with dose between 0.8 mSv and 1.2 mSv:\n", around_threshold['Participant Name'].tolist())
    print("\nParticipants with dose > 1.2 mSv:\n", above_threshold['Participant Name'].tolist())
    
   ## Calculate total dose for sorting
#
#dose_threshold = 0
#
## Calculate the total number of participants and find late entries for CHEST monitors
#for year in years_to_analyze:
#    start_date = f'{year}-07-01'
#    end_date = f'{year + 1}-06-30'
#    filtered_data = df[(df['Period Begin Date'] >= start_date) & (df['Period End Date'] <= end_date)]
#    chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']
#
#    # Filter for participants with 'Unused' in their 'NoteCode'
#    unused_participants = chest_monitors[chest_monitors['NoteCode'] == 'Unused']
#
#    # Remove 'Unused' participants from the main data
#    #chest_monitors = chest_monitors[chest_monitors['NoteCode'] != 'Unused']
#
#    chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum()
#    total_participants = chest_doses.index.nunique()
#
#    # Find late entries for CHEST monitors for the current year
#    late_entries = chest_monitors[(chest_monitors['Scan Date'] - chest_monitors['Period End Date']).dt.days >= 60]
#    num_late_entries = late_entries['Participant Name'].nunique()
#
#    print(f"Total number of participants for {year}-{year + 1}: {total_participants}")
#    print(f"Number of late entries for {year}-{year + 1}: {num_late_entries}")
#    print(f"Number of participants with 'Unused' in NoteCode for {year}-{year + 1}: {unused_participants['Participant Name'].nunique()}")
#
## Calculate the total number of participants and find late entries for CHEST monitors
#for year in years_to_analyze:
#    start_date = f'{year}-07-01'
#    end_date = f'{year + 1}-06-30'
#    filtered_data = df[(df['Period Begin Date'] >= start_date) & (df['Period End Date'] <= end_date)]
#    chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']
#    chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum()
#    total_participants = chest_doses.index.nunique()
#
#    # Find late entries for CHEST monitors for the current year
#    late_entries = chest_monitors[(chest_monitors['Scan Date'] - chest_monitors['Period End Date']).dt.days >= 60]
#    num_late_entries = late_entries['Participant Name'].nunique()
#
#    print(f"Total number of participants for {year}-{year + 1}: {total_participants}")
#    print(f"Number of late entries for {year}-{year + 1}: {num_late_entries}")
#
#
sys.exit()



# Save the plot as a high-resolution PNG
fig_file_path = base_dir + "current_dde_hist_2018_2022.png"
fig.write_image(
    fig_file_path, width=1200, height=800, scale=3
)  # Increase scale for higher resolution

# Calculate the total number of participants and those above the dose threshold for each year
for year in years_to_analyze:
    start_date = f"{year}-07-01"
    end_date = f"{year + 1}-06-30"
    filtered_data = df[
        (df["Period Begin Date"] >= start_date) & (df["Period End Date"] <= end_date)
    ]
    chest_monitors = filtered_data[filtered_data["Use"] == "CHEST"]
    chest_doses = chest_monitors.groupby("Participant Name")["Current DDE"].sum()
    total_participants = chest_doses.index.nunique()
    participants_above_threshold = chest_doses[
        chest_doses > dose_threshold
    ].index.nunique()
    participants_above = chest_doses[chest_doses > 1].index.nunique()
    # Filter entries where 'Scan Date' is at least a year later than 'Period End Date'
    df["Scan Date"] = pd.to_datetime(df["Scan Date"], errors="coerce")
    filtered_entries = df[
        (df["Period Begin Date"] >= start_date)
        & (df["Period End Date"] <= end_date)
        & (df["Scan Date"] >= df["Period End Date"] + pd.DateOffset(years=1))
    ]

    # total cost per year with a quartely cost of $10.70 per badge
    total_yearly_cost = total_participants * 4 * 10.70
    # Get the number of late entries for the year
    num_late_entries = filtered_entries.shape[0]

    # Get the cost for late badges with a cost of $17.30 per late badge
    late_cost = num_late_entries * 17.30

    print(f"Total number of participants for {year}-{year + 1}: {total_participants}")
    print(
        f"Number of participants with Current DDE > {dose_threshold} mSv for {year}-{year + 1}: {participants_above_threshold}"
    )
    print(
        f"Number of participants with Current DDE > 1 mSv for {year}-{year + 1}: {participants_above}"
    )
    print(
        f"Number of late entries (Scan Date at least a year after Period End Date) for {year}-{year + 1}: {num_late_entries}"
    )
    print(
        f"Cost for late entries ($17.30 per late badge) for {year}-{year + 1}: {late_cost}"
    )
    print(
        f"Cost for quarterly badges ($10.70 per badge) for {year}-{year + 1}: {total_yearly_cost}"
    )


# Thank you for your enquiry, please find below a list of the pricing for your badges as requested.
#
# Monthly badges - $10.50 per badge
#
# Quarterly badges - $10.70 per badge
#
# Lost badges - $17.30 per badge
#
# Late return - $17.30 per badge
#
#
#
# If you have any further questions, please do not hesitate to contact me.
#
#
#
# Kind Regards,
#
# Rebecca Ruming
#
# Customer Service Representative
#
# Landauer Australasia Pty Ltd
#
# Phone: 02 8651 4000
#
# Email: rebecca.ruming@landauer.com
#
