import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import datetime
#import webbrowser
import subprocess

############################
#Deep dose equivalent (DDE) is the dose equivalent at a tissue depth of 1 cm (i.e., the minimum
#depth of internal organs) and is used to compute your whole-body dose.
# The whole-body dose should be compared to the deep dose equivalent or effective dose equivalent limit (5,000 mrem/year).
#• Lens dose equivalent (LDE) is the dose equivalent at a tissue depth of 0.3 cm (i.e., to the lens of the
#eye) and should be compared to the lens dose limit (15,000 mrem/year).
#• Shallow dose equivalent (SDE) is the dose equivalent at a tissue depth of 0.007 cm (i.e., to the
#dermis) and should be compared to the skin and extremity dose limits (50,000 mrem/year).
#YTD DDE stands for "Year-to-Date Diagnostic Dose Equivalent." It is a term commonly used in dose monitoring systems in the medical field. YTD DDE refers to the cumulative diagnostic dose equivalent received by an individual or a group of individuals over the course of the current calendar year up to the present date.
#############################
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
# load dose data
#file_path = '/home/peter/Documents/research/dose-monitoring/python_/data/DoseReport_20180101_20230301.xlsx'
#df = pd.read_excel(file_path, sheet_name='DoseReport')

#####################
# Load the data from the ODS file
file_path = '/home/peter/Documents/research/dose-monitoring/python_/data/DoseReport_20180101_20230301.ods'
#df = pd.read_excel(file_path, engine='odf')
df = pd.read_excel(file_path, sheet_name='DoseReport')

# Convert dose columns to numeric (assuming they are not already)
dose_columns = ['Current DDE', 'Current LDE', 'Current SDE']
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Period Begin Date' and 'Period End Date' to datetime objects
df['Period Begin Date'] = pd.to_datetime(df['Period Begin Date'], errors='coerce')
df['Period End Date'] = pd.to_datetime(df['Period End Date'], errors='coerce')

# Filter data for the specified period (2021-07 to 2022-07)
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
#non_zero_total_sum = total_doses[total_doses['Total Sum Current'] > 0.2]

# Sort the data by 'Total Sum Current' in descending order
#sorted_total_doses = non_zero_total_sum.sort_values(by='Total Sum Current', ascending=False)
sorted_total_doses = total_doses.sort_values(by='Total Sum Current', ascending=False)

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
#fig.show()

#Define the base dir for saving plots ########
base_dir = '/home/peter/Documents/research/dose-monitoring/python_/results/'

# Save the plot as a high-resolution PNG
#fig_file_path = base_dir + 'current_dose_histogram.png'
#fig.write_image(fig_file_path, width=1200, height=800, scale=3)  # Increase scale for higher resolution

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

################

# Convert dose columns to numeric
dose_columns = ['Current DDE', 'Current LDE', 'Current SDE']
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Period Begin Date' and 'Period End Date' to datetime objects
df['Period Begin Date'] = pd.to_datetime(df['Period Begin Date'], errors='coerce')
df['Period End Date'] = pd.to_datetime(df['Period End Date'], errors='coerce')

# Define the base directory for saving plots
base_dir = '/home/peter/Documents/research/dose-monitoring/python_/results/'

# Define the years to analyze
years_to_analyze = [2018, 2019, 2020, 2021]

for year in years_to_analyze:
    # Filter data for the specified period (July of the current year to July of the next year)
    start_date = f'{year}-07-01'
    end_date = f'{year + 1}-06-30'
    filtered_data = df[(df['Period Begin Date'] >= start_date) & (df['Period End Date'] <= end_date)]

    # Filter data for CHEST monitors only
    chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']

    # Group by 'Participant Name' and sum the DDE
    chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum().reset_index()

    # Filter out participants with non-zero DDE
    non_zero_dde = chest_doses[chest_doses['Current DDE'] > 0]

    # Sort the data by 'Current DDE' in descending order
    sorted_dde_sum = non_zero_dde.sort_values(by='Current DDE', ascending=False)

    # Create a bar chart with Plotly
    fig = px.bar(sorted_dde_sum, x='Participant Name',
                 y='Current DDE',
                 labels={'value': 'Dose (mSv)'},
                 color_discrete_sequence=['red'],
                 title=f'Current DDE for Participants with CHEST Monitors ({year}-{year + 1})')

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

#    # Show the interactive plot
#    fig.show()
#
#    # Save the plot as a high-resolution PNG
#    fig_file_path = f'{base_dir}current_dde_hist_{year}_{year + 1}.png'
#    fig.write_image(fig_file_path, width=1200, height=800, scale=3)  # Increase scale for higher resolution
#

### combined stacked histogram #######

# Convert dose columns to numeric
dose_columns = ['Current DDE', 'Current LDE', 'Current SDE']
df[dose_columns] = df[dose_columns].apply(pd.to_numeric, errors='coerce')

# Convert 'Period Begin Date' and 'Period End Date' to datetime objects
df['Period Begin Date'] = pd.to_datetime(df['Period Begin Date'], errors='coerce')
df['Period End Date'] = pd.to_datetime(df['Period End Date'], errors='coerce')

# Define the base directory for saving plots
base_dir = '/home/peter/Documents/research/dose-monitoring/python_/results/'

# Define the years to analyze
years_to_analyze = [2018, 2019, 2020, 2021]

# Get a list of all unique participant names
all_participants = df['Participant Name'].unique()

# Create an empty DataFrame to store combined data
combined_data = pd.DataFrame()

for year in years_to_analyze:
    # Filter data for the specified period (July of the current year to July of the next year)
    start_date = f'{year}-07-01'
    end_date = f'{year + 1}-06-30'  # Ending on June 30th of the next year
    filtered_data = df[(df['Period Begin Date'] >= start_date) & (df['Period End Date'] <= end_date)]

    # Filter data for CHEST monitors only
    chest_monitors = filtered_data[filtered_data['Use'] == 'CHEST']

    # Group by 'Participant Name' and sum the DDE
    chest_doses = chest_monitors.groupby('Participant Name')['Current DDE'].sum()

    # Reindex to include all participants and fill missing values with 0
    chest_doses = chest_doses.reindex(all_participants, fill_value=0).reset_index()
    chest_doses.columns = ['Participant Name', 'Current DDE']
    chest_doses['Year'] = f'{year}-{year + 1}'

    # Append to combined data
    combined_data = pd.concat([combined_data, chest_doses], ignore_index=True)

# Calculate total dose for sorting
total_dose = combined_data.groupby('Participant Name')['Current DDE'].sum().reset_index()
sorted_participants = total_dose.sort_values(by='Current DDE', ascending=False)['Participant Name']

# Sort combined_data by sorted_participants
combined_data['Participant Name'] = pd.Categorical(combined_data['Participant Name'], categories=sorted_participants, ordered=True)
sorted_combined_data = combined_data.sort_values('Participant Name')

# Create a stacked bar chart with Plotly
fig = px.bar(sorted_combined_data, x='Participant Name', y='Current DDE', color='Year',
             labels={'Current DDE': 'Dose (mSv)'},
             title='Current DDE for Participants with CHEST Monitors 2018-2022',
             color_discrete_sequence=px.colors.qualitative.Bold)

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


# Filter the data for participants with Current DDE greater than 1 mSv
greater_than_1mSv = sorted_combined_data[sorted_combined_data['Current DDE'] >= 1]

# Filter the data for participants with Current DDE less greater than 1 mSv
less_than_1mSv = sorted_combined_data[sorted_combined_data['Current DDE'] < 1]

# Count the number of unique participants
num_greater_than_1mSv = greater_than_1mSv['Participant Name'].nunique()
num_less_than_1mSv = less_than_1mSv['Participant Name'].nunique()

print(f"Number of participants with Current DDE >= 1 mSv: {num_greater_than_1mSv}")
print(f"Number of participants with Current DDE < 1 mSv: {num_less_than_1mSv}")

# Show the interactive plot
#fig.show()

# Save the plot as a high-resolution PNG
#fig_file_path = base_dir + 'current_dde_hist_2018_2022.png'
#fig.write_image(fig_file_path, width=1200, height=800, scale=3)  # Increase scale for higher resolution

