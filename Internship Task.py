#!/usr/bin/env python
# coding: utf-8

# # Task 1.1

# In[4]:


import pandas as pd


# In[5]:


df = pd.read_csv('Railway_info.csv')


# In[6]:


print(df.head(10))


# In[7]:


print(df.info())


# In[8]:


print(df.describe())


# In[9]:


df['days'] = df['days'].replace({
    'Saturdayd' : 'Saturday',
    'Thursdayd' : 'Thursday',
    'Sundayd'  : 'Sunday',
    'Mondayd' : 'Monday',
    'Tuesdayd' : 'Tuesday',
    'Wednesdayd' : 'Wednesday',
    'Fridayd' : 'Friday'  
})
df['days'].unique()


# In[10]:


print(df.isnull().sum())


# # Task 1.2

# In[11]:


unique_train = df['Train_Name'].unique()
print(unique_train)


# In[12]:


count_unique_train = df["Train_Name"].nunique()
print(count_unique_train)


# In[13]:


count_each_train_count = df["Train_Name"].value_counts()
print(count_each_train_count)


# In[14]:


unique_sourcestations = df['Source_Station_Name'].unique()
print(unique_sourcestations)


# In[15]:


unique_sourcestations = df['Source_Station_Name'].nunique()
print(unique_sourcestations)


# In[16]:


unique_Destination_Station = df['Destination_Station_Name'].unique()
print(unique_Destination_Station)


# In[17]:


unique_Destination_Station_count = df['Destination_Station_Name'].nunique()
print(unique_Destination_Station_count)


# In[18]:


count_each_s_station_count = df['Source_Station_Name'].value_counts()
print(count_each_train_count)


# In[19]:


top5_source_stations = count_each_train_count.head(5)
print(top5_source_stations)


# In[20]:


count_d_station_count = df['Destination_Station_Name'].value_counts()
print(count_each_train_count)


# In[21]:


top5_destination = count_d_station_count.head(5)
print(top5_destination)


# # Task 2.1

# In[22]:


filtered_df = df[df['days'] == "Saturday"]
print(filtered_df.head())


# In[23]:


train_cstmumbai_df = df[df['Source_Station_Name'] == 'CST-MUMBAI']
print(train_df)


# In[24]:


top_5_destinations_from_CSTmumbai  = (
    train_cstmumbai_df["Destination_Station_Name"]
    .value_counts()
    .head(5)
)
top_5_destinations_from_CSTmumbai


# In[25]:


days_of_CSTmumbai = train_cstmumbai_df["days"].value_counts()
days_of_CSTmumbai


# # TASK 2.2

# In[26]:


station_train_count = (
    df.groupby("Source_Station_Name")
    .size()                              
    .reset_index(name="Train_Count")    
    .sort_values(by="Train_Count", ascending=False)
)
station_train_count.head(15)


# In[27]:


df["Days_List"] = df["days"].apply(lambda x: [day.strip() for day in x.split(",") if day.strip()])
df["Days_List"]


# In[28]:


df["Operating_Days_Count"] = df["Days_List"].apply(len)


# In[29]:


avg_trains_per_day = (
    df.groupby("Source_Station_Name")["Operating_Days_Count"].sum() / 7
).reset_index(name="Avg_Trains_Per_Day")

avg_trains_per_day = avg_trains_per_day.sort_values(by="Avg_Trains_Per_Day", ascending=False)

avg_trains_per_day.head(10)


# # Task 2.3: Data Enrichment

# In[30]:


def categorize_train(days):
    weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
    weekends = {"Saturday", "Sunday"}
    day_set = set(days)

    if day_set == weekdays.union(weekends):
        return "AllDays"
    elif day_set.issubset(weekdays):
        return "Weekday"
    elif day_set.issubset(weekends):
        return "Weekend"
    elif day_set.intersection(weekdays) and day_set.intersection(weekends):
        return "Both"
    else:
        return "Other"

df["Day_Type"] = df["Days_List"].apply(categorize_train)


print(df[["Train_No", "days", "Day_Type"]].head(10))


print("\nCategory count:")
print(df["Day_Type"].value_counts())



# In[31]:


df.head(10)


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
import ast

def safe_literal_eval(val):
    """
    Safely evaluate string representations of lists.
    
    Args:
        val: Value to evaluate (string, list, or other)
        
    Returns:
        list: Parsed list or empty list if parsing fails
    """
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return val if isinstance(val, list) else []

def analyze_train_schedule(df):
    """
    Analyze and visualize train schedule distribution across weekdays.
    
    Args:
        df: DataFrame containing a 'Days_List' column with day information
        
    Returns:
        pd.DataFrame: Summary of trains per day
    """
    # Safely parse Days_List column
    df = df.copy()  # Avoid modifying original dataframe
    df["Days_List"] = df["Days_List"].apply(safe_literal_eval)
    
    # Explode Days_List so each day has its own row
    df_exploded = df.explode("Days_List")
    
    # Define correct day order
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Group by Days_List and count rows (journeys per day)
    day_counts = (df_exploded["Days_List"]
                  .value_counts()
                  .reindex(day_order, fill_value=0))
    
    # Convert to DataFrame
    day_df = day_counts.reset_index()
    day_df.columns = ["Day", "No_of_Trains"]
    
    return day_df

def plot_train_distribution(day_df):
    """
    Create a bar chart showing train distribution across weekdays.
    
    Args:
        day_df: DataFrame with 'Day' and 'No_of_Trains' columns
    """
    # Create figure and axis with better styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(day_df["Day"], day_df["No_of_Trains"], 
                  color='steelblue', alpha=0.8, edgecolor='navy')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_title("Distribution of Train Journeys Throughout the Week", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Day of the Week", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Trains Operating", fontsize=12, fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Assuming df is already loaded
    # df = pd.read_csv("your_data.csv")  # Uncomment and modify as needed
    
    # Analyze the data
    day_summary = analyze_train_schedule(df)
    
    # Display summary statistics
    print("Train Schedule Summary:")
    print(day_summary.to_string(index=False))
    print(f"\nTotal trains: {day_summary['No_of_Trains'].sum()}")
    print(f"Average trains per day: {day_summary['No_of_Trains'].mean():.2f}")
    print(f"Busiest day: {day_summary.loc[day_summary['No_of_Trains'].idxmax(), 'Day']}")
    print(f"Quietest day: {day_summary.loc[day_summary['No_of_Trains'].idxmin(), 'Day']}")
    
    # Create visualization
    plot_train_distribution(day_summary)


# In[33]:


import matplotlib.pyplot as plt
import pandas as pd

# Day Type distribution analyze kara
day_type_count = df["Day_Type"].value_counts().reset_index()
day_type_count.columns = ["Day_Type", "No_of_Trains"]

# Summary print kara
print("Day Type Summary:")
print(day_type_count)
print(f"\nTotal Trains: {day_type_count['No_of_Trains'].sum()}")

# Colors define kara
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Pie Chart
plt.figure(figsize=(4, 5))
plt.pie(
    day_type_count["No_of_Trains"],
    labels=day_type_count["Day_Type"],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors[:len(day_type_count)],
    explode=[0.05] * len(day_type_count),
    shadow=True,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)
plt.title("Train Operations: Weekday vs Weekend", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Bar Chart (Alternative)
plt.figure(figsize=(4, 3))
bars = plt.bar(
    day_type_count["Day_Type"],
    day_type_count["No_of_Trains"],
    color=colors[:len(day_type_count)],
    alpha=0.8,
    edgecolor='black'
)

# Bar var values show kara
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Train Operations: Weekday vs Weekend", fontsize=14, fontweight='bold')
plt.xlabel("Day Type", fontsize=12, fontweight='bold')
plt.ylabel("Number of Trains", fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# In[39]:


keywords = ['Express', 'Mail', 'Superfast', 'Passenger', 'Rajdhani', 'Shatabdi']
df_words = df.assign(day=df['days'].str.split(',')).explode('day')
df_words['day'] = df_words['day'].str.strip()

# Create keyword columns
for word in keywords:
    df_words[word] = df_words['Train_Name'].str.contains(word, case=False, na=False).astype(int)

# Group and sum
heatmap_data = df_words.groupby('day')[keywords].sum()

plt.figure(figsize=(8,5))
sns.heatmap(heatmap_data, annot=True, cmap='crest')
plt.title('Keyword Frequency in Train Names by Day')
plt.xlabel('Keyword')
plt.ylabel('Day')
plt.tight_layout()
plt.show()


# In[46]:


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ✅ Step 1: Find top 1 source stations by train count
top_sources = df['Source_Station_Name'].value_counts().nlargest(1).index

# ✅ Step 2: Filter dataset for those stations (as source or destination)
filtered_df = df[(df['Source_Station_Name'].isin(top_sources)) | 
                 (df['Destination_Station_Name'].isin(top_sources))]

# ✅ Step 3: Create graph
G = nx.from_pandas_edgelist(filtered_df, 
                            source='Source_Station_Name', 
                            target='Destination_Station_Name')

# ✅ Step 4: Plot the network
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3, seed=42)  # better spacing

nx.draw_networkx_nodes(G, pos, node_size=100, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=7)

plt.title('Connectivity Network of Top 1 Busiest Source Stations', fontsize=13)
plt.axis('off')
plt.show()



# In[54]:


import plotly.express as px
import pandas as pd

# Count trains per source station
top_sources = df['Source_Station_Name'].value_counts().nlargest(10).reset_index()
top_sources.columns = ['Source_Station_Name', 'Train_Count']

# Plotly bar chart
fig = px.bar(
    top_sources,
    x='Source_Station_Name',
    y='Train_Count',
    color='Train_Count',
    color_continuous_scale='Blues',
    title='Top 10 Source Stations by Number of Trains'
)

fig.update_layout(
    xaxis_title='Source Station',
    yaxis_title='Number of Trains',
    template='plotly_white'
)

fig.show()


# In[58]:


top_src = df['Source_Station_Name'].value_counts().nlargest(10).index
top_dst = df['Destination_Station_Name'].value_counts().nlargest(10).index
subset = df[df['Source_Station_Name'].isin(top_src) & df['Destination_Station_Name'].isin(top_dst)]

heat = subset.groupby(['Source_Station_Name', 'Destination_Station_Name']).size().unstack(fill_value=0)

plt.figure(figsize=(8,4))
sns.heatmap(heat, cmap='YlOrRd', annot=True, fmt='d')
plt.title('Heatmap: Train Counts Between Top 10 Stations')
plt.xlabel('Destination Station')
plt.ylabel('Source Station')
plt.tight_layout()
plt.show()

