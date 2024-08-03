import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash import dcc, html, Dash
import os
import pickle

# Ensure the directories exist
def create_directories():
    """
    Create directories for saving visualizations if they do not exist.
    """
    if not os.path.exists("data/analyzed"):
        os.makedirs("data/analyzed")

# Load data
def load_data():
    """
    Load the analyzed data from the CSV file.
    
    Returns:
        pd.DataFrame: The DataFrame containing the flight delay data.
    """
    data_path = "data/processed/flight_data_transformed.csv"
    return pd.read_csv(data_path)

# Load the model
def load_model():
    """
    Load the pre-trained model from a pickle file.
    
    Returns:
        model: The pre-trained model.
    """
    model_path = "model.pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Make predictions using the model
def predict_delays(df, model):
    """
    Predict delay probabilities using the pre-trained model.

    Args:
        df (pd.DataFrame): The DataFrame containing flight data.
        model: The pre-trained model.

    Returns:
        pd.DataFrame: The DataFrame with added predictions.
    """
    # Assume the model expects features such as 'historical_on_time_performance' and 'current_conditions'
    features = ['historical_on_time_performance', 'current_conditions']  # Adjust based on actual model features
    X = df[features].fillna(0)  # Replace missing values with 0 or appropriate value
    df['delay_probability'] = model.predict_proba(X)[:, 1]  # Probability of delays, adjust indexing if needed
    return df

# Static Visualizations
def create_static_visualizations(df, airport_code):
    """
    Create static visualizations using Matplotlib and Seaborn.

    Args:
        df (pd.DataFrame): The DataFrame containing flight delay data.
        airport_code (str): The 3-letter airport code to filter the data.
    """
    # Filter data for the selected airport
    filtered_df = df[df['destination_airport'] == airport_code]

    # Histogram of delay probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df['delay_probability'], bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of Flight Delay Probabilities to {airport_code}')
    plt.xlabel('Delay Probability')
    plt.ylabel('Frequency')
    plt.savefig('data/analyzed/flight_delay_histogram.png')
    plt.close()
    print("Static histogram saved to data/analyzed/flight_delay_histogram.png")

    # Bar chart of average delay probabilities by airline
    plt.figure(figsize=(12, 8))
    sns.barplot(x='airline', y='delay_probability', data=filtered_df, palette='viridis')
    plt.title(f'Average Delay Probability by Airline to {airport_code}')
    plt.xlabel('Airline')
    plt.ylabel('Average Delay Probability')
    plt.xticks(rotation=45)
    plt.savefig('data/analyzed/average_delay_by_airline.png')
    plt.close()
    print("Static bar chart saved to data/analyzed/average_delay_by_airline.png")

# Interactive Scatter Plot
def create_interactive_scatter_plot(df, airport_code):
    """
    Create an interactive scatter plot to visualize the relationship between different factors and delay probabilities.

    Args:
        df (pd.DataFrame): The DataFrame containing flight delay data.
        airport_code (str): The 3-letter airport code to filter the data.
    
    Returns:
        fig: The Plotly figure object for the scatter plot.
    """
    filtered_df = df[df['destination_airport'] == airport_code]
    
    fig = px.scatter(filtered_df, x='historical_on_time_performance', y='current_conditions',
                     color='delay_probability', size='delay_probability',
                     hover_name='flight_id',  # Example field, adjust as necessary
                     title=f'Impact of Historical Performance and Current Conditions on Delay Probability to {airport_code}',
                     labels={'historical_on_time_performance': 'Historical On-Time Performance',
                             'current_conditions': 'Current Conditions',
                             'delay_probability': 'Delay Probability'})
    
    return fig

# Create airline delay table
def create_airline_delay_table(df, airport_code):
    """
    Create a table of airlines and their delay percentages.

    Args:
        df (pd.DataFrame): The DataFrame containing flight delay data.
        airport_code (str): The 3-letter airport code to filter the data.
    
    Returns:
        pd.DataFrame: The DataFrame containing airlines and their average delay probabilities.
    """
    filtered_df = df[df['destination_airport'] == airport_code]
    airline_delay_df = filtered_df.groupby('airline')['delay_probability'].mean().reset_index()
    airline_delay_df = airline_delay_df.sort_values(by='delay_probability', ascending=False)
    return airline_delay_df

# Interactive Dashboard
def create_dashboard(df, airport_code):
    """
    Create an interactive dashboard using Dash.

    Args:
        df (pd.DataFrame): The DataFrame containing flight delay data.
        airport_code (str): The 3-letter airport code to filter the data.
    """
    app = Dash(__name__)

    # Create dashboard layout
    app.layout = html.Div([
        html.H1(f"Flight Delay Probability Dashboard to {airport_code}"),
        dcc.Graph(id='delay-scatter-plot'),
        dash_table.DataTable(
            id='airline-delay-table',
            columns=[{"name": i, "id": i} for i in ['airline', 'delay_probability']],
            data=create_airline_delay_table(df, airport_code).to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'fontWeight': 'bold'}
        )
    ])

    # Define callback to update visualizations
    @app.callback(
        dcc.Output('delay-scatter-plot', 'figure')
    )
    def update_scatter_plot():
        scatter_fig = create_interactive_scatter_plot(df, airport_code)
        return scatter_fig

    # Run the app
    app.run_server(debug=True)