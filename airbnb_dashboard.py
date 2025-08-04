import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Airbnb Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF5A5F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF5A5F;
    }
    .stDataFrame {
        border: 1px solid #e6e6e6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        # Coba load file CSV yang sebenarnya
        df = pd.read_csv('airbnb_features.csv')
        return df
    except FileNotFoundError:
        # Jika file tidak ditemukan, buat sample data
        st.warning("File 'airbnb_features.csv' tidak ditemukan. Menggunakan sample data.")
        
        np.random.seed(42)
        neighbourhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        room_types = ['Entire home/apt', 'Private room', 'Shared room']
        price_categories = ['Budget', 'Mid-range', 'Luxury']
        performance_tiers = ['Low', 'Medium', 'High']
        
        n_samples = 500
        data = {
            'id': range(1, n_samples + 1),
            'name': [f'Property {i}' for i in range(1, n_samples + 1)],
            'host_id': np.random.randint(1000, 9999, n_samples),
            'host_name': [f'Host {i}' for i in range(1, n_samples + 1)],
            'neighbourhood_group': np.random.choice(neighbourhoods, n_samples),
            'neighbourhood': [f'Area {i}' for i in range(1, n_samples + 1)],
            'room_type': np.random.choice(room_types, n_samples),
            'price': np.random.randint(50, 500, n_samples),
            'minimum_nights': np.random.randint(1, 30, n_samples),
            'number_of_reviews': np.random.randint(0, 300, n_samples),
            'availability_365': np.random.randint(0, 365, n_samples),
            'price_category': np.random.choice(price_categories, n_samples),
            'estimated_annual_revenue': np.random.randint(10000, 100000, n_samples),
            'estimated_occupancy_rate': np.round(np.random.uniform(0.3, 0.9, n_samples), 2),
            'performance_tier': np.random.choice(performance_tiers, n_samples),
            'market_position': np.random.choice(['Niche', 'Mainstream', 'Premium'], n_samples)
        }
        
        return pd.DataFrame(data)

def create_summary_table(data, group_col, value_col, title):
    """Create summary table for groupby analysis"""
    if value_col:
        summary = data.groupby(group_col).agg({
            value_col: ['count', 'mean', 'sum', 'min', 'max']
        }).round(2)
        summary.columns = ['Count', f'Avg {value_col}', f'Total {value_col}', f'Min {value_col}', f'Max {value_col}']
    else:
        summary = data[group_col].value_counts().to_frame()
        summary.columns = ['Count']
        summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(1)
    
    st.subheader(title)
    st.dataframe(summary, use_container_width=True)
    return summary

def display_chart_using_streamlit(data, x_col, y_col=None, chart_type='bar'):
    """Create charts using Streamlit's built-in chart functions"""
    if chart_type == 'bar' and y_col:
        # Group data for bar chart
        grouped_data = data.groupby(x_col)[y_col].mean().reset_index()
        st.bar_chart(grouped_data.set_index(x_col)[y_col])
    elif chart_type == 'bar':
        # Count data for bar chart
        count_data = data[x_col].value_counts()
        st.bar_chart(count_data)
    elif chart_type == 'line' and y_col:
        # Line chart
        grouped_data = data.groupby(x_col)[y_col].mean().reset_index()
        st.line_chart(grouped_data.set_index(x_col)[y_col])
    elif chart_type == 'scatter' and y_col:
        # For scatter plot, we'll show a sample of data points
        sample_data = data.sample(min(100, len(data)))[[x_col, y_col]]
        st.scatter_chart(sample_data.set_index(x_col)[y_col])

def show_statistical_summary(data, columns):
    """Display statistical summary for numeric columns"""
    st.subheader("Statistical Summary")
    
    numeric_data = data[columns].select_dtypes(include=[np.number])
    if not numeric_data.empty:
        summary_stats = numeric_data.describe().round(2)
        st.dataframe(summary_stats, use_container_width=True)
    else:
        st.info("No numeric columns available for statistical summary")

def main():
    # Header
    st.markdown('<h1 class="main-header">Airbnb Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Neighbourhood filter
    if 'neighbourhood_group' in df.columns:
        selected_neighbourhoods = st.sidebar.multiselect(
            "Select Neighbourhood Groups",
            options=df['neighbourhood_group'].unique(),
            default=df['neighbourhood_group'].unique()
        )
    else:
        selected_neighbourhoods = None
    
    # Room type filter
    if 'room_type' in df.columns:
        selected_room_types = st.sidebar.multiselect(
            "Select Room Types",
            options=df['room_type'].unique(),
            default=df['room_type'].unique()
        )
    else:
        selected_room_types = None
    
    # Price range filter
    if 'price' in df.columns:
        price_range = st.sidebar.slider(
            "Price Range ($)",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=(int(df['price'].min()), int(df['price'].max()))
        )
    else:
        price_range = None
    
    # Filter data
    filtered_df = df.copy()
    
    if selected_neighbourhoods and 'neighbourhood_group' in df.columns:
        filtered_df = filtered_df[filtered_df['neighbourhood_group'].isin(selected_neighbourhoods)]
    
    if selected_room_types and 'room_type' in df.columns:
        filtered_df = filtered_df[filtered_df['room_type'].isin(selected_room_types)]
    
    if price_range and 'price' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['price'] >= price_range[0]) &
            (filtered_df['price'] <= price_range[1])
        ]
    
    # Key Metrics
    st.header("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Properties",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)} from total"
        )
    
    with col2:
        if 'price' in filtered_df.columns:
            avg_price = filtered_df['price'].mean()
            st.metric(
                label="Average Price",
                value=f"${avg_price:.0f}",
                delta=f"${avg_price - df['price'].mean():.0f} vs overall"
            )
        else:
            st.metric(label="Average Price", value="N/A")
    
    with col3:
        if 'estimated_annual_revenue' in filtered_df.columns:
            total_revenue = filtered_df['estimated_annual_revenue'].sum()
            st.metric(
                label="Total Est. Revenue",
                value=f"${total_revenue:,.0f}",
            )
        else:
            st.metric(label="Total Est. Revenue", value="N/A")
    
    with col4:
        if 'estimated_occupancy_rate' in filtered_df.columns:
            avg_occupancy = filtered_df['estimated_occupancy_rate'].mean()
            st.metric(
                label="Avg Occupancy Rate",
                value=f"{avg_occupancy:.1%}",
            )
        else:
            st.metric(label="Avg Occupancy Rate", value="N/A")
    
    with col5:
        if 'number_of_reviews' in filtered_df.columns:
            avg_reviews = filtered_df['number_of_reviews'].mean()
            st.metric(
                label="Avg Reviews",
                value=f"{avg_reviews:.0f}",
            )
        else:
            st.metric(label="Avg Reviews", value="N/A")
    
    # Market Analysis with built-in charts
    st.header("Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'neighbourhood_group' in filtered_df.columns and 'price' in filtered_df.columns:
            st.subheader("Average Price by Neighbourhood Group")
            display_chart_using_streamlit(filtered_df, 'neighbourhood_group', 'price', 'bar')
        else:
            st.info("Price by neighbourhood data not available")
    
    with col2:
        if 'room_type' in filtered_df.columns:
            st.subheader("Room Type Distribution")
            display_chart_using_streamlit(filtered_df, 'room_type', None, 'bar')
        else:
            st.info("Room type data not available")
    
    # Performance Analysis with tables
    st.header("Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'performance_tier' in filtered_df.columns and 'estimated_annual_revenue' in filtered_df.columns:
            create_summary_table(filtered_df, 'performance_tier', 'estimated_annual_revenue',
                                'Revenue by Performance Tier')
        else:
            st.info("Performance tier data not available")
    
    with col2:
        if 'market_position' in filtered_df.columns and 'price' in filtered_df.columns:
            create_summary_table(filtered_df, 'market_position', 'price',
                                'Price by Market Position')
        else:
            st.info("Market position data not available")
    
    # Additional Analysis with Streamlit charts
    st.header("Additional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'price' in filtered_df.columns and 'estimated_occupancy_rate' in filtered_df.columns:
            st.subheader("Price vs Occupancy Rate")
            display_chart_using_streamlit(filtered_df, 'price', 'estimated_occupancy_rate', 'scatter')
    
    with col2:
        if 'availability_365' in filtered_df.columns:
            st.subheader("Availability Distribution")
            display_chart_using_streamlit(filtered_df, 'availability_365', None, 'bar')
    
    # Statistical Summary
    numeric_cols = []
    potential_cols = ['price', 'minimum_nights', 'number_of_reviews', 
                     'availability_365', 'estimated_annual_revenue', 'estimated_occupancy_rate']
    
    for col in potential_cols:
        if col in filtered_df.columns:
            numeric_cols.append(col)
    
    if numeric_cols:
        show_statistical_summary(filtered_df, numeric_cols)
    
    # Detailed Analysis Tables
    st.header("Detailed Analysis")
    
    # Top performers table
    st.subheader("Top Performing Properties")
    
    if 'estimated_annual_revenue' in filtered_df.columns:
        display_cols = ['name'] if 'name' in filtered_df.columns else []
        for col in ['neighbourhood_group', 'room_type', 'price', 
                   'estimated_annual_revenue', 'estimated_occupancy_rate', 'number_of_reviews']:
            if col in filtered_df.columns:
                display_cols.append(col)
        
        if display_cols:
            top_performers = filtered_df.nlargest(10, 'estimated_annual_revenue')[display_cols]
            st.dataframe(top_performers, use_container_width=True)
    else:
        st.info("Revenue data not available for ranking")
    
    # Category Analysis
    st.subheader("Category Analysis")
    
    analysis_tabs = st.tabs(["Room Type Analysis", "Price Category Analysis", "Performance Tier Analysis"])
    
    with analysis_tabs[0]:
        if 'room_type' in filtered_df.columns:
            create_summary_table(filtered_df, 'room_type', 'price', 'Room Type Summary')
    
    with analysis_tabs[1]:
        if 'price_category' in filtered_df.columns:
            create_summary_table(filtered_df, 'price_category', 'estimated_annual_revenue', 
                                'Price Category Summary')
    
    with analysis_tabs[2]:
        if 'performance_tier' in filtered_df.columns:
            create_summary_table(filtered_df, 'performance_tier', None, 'Performance Tier Distribution')
    
    # Data Preview
    st.header("Data Preview")
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(filtered_df))
    with col2:
        st.metric("Total Columns", len(filtered_df.columns))
    with col3:
        st.metric("Memory Usage", f"{filtered_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Show column info
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': filtered_df.columns,
        'Data Type': filtered_df.dtypes,
        'Non-Null Count': filtered_df.count(),
        'Null Count': filtered_df.isnull().sum(),
        'Null Percentage': (filtered_df.isnull().sum() / len(filtered_df) * 100).round(2)
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(20), use_container_width=True)
    
    # Data Export
    st.header("Data Export")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"airbnb_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info(f"Showing {len(filtered_df)} properties out of {len(df)} total properties")
        
        # Additional export options
        if st.button("Show Data Summary"):
            st.json({
                "total_properties": len(filtered_df),
                "avg_price": float(filtered_df['price'].mean()) if 'price' in filtered_df.columns else None,
                "total_revenue": float(filtered_df['estimated_annual_revenue'].sum()) if 'estimated_annual_revenue' in filtered_df.columns else None,
                "avg_occupancy": float(filtered_df['estimated_occupancy_rate'].mean()) if 'estimated_occupancy_rate' in filtered_df.columns else None
            })

if __name__ == "__main__":
    main()
