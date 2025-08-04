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

# Load and clean data function
@st.cache_data
def load_data():
    try:
        # Load file CSV
        df = pd.read_csv('airbnb_features.csv')
        
        # Data cleaning
        df = clean_data(df)
        return df
        
    except FileNotFoundError:
        st.error("File 'airbnb_features.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan script ini.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat membaca file CSV: {str(e)}")
        st.stop()

def clean_data(df):
    """Clean and prepare data for analysis"""
    try:
        # Clean numeric columns
        numeric_columns = ['price', 'minimum_nights', 'number_of_reviews', 
                          'availability_365', 'estimated_annual_revenue', 'estimated_occupancy_rate']
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to string first, then clean
                df[col] = df[col].astype(str)
                
                # Remove problematic characters
                df[col] = df[col].str.replace('{', '', regex=False)
                df[col] = df[col].str.replace('}', '', regex=False)
                df[col] = df[col].str.replace('x', '', regex=False)
                df[col] = df[col].str.replace('[^0-9.-]', '', regex=True)
                
                # Replace empty strings with NaN
                df[col] = df[col].replace('', np.nan)
                
                # Convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill NaN values with median for numeric columns
                if col in ['price', 'estimated_annual_revenue']:
                    # For price and revenue, use median
                    df[col] = df[col].fillna(df[col].median())
                elif col == 'estimated_occupancy_rate':
                    # For occupancy rate, use mean
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    # For others, use 0
                    df[col] = df[col].fillna(0)
        
        # Clean string columns
        string_columns = ['name', 'host_name', 'neighbourhood_group', 'neighbourhood', 
                         'room_type', 'price_category', 'performance_tier', 'market_position']
        
        for col in string_columns:
            if col in df.columns:
                # Fill NaN values with 'Unknown'
                df[col] = df[col].fillna('Unknown')
                # Remove extra whitespace
                df[col] = df[col].astype(str).str.strip()
        
        # Ensure estimated_occupancy_rate is between 0 and 1
        if 'estimated_occupancy_rate' in df.columns:
            # If values are greater than 1, assume they are percentages (divide by 100)
            df.loc[df['estimated_occupancy_rate'] > 1, 'estimated_occupancy_rate'] = \
                df.loc[df['estimated_occupancy_rate'] > 1, 'estimated_occupancy_rate'] / 100
            
            # Clip values to be between 0 and 1
            df['estimated_occupancy_rate'] = df['estimated_occupancy_rate'].clip(0, 1)
        
        # Ensure price is positive
        if 'price' in df.columns:
            df['price'] = df['price'].abs()
            df.loc[df['price'] == 0, 'price'] = df['price'].median()
        
        return df
        
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return df

def safe_calculate_mean(series):
    """Safely calculate mean, handling any remaining issues"""
    try:
        # Remove any non-numeric values
        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.mean()
    except:
        return 0

def safe_calculate_sum(series):
    """Safely calculate sum, handling any remaining issues"""
    try:
        # Remove any non-numeric values
        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.sum()
    except:
        return 0

def create_summary_table(data, group_col, value_col, title):
    """Create summary table for groupby analysis"""
    try:
        if value_col and value_col in data.columns:
            summary = data.groupby(group_col).agg({
                value_col: ['count', safe_calculate_mean, safe_calculate_sum, 'min', 'max']
            }).round(2)
            summary.columns = ['Count', f'Avg {value_col}', f'Total {value_col}', f'Min {value_col}', f'Max {value_col}']
        else:
            summary = data[group_col].value_counts().to_frame()
            summary.columns = ['Count']
            summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(1)
        
        st.subheader(title)
        st.dataframe(summary, use_container_width=True)
        return summary
    except Exception as e:
        st.error(f"Error creating summary table: {str(e)}")
        return pd.DataFrame()

def display_chart_using_streamlit(data, x_col, y_col=None, chart_type='bar'):
    """Create charts using Streamlit's built-in chart functions"""
    try:
        if chart_type == 'bar' and y_col and y_col in data.columns:
            # Group data for bar chart
            grouped_data = data.groupby(x_col)[y_col].apply(safe_calculate_mean).reset_index()
            if not grouped_data.empty:
                st.bar_chart(grouped_data.set_index(x_col)[y_col])
        elif chart_type == 'bar':
            # Count data for bar chart
            count_data = data[x_col].value_counts()
            if not count_data.empty:
                st.bar_chart(count_data)
        elif chart_type == 'line' and y_col and y_col in data.columns:
            # Line chart
            grouped_data = data.groupby(x_col)[y_col].apply(safe_calculate_mean).reset_index()
            if not grouped_data.empty:
                st.line_chart(grouped_data.set_index(x_col)[y_col])
        elif chart_type == 'scatter' and y_col and y_col in data.columns:
            # For scatter plot, we'll show a sample of data points
            sample_data = data.sample(min(100, len(data)))[[x_col, y_col]]
            # Remove any non-numeric values
            sample_data = sample_data.dropna()
            if not sample_data.empty:
                st.scatter_chart(sample_data.set_index(x_col)[y_col])
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")

def show_statistical_summary(data, columns):
    """Display statistical summary for numeric columns"""
    st.subheader("Statistical Summary")
    
    try:
        # Clean the data first
        clean_data_for_stats = pd.DataFrame()
        for col in columns:
            if col in data.columns:
                clean_series = pd.to_numeric(data[col], errors='coerce')
                clean_data_for_stats[col] = clean_series
        
        if not clean_data_for_stats.empty:
            summary_stats = clean_data_for_stats.describe().round(2)
            st.dataframe(summary_stats, use_container_width=True)
        else:
            st.info("No valid numeric columns available for statistical summary")
    except Exception as e:
        st.error(f"Error creating statistical summary: {str(e)}")

def main():
    # Header
    st.markdown('<h1 class="main-header">Airbnb Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("Dataset kosong atau tidak dapat dibaca")
        return
    
    # Show data info
    st.sidebar.success(f"Dataset berhasil dimuat: {len(df)} baris, {len(df.columns)} kolom")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Neighbourhood filter
    if 'neighbourhood_group' in df.columns:
        unique_neighbourhoods = df['neighbourhood_group'].dropna().unique()
        if len(unique_neighbourhoods) > 0:
            selected_neighbourhoods = st.sidebar.multiselect(
                "Select Neighbourhood Groups",
                options=unique_neighbourhoods,
                default=unique_neighbourhoods
            )
        else:
            selected_neighbourhoods = None
    else:
        selected_neighbourhoods = None
    
    # Room type filter
    if 'room_type' in df.columns:
        unique_room_types = df['room_type'].dropna().unique()
        if len(unique_room_types) > 0:
            selected_room_types = st.sidebar.multiselect(
                "Select Room Types",
                options=unique_room_types,
                default=unique_room_types
            )
        else:
            selected_room_types = None
    else:
        selected_room_types = None
    
    # Price range filter
    if 'price' in df.columns:
        price_min = float(df['price'].min())
        price_max = float(df['price'].max())
        if price_min != price_max and not (np.isnan(price_min) or np.isnan(price_max)):
            price_range = st.sidebar.slider(
                "Price Range ($)",
                min_value=price_min,
                max_value=price_max,
                value=(price_min, price_max)
            )
        else:
            price_range = None
    else:
        price_range = None
    
    # Filter data
    filtered_df = df.copy()
    
    try:
        if selected_neighbourhoods and 'neighbourhood_group' in df.columns:
            filtered_df = filtered_df[filtered_df['neighbourhood_group'].isin(selected_neighbourhoods)]
        
        if selected_room_types and 'room_type' in df.columns:
            filtered_df = filtered_df[filtered_df['room_type'].isin(selected_room_types)]
        
        if price_range and 'price' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['price'] >= price_range[0]) &
                (filtered_df['price'] <= price_range[1])
            ]
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        filtered_df = df.copy()
    
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
            avg_price = safe_calculate_mean(filtered_df['price'])
            overall_avg = safe_calculate_mean(df['price'])
            st.metric(
                label="Average Price",
                value=f"${avg_price:.0f}",
                delta=f"${avg_price - overall_avg:.0f} vs overall"
            )
        else:
            st.metric(label="Average Price", value="N/A")
    
    with col3:
        if 'estimated_annual_revenue' in filtered_df.columns:
            total_revenue = safe_calculate_sum(filtered_df['estimated_annual_revenue'])
            st.metric(
                label="Total Est. Revenue",
                value=f"${total_revenue:,.0f}"
            )
        else:
            st.metric(label="Total Est. Revenue", value="N/A")
    
    with col4:
        if 'estimated_occupancy_rate' in filtered_df.columns:
            avg_occupancy = safe_calculate_mean(filtered_df['estimated_occupancy_rate'])
            st.metric(
                label="Avg Occupancy Rate",
                value=f"{avg_occupancy:.1%}"
            )
        else:
            st.metric(label="Avg Occupancy Rate", value="N/A")
    
    with col5:
        if 'number_of_reviews' in filtered_df.columns:
            avg_reviews = safe_calculate_mean(filtered_df['number_of_reviews'])
            st.metric(
                label="Avg Reviews",
                value=f"{avg_reviews:.0f}"
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
    
    # Additional Analysis
    st.header("Additional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'price' in filtered_df.columns and 'estimated_occupancy_rate' in filtered_df.columns:
            st.subheader("Price vs Occupancy Rate")
            display_chart_using_streamlit(filtered_df, 'price', 'estimated_occupancy_rate', 'scatter')
        else:
            st.info("Price vs occupancy data not available")
    
    with col2:
        if 'availability_365' in filtered_df.columns:
            st.subheader("Availability Distribution")
            # Create bins for availability
            if not filtered_df['availability_365'].isna().all():
                availability_bins = pd.cut(filtered_df['availability_365'], bins=5)
                availability_counts = availability_bins.value_counts().sort_index()
                st.bar_chart(availability_counts)
            else:
                st.info("No availability data to display")
        else:
            st.info("Availability data not available")
    
    # Statistical Summary
    numeric_cols = []
    potential_cols = ['price', 'minimum_nights', 'number_of_reviews', 
                     'availability_365', 'estimated_annual_revenue', 'estimated_occupancy_rate']
    
    for col in potential_cols:
        if col in filtered_df.columns:
            numeric_cols.append(col)
    
    if numeric_cols:
        show_statistical_summary(filtered_df, numeric_cols)
    
    # Top performers table
    st.header("Top Performing Properties")
    
    if 'estimated_annual_revenue' in filtered_df.columns:
        try:
            display_cols = []
            for col in ['name', 'neighbourhood_group', 'room_type', 'price', 
                       'estimated_annual_revenue', 'estimated_occupancy_rate', 'number_of_reviews']:
                if col in filtered_df.columns:
                    display_cols.append(col)
            
            if display_cols:
                # Clean the revenue column for sorting
                revenue_clean = pd.to_numeric(filtered_df['estimated_annual_revenue'], errors='coerce')
                top_indices = revenue_clean.nlargest(10).index
                top_performers = filtered_df.loc[top_indices, display_cols]
                st.dataframe(top_performers, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating top performers table: {str(e)}")
    else:
        st.info("Revenue data not available for ranking")
    
    # Data Preview
    st.header("Data Preview")
    st.subheader("Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(filtered_df))
    with col2:
        st.metric("Total Columns", len(filtered_df.columns))
    with col3:
        memory_usage = filtered_df.memory_usage(deep=True).sum() / 1024
        st.metric("Memory Usage", f"{memory_usage:.1f} KB")
    
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
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Data Export
    st.header("Data Export")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"airbnb_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")
    
    with col2:
        st.info(f"Showing {len(filtered_df)} properties out of {len(df)} total properties")

if __name__ == "__main__":
    main()
