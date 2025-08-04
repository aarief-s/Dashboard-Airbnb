import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
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

def create_bar_chart(data, x_col, y_col, title):
    """Create a simple bar chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if y_col:
        # Grouped bar chart
        grouped_data = data.groupby(x_col)[y_col].mean()
        bars = ax.bar(grouped_data.index, grouped_data.values, color='#FF5A5F', alpha=0.7)
        ax.set_ylabel(y_col.replace('_', ' ').title())
    else:
        # Count bar chart
        counts = data[x_col].value_counts()
        bars = ax.bar(counts.index, counts.values, color='#FF5A5F', alpha=0.7)
        ax.set_ylabel('Count')
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_pie_chart(data, column, title):
    """Create a pie chart using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    counts = data[column].value_counts()
    colors = ['#FF5A5F', '#00A699', '#FC642D', '#484848', '#767676']
    
    wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                                     colors=colors[:len(counts)], startangle=90)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    return fig

def create_scatter_plot(data, x_col, y_col, title):
    """Create a scatter plot using matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(data[x_col], data[y_col], alpha=0.6, color='#FF5A5F')
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    ax.set_title(title)
    
    # Add trend line
    z = np.polyfit(data[x_col], data[y_col], 1)
    p = np.poly1d(z)
    ax.plot(data[x_col], p(data[x_col]), "r--", alpha=0.8)
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(data, numeric_cols):
    """Create correlation heatmap using seaborn"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    correlation_matrix = data[numeric_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0,
                square=True, ax=ax, cbar_kws={"shrink": .8})
    
    ax.set_title('Correlation Matrix of Key Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

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
    
    # Charts Row 1
    st.header("Market Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'neighbourhood_group' in filtered_df.columns and 'price' in filtered_df.columns:
            fig = create_bar_chart(filtered_df, 'neighbourhood_group', 'price', 
                                 'Average Price by Neighbourhood Group')
            st.pyplot(fig)
        else:
            st.info("Price by neighbourhood data not available")
    
    with col2:
        if 'room_type' in filtered_df.columns:
            fig = create_pie_chart(filtered_df, 'room_type', 'Room Type Distribution')
            st.pyplot(fig)
        else:
            st.info("Room type data not available")
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        if 'performance_tier' in filtered_df.columns and 'estimated_annual_revenue' in filtered_df.columns:
            fig = create_bar_chart(filtered_df, 'performance_tier', 'estimated_annual_revenue',
                                 'Average Revenue by Performance Tier')
            st.pyplot(fig)
        else:
            st.info("Performance tier data not available")
    
    with col2:
        if 'price' in filtered_df.columns and 'estimated_occupancy_rate' in filtered_df.columns:
            fig = create_scatter_plot(filtered_df, 'price', 'estimated_occupancy_rate',
                                    'Occupancy Rate vs Price')
            st.pyplot(fig)
        else:
            st.info("Price vs occupancy data not available")
    
    # Performance Analysis
    st.header("Performance Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'performance_tier' in filtered_df.columns:
            fig = create_bar_chart(filtered_df, 'performance_tier', None,
                                 'Performance Tier Distribution')
            st.pyplot(fig)
    
    with col2:
        if 'market_position' in filtered_df.columns:
            fig = create_bar_chart(filtered_df, 'market_position', None,
                                 'Market Position Distribution')
            st.pyplot(fig)
    
    with col3:
        if 'price_category' in filtered_df.columns:
            fig = create_bar_chart(filtered_df, 'price_category', None,
                                 'Price Category Distribution')
            st.pyplot(fig)
    
    # Correlation Analysis
    st.header("Detailed Analysis")
    
    numeric_cols = []
    potential_cols = ['price', 'minimum_nights', 'number_of_reviews', 
                     'availability_365', 'estimated_annual_revenue', 'estimated_occupancy_rate']
    
    for col in potential_cols:
        if col in filtered_df.columns:
            numeric_cols.append(col)
    
    if len(numeric_cols) > 1:
        fig = create_correlation_heatmap(filtered_df, numeric_cols)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation analysis")
    
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
    
    # Raw Data Preview
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
        'Null Count': filtered_df.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
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

if __name__ == "__main__":
    main()

