import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    return pd.read_csv('airbnb_features.csv')
    np.random.seed(42)
    
    neighbourhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    room_types = ['Entire home/apt', 'Private room', 'Shared room']
    price_categories = ['Budget', 'Mid-range', 'Luxury']
    
    data = {
        'id': range(1, 501),
        'name': [f'Property {i}' for i in range(1, 501)],
        'host_id': np.random.randint(1000, 9999, 500),
        'host_name': [f'Host {i}' for i in range(1, 501)],
        'neighbourhood_group': np.random.choice(neighbourhoods, 500),
        'neighbourhood': [f'Area {i}' for i in range(1, 501)],
        'room_type': np.random.choice(room_types, 500),
        'price': np.random.randint(50, 500, 500),
        'minimum_nights': np.random.randint(1, 30, 500),
        'number_of_reviews': np.random.randint(0, 300, 500),
        'availability_365': np.random.randint(0, 365, 500),
        'price_category': np.random.choice(price_categories, 500),
        'estimated_annual_revenue': np.random.randint(10000, 100000, 500),
        'estimated_occupancy_rate': np.random.uniform(0.3, 0.9, 500),
        'performance_tier': np.random.choice(['Low', 'Medium', 'High'], 500),
        'market_position': np.random.choice(['Niche', 'Mainstream', 'Premium'], 500)
    }
    
    return pd.DataFrame(data)

def main():
    # Header
    st.markdown('<h1 class="main-header"> Airbnb Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header(" Filters")
    
    # Neighbourhood filter
    selected_neighbourhoods = st.sidebar.multiselect(
        "Select Neighbourhood Groups",
        options=df['neighbourhood_group'].unique(),
        default=df['neighbourhood_group'].unique()
    )
    
    # Room type filter
    selected_room_types = st.sidebar.multiselect(
        "Select Room Types",
        options=df['room_type'].unique(),
        default=df['room_type'].unique()
    )
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['neighbourhood_group'].isin(selected_neighbourhoods)) &
        (df['room_type'].isin(selected_room_types)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ]
    
    # Key Metrics
    st.header(" Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Properties",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)} from total"
        )
    
    with col2:
        avg_price = filtered_df['price'].mean()
        st.metric(
            label="Average Price",
            value=f"${avg_price:.0f}",
            delta=f"${avg_price - df['price'].mean():.0f} vs overall"
        )
    
    with col3:
        total_revenue = filtered_df['estimated_annual_revenue'].sum()
        st.metric(
            label="Total Est. Revenue",
            value=f"${total_revenue:,.0f}",
        )
    
    with col4:
        avg_occupancy = filtered_df['estimated_occupancy_rate'].mean()
        st.metric(
            label="Avg Occupancy Rate",
            value=f"{avg_occupancy:.1%}",
        )
    
    with col5:
        avg_reviews = filtered_df['number_of_reviews'].mean()
        st.metric(
            label="Avg Reviews",
            value=f"{avg_reviews:.0f}",
        )
    
    # Charts Row 1
    st.header(" Market Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by neighbourhood
        fig_price_dist = px.box(
            filtered_df, 
            x='neighbourhood_group', 
            y='price',
            title="Price Distribution by Neighbourhood Group",
            color='neighbourhood_group'
        )
        fig_price_dist.update_layout(height=400)
        st.plotly_chart(fig_price_dist, use_container_width=True)
    
    with col2:
        # Room type distribution
        room_type_counts = filtered_df['room_type'].value_counts()
        fig_room_type = px.pie(
            values=room_type_counts.values,
            names=room_type_counts.index,
            title="Room Type Distribution"
        )
        fig_room_type.update_layout(height=400)
        st.plotly_chart(fig_room_type, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by performance tier
        revenue_by_tier = filtered_df.groupby('performance_tier')['estimated_annual_revenue'].mean().reset_index()
        fig_revenue_tier = px.bar(
            revenue_by_tier,
            x='performance_tier',
            y='estimated_annual_revenue',
            title="Average Revenue by Performance Tier",
            color='performance_tier'
        )
        fig_revenue_tier.update_layout(height=400)
        st.plotly_chart(fig_revenue_tier, use_container_width=True)
    
    with col2:
        # Occupancy vs Price scatter
        fig_scatter = px.scatter(
            filtered_df,
            x='price',
            y='estimated_occupancy_rate',
            color='performance_tier',
            size='number_of_reviews',
            title="Occupancy Rate vs Price",
            hover_data=['neighbourhood_group']
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Performance Analysis
    st.header("Performance Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Performance tier distribution
        perf_counts = filtered_df['performance_tier'].value_counts()
        fig_perf = px.bar(
            x=perf_counts.index,
            y=perf_counts.values,
            title="Performance Tier Distribution",
            color=perf_counts.index
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        # Market position analysis
        market_counts = filtered_df['market_position'].value_counts()
        fig_market = px.bar(
            x=market_counts.index,
            y=market_counts.values,
            title="Market Position Distribution",
            color=market_counts.index
        )
        st.plotly_chart(fig_market, use_container_width=True)
    
    with col3:
        # Price category analysis
        price_cat_counts = filtered_df['price_category'].value_counts()
        fig_price_cat = px.bar(
            x=price_cat_counts.index,
            y=price_cat_counts.values,
            title="Price Category Distribution",
            color=price_cat_counts.index
        )
        st.plotly_chart(fig_price_cat, use_container_width=True)
    
    # Detailed Analysis
    st.header("Detailed Analysis")
    
    # Correlation heatmap
    numeric_cols = ['price', 'minimum_nights', 'number_of_reviews', 
                   'availability_365', 'estimated_annual_revenue', 'estimated_occupancy_rate']
    correlation_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Correlation Matrix of Key Metrics",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Top performers table
    st.subheader("Top Performing Properties")
    top_performers = filtered_df.nlargest(10, 'estimated_annual_revenue')[
        ['name', 'neighbourhood_group', 'room_type', 'price', 
         'estimated_annual_revenue', 'estimated_occupancy_rate', 'number_of_reviews']
    ]
    st.dataframe(top_performers, use_container_width=True)
    
    # Data Export
    st.header("Data Export")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="airbnb_filtered_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info(f"Showing {len(filtered_df)} properties out of {len(df)} total properties")

if __name__ == "__main__":
    main()