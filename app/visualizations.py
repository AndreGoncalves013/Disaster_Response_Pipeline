import pandas as pd
import plotly.express as px

def format_category_text(category):
    return category.capitalize().replace('_', ' ')


def create_occurrency_category_dataframe(df):

    categories_names = df.columns[4:]

    category_mean_df = df[categories_names].mean(axis=0).reset_index()
    
    category_mean_df = category_mean_df\
        .rename(columns={'index':'category', 0:'percentage'})
    
    category_mean_df['category'] = category_mean_df['category']\
        .apply(format_category_text)
    
    category_mean_df = category_mean_df.sort_values(by='percentage')

    return category_mean_df

def create_percentage_occurency_per_category_chart(df):
    

    category_mean_df = create_occurrency_category_dataframe(df)
    
    fig = px.bar(
        category_mean_df, 
        x="percentage", 
        y="category", 
        orientation='h'
    )
    
    fig.update_layout(
        title={
            'text': 'Percentage of occurrences by category',
            'xanchor': 'center',
            'yanchor': 'top',
            'x':0.5
        }
    )

    return fig


def create_category_correlation_dataframe(df):

    categories_names = df.columns[4:]

    categories_df = df[categories_names]
    categories_df = categories_df.sum(axis=0).reset_index()
    categories_df = categories_df.rename(columns={'index':'category', 0:'occurrences'})
    
    most_common_categories_df = categories_df.nlargest(10, 'occurrences')
    most_common_categories = most_common_categories_df.category.values
    
    heatmap_df = df[most_common_categories].copy()
    heatmap_df = heatmap_df.rename(columns={col: format_category_text(col) for col in heatmap_df.columns})
    
    corr_df = heatmap_df.corr().fillna(0)

    return corr_df


def create_category_correlation_heatmap_chart(df):

    corr_df = create_category_correlation_dataframe(df)
    
    fig = px.imshow(
        corr_df, 
        color_continuous_scale='BuPu'
    )

    fig.update_layout(
        title={
            'text': "Categories correlation heatmap",
            'xanchor': 'center',
            'yanchor': 'top',
            'x':0.5
        }
    )
    
    return fig