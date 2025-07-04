import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🤖 LLM-Powered Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .trending-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: 600;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data files"""
    try:
        # Load main analysis
        analysis_df = pd.read_csv('outputs/feedback_analysis.csv')
        
        # Load clusters
        clusters_df = pd.read_csv('outputs/category_clusters.csv')
        
        # Load insights
        with open('outputs/insights_summary.json', 'r') as f:
            insights = json.load(f)
        
        return analysis_df, clusters_df, insights
    except FileNotFoundError:
        st.error("❌ Data files not found! Please run 'analyze_feedback.py' first.")
        return None, None, None

def get_cluster_for_category(category, clusters_df):
    """Find which cluster a category belongs to"""
    for _, cluster in clusters_df.iterrows():
        if category in cluster['categories'].split(','):
            return cluster['cluster_name']
    return 'Other'

def main():
    st.title("🤖 LLM-Powered Sentiment Analysis Dashboard")
    st.markdown("*Automated feedback analysis with OpenAI and intelligent clustering*")
    
    # Load data
    analysis_df, clusters_df, insights = load_data()
    
    if analysis_df is None:
        st.stop()
    
    # Add cluster information to analysis_df
    analysis_df['cluster'] = analysis_df['category'].apply(
        lambda x: get_cluster_for_category(x, clusters_df)
    )
    
    # Overview metrics
    st.header("📊 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Feedback",
            value=len(analysis_df),
            delta=f"Analyzed: {insights['analysis_date'][:10]}"
        )
    
    with col2:
        avg_rating = insights['average_rating']
        st.metric(
            label="Average Rating",
            value=f"{avg_rating}/5",
            delta="⭐" * int(round(avg_rating))
        )
    
    with col3:
        avg_severity = insights['average_severity']
        st.metric(
            label="Average Severity",
            value=f"{avg_severity}/5",
            delta="🚨" if avg_severity >= 4 else "⚠️" if avg_severity >= 3 else "✅"
        )
    
    with col4:
        total_sentiment = sum(insights['sentiment_distribution'].values())
        negative_pct = (insights['sentiment_distribution'].get('negative', 0) / total_sentiment * 100) if total_sentiment > 0 else 0
        st.metric(
            label="Negative Sentiment",
            value=f"{negative_pct:.1f}%",
            delta="🔴 High" if negative_pct > 60 else "🟡 Medium" if negative_pct > 30 else "🟢 Low"
        )
    
    # Trending Issues Alert
    if insights['trending_issues']:
        st.header("🚨 Trending Issues Alert")
        for issue in insights['trending_issues']:
            issue_cluster = get_cluster_for_category(issue['category'], clusters_df)
            st.error(f"**{issue['category']}** ({issue_cluster}) - {issue['count']} recent reports (Severity: {issue['severity']:.1f})")
    
    # Analysis Charts
    st.header("📈 Analysis Charts")
    
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution
        sentiment_counts = analysis_df['sentiment'].value_counts()
        fig_sentiment = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#00c851',
                'negative': '#ff4444',
                'neutral': '#ffbb33'
            }
        )
        fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Issue Clusters
        cluster_counts = analysis_df['cluster'].value_counts().head(8)
        fig_cluster = px.bar(
            x=cluster_counts.values,
            y=cluster_counts.index,
            orientation='h',
            title=f"Issue Clusters (Filtered: {len(analysis_df)} feedback)",
            labels={'x': 'Count', 'y': 'Cluster'},
            color=cluster_counts.values,
            color_continuous_scale='viridis'
        )
        fig_cluster.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top LLM Categories
        category_counts = analysis_df['category'].value_counts().head(10)
        fig_category = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title=f"Top LLM Categories (Filtered: {len(analysis_df)} feedback)",
            labels={'x': 'Count', 'y': 'Category'},
            color=category_counts.values,
            color_continuous_scale='blues'
        )
        fig_category.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Severity vs Sentiment by Cluster
        fig_scatter = px.scatter(
            analysis_df,
            x='severity',
            y='sentiment_score',
            color='cluster',
            size='feedback_rating',
            hover_data=['category', 'feedback_text'],
            title="Severity vs Sentiment by Cluster",
            labels={'severity': 'Severity (1-5)', 'sentiment_score': 'Sentiment Score (-1 to 1)'}
        )
        fig_scatter.update_layout(showlegend=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # LLM-Generated Category Clusters Analysis
    st.header("🔗 LLM-Generated Category Clusters")
    
    # Show cluster overview table
    st.subheader("📊 Cluster Overview")
    cluster_summary = []
    for _, cluster in clusters_df.iterrows():
        cluster_feedback = analysis_df[analysis_df['cluster'] == cluster['cluster_name']]
        if len(cluster_feedback) > 0:
            avg_sentiment_score = cluster_feedback['sentiment_score'].mean()
            neg_pct = (cluster_feedback['sentiment'] == 'negative').mean() * 100
            avg_severity = cluster_feedback['severity'].mean()
            avg_rating = cluster_feedback['feedback_rating'].mean()
            
            cluster_summary.append({
                'Cluster': cluster['cluster_name'],
                'Feedback Count': cluster['count'],
                'Avg Rating': f"{avg_rating:.1f}/5",
                'Avg Sentiment Score': f"{avg_sentiment_score:.2f}",
                'Negative %': f"{neg_pct:.1f}%",
                'Avg Severity': f"{avg_severity:.1f}/5"
            })
    
    if cluster_summary:
        summary_df = pd.DataFrame(cluster_summary)
        st.dataframe(
            summary_df, 
            use_container_width=True,
            column_config={
                "Cluster": st.column_config.TextColumn("Cluster Name", width="large"),
                "Feedback Count": st.column_config.NumberColumn("Count"),
                "Avg Rating": st.column_config.TextColumn("Avg Rating"),
                "Avg Sentiment Score": st.column_config.TextColumn("Sentiment Score"),
                "Negative %": st.column_config.TextColumn("Negative %"),
                "Avg Severity": st.column_config.TextColumn("Severity")
            }
        )
    
    # Detailed cluster analysis
    st.subheader("🔍 Detailed Cluster Analysis")
    
    for _, cluster in clusters_df.iterrows():
        with st.expander(f"🎯 {cluster['cluster_name']} ({cluster['count']} feedback)", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {cluster['description']}")
                st.markdown(f"**LLM-Generated Categories:** `{cluster['categories'].replace(',', '`, `')}`")
                
                # Show category breakdown within cluster
                cluster_categories = cluster['categories'].split(',')
                cluster_feedback = analysis_df[analysis_df['category'].isin(cluster_categories)]
                
                if not cluster_feedback.empty:
                    category_breakdown = cluster_feedback['category'].value_counts()
                    st.markdown("**Category Breakdown:**")
                    for cat, count in category_breakdown.items():
                        st.markdown(f"  • {cat}: {count} feedback")
            
            with col2:
                if not cluster_feedback.empty:
                    # Cluster metrics
                    cluster_avg_rating = cluster_feedback['feedback_rating'].mean()
                    cluster_neg_pct = (cluster_feedback['sentiment'] == 'negative').mean() * 100
                    cluster_avg_severity = cluster_feedback['severity'].mean()
                    
                    st.metric("Avg Rating", f"{cluster_avg_rating:.1f}/5")
                    st.metric("Negative %", f"{cluster_neg_pct:.1f}%")
                    st.metric("Avg Severity", f"{cluster_avg_severity:.1f}/5")
            
            # Sample feedback
            if not cluster_feedback.empty:
                st.markdown("**Sample Feedback:**")
                for _, row in cluster_feedback.head(3).iterrows():
                    sentiment_color = "#00c851" if row['sentiment'] == 'positive' else "#ff4444" if row['sentiment'] == 'negative' else "#ffbb33"
                    st.markdown(f"""
                    <div style="border-left: 3px solid {sentiment_color}; padding-left: 10px; margin: 5px 0; background-color: #f8f9fa; border-radius: 3px; padding: 10px;">
                        <strong>{row['feedback_user']}</strong> ({row['feedback_rating']}/5) - 
                        <span style="color: {sentiment_color}; font-weight: bold;">{row['sentiment'].upper()}</span> - 
                        <span style="background-color: #e9ecef; padding: 2px 6px; border-radius: 10px; font-size: 0.8em;">{row['category']}</span><br>
                        <em>"{row['feedback_text']}"</em><br>
                        <small><strong>Keywords:</strong> {row['keywords']}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # AI-Generated Recommendations
    st.header("💡 AI-Generated Recommendations")
    
    for recommendation in insights['recommendations']:
        st.info(recommendation)
    
    # Cluster-Based Insights
    st.subheader("🎯 Cluster-Based Insights")
    
    # Find most problematic cluster
    cluster_problems = []
    for _, cluster in clusters_df.iterrows():
        cluster_feedback = analysis_df[analysis_df['cluster'] == cluster['cluster_name']]
        if len(cluster_feedback) > 0:
            neg_pct = (cluster_feedback['sentiment'] == 'negative').mean() * 100
            avg_severity = cluster_feedback['severity'].mean()
            problem_score = (neg_pct / 100) * avg_severity
            cluster_problems.append((cluster['cluster_name'], problem_score, neg_pct, avg_severity, len(cluster_feedback)))
    
    cluster_problems.sort(key=lambda x: x[1], reverse=True)
    
    if cluster_problems:
        most_problematic = cluster_problems[0]
        st.warning(f"🚨 **Most Problematic Cluster:** {most_problematic[0]} - {most_problematic[2]:.1f}% negative sentiment, {most_problematic[3]:.1f} avg severity ({most_problematic[4]} feedback)")
        
        if len(cluster_problems) > 1:
            least_problematic = cluster_problems[-1]
            st.success(f"✅ **Best Performing Cluster:** {least_problematic[0]} - {least_problematic[2]:.1f}% negative sentiment, {least_problematic[3]:.1f} avg severity ({least_problematic[4]} feedback)")
    
    # Detailed Feedback Table
    st.header("📋 Detailed Feedback Analysis")
    
    # Add search functionality
    search_term = st.text_input("🔍 Search feedback text:")
    display_df = analysis_df.copy()
    
    if search_term:
        display_df = display_df[display_df['feedback_text'].str.contains(search_term, case=False, na=False)]
        st.info(f"Found {len(display_df)} feedback items matching '{search_term}'")
    
    # Display table with cluster information
    if len(display_df) > 0:
        table_df = display_df[['feedback_user', 'feedback_text', 'cluster', 'category', 'sentiment', 'severity', 'feedback_rating', 'keywords']].copy()
        st.dataframe(
            table_df,
            use_container_width=True,
            column_config={
                "feedback_text": st.column_config.TextColumn("Feedback", width="large"),
                "cluster": st.column_config.TextColumn("LLM Cluster", width="medium"),
                "category": st.column_config.TextColumn("LLM Category", width="medium"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "severity": st.column_config.NumberColumn("Severity", min_value=1, max_value=5),
                "feedback_rating": st.column_config.NumberColumn("Rating", min_value=1, max_value=5),
                "keywords": st.column_config.TextColumn("Keywords", width="medium")
            }
        )
    else:
        st.info("No feedback matches the search term")
    
    # Download section
    st.header("💾 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = analysis_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Full Analysis",
            data=csv_data,
            file_name=f"feedback_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        cluster_csv = clusters_df.to_csv(index=False)
        st.download_button(
            label="🔗 Download LLM Clusters",
            data=cluster_csv,
            file_name=f"llm_clusters_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col3:
        insights_json = json.dumps(insights, indent=2)
        st.download_button(
            label="📋 Download AI Insights",
            data=insights_json,
            file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()