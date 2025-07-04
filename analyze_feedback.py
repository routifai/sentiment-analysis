import psycopg2
import pandas as pd
import openai
import instructor
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time
from config import DB_CONFIG, OPENAI_API_KEY, FEEDBACK_TABLE, OUTPUT_DIR, TRENDING_THRESHOLD, RECENT_DAYS, MONTHS_BACK
from concurrent.futures import ThreadPoolExecutor, as_completed

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pydantic models for structured LLM output
class FeedbackAnalysis(BaseModel):
    category: str = Field(description="Main category of the feedback (e.g., 'performance', 'ui_design', 'login_issues')")
    subcategory: str = Field(description="Specific subcategory or issue type")
    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
    sentiment_score: float = Field(description="Sentiment score from -1 (very negative) to 1 (very positive)")
    keywords: List[str] = Field(description="Key terms and phrases from the feedback")
    severity: int = Field(description="Issue severity from 1 (minor) to 5 (critical)")

class CategoryCluster(BaseModel):
    cluster_name: str = Field(description="Descriptive name for this cluster of similar categories")
    categories: List[str] = Field(description="List of categories that belong in this cluster")
    description: str = Field(description="Brief description of what issues this cluster represents")
    reasoning: str = Field(description="Why these categories were grouped together")

class CategoryClusters(BaseModel):
    clusters: List[CategoryCluster] = Field(description="List of category clusters")

def connect_to_database():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to database successfully")
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

def fetch_feedback_data(conn, months_back=3):
    """Fetch feedback data from database with optional date filter"""
    if months_back:
        # Calculate the date X months back from now
        cutoff_date = datetime.now() - timedelta(days=months_back * 30)
        query = f"""
        SELECT 
            message_id,
            feedback_user,
            feedback_text,
            feedback_timestamp,
            feedback_rating,
            feedback_section
        FROM {FEEDBACK_TABLE}
        WHERE feedback_timestamp >= %s
        ORDER BY feedback_timestamp DESC
        """
        try:
            df = pd.read_sql_query(query, conn, params=[cutoff_date])
            print(f"✅ Fetched {len(df)} feedback entries from the last {months_back} months")
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    else:
        # Fetch all data
        query = f"""
        SELECT 
            message_id,
            feedback_user,
            feedback_text,
            feedback_timestamp,
            feedback_rating,
            feedback_section
        FROM {FEEDBACK_TABLE}
        ORDER BY feedback_timestamp DESC
        """
        try:
            df = pd.read_sql_query(query, conn)
            print(f"✅ Fetched {len(df)} feedback entries (all time)")
            return df
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

def analyze_single_feedback(client, feedback_text, rating):
    """Analyze single feedback using OpenAI + Instructor"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=FeedbackAnalysis,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at analyzing user feedback. 
                    Categorize the feedback, determine sentiment, and extract key insights.
                    Be specific with categories - use descriptive names like 'slow_loading', 'login_errors', 'ui_confusing' etc."""
                },
                {
                    "role": "user",
                    "content": f"Feedback: '{feedback_text}' | Rating: {rating}/5"
                }
            ]
        )
        return response
    except Exception as e:
        print(f"❌ Error analyzing feedback: {e}")
        return None

def process_feedback_with_llm(df):
    """Process all feedback through LLM analysis in parallel"""
    client = instructor.patch(openai.OpenAI(api_key=OPENAI_API_KEY))
    results = []
    total = len(df)
    print(f"🤖 Starting LLM analysis of {total} feedback items (parallel)...")

    def analyze_row(row):
        print(f"Processing: {row['feedback_text'][:50]}...")
        analysis = analyze_single_feedback(
            client,
            row['feedback_text'],
            row['feedback_rating']
        )
        if analysis:
            return {
                'message_id': row['message_id'],
                'feedback_user': row['feedback_user'],
                'feedback_text': row['feedback_text'],
                'feedback_timestamp': row['feedback_timestamp'],
                'feedback_rating': row['feedback_rating'],
                'feedback_section': row['feedback_section'],
                'category': analysis.category,
                'subcategory': analysis.subcategory,
                'sentiment': analysis.sentiment,
                'sentiment_score': analysis.sentiment_score,
                'keywords': ','.join(analysis.keywords),
                'severity': analysis.severity
            }
        return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {executor.submit(analyze_row, row): idx for idx, row in df.iterrows()}
        for future in as_completed(future_to_idx):
            result = future.result()
            if result:
                results.append(result)

    print("✅ LLM analysis completed (parallel)")
    return pd.DataFrame(results)

def create_category_clusters(df, client):
    """Use LLM to intelligently cluster categories based on meaning and context"""
    print("🔗 Creating LLM-powered category clusters...")
    
    categories = df['category'].unique()
    category_counts = df['category'].value_counts().to_dict()
    
    if len(categories) <= 1:
        return pd.DataFrame([{
            'cluster_name': 'General Issues',
            'categories': ','.join(categories),
            'description': 'All feedback categories',
            'count': len(df)
        }])
    
    # Get sample feedback for each category to give LLM context
    category_examples = {}
    for category in categories:
        examples = df[df['category'] == category]['feedback_text'].head(2).tolist()
        category_examples[category] = examples
    
    # Prepare data for LLM
    category_info = []
    for category in categories:
        category_info.append({
            'category': category,
            'count': category_counts[category],
            'examples': category_examples[category]
        })
    
    # Format for LLM
    formatted_categories = []
    for info in category_info:
        examples_text = "; ".join(info['examples'])
        formatted_categories.append(f"- **{info['category']}** ({info['count']} feedback): {examples_text}")
    
    categories_text = "\n".join(formatted_categories)
    
    try:
        # Use LLM to create intelligent clusters
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=CategoryClusters,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at analyzing user feedback categories and grouping them into meaningful clusters.

Your task is to group similar categories together based on:
1. Semantic similarity (e.g., 'performance' and 'slow_loading' are similar)
2. Problem domain (e.g., 'login_issues' and 'authentication' are both auth-related)  
3. User impact (e.g., 'ui_design' and 'navigation' both affect user experience)

Create 3-7 clusters that make business sense. Each cluster should represent a distinct area of concern.
Give clusters clear, actionable names that a product team would understand.

Example good cluster names: "Performance & Speed Issues", "Authentication & Access Problems", "UI/UX Design Issues", "Data Management Issues", "Feature Functionality Problems"""
                },
                {
                    "role": "user",
                    "content": f"""Please analyze these feedback categories and group them into meaningful clusters:

{categories_text}

Group these categories into 3-7 clusters based on their semantic meaning and business impact. Make sure each category is assigned to exactly one cluster."""
                }
            ]
        )
        
        # Convert LLM response to DataFrame
        cluster_results = []
        for cluster in response.clusters:
            # Calculate actual count for this cluster
            cluster_count = df[df['category'].isin(cluster.categories)].shape[0]
            
            cluster_results.append({
                'cluster_name': cluster.cluster_name,
                'categories': ','.join(cluster.categories),
                'description': cluster.description,
                'count': cluster_count
            })
        
        # Sort by count (most feedback first)
        cluster_results.sort(key=lambda x: x['count'], reverse=True)
        
        print(f"✅ LLM created {len(cluster_results)} intelligent clusters")
        return pd.DataFrame(cluster_results)
        
    except Exception as e:
        print(f"❌ Error in LLM clustering: {e}")
        # Fallback to simple grouping
        return simple_fallback_clustering(df)

def simple_fallback_clustering(df):
    """Fallback clustering if LLM fails"""
    print("🔄 Using fallback clustering...")
    
    categories = df['category'].unique()
    
    # Simple keyword-based grouping
    performance_keywords = ['performance', 'slow', 'speed', 'loading', 'lag']
    ui_keywords = ['ui', 'design', 'interface', 'navigation', 'usability']
    auth_keywords = ['login', 'auth', 'access', 'password', 'account']
    data_keywords = ['data', 'sync', 'export', 'import', 'integration']
    feature_keywords = ['feature', 'function', 'tool', 'capability']
    
    clusters = {
        'Performance Issues': [],
        'UI/UX Issues': [],
        'Authentication Issues': [],
        'Data & Integration Issues': [],
        'Feature Issues': [],
        'Other Issues': []
    }
    
    for category in categories:
        category_lower = category.lower()
        assigned = False
        
        if any(keyword in category_lower for keyword in performance_keywords):
            clusters['Performance Issues'].append(category)
            assigned = True
        elif any(keyword in category_lower for keyword in ui_keywords):
            clusters['UI/UX Issues'].append(category)
            assigned = True
        elif any(keyword in category_lower for keyword in auth_keywords):
            clusters['Authentication Issues'].append(category)
            assigned = True
        elif any(keyword in category_lower for keyword in data_keywords):
            clusters['Data & Integration Issues'].append(category)
            assigned = True
        elif any(keyword in category_lower for keyword in feature_keywords):
            clusters['Feature Issues'].append(category)
            assigned = True
        
        if not assigned:
            clusters['Other Issues'].append(category)
    
    # Convert to DataFrame format
    cluster_results = []
    for cluster_name, cats in clusters.items():
        if cats:  # Only include clusters with categories
            cluster_count = df[df['category'].isin(cats)].shape[0]
            cluster_results.append({
                'cluster_name': cluster_name,
                'categories': ','.join(cats),
                'description': f"Issues related to {cluster_name.lower()}",
                'count': cluster_count
            })
    
    cluster_results.sort(key=lambda x: x['count'], reverse=True)
    return pd.DataFrame(cluster_results)

def identify_trending_issues(df):
    """Identify trending issues from recent feedback"""
    print("📈 Identifying trending issues...")
    
    # Get recent feedback (last N days)
    recent_date = datetime.now() - timedelta(days=RECENT_DAYS)
    df['feedback_timestamp'] = pd.to_datetime(df['feedback_timestamp'])
    recent_df = df[df['feedback_timestamp'] >= recent_date]
    
    if recent_df.empty:
        return []
    
    # Count category occurrences in recent feedback
    category_counts = recent_df['category'].value_counts()
    
    # Identify trending (categories appearing multiple times recently)
    trending = []
    for category, count in category_counts.items():
        if count >= TRENDING_THRESHOLD:
            recent_examples = recent_df[recent_df['category'] == category]['feedback_text'].head(3).tolist()
            trending.append({
                'category': category,
                'count': count,
                'examples': recent_examples,
                'severity': recent_df[recent_df['category'] == category]['severity'].mean()
            })
    
    # Sort by count (most frequent first)
    trending.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"✅ Found {len(trending)} trending issues")
    return trending

def generate_insights_summary(df, clusters_df, trending_issues):
    """Generate overall insights summary"""
    print("📊 Generating insights summary...")
    
    total_feedback = len(df)
    sentiment_dist = df['sentiment'].value_counts().to_dict()
    avg_rating = df['feedback_rating'].mean()
    avg_severity = df['severity'].mean()
    
    # Most common categories
    top_categories = df['category'].value_counts().head(5).to_dict()
    
    # Common negative feedback
    negative_feedback = df[df['sentiment'] == 'negative']
    common_negative = negative_feedback['category'].value_counts().head(5).to_dict()
    
    insights = {
        'analysis_date': datetime.now().isoformat(),
        'total_feedback': total_feedback,
        'sentiment_distribution': sentiment_dist,
        'average_rating': round(avg_rating, 2),
        'average_severity': round(avg_severity, 2),
        'top_categories': top_categories,
        'common_negative_issues': common_negative,
        'trending_issues': trending_issues,
        'cluster_summary': clusters_df.to_dict('records'),
        'recommendations': generate_recommendations(df, trending_issues)
    }
    
    print("✅ Insights summary generated")
    return insights

def generate_recommendations(df, trending_issues):
    """Generate actionable recommendations"""
    recommendations = []
    
    # High severity issues
    high_severity = df[df['severity'] >= 4]
    if not high_severity.empty:
        recommendations.append(f"🚨 {len(high_severity)} high-severity issues need immediate attention")
    
    # Trending issues
    if trending_issues:
        for issue in trending_issues[:3]:  # Top 3 trending
            recommendations.append(f"📈 '{issue['category']}' is trending with {issue['count']} recent reports")
    
    # Negative sentiment patterns
    negative_pct = (df['sentiment'] == 'negative').mean() * 100
    if negative_pct > 60:
        recommendations.append(f"⚠️ {negative_pct:.1f}% negative sentiment indicates systemic issues")
    
    # Low ratings
    low_ratings = df[df['feedback_rating'] <= 2]
    if len(low_ratings) > len(df) * 0.3:
        recommendations.append("📉 High percentage of low ratings needs investigation")
    
    return recommendations

def main():
    """Main execution function"""
    print("🚀 Starting Feedback Analysis Pipeline...")
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        return
    
    try:
        # Fetch data
        df = fetch_feedback_data(conn, months_back=MONTHS_BACK)
        if df is None or df.empty:
            print("❌ No data to process")
            return
        
        # Process with LLM
        analyzed_df = process_feedback_with_llm(df)
        
        # Create LLM-powered category clusters (pass the client)
        client = instructor.patch(openai.OpenAI(api_key=OPENAI_API_KEY))
        clusters_df = create_category_clusters(analyzed_df, client)
        
        # Identify trending issues
        trending_issues = identify_trending_issues(analyzed_df)
        
        # Generate insights
        insights = generate_insights_summary(analyzed_df, clusters_df, trending_issues)
        
        # Save outputs
        print("💾 Saving results...")
        
        # Save main analysis
        analyzed_df.to_csv(f'{OUTPUT_DIR}/feedback_analysis.csv', index=False)
        print(f"✅ Saved feedback_analysis.csv ({len(analyzed_df)} rows)")
        
        # Save clusters
        clusters_df.to_csv(f'{OUTPUT_DIR}/category_clusters.csv', index=False)
        print(f"✅ Saved category_clusters.csv ({len(clusters_df)} clusters)")
        
        # Save insights
        with open(f'{OUTPUT_DIR}/insights_summary.json', 'w') as f:
            json.dump(insights, f, indent=2)
        print("✅ Saved insights_summary.json")
        
        print("\n🎉 Analysis complete! Files saved in 'outputs' directory")
        print(f"\n📊 Created {len(clusters_df)} intelligent clusters:")
        for _, cluster in clusters_df.iterrows():
            print(f"  • {cluster['cluster_name']}: {cluster['count']} feedback")
        
        print("\nNext steps:")
        print("1. Run: streamlit run dashboard.py")
        print("2. Or open: index.html in your browser")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()