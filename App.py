import streamlit as st
import pandas as pd
import requests
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

def is_structured(df):
    """Check if the DataFrame has the required structured columns."""
    required_columns = {'Technology Name', 'Technology Type'}
    return required_columns.issubset(set(df.columns))

def generate_search_queries(df):
    """Generate search queries for Method 1."""
    queries = set()
    for _, row in df.iterrows():
        if len(queries) >= 30:  
            break
        tech_name = str(row['Technology Name']).strip()
        tech_type = str(row['Technology Type']).strip()
        if pd.notna(tech_name) and pd.notna(tech_type):
            query = f"{tech_name} {tech_type} methane mitigation"
            queries.add(query)
    return list(queries)

def search_crossref(query, max_results=30):  
    """Search for research papers using CrossRef API."""
    url = f"https://api.crossref.org/works?query={query}&rows={max_results}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('message', {}).get('items', [])
    else:
        st.warning(f"Failed to fetch results for query: {query}")
        return []

def extract_keywords(text, api_key, max_keywords=30):  
    """Extract keywords using OpenAI LLM for Method 2."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for extracting keywords from text. Please provide exactly 10 keywords or fewer."},
            {"role": "user", "content": f"Extract keywords or phrases related to methane mitigation technologies in the oil and gas sector from the following text:\n{text}"}
        ]
    )
    keywords = [keyword.strip() for keyword in response.choices[0].message.content.split(',')]
    return keywords[:max_keywords]  

def make_clickable(url):
    """Make a URL clickable in Streamlit."""
    if pd.isna(url) or not url:
        return ""
    return f'<a href="{url}" target="_blank">Link</a>'

def run_method_1(df):
    """Execute Method 1 for structured files."""
    queries = generate_search_queries(df)
    results = []
    seen_titles = set()
    
    for query in queries:
        if len(results) >= 30:  
            break
        papers = search_crossref(query)
        for paper in papers:
            if len(results) >= 30:  
                break
            title = paper.get('title', [''])[0]
            if title not in seen_titles:
                seen_titles.add(title)
                results.append({
                    'Title': title,
                    'Authors': ', '.join([author.get('family', '') for author in paper.get('author', [])]),
                    'DOI': paper.get('DOI', ''),
                    'URL': paper.get('URL', '')
                })
    return pd.DataFrame(results)

def run_method_2(df, api_key):
    """Execute Method 2 for unstructured files."""
    combined_text = ' '.join(df.fillna('').astype(str).values.flatten())
    keywords = extract_keywords(combined_text, api_key)
    results = []
    seen_titles = set()
    
    for keyword in keywords:
        if len(results) >= 30:  
            break
        papers = search_crossref(keyword.strip())
        for paper in papers:
            if len(results) >= 30:  
                break
            title = paper.get('title', [''])[0]
            if title not in seen_titles:
                seen_titles.add(title)
                results.append({
                    'Title': title,
                    'Authors': ', '.join([author.get('family', '') for author in paper.get('author', [])]),
                    'DOI': paper.get('DOI', ''),
                    'URL': paper.get('URL', '')
                })
    return pd.DataFrame(results)

# Streamlit UI
st.title("Methane Mitigation Literature Finder")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

api_key = os.getenv("OPENAI_API_KEY")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if is_structured(df):
        results_df = run_method_1(df)
    else:
        if not api_key:
            st.error("OpenAI API key is not set in the environment.")
        else:
            results_df = run_method_2(df, api_key)

    if 'results_df' in locals() and not results_df.empty:
        st.subheader("Search Results")
        
        results_df['URL'] = results_df['URL'].apply(make_clickable)
        
        st.write(results_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        csv = results_df.copy()
        csv['URL'] = csv['URL'].apply(lambda x: x.split('"')[1] if pd.notna(x) and x else "")  # Extract original URL from HTML
        csv_data = csv.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv_data, file_name="literature_results.csv", mime="text/csv")
    elif 'results_df' in locals():
        st.warning("No results found.")