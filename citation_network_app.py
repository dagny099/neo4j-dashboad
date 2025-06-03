import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import os
from collections import Counter

# Page config
st.set_page_config(
    page_title="Citation Network Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-bottom: 10px;
    }
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📚 Citation Network Explorer")
st.markdown("""
Explore the hidden connections in academic literature through interactive network analysis.
This tool reveals the "intellectual DNA" linking papers across different citation universes.
""")

# Initialize Neo4j connection
@st.cache_resource
def init_neo4j_connection():
    """Initialize Neo4j database connection using environment variables"""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PWD")
    
    if not all([uri, user, password]):
        st.error("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PWD environment variables")
        return None
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test connection
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        return None

# Helper function to run queries
def run_query(driver, query, parameters=None):
    """Execute a Cypher query and return results as DataFrame"""
    with driver.session() as session:
        result = session.run(query, parameters or {})
        return pd.DataFrame([record.data() for record in result])

# Initialize connection
driver = init_neo4j_connection()

if driver:
    # Sidebar navigation
    st.sidebar.header("🔍 Navigation")
    page = st.sidebar.radio(
        "Choose Analysis",
        ["📊 Network Overview", "📄 Paper Explorer", "🔗 Missing Connections", 
         "📈 Network Analytics", "🌐 Interactive Visualization", "💡 Insights Dashboard"]
    )
    
    # 1. NETWORK OVERVIEW
    if page == "📊 Network Overview":
        st.header("Network Overview")
        st.markdown("Get a bird's-eye view of the citation network structure")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Count queries with custom styling
        with st.spinner("Loading network statistics..."):
            # Papers count
            papers_count = run_query(driver, "MATCH (p:Paper) RETURN count(p) as count")
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📄</div>
                    <div class="metric-label">Papers</div>
                    <div class="metric-value">{papers_count.iloc[0]['count']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Authors count
            authors_count = run_query(driver, "MATCH (a:Author) RETURN count(a) as count")
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">👥</div>
                    <div class="metric-label">Authors</div>
                    <div class="metric-value">{authors_count.iloc[0]['count']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Venues count
            venues_count = run_query(driver, "MATCH (v:PubVenue) RETURN count(v) as count")
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📚</div>
                    <div class="metric-label">Venues</div>
                    <div class="metric-value">{venues_count.iloc[0]['count']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Total relationships
            rels_count = run_query(driver, "MATCH ()-[r]->() RETURN count(r) as count")
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">🔗</div>
                    <div class="metric-label">Relationships</div>
                    <div class="metric-value">{rels_count.iloc[0]['count']:,}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Relationship breakdown
        st.subheader("Relationship Types Distribution")
        rel_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        rel_types_df = run_query(driver, rel_types_query)
        
        if not rel_types_df.empty:
            fig = px.bar(rel_types_df, x='relationship_type', y='count',
                        title="Distribution of Relationship Types",
                        labels={'count': 'Number of Relationships', 'relationship_type': 'Type'},
                        color='count', color_continuous_scale='viridis')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Node type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Citation Distribution")
            citation_dist_query = """
            MATCH (p:Paper)
            WHERE p.citationCount IS NOT NULL
            RETURN p.citationCount as citations
            """
            citations_df = run_query(driver, citation_dist_query)
            
            if not citations_df.empty:
                fig = px.histogram(citations_df, x='citations', nbins=50,
                                 title="Paper Citation Count Distribution",
                                 labels={'citations': 'Number of Citations', 'count': 'Number of Papers'})
                fig.update_xaxes(range=[0, np.percentile(citations_df['citations'], 95)])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Publication Years")
            year_dist_query = """
            MATCH (p:Paper)-[:PUB_YEAR]->(y:PubYear)
            RETURN y.year as year, count(p) as papers
            ORDER BY year
            """
            years_df = run_query(driver, year_dist_query)
            
            if not years_df.empty:
                fig = px.line(years_df, x='year', y='papers',
                            title="Papers Published per Year",
                            labels={'papers': 'Number of Papers', 'year': 'Year'})
                st.plotly_chart(fig, use_container_width=True)
    
    # 2. PAPER EXPLORER
    elif page == "📄 Paper Explorer":
        st.header("Paper Explorer")
        st.markdown("Deep dive into individual papers and their connections")
        
        # Paper search
        search_method = st.radio("Search by:", ["Paper ID", "Title Keyword", "Top Cited Papers"])
        
        paper_id = None
        if search_method == "Paper ID":
            paper_id = st.text_input("Enter Paper ID:", placeholder="e.g., 1234567")
        
        elif search_method == "Title Keyword":
            keyword = st.text_input("Enter keyword:", placeholder="e.g., neural network")
            if keyword:
                search_query = """
                MATCH (p:Paper)
                WHERE toLower(p.title) CONTAINS toLower($keyword)
                RETURN p.paperId as id, p.title as title, p.citationCount as citations
                ORDER BY p.citationCount DESC
                LIMIT 10
                """
                results = run_query(driver, search_query, {"keyword": keyword})
                if not results.empty:
                    selected = st.selectbox("Select a paper:", 
                                          results['title'].tolist(),
                                          format_func=lambda x: f"{x[:80]}...")
                    paper_id = results[results['title'] == selected]['id'].iloc[0]
        
        else:  # Top Cited Papers
            top_papers_query = """
            MATCH (p:Paper)
            WHERE p.citationCount IS NOT NULL
            RETURN p.paperId as id, p.title as title, p.citationCount as citations
            ORDER BY p.citationCount DESC
            LIMIT 20
            """
            top_papers = run_query(driver, top_papers_query)
            if not top_papers.empty:
                selected = st.selectbox("Select a paper:", 
                                      top_papers.apply(lambda x: f"{x['title'][:60]}... ({x['citations']} citations)", axis=1).tolist())
                paper_id = top_papers.iloc[top_papers.index[top_papers.apply(lambda x: f"{x['title'][:60]}... ({x['citations']} citations)", axis=1) == selected].tolist()[0]]['id']
        
        if paper_id:
            # Get paper details
            paper_query = """
            MATCH (p:Paper {paperId: $paperId})
            RETURN p
            """
            paper_data = run_query(driver, paper_query, {"paperId": paper_id})
            
            if not paper_data.empty:
                paper = paper_data.iloc[0]['p']
                
                # Display paper info
                st.subheader(f"📄 {paper.get('title', 'Unknown Title')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Citations", paper.get('citationCount', 0))
                col2.metric("References", paper.get('referenceCount', 0))
                col3.metric("Year", paper.get('year', 'Unknown'))
                col4.metric("Open Access", "✅" if paper.get('isOpenAccess', False) else "❌")
                
                # Abstract
                if paper.get('abstract'):
                    with st.expander("Abstract"):
                        st.write(paper['abstract'])
                
                # Authors
                st.subheader("👥 Authors")
                authors_query = """
                MATCH (a:Author)-[:AUTHORED]->(p:Paper {paperId: $paperId})
                RETURN a.authorName as name, a.authorId as id
                """
                authors_df = run_query(driver, authors_query, {"paperId": paper_id})
                if not authors_df.empty:
                    st.dataframe(authors_df, use_container_width=True)
                
                # Related papers
                st.subheader("🔗 Connected Papers")
                
                # Co-authored papers
                coauthored_query = """
                MATCH (p1:Paper {paperId: $paperId})<-[:AUTHORED]-(a:Author)-[:AUTHORED]->(p2:Paper)
                WHERE p1 <> p2
                RETURN DISTINCT p2.title as title, p2.paperId as id, 
                       p2.citationCount as citations, count(a) as shared_authors
                ORDER BY shared_authors DESC, citations DESC
                LIMIT 10
                """
                coauthored_df = run_query(driver, coauthored_query, {"paperId": paper_id})
                
                if not coauthored_df.empty:
                    st.write("**Papers by Same Authors:**")
                    st.dataframe(coauthored_df[['title', 'citations', 'shared_authors']], 
                               use_container_width=True)
                
                # Same venue papers
                venue_query = """
                MATCH (p1:Paper {paperId: $paperId})-[:PUBLISHED_IN]->(v:PubVenue)<-[:PUBLISHED_IN]-(p2:Paper)
                WHERE p1 <> p2
                RETURN p2.title as title, p2.citationCount as citations, v.venue as venue
                ORDER BY p2.citationCount DESC
                LIMIT 10
                """
                venue_df = run_query(driver, venue_query, {"paperId": paper_id})
                
                if not venue_df.empty:
                    st.write(f"**Papers in Same Venue ({venue_df.iloc[0]['venue']}):**")
                    st.dataframe(venue_df[['title', 'citations']], use_container_width=True)
    
    # 3. MISSING CONNECTIONS
    elif page == "🔗 Missing Connections":
        st.header("Missing Connections Analysis")
        st.markdown("Identify papers and nodes with incomplete relationships")
        
        # Papers without authors
        st.subheader("📄 Papers Without Authors")
        count_query = """
        MATCH (p:Paper)
        WHERE NOT EXISTS((p)<-[:AUTHORED]-(:Author))
        RETURN count(p) as total
        """
        count_result = run_query(driver, count_query)
        total_count = count_result.iloc[0]['total'] if not count_result.empty else 0

        if total_count > 0:
            no_authors_query = """
            MATCH (p:Paper)
            WHERE NOT EXISTS((p)<-[:AUTHORED]-(:Author))
            RETURN p.paperId as id, p.title as title, p.citationCount as citations
            ORDER BY p.citationCount DESC
            LIMIT 20
            """
            no_authors_df = run_query(driver, no_authors_query)
            
            st.warning(f"Found {total_count} papers without authors (showing top 20)")
            st.dataframe(no_authors_df, use_container_width=True)
        else:
            st.success("All papers have at least one author!")
        
        # Papers without venues
        st.subheader("📄 Papers Without Publication Venues")
        count_query = """
        MATCH (p:Paper)
        WHERE NOT EXISTS((p)-[:PUBLISHED_IN]->(:PubVenue))
        RETURN count(p) as total
        """
        count_result = run_query(driver, count_query)
        total_count = count_result.iloc[0]['total'] if not count_result.empty else 0
        
        if total_count > 0:
            no_venue_query = """
            MATCH (p:Paper)
            WHERE NOT EXISTS((p)-[:PUBLISHED_IN]->(:PubVenue))
            RETURN p.paperId as id, p.title as title, p.year as year
            ORDER BY p.year DESC
            LIMIT 20
            """
            no_venue_df = run_query(driver, no_venue_query)
            
            st.warning(f"Found {total_count} papers without publication venues (showing top 20)")
            st.dataframe(no_venue_df, use_container_width=True)
        
        # Papers without fields
        st.subheader("📄 Papers Without Research Fields")
        count_query = """
        MATCH (p:Paper)
        WHERE NOT EXISTS((p)-[:IS_ABOUT]->(:Field))
        RETURN count(p) as total
        """
        count_result = run_query(driver, count_query)
        total_count = count_result.iloc[0]['total'] if not count_result.empty else 0
        
        if total_count > 0:
            no_fields_query = """
            MATCH (p:Paper)
            WHERE NOT EXISTS((p)-[:IS_ABOUT]->(:Field))
            RETURN p.paperId as id, p.title as title, p.year as year
            ORDER BY p.citationCount DESC
            LIMIT 20
            """
            no_fields_df = run_query(driver, no_fields_query)
            
            st.warning(f"Found {total_count} papers without research fields (showing top 20)")
            st.dataframe(no_fields_df, use_container_width=True)
        
        # Isolated authors
        st.subheader("👤 Isolated Authors")
        count_query = """
        MATCH (a:Author)
        WHERE NOT EXISTS((a)-[:AUTHORED]->(:Paper)<-[:AUTHORED]-(:Author))
        OR NOT EXISTS((a)-[:AUTHORED]->(:Paper))
        RETURN count(a) as total
        """
        count_result = run_query(driver, count_query)
        total_count = count_result.iloc[0]['total'] if not count_result.empty else 0
        
        if total_count > 0:
            isolated_authors_query = """
            MATCH (a:Author)
            WHERE NOT EXISTS((a)-[:AUTHORED]->(:Paper)<-[:AUTHORED]-(:Author))
            OR NOT EXISTS((a)-[:AUTHORED]->(:Paper))
            RETURN a.authorName as name, a.authorId as id
            LIMIT 20
            """
            isolated_df = run_query(driver, isolated_authors_query)
            
            st.info(f"Found {total_count} authors with no co-authors (showing top 20)")
            st.dataframe(isolated_df, use_container_width=True)
    
    # 4. NETWORK ANALYTICS
    elif page == "📈 Network Analytics":
        st.header("Network Analytics")
        st.markdown("Advanced graph algorithms and network metrics")
        
        # Most influential authors
        st.subheader("🌟 Most Influential Authors")
        
        metric = st.selectbox("Rank by:", 
                            ["Total Citations", "Number of Papers", "Average Citations per Paper", "Number of Co-authors"])
        
        if metric == "Total Citations":
            query = """
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)
            WITH a, sum(p.citationCount) as total_citations, count(p) as paper_count
            RETURN a.authorName as author, total_citations, paper_count
            ORDER BY total_citations DESC
            LIMIT 20
            """
        elif metric == "Number of Papers":
            query = """
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)
            WITH a, count(p) as paper_count, sum(p.citationCount) as total_citations
            RETURN a.authorName as author, paper_count, total_citations
            ORDER BY paper_count DESC
            LIMIT 20
            """
        elif metric == "Average Citations per Paper":
            query = """
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)
            WITH a, avg(p.citationCount) as avg_citations, count(p) as paper_count
            WHERE paper_count >= 3
            RETURN a.authorName as author, avg_citations, paper_count
            ORDER BY avg_citations DESC
            LIMIT 20
            """
        else:  # Number of Co-authors
            query = """
            MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
            WHERE a1 <> a2
            WITH a1, count(DISTINCT a2) as coauthor_count
            RETURN a1.authorName as author, coauthor_count
            ORDER BY coauthor_count DESC
            LIMIT 20
            """
        
        results_df = run_query(driver, query)
        if not results_df.empty:
            fig = px.bar(results_df.head(15), 
                        x='author', 
                        y=results_df.columns[1],
                        title=f"Top Authors by {metric}",
                        color=results_df.columns[1],
                        color_continuous_scale='viridis')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Collaboration network metrics
        st.subheader("🤝 Collaboration Network Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average collaborators
            avg_collab_query = """
            MATCH (a:Author)
            OPTIONAL MATCH (a)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(other:Author)
            WHERE a <> other
            WITH a, count(DISTINCT other) as collaborators
            RETURN avg(collaborators) as avg_collaborators, 
                   max(collaborators) as max_collaborators,
                   min(collaborators) as min_collaborators
            """
            collab_stats = run_query(driver, avg_collab_query)
            if not collab_stats.empty:
                st.metric("Average Collaborators per Author", 
                        f"{collab_stats.iloc[0]['avg_collaborators']:.2f}")
                st.metric("Most Connected Author", 
                        f"{collab_stats.iloc[0]['max_collaborators']} collaborators")
        
        with col2:
            # Clustering coefficient
            st.info("💡 Network clustering analysis shows how tightly connected research communities are")
        
        # Research field analysis
        st.subheader("🔬 Research Field Analysis")
        
        field_query = """
        MATCH (f:Field)<-[:IS_ABOUT]-(p:Paper)
        WITH f, count(p) as paper_count, avg(p.citationCount) as avg_citations
        RETURN f.field as field, paper_count, avg_citations
        ORDER BY paper_count DESC
        LIMIT 15
        """
        fields_df = run_query(driver, field_query)
        
        if not fields_df.empty:
            fig = px.scatter(fields_df, 
                           x='paper_count', 
                           y='avg_citations',
                           size='paper_count',
                           hover_data=['field'],
                           title="Research Fields: Papers vs Average Citations",
                           labels={'paper_count': 'Number of Papers', 
                                  'avg_citations': 'Average Citations'})
            
            # Add field labels
            for idx, row in fields_df.iterrows():
                fig.add_annotation(x=row['paper_count'], 
                                 y=row['avg_citations'],
                                 text=row['field'][:20],
                                 showarrow=False,
                                 font=dict(size=9))
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-based analysis
        st.subheader("📅 Temporal Analysis")
        
        yearly_growth_query = """
        MATCH (p:Paper)-[:PUB_YEAR]->(y:PubYear)
        MATCH (p)-[:IS_ABOUT]->(f:Field)
        WHERE y.year >= 2010
        WITH y.year as year, f.field as field, count(p) as papers
        RETURN year, field, papers
        ORDER BY year, papers DESC
        """
        yearly_df = run_query(driver, yearly_growth_query)
        
        if not yearly_df.empty:
            # Get top 5 fields
            top_fields = yearly_df.groupby('field')['papers'].sum().nlargest(5).index.tolist()
            filtered_df = yearly_df[yearly_df['field'].isin(top_fields)]
            
            fig = px.line(filtered_df, 
                         x='year', 
                         y='papers', 
                         color='field',
                         title="Growth of Top Research Fields Over Time",
                         labels={'papers': 'Number of Papers', 'year': 'Year'})
            st.plotly_chart(fig, use_container_width=True)
    
    # 5. INTERACTIVE VISUALIZATION
    elif page == "🌐 Interactive Visualization":
        st.header("Interactive Network Visualization")
        st.markdown("Explore the citation network through interactive visualizations")
        
        viz_type = st.selectbox("Visualization Type:", 
                              ["Author Collaboration Network", "Paper Citation Network", 
                               "Field Relationship Network", "Venue Network"])
        
        sample_size = st.slider("Number of nodes to display:", 10, 100, 30)
        
        if st.button("Generate Visualization", type="primary"):
            with st.spinner("Creating network visualization..."):
                
                if viz_type == "Author Collaboration Network":
                    # Create author collaboration network
                    query = """
                    MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
                    WHERE id(a1) < id(a2)
                    WITH a1, a2, count(distinct p) as collaborations
                    ORDER BY collaborations DESC
                    LIMIT $limit
                    RETURN a1.authorName as source, a2.authorName as target, collaborations as weight
                    """
                    edges_df = run_query(driver, query, {"limit": sample_size})
                    
                    if not edges_df.empty:
                        # Create NetworkX graph
                        G = nx.Graph()
                        
                        # Add edges with weights
                        for _, row in edges_df.iterrows():
                            G.add_edge(row['source'], row['target'], weight=row['weight'])
                        
                        # Create layout
                        pos = nx.spring_layout(G, k=1, iterations=50)
                        
                        # Create edge traces
                        edge_traces = []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            weight = edge[2]['weight']
                            
                            edge_trace = go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=min(weight*2, 10), color='#888'),
                                hoverinfo='text',
                                text=f"{edge[0]} - {edge[1]}: {weight} collaborations",
                                showlegend=False
                            )
                            edge_traces.append(edge_trace)
                        
                        # Create node trace
                        node_x = []
                        node_y = []
                        node_text = []
                        node_connections = []
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(f"{node}<br>{G.degree(node)} connections")
                            node_connections.append(G.degree(node))
                        
                        node_trace = go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode='markers+text',
                            text=[n.split()[0] if len(n.split()[0]) < 15 else n.split()[0][:12]+'...' for n in G.nodes()],
                            textposition="top center",
                            hoverinfo='text',
                            hovertext=node_text,
                            marker=dict(
                                size=[15 + c*3 for c in node_connections],
                                color=node_connections,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title="Connections",
                                    thickness=15,
                                    xanchor='left'
                                ),
                                line=dict(width=2, color='white')
                            )
                        )
                        
                        # Create figure
                        fig = go.Figure(data=edge_traces + [node_trace])
                        
                        fig.update_layout(
                            title=f"Author Collaboration Network ({len(G.nodes())} authors, {len(G.edges())} collaborations)",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Network statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Number of Authors", len(G.nodes()))
                        with col2:
                            st.metric("Number of Collaborations", len(G.edges()))
                        with col3:
                            if len(G.nodes()) > 0:
                                avg_connections = sum(dict(G.degree()).values()) / len(G.nodes())
                                st.metric("Avg Connections", f"{avg_connections:.1f}")
                    else:
                        st.warning("No collaboration data found")
                
                elif viz_type == "Field Relationship Network":
                    # Field co-occurrence network
                    query = """
                    MATCH (p:Paper)-[:IS_ABOUT]->(f1:Field)
                    MATCH (p)-[:IS_ABOUT]->(f2:Field)
                    WHERE f1.field < f2.field
                    WITH f1.field as field1, f2.field as field2, count(p) as papers
                    ORDER BY papers DESC
                    LIMIT $limit
                    RETURN field1, field2, papers
                    """
                    field_edges = run_query(driver, query, {"limit": sample_size})
                    
                    if not field_edges.empty:
                        # Create NetworkX graph
                        G = nx.Graph()
                        
                        # Add edges
                        for _, row in field_edges.iterrows():
                            G.add_edge(row['field1'], row['field2'], weight=row['papers'])
                        
                        # Create layout
                        pos = nx.spring_layout(G, k=2, iterations=50)
                        
                        # Create edge traces
                        edge_traces = []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            weight = edge[2]['weight']
                            
                            edge_trace = go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=min(weight/5, 10), color='#888'),
                                hoverinfo='text',
                                text=f"{edge[0]} - {edge[1]}: {weight} papers",
                                showlegend=False
                            )
                            edge_traces.append(edge_trace)
                        
                        # Create node trace
                        node_x = []
                        node_y = []
                        node_text = []
                        node_sizes = []
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            degree = G.degree(node)
                            node_text.append(f"{node}<br>{degree} connections")
                            node_sizes.append(20 + degree * 5)
                        
                        node_trace = go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode='markers+text',
                            text=[f if len(f) < 20 else f[:17]+'...' for f in G.nodes()],
                            textposition="top center",
                            hoverinfo='text',
                            hovertext=node_text,
                            marker=dict(
                                size=node_sizes,
                                color='lightblue',
                                line=dict(width=2, color='darkblue')
                            )
                        )
                        
                        # Create figure
                        fig = go.Figure(data=edge_traces + [node_trace])
                        
                        fig.update_layout(
                            title=f"Field Co-occurrence Network ({len(G.nodes())} fields, {len(G.edges())} connections)",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No field relationship data found")
                
                elif viz_type == "Paper Citation Network":
                    # Since we don't have direct citations, show papers with shared attributes
                    st.info("Showing papers connected through shared authors and fields")
                    
                    query = """
                    MATCH (p1:Paper)<-[:AUTHORED]-(a:Author)-[:AUTHORED]->(p2:Paper)
                    WHERE id(p1) < id(p2) AND p1.citationCount > 10 AND p2.citationCount > 10
                    WITH p1, p2, count(distinct a) as shared_authors
                    WHERE shared_authors >= 2
                    ORDER BY shared_authors DESC
                    LIMIT $limit
                    RETURN substring(p1.title, 0, 50) as source, 
                           substring(p2.title, 0, 50) as target, 
                           shared_authors as weight,
                           p1.citationCount as source_citations,
                           p2.citationCount as target_citations
                    """
                    paper_edges = run_query(driver, query, {"limit": sample_size})
                    
                    if not paper_edges.empty:
                        # Create NetworkX graph
                        G = nx.Graph()
                        
                        # Track citation counts for node sizing
                        node_citations = {}
                        
                        for _, row in paper_edges.iterrows():
                            G.add_edge(row['source'] + '...', row['target'] + '...', weight=row['weight'])
                            node_citations[row['source'] + '...'] = row['source_citations']
                            node_citations[row['target'] + '...'] = row['target_citations']
                        
                        pos = nx.spring_layout(G, k=3, iterations=50)
                        
                        # Create edge traces
                        edge_traces = []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            weight = edge[2]['weight']
                            
                            edge_trace = go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=weight, color='#888'),
                                hoverinfo='text',
                                text=f"{weight} shared authors",
                                showlegend=False
                            )
                            edge_traces.append(edge_trace)
                        
                        # Create node trace
                        node_x = []
                        node_y = []
                        node_text = []
                        node_sizes = []
                        node_labels = []
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            citations = node_citations.get(node, 0)
                            node_text.append(f"{node}<br>Citations: {citations}<br>Connections: {G.degree(node)}")
                            node_sizes.append(10 + min(citations/10, 50))
                            # Truncate label for display
                            node_labels.append(node[:30] + '...' if len(node) > 30 else node)
                        
                        node_trace = go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode='markers+text',
                            text=node_labels,
                            textposition="top center",
                            hoverinfo='text',
                            hovertext=node_text,
                            marker=dict(
                                size=node_sizes,
                                color=[node_citations.get(n, 0) for n in G.nodes()],
                                colorscale='YlOrRd',
                                showscale=True,
                                colorbar=dict(
                                    title="Citations",
                                    thickness=15,
                                    xanchor='left'
                                ),
                                line=dict(width=2, color='white')
                            )
                        )
                        
                        # Create figure
                        fig = go.Figure(data=edge_traces + [node_trace])
                        
                        fig.update_layout(
                            title=f"Paper Network ({len(G.nodes())} papers, {len(G.edges())} connections based on shared authors)",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Papers", len(G.nodes()))
                        with col2:
                            st.metric("Connections", len(G.edges()))
                        with col3:
                            avg_citations = sum(node_citations.values()) / len(node_citations) if node_citations else 0
                            st.metric("Avg Citations", f"{avg_citations:.0f}")
                    else:
                        st.warning("No paper connections found with current criteria. Try reducing the citation threshold or shared author requirement.")
                
                elif viz_type == "Venue Network":
                    # Venues connected by authors who publish in both
                    query = """
                    MATCH (v1:PubVenue)<-[:PUBLISHED_IN]-(p1:Paper)<-[:AUTHORED]-(a:Author)
                          -[:AUTHORED]->(p2:Paper)-[:PUBLISHED_IN]->(v2:PubVenue)
                    WHERE v1.venue < v2.venue
                    WITH v1.venue as venue1, v2.venue as venue2, count(distinct a) as shared_authors
                    WHERE shared_authors >= 2
                    ORDER BY shared_authors DESC
                    LIMIT $limit
                    RETURN venue1, venue2, shared_authors
                    """
                    venue_edges = run_query(driver, query, {"limit": sample_size})
                    
                    if not venue_edges.empty:
                        # Create NetworkX graph
                        G = nx.Graph()
                        
                        for _, row in venue_edges.iterrows():
                            G.add_edge(row['venue1'], row['venue2'], weight=row['shared_authors'])
                        
                        pos = nx.spring_layout(G, k=2, iterations=50)
                        
                        # Create edge traces
                        edge_traces = []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            weight = edge[2]['weight']
                            
                            edge_trace = go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=min(weight/2, 10), color='#888'),
                                hoverinfo='text',
                                text=f"{weight} shared authors",
                                showlegend=False
                            )
                            edge_traces.append(edge_trace)
                        
                        # Create node trace
                        node_x = []
                        node_y = []
                        node_text = []
                        node_sizes = []
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            degree = G.degree(node)
                            node_text.append(f"{node}<br>{degree} connections")
                            node_sizes.append(20 + degree * 5)
                        
                        node_trace = go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode='markers+text',
                            text=[v[:30] + '...' if len(v) > 30 else v for v in G.nodes()],
                            textposition="top center",
                            hoverinfo='text',
                            hovertext=node_text,
                            marker=dict(
                                size=node_sizes,
                                color='lightcoral',
                                line=dict(width=2, color='darkred')
                            )
                        )
                        
                        # Create figure
                        fig = go.Figure(data=edge_traces + [node_trace])
                        
                        fig.update_layout(
                            title=f"Venue Network ({len(G.nodes())} venues connected by authors publishing in both)",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Venues", len(G.nodes()))
                        with col2:
                            st.metric("Connections", len(G.edges()))
                        with col3:
                            if len(G.nodes()) > 0:
                                avg_connections = sum(dict(G.degree()).values()) / len(G.nodes())
                                st.metric("Avg Connections", f"{avg_connections:.1f}")
                    else:
                        st.warning("No venue connections found. Try lowering the shared author threshold.")
        
        else:
            st.info("👆 Click 'Generate Visualization' to create the network graph")
    
    # 6. INSIGHTS DASHBOARD
    elif page == "💡 Insights Dashboard":
        st.header("Key Insights & Discoveries")
        st.markdown("Uncovering hidden patterns and intellectual connections")
        
        # Rising stars
        st.subheader("🌟 Rising Stars (Authors with Growing Influence)")
        
        rising_query = """
        MATCH (a:Author)-[:AUTHORED]->(p:Paper)-[:PUB_YEAR]->(y:PubYear)
        WHERE y.year >= 2018
        WITH a, y.year as year, sum(p.citationCount) as yearly_citations
        WITH a, collect({year: year, citations: yearly_citations}) as yearly_data
        WHERE size(yearly_data) >= 3
        RETURN a.authorName as author, yearly_data
        LIMIT 20
        """
        rising_df = run_query(driver, rising_query)
        
        if not rising_df.empty:
            # Calculate growth rates
            growth_rates = []
            for _, row in rising_df.iterrows():
                data = sorted(row['yearly_data'], key=lambda x: x['year'])
                if len(data) >= 2:
                    growth = (data[-1]['citations'] - data[0]['citations']) / (data[-1]['year'] - data[0]['year'])
                    growth_rates.append({'author': row['author'], 'growth_rate': growth})
            
            if growth_rates:
                growth_df = pd.DataFrame(growth_rates).sort_values('growth_rate', ascending=False).head(10)
                
                fig = px.bar(growth_df, x='author', y='growth_rate',
                           title="Authors with Fastest Growing Citation Rates",
                           labels={'growth_rate': 'Citation Growth Rate (per year)'})
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Cross-field connections
        st.subheader("🌉 Cross-Disciplinary Bridges")
        
        bridge_query = """
        MATCH (f1:Field)<-[:IS_ABOUT]-(p:Paper)-[:IS_ABOUT]->(f2:Field)
        WHERE f1.field < f2.field
        WITH f1.field as field1, f2.field as field2, count(p) as connections
        ORDER BY connections DESC
        LIMIT 15
        RETURN field1, field2, connections
        """
        bridges_df = run_query(driver, bridge_query)
        
        if not bridges_df.empty:
            bridges_df['pair'] = bridges_df['field1'] + ' ↔ ' + bridges_df['field2']
            
            fig = px.bar(bridges_df, x='pair', y='connections',
                        title="Strongest Cross-Disciplinary Connections",
                        labels={'connections': 'Number of Papers', 'pair': 'Field Pairs'})
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Hidden gems
        st.subheader("💎 Hidden Gems (Low Citations but High Potential)")
        
        gems_query = """
        MATCH (p:Paper)-[:AUTHORED]-(a:Author)
        MATCH (a)-[:AUTHORED]->(other:Paper)
        WHERE p.citationCount < 10 AND p.year >= 2020
        WITH p, avg(other.citationCount) as author_avg_citations, count(other) as other_papers
        WHERE author_avg_citations > 50 AND other_papers > 5
        RETURN p.title as title, p.citationCount as citations, 
               author_avg_citations, p.year as year
        ORDER BY author_avg_citations DESC
        LIMIT 10
        """
        gems_df = run_query(driver, gems_query)
        
        if not gems_df.empty:
            st.write("Papers with low citations but authored by highly-cited researchers:")
            st.dataframe(gems_df, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Network Health Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">🌐</div>
                <div class="metric-label">Network Density</div>
                <div class="metric-value">Calculating...</div>
                <div style="font-size: 0.8rem; color: #999; margin-top: 10px;">
                    How connected the network is compared to a fully connected graph
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_path_query = """
            MATCH (a:Author)
            WITH count(a) as total_authors
            RETURN total_authors
            """
            result = run_query(driver, avg_path_query)
            if not result.empty:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">👤</div>
                    <div class="metric-label">Total Authors</div>
                    <div class="metric-value">{result.iloc[0]['total_authors']:,}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            orphan_query = """
            MATCH (p:Paper)
            WHERE NOT EXISTS((p)<-[:AUTHORED]-(:Author))
            RETURN count(p) as orphans
            """
            orphans = run_query(driver, orphan_query)
            if not orphans.empty:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📄</div>
                    <div class="metric-label">Orphan Papers</div>
                    <div class="metric-value">{orphans.iloc[0]['orphans']}</div>
                    <div style="font-size: 0.8rem; color: #999; margin-top: 10px;">
                        Papers without author information
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("Unable to connect to Neo4j database. Please check your environment variables.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Citation Network Explorer | Built with Streamlit & Neo4j
</div>
""", unsafe_allow_html=True)