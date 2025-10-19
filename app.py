#!/usr/bin/env python3
"""
Streamlit app for Faculty Workload Management AI Agent
"""

import streamlit as st
import pandas as pd
from data_loader import FacultyDataLoader
from vector_store import PolicyVectorStore
from agent import create_rag_policy_tool, create_timetable_query_tool, create_workload_report_tool, IntelligentQueryProcessor

# Page configuration
st.set_page_config(
    page_title="Faculty Workload Management AI Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #000000;
    }
    .response-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tool-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1rem;
        color: #000000;
        line-height: 1.5;
        border: 1px solid #dee2e6;
        white-space: pre-wrap;
    }
    .tool-used {
        font-weight: bold;
        color: #1f77b4;
        font-size: 1.1rem;
    }
    .response-label {
        font-weight: bold;
        color: #000000;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """Initialize all components with caching."""
    try:
        # Initialize data loader
        data_loader = FacultyDataLoader()
        
        # Initialize vector store
        vector_store = PolicyVectorStore()
        vector_store.load_policies_from_file("policies.txt")
        
        # Create tools
        rag_tool = create_rag_policy_tool(vector_store)
        timetable_tool = create_timetable_query_tool(data_loader)
        workload_tool = create_workload_report_tool(data_loader)
        
        # Create intelligent query processor
        intelligent_processor = IntelligentQueryProcessor(data_loader, vector_store)
        
        return data_loader, vector_store, rag_tool, timetable_tool, workload_tool, intelligent_processor
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None, None, None

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🎓 Faculty Workload Management AI Agent</h1>', unsafe_allow_html=True)
    
    # Initialize components
    with st.spinner("Initializing AI Agent components..."):
        data_loader, vector_store, rag_tool, timetable_tool, workload_tool, intelligent_processor = initialize_components()
    
    if not all([data_loader, vector_store, rag_tool, timetable_tool, workload_tool, intelligent_processor]):
        st.error("Failed to initialize components. Please check the error messages above.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Agent Status")
        st.success("✅ All components loaded successfully")
        
        st.header("📊 Quick Stats")
        if data_loader:
            faculty_count = len(data_loader.get_all_faculty())
            dept_count = len(data_loader.get_all_departments())
            st.metric("Total Faculty", faculty_count)
            st.metric("Departments", dept_count)
        
        st.header("🔧 Available Tools")
        st.info("""
        **RAG Policy Tool**: Search university policies
        **Timetable Query Tool**: Check schedules and availability
        **Workload Report Tool**: Generate workload reports
        """)
        
        st.header("📝 Sample Queries")
        sample_queries = [
            "What is Prof. Sharma's workload?",
            "Which faculty is free on Tuesday at 2 PM?",
            "What are the university policies on maximum workload?",
            "Give me a summary of the CSE department workload",
            "Show me Prof. Mehta's schedule",
            "What courses does Prof. Verma teach?",
            "Which room is allocated Prof. Sharma on Monday?"
        ]
        
        for query in sample_queries:
            if st.button(f"💡 {query}", key=f"sample_{query}"):
                st.session_state.user_query = query
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask the AI Agent")
        
        # Query input
        user_query = st.text_input(
            "Enter your question about faculty workload, schedules, or policies:",
            value=st.session_state.get('user_query', ''),
            placeholder="e.g., What is Prof. Sharma's workload?",
            key="query_input"
        )
        
        # Query buttons
        col_query1, col_query2, col_query3 = st.columns(3)
        
        with col_query1:
            if st.button("🔍 Search Policies", use_container_width=True):
                user_query = "What are the university policies on faculty workload?"
                st.session_state.user_query = user_query
                st.rerun()
        
        with col_query2:
            if st.button("📅 Check Schedule", use_container_width=True):
                user_query = "Which faculty is free on Tuesday at 2 PM?"
                st.session_state.user_query = user_query
                st.rerun()
        
        with col_query3:
            if st.button("📊 Workload Report", use_container_width=True):
                user_query = "What is Prof. Sharma's workload?"
                st.session_state.user_query = user_query
                st.rerun()
        
        # Process query
        if user_query:
            st.markdown('<div class="query-box">', unsafe_allow_html=True)
            st.write(f"**Your Query:** {user_query}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Determine which tool to use
            query_lower = user_query.lower()
            
            with st.spinner("Processing your query..."):
                # Use intelligent query processor for better human-like responses
                st.info("🧠 Analyzing your query with AI intelligence...")
                result = intelligent_processor.process_query(user_query)
                tool_used = "Intelligent AI Processor"
            
            # Display result
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="tool-used">Tool Used: {tool_used}</div>', unsafe_allow_html=True)
            st.markdown('<div class="response-label">Response:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="tool-result">{result}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.header("📈 Data Overview")
        
        if data_loader:
            # Faculty workload summary
            st.subheader("Faculty Workload Summary")
            all_faculty = data_loader.get_all_faculty()
            
            # Create a simple workload summary
            workload_data = []
            for faculty_name in all_faculty[:10]:  # Show first 10
                try:
                    workload = data_loader.get_faculty_workload(faculty_name)
                    if "error" not in workload:
                        workload_data.append({
                            "Faculty": faculty_name,
                            "Department": workload.get("department", "N/A"),
                            "Hours": workload.get("total_hours", 0)
                        })
                except:
                    continue
            
            if workload_data:
                df = pd.DataFrame(workload_data)
                st.dataframe(df, use_container_width=True)
            
            # Department summary
            st.subheader("Department Summary")
            departments = data_loader.get_all_departments()
            dept_data = []
            
            for dept in departments:
                try:
                    dept_summary = data_loader.get_department_summary(dept)
                    if "error" not in dept_summary:
                        dept_data.append({
                            "Department": dept,
                            "Faculty Count": dept_summary.get("total_faculty", 0),
                            "Total Hours": dept_summary.get("total_hours", 0)
                        })
                except:
                    continue
            
            if dept_data:
                dept_df = pd.DataFrame(dept_data)
                st.dataframe(dept_df, use_container_width=True)
        
        st.header("ℹ️ About")
        st.info("""
        This AI agent helps manage faculty workload and schedules by:
        
        • Searching university policies
        • Checking faculty availability
        • Generating workload reports
        • Analyzing department summaries
        
        All data is processed locally for privacy and security.
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ''
    
    main()