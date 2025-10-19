from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from typing import Optional, Type, Dict, Any
import json

from data_loader import FacultyDataLoader
from vector_store import PolicyVectorStore

# Global variables to store the instances
_vector_store = None
_data_loader = None

def create_rag_policy_tool(vector_store: PolicyVectorStore):
    """Create RAG policy tool."""
    global _vector_store
    _vector_store = vector_store
    
    def rag_policy_search(query: str) -> str:
        """Search for university policies using RAG."""
        try:
            results = _vector_store.search_policies(query, n_results=3)
            
            if not results:
                return "No relevant policies found for your query."
            
            response = "Relevant policies found:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['text']}\n"
                response += f"   Category: {result['metadata']['category']}\n\n"
            
            return response
            
        except Exception as e:
            return f"Error searching policies: {str(e)}"
    
    return Tool(
        name="rag_policy_tool",
        description="Use this tool to search for university policies related to faculty workload, scheduling rules, and department management. Input should be a natural language query about policies.",
        func=rag_policy_search
    )

def create_timetable_query_tool(data_loader: FacultyDataLoader):
    """Create timetable query tool."""
    global _data_loader
    _data_loader = data_loader
    
    def timetable_query(query: str) -> str:
        """Execute the timetable query."""
        try:
            query_lower = query.lower()
            
            if "free" in query_lower and ("faculty" in query_lower or "professor" in query_lower):
                return _handle_free_faculty_query(query)
            elif "room" in query_lower and ("allocated" in query_lower or "allotted" in query_lower):
                return _handle_room_allocation_query(query)
            elif "schedule" in query_lower or "timetable" in query_lower:
                return _handle_schedule_query(query)
            elif "room" in query_lower:
                return _handle_room_query(query)
            elif "course" in query_lower:
                return _handle_course_query(query)
            else:
                return _handle_general_query(query)
                
        except Exception as e:
            return f"Error processing timetable query: {str(e)}"
    
    return Tool(
        name="timetable_query_tool",
        description="Use this tool to query faculty schedules, find free faculty at specific times, check room availability, or search for specific courses. Input should describe what you're looking for (e.g., 'faculty free on Tuesday at 2 PM', 'Prof. Sharma schedule', 'room 201 availability').",
        func=timetable_query
    )

def create_workload_report_tool(data_loader: FacultyDataLoader):
    """Create workload report tool."""
    global _data_loader
    _data_loader = data_loader
    
    def workload_report(query: str) -> str:
        """Execute the workload report generation."""
        try:
            query_lower = query.lower()
            
            if "department" in query_lower:
                return _handle_department_query(query)
            elif "prof" in query_lower or "professor" in query_lower:
                return _handle_faculty_query(query)
            elif "all" in query_lower and "faculty" in query_lower:
                return _handle_all_faculty_query(query)
            else:
                return _handle_general_workload_query(query)
                
        except Exception as e:
            return f"Error generating workload report: {str(e)}"
    
    return Tool(
        name="workload_report_tool",
        description="Use this tool to generate workload reports for individual faculty members or entire departments. Input should specify what report you want (e.g., 'Prof. Sharma workload', 'CSE department summary', 'all faculty workload').",
        func=workload_report
    )

# Helper functions for timetable queries
def _handle_free_faculty_query(query: str) -> str:
    """Handle queries about free faculty at specific times."""
    # Extract day and time from query
    words = query.lower().split()
    day = None
    time = None
    
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for word in words:
        if word in days:
            day = word.capitalize()
            break
    
    # Look for time patterns
    for i, word in enumerate(words):
        if word == 'at' and i + 1 < len(words):
            time = words[i + 1]
            if i + 2 < len(words) and words[i + 2] in ['am', 'pm']:
                time += " " + words[i + 2]
            break
    
    # If no time found with 'at', look for time patterns directly
    if not time:
        for word in words:
            if any(char.isdigit() for char in word) and ('am' in word or 'pm' in word or ':' in word):
                time = word
                break
    
    if not day or not time:
        return "Please specify both day and time (e.g., 'faculty free on Tuesday at 2 PM')"
    
    result = _data_loader.get_free_faculty(day, time)
    
    if "error" in result:
        return result["error"]
    
    response = f"Free faculty on {result['day']} at {result['time']}:\n\n"
    response += f"Available faculty ({len(result['free_faculty'])}):\n"
    for faculty in result['free_faculty']:
        response += f"- {faculty}\n"
    
    if result['busy_faculty']:
        response += f"\nBusy faculty ({len(result['busy_faculty'])}):\n"
        for faculty in result['busy_faculty']:
            response += f"- {faculty['name']} (teaching {faculty['course']} in {faculty['room']})\n"
    
    return response

def _handle_schedule_query(query: str) -> str:
    """Handle queries about faculty schedules."""
    # Extract faculty name and day from query
    words = query.lower().split()
    faculty_name = None
    day = None
    
    # Look for "Prof." or "Professor"
    for i, word in enumerate(words):
        if word in ['prof.', 'professor'] and i + 1 < len(words):
            # Remove any punctuation from the name
            name_part = words[i + 1].replace("'s", "").replace("'", "").replace(".", "")
            faculty_name = "Prof." + name_part.capitalize()
            break
    
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for word in words:
        if word in days:
            day = word.capitalize()
            break
    
    if not faculty_name:
        return "Please specify a faculty member (e.g., 'Prof. Sharma schedule')"
    
    if day:
        result = _data_loader.get_faculty_schedule(faculty_name, day)
    else:
        # Get schedule for all days
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        response = f"Schedule for {faculty_name}:\n\n"
        for d in all_days:
            result = _data_loader.get_faculty_schedule(faculty_name, d)
            if "sessions" in result and result["sessions"]:
                response += f"{d}:\n"
                for session in result["sessions"]:
                    response += f"  {session['time']} - {session['course']} ({session['room']})\n"
                response += "\n"
        return response
    
    if "error" in result:
        return result["error"]
    
    response = f"Schedule for {result['name']} on {result['day']}:\n\n"
    if result['sessions']:
        for session in result['sessions']:
            response += f"- {session['time']}: {session['course']} ({session['room']})\n"
    else:
        response += "No classes scheduled.\n"
    
    return response


def _handle_room_query(query: str) -> str:
    """Handle queries about room availability."""
    # Extract room number from query
    words = query.lower().split()
    room = None
    
    for word in words:
        if 'room' in word:
            room = word.capitalize()
            break
    
    if not room:
        return "Please specify a room (e.g., 'room 201 availability')"
    
    result = _data_loader.get_room_schedule(room)
    
    if "error" in result:
        return result["error"]
    
    response = f"Schedule for {result['room']}:\n\n"
    if result['sessions']:
        for session in result['sessions']:
            response += f"- {session['day']} {session['time']}: {session['course']} (Prof. {session['faculty']})\n"
    else:
        response += "No classes scheduled in this room.\n"
    
    return response

def _handle_course_query(query: str) -> str:
    """Handle queries about courses."""
    # Extract course name from query
    words = query.lower().split()
    course_name = None
    
    # Look for course-related keywords
    course_keywords = ['course', 'subject', 'class']
    for i, word in enumerate(words):
        if word in course_keywords and i + 1 < len(words):
            course_name = words[i + 1].capitalize()
            break
    
    if not course_name:
        return "Please specify a course name (e.g., 'Data Structures course')"
    
    results = _data_loader.search_faculty_by_course(course_name)
    
    if "error" in results:
        return results["error"]
    
    response = f"Faculty teaching {course_name}:\n\n"
    for result in results:
        response += f"- {result['name']} ({result['department']}) - {result['hours_per_week']} hours/week\n"
    
    return response

def _handle_room_allocation_query(query: str) -> str:
    """Handle queries about room allocation for specific faculty."""
    words = query.lower().split()
    faculty_name = None
    day = None
    
    # Look for "Prof." or "Professor" - improved matching
    query_lower = query.lower()
    if 'prof.' in query_lower:
        # Find the position of "prof."
        prof_pos = query_lower.find('prof.')
        if prof_pos != -1:
            # Extract the name after "prof."
            after_prof = query_lower[prof_pos + 5:].strip()
            # Get the first word after "prof."
            name_parts = after_prof.split()
            if name_parts:
                name_part = name_parts[0].replace("'s", "").replace("'", "").replace(".", "")
                faculty_name = "Prof." + name_part.capitalize()
    
    # Look for day - improved matching
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for day_name in days:
        if day_name in query_lower:
            day = day_name.capitalize()
            break
    
    if not faculty_name:
        return "Please specify a faculty member (e.g., 'which room is allocated Prof. Sharma on Monday')"
    
    if not day:
        return "Please specify a day (e.g., 'which room is allocated Prof. Sharma on Monday')"
    
    # Get faculty schedule for the specific day
    result = _data_loader.get_faculty_schedule(faculty_name, day)
    
    if "error" in result:
        return result["error"]
    
    if result['sessions']:
        response = f"Room allocation for {faculty_name} on {day}:\n\n"
        for session in result['sessions']:
            response += f"- {session['time']}: {session['room']} (teaching {session['course']})\n"
    else:
        response = f"{faculty_name} has no classes scheduled on {day}.\n"
    
    return response

class IntelligentQueryProcessor:
    """Enhanced query processor that thinks more like a human."""
    
    def __init__(self, data_loader, vector_store):
        self.data_loader = data_loader
        self.vector_store = vector_store
    
    def process_query(self, query: str) -> str:
        """Process query with human-like intelligence and context awareness."""
        query_lower = query.lower().strip()
        
        # Analyze query intent and context
        intent = self._analyze_intent(query_lower)
        context = self._extract_context(query_lower)
        
        # Generate intelligent response based on intent and context
        if intent == "room_allocation":
            return self._handle_room_allocation_intelligent(query, context)
        elif intent == "faculty_schedule":
            return self._handle_faculty_schedule_intelligent(query, context)
        elif intent == "workload_inquiry":
            return self._handle_workload_inquiry_intelligent(query, context)
        elif intent == "policy_search":
            return self._handle_policy_search_intelligent(query, context)
        elif intent == "availability_check":
            return self._handle_availability_check_intelligent(query, context)
        else:
            return self._handle_general_intelligent(query, context)
    
    def _analyze_intent(self, query: str) -> str:
        """Analyze the user's intent from the query."""
        if any(word in query for word in ['room', 'allocated', 'allotted', 'assigned']):
            return "room_allocation"
        elif any(word in query for word in ['schedule', 'timetable', 'when', 'time']):
            return "faculty_schedule"
        elif any(word in query for word in ['workload', 'hours', 'teaching', 'courses']):
            return "workload_inquiry"
        elif any(word in query for word in ['policy', 'rule', 'regulation', 'guideline']):
            return "policy_search"
        elif any(word in query for word in ['free', 'available', 'busy']):
            return "availability_check"
        else:
            return "general"
    
    def _extract_context(self, query: str) -> dict:
        """Extract contextual information from the query."""
        context = {
            'faculty': None,
            'day': None,
            'time': None,
            'room': None,
            'department': None,
            'course': None
        }
        
        # Extract faculty name
        if 'prof.' in query:
            prof_pos = query.find('prof.')
            after_prof = query[prof_pos + 5:].strip()
            name_parts = after_prof.split()
            if name_parts:
                name_part = name_parts[0].replace("'s", "").replace("'", "").replace(".", "")
                context['faculty'] = "Prof." + name_part.capitalize()
        
        # Extract day
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for day in days:
            if day in query:
                context['day'] = day.capitalize()
                break
        
        # Extract time
        import re
        time_patterns = [
            r'\d{1,2}:\d{2}',  # 09:00, 14:30
            r'\d{1,2}\s*(am|pm)',  # 9 am, 2 pm
            r'\d{1,2}\s*(am|pm)',  # 9am, 2pm
        ]
        for pattern in time_patterns:
            match = re.search(pattern, query)
            if match:
                context['time'] = match.group()
                break
        
        # Extract room
        room_match = re.search(r'room\s*(\d+)', query)
        if room_match:
            context['room'] = f"Room {room_match.group(1)}"
        
        # Extract department
        departments = ['cse', 'eee', 'me', 'civil', 'ece', 'it']
        for dept in departments:
            if dept in query:
                context['department'] = dept.upper()
                break
        
        return context
    
    def _handle_room_allocation_intelligent(self, query: str, context: dict) -> str:
        """Handle room allocation queries with human-like intelligence."""
        if not context['faculty'] or not context['day']:
            return self._suggest_clarification(query, context, "room_allocation")
        
        result = self.data_loader.get_faculty_schedule(context['faculty'], context['day'])
        
        if "error" in result:
            return f"I couldn't find any classes for {context['faculty']} on {context['day']}. They might be free that day or the information might not be available."
        
        if result['sessions']:
            response = f"Based on the schedule, {context['faculty']} is allocated the following rooms on {context['day']}:\n\n"
            for session in result['sessions']:
                response += f"• {session['time']}: {session['room']} (teaching {session['course']})\n"
            
            # Add helpful context
            if len(result['sessions']) == 1:
                response += f"\n{context['faculty']} has only one class on {context['day']}."
            else:
                response += f"\n{context['faculty']} has {len(result['sessions'])} classes scheduled on {context['day']}."
        else:
            response = f"{context['faculty']} doesn't have any classes scheduled on {context['day']}. They're free that day!"
        
        return response
    
    def _handle_faculty_schedule_intelligent(self, query: str, context: dict) -> str:
        """Handle faculty schedule queries with intelligence."""
        if not context['faculty']:
            return self._suggest_clarification(query, context, "faculty_schedule")
        
        if context['day']:
            result = self.data_loader.get_faculty_schedule(context['faculty'], context['day'])
            if "error" in result:
                return f"I don't see any classes scheduled for {context['faculty']} on {context['day']}."
            
            response = f"Here's {context['faculty']}'s schedule for {context['day']}:\n\n"
            if result['sessions']:
                for session in result['sessions']:
                    response += f"• {session['time']}: {session['course']} in {session['room']}\n"
            else:
                response += "No classes scheduled - they're free that day!"
        else:
            # Get full week schedule
            response = f"Here's {context['faculty']}'s weekly schedule:\n\n"
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            has_classes = False
            
            for day in days:
                result = self.data_loader.get_faculty_schedule(context['faculty'], day)
                if result.get('sessions'):
                    has_classes = True
                    response += f"{day}:\n"
                    for session in result['sessions']:
                        response += f"  • {session['time']}: {session['course']} in {session['room']}\n"
                    response += "\n"
            
            if not has_classes:
                response += "No classes scheduled for this faculty member."
        
        return response
    
    def _handle_workload_inquiry_intelligent(self, query: str, context: dict) -> str:
        """Handle workload inquiries with intelligent analysis."""
        if context['faculty']:
            result = self.data_loader.get_faculty_workload(context['faculty'])
            if "error" in result:
                return f"I couldn't find workload information for {context['faculty']}."
            
            response = f"Here's {context['faculty']}'s teaching workload:\n\n"
            response += f"📊 **Total Hours**: {result['total_hours']} hours per week\n"
            response += f"🏢 **Department**: {result['department']}\n\n"
            response += "📚 **Courses Teaching**:\n"
            
            for course in result['course_details']:
                response += f"• {course['course']}: {course['hours_per_week']} hours/week\n"
            
            # Add intelligent analysis
            if result['total_hours'] > 10:
                response += f"\n💡 **Note**: {context['faculty']} has a heavy teaching load ({result['total_hours']} hours)."
            elif result['total_hours'] < 6:
                response += f"\n💡 **Note**: {context['faculty']} has a light teaching load ({result['total_hours']} hours)."
            else:
                response += f"\n💡 **Note**: {context['faculty']} has a balanced teaching load ({result['total_hours']} hours)."
        
        elif context['department']:
            result = self.data_loader.get_department_summary(context['department'])
            if "error" in result:
                return f"I couldn't find information for the {context['department']} department."
            
            response = f"Here's the workload summary for the {context['department']} department:\n\n"
            response += f"👥 **Total Faculty**: {result['total_faculty']}\n"
            response += f"⏰ **Total Teaching Hours**: {result['total_hours']} hours/week\n"
            response += f"📊 **Average Hours per Faculty**: {result['total_hours'] / result['total_faculty']:.1f} hours/week\n\n"
            
            response += "👨‍🏫 **Faculty Details**:\n"
            for faculty in result['faculty_details']:
                response += f"• {faculty['name']}: {faculty['courses']} ({faculty['hours_per_week']} hours/week)\n"
        
        else:
            return "I'd be happy to help with workload information! Please specify either a faculty member (e.g., 'Prof. Sharma workload') or a department (e.g., 'CSE department workload')."
        
        return response
    
    def _handle_policy_search_intelligent(self, query: str, context: dict) -> str:
        """Handle policy searches with intelligent responses."""
        results = self.vector_store.search_policies(query, n_results=3)
        
        if not results:
            return "I couldn't find any policies related to your query. The policy database might not contain information about this topic."
        
        response = "Here are the relevant university policies:\n\n"
        for i, result in enumerate(results, 1):
            response += f"**{i}. {result['metadata']['category'].replace('_', ' ').title()}**\n"
            response += f"{result['text']}\n\n"
        
        response += "💡 **Tip**: These policies are guidelines for faculty workload and scheduling. If you need more specific information, please ask!"
        
        return response
    
    def _handle_availability_check_intelligent(self, query: str, context: dict) -> str:
        """Handle availability checks with intelligent responses."""
        if not context['day'] or not context['time']:
            return "To check faculty availability, please specify both a day and time (e.g., 'which faculty is free on Tuesday at 2 PM')."
        
        result = self.data_loader.get_free_faculty(context['day'], context['time'])
        
        response = f"Here's the faculty availability for {context['day']} at {context['time']}:\n\n"
        
        if result['free_faculty']:
            response += f"✅ **Available Faculty** ({len(result['free_faculty'])}):\n"
            for faculty in result['free_faculty']:
                response += f"• {faculty}\n"
        else:
            response += "❌ **No faculty available** at this time.\n"
        
        if result['busy_faculty']:
            response += f"\n🚫 **Busy Faculty** ({len(result['busy_faculty'])}):\n"
            for faculty in result['busy_faculty']:
                response += f"• {faculty['name']} (teaching {faculty['course']} in {faculty['room']})\n"
        
        return response
    
    def _handle_general_intelligent(self, query: str, context: dict) -> str:
        """Handle general queries with intelligent suggestions."""
        response = "I can help you with various faculty and scheduling questions! Here's what I can do:\n\n"
        
        suggestions = [
            "🔍 **Room Allocation**: 'Which room is allocated Prof. Sharma on Monday?'",
            "📅 **Faculty Schedule**: 'Show me Prof. Mehta's schedule'",
            "📊 **Workload Reports**: 'What is Prof. Verma's workload?'",
            "👥 **Department Summary**: 'Give me CSE department workload summary'",
            "🆓 **Availability Check**: 'Which faculty is free on Tuesday at 2 PM?'",
            "📋 **Policy Search**: 'What are the university policies on maximum workload?'"
        ]
        
        for suggestion in suggestions:
            response += f"{suggestion}\n"
        
        response += "\n💡 **Tip**: Be specific about what you're looking for, and I'll provide detailed information!"
        
        return response
    
    def _suggest_clarification(self, query: str, context: dict, intent: str) -> str:
        """Suggest clarifications for incomplete queries."""
        if intent == "room_allocation":
            if not context['faculty']:
                return "I'd be happy to help with room allocation! Please specify a faculty member (e.g., 'Which room is allocated Prof. Sharma on Monday?')."
            elif not context['day']:
                return f"I can find room allocation for {context['faculty']}, but please specify a day (e.g., 'Which room is allocated {context['faculty']} on Monday?')."
        
        return "I need a bit more information to help you. Could you please be more specific about what you're looking for?"

def _handle_general_query(query: str) -> str:
    """Handle general timetable queries."""
    return "I can help you with:\n- Finding free faculty at specific times\n- Checking faculty schedules\n- Room availability\n- Course information\n\nPlease be more specific about what you're looking for."

# Helper functions for workload reports
def _handle_department_query(query: str) -> str:
    """Handle department workload queries."""
    words = query.lower().split()
    department = None
    
    # Look for department names
    departments = ['cse', 'eee', 'me', 'civil', 'ece', 'it']
    for word in words:
        if word in departments:
            department = word.upper()
            break
    
    if not department:
        return "Please specify a department (e.g., 'CSE department summary')"
    
    result = _data_loader.get_department_summary(department)
    
    if "error" in result:
        return result["error"]
    
    response = f"Department Summary for {result['department']}:\n\n"
    response += f"Total Faculty: {result['total_faculty']}\n"
    response += f"Total Hours: {result['total_hours']}\n\n"
    response += "Faculty Details:\n"
    for faculty in result['faculty_details']:
        response += f"- {faculty['name']}: {faculty['courses']} ({faculty['hours_per_week']} hours/week)\n"
    
    return response

def _handle_faculty_query(query: str) -> str:
    """Handle individual faculty workload queries."""
    words = query.lower().split()
    faculty_name = None
    
    # Look for "Prof." or "Professor"
    for i, word in enumerate(words):
        if word in ['prof.', 'professor'] and i + 1 < len(words):
            # Remove any punctuation from the name
            name_part = words[i + 1].replace("'s", "").replace("'", "").replace(".", "")
            faculty_name = "Prof." + name_part.capitalize()
            break
    
    if not faculty_name:
        return "Please specify a faculty member (e.g., 'Prof. Sharma workload')"
    
    result = _data_loader.get_faculty_workload(faculty_name)
    
    if "error" in result:
        return result["error"]
    
    response = f"Workload Report for {result['name']}:\n\n"
    response += f"Department: {result['department']}\n"
    response += f"Total Hours: {result['total_hours']}\n\n"
    response += "Courses:\n"
    for course in result['course_details']:
        response += f"- {course['course']}: {course['hours_per_week']} hours/week\n"
    
    return response

def _handle_all_faculty_query(query: str) -> str:
    """Handle all faculty workload query."""
    all_faculty = _data_loader.get_all_faculty()
    
    response = f"All Faculty Workload Summary:\n\n"
    response += f"Total Faculty: {len(all_faculty)}\n\n"
    
    # Get workload for each faculty
    for faculty_name in all_faculty[:10]:  # Limit to first 10 for readability
        result = _data_loader.get_faculty_workload(faculty_name)
        if "error" not in result:
            response += f"- {result['name']} ({result['department']}): {result['total_hours']} hours\n"
    
    if len(all_faculty) > 10:
        response += f"... and {len(all_faculty) - 10} more faculty members.\n"
    
    return response

def _handle_general_workload_query(query: str) -> str:
    """Handle general workload queries."""
    return "I can help you with:\n- Individual faculty workload reports\n- Department workload summaries\n- All faculty workload overview\n\nPlease be more specific about what you want."

class FacultyWorkloadAgent:
    """Main agent class for faculty workload management."""
    
    def __init__(self, model_name: str = "llama2"):
        """Initialize the agent with tools and LLM."""
        self.model_name = model_name
        self.llm = None
        self.agent = None
        self.agent_executor = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize data loader
            self.data_loader = FacultyDataLoader()
            
            # Initialize vector store
            self.vector_store = PolicyVectorStore()
            self.vector_store.load_policies_from_file("policies.txt")
            
            # Initialize LLM
            self.llm = Ollama(model=self.model_name)
            
            # Create tools
            self.rag_tool = create_rag_policy_tool(self.vector_store)
            self.timetable_tool = create_timetable_query_tool(self.data_loader)
            self.workload_tool = create_workload_report_tool(self.data_loader)
            
            # Create agent
            self._create_agent()
            
            print(f"Agent initialized successfully with model: {self.model_name}")
            
        except Exception as e:
            print(f"Error initializing agent: {e}")
            raise
    
    def _create_agent(self):
        """Create the LangChain agent."""
        tools = [self.rag_tool, self.timetable_tool, self.workload_tool]
        
        # Create prompt template
        prompt = PromptTemplate(
            template="""You are a helpful university faculty workload management assistant. You can help with:

1. Faculty workload queries - Get information about individual faculty members' teaching loads
2. Timetable queries - Find free faculty, check schedules, room availability
3. Policy queries - Search university policies related to workload and scheduling
4. Department summaries - Get workload overview for entire departments

Use the available tools to answer user questions. Always provide clear, helpful responses.

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}""",
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
        
        # Create agent
        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def query(self, question: str) -> str:
        """Process a user query."""
        try:
            response = self.agent_executor.invoke({"input": question})
            return response["output"]
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    # Initialize agent
    agent = FacultyWorkloadAgent()
    
    # Test queries
    test_queries = [
        "What is Prof. Sharma's workload?",
        "Which faculty is free on Tuesday at 2 PM?",
        "What are the university policies on maximum workload?",
        "Give me a summary of the CSE department workload"
    ]
    
    print("=== Testing Faculty Workload Agent ===")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        response = agent.query(query)
        print(f"Response: {response}")
        print("=" * 50)