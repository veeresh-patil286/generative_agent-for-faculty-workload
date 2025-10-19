import pandas as pd
from typing import List, Dict, Optional, Tuple
import re

class FacultyDataLoader:
    """Data loader for faculty workload and timetable management."""
    
    def __init__(self, faculty_file: str = "faculty_workload.csv", timetable_file: str = "timetable.csv"):
        """Initialize the data loader with CSV files."""
        self.faculty_df = pd.read_csv(faculty_file)
        self.timetable_df = pd.read_csv(timetable_file)
        
        # Clean data
        self.faculty_df = self.faculty_df.dropna()
        self.timetable_df = self.timetable_df.dropna()
        
        # Convert time format for easier parsing
        self.timetable_df['StartTime'] = self.timetable_df['Time'].str.split('-').str[0]
        self.timetable_df['EndTime'] = self.timetable_df['Time'].str.split('-').str[1]
    
    def get_faculty_workload(self, faculty_name: str) -> Dict:
        """Get workload information for a specific faculty member."""
        faculty_data = self.faculty_df[self.faculty_df['Name'].str.contains(faculty_name, case=False, na=False)]
        
        if faculty_data.empty:
            return {"error": f"Faculty member '{faculty_name}' not found"}
        
        result = {
            "name": faculty_data.iloc[0]['Name'],
            "department": faculty_data.iloc[0]['Department'],
            "courses": faculty_data['Course'].tolist(),
            "total_hours": faculty_data['HoursPerWeek'].sum(),
            "course_details": []
        }
        
        for _, row in faculty_data.iterrows():
            result["course_details"].append({
                "course": row['Course'],
                "hours_per_week": row['HoursPerWeek']
            })
        
        return result
    
    def get_faculty_schedule(self, faculty_name: str, day: Optional[str] = None) -> Dict:
        """Get schedule information for a specific faculty member."""
        faculty_schedule = self.timetable_df[
            self.timetable_df['Faculty'].str.contains(faculty_name, case=False, na=False)
        ]
        
        if day:
            faculty_schedule = faculty_schedule[
                faculty_schedule['Day'].str.contains(day, case=False, na=False)
            ]
        
        if faculty_schedule.empty:
            return {"error": f"No schedule found for '{faculty_name}'" + (f" on {day}" if day else "")}
        
        result = {
            "name": faculty_name,
            "day": day,
            "sessions": []
        }
        
        for _, row in faculty_schedule.iterrows():
            result["sessions"].append({
                "day": row['Day'],
                "time": row['Time'],
                "course": row['Course'],
                "room": row['Room']
            })
        
        return result
    
    def get_free_faculty(self, day: str, time: str) -> Dict:
        """Find faculty members who are free at a specific day and time."""
        # Convert time to 24-hour format for comparison
        time_24 = self._convert_to_24hour(time)
        
        # Find all faculty teaching at the specified day and time
        busy_faculty = self.timetable_df[
            (self.timetable_df['Day'].str.contains(day, case=False, na=False)) &
            (self.timetable_df['StartTime'] <= time_24) &
            (self.timetable_df['EndTime'] > time_24)
        ]['Faculty'].tolist()
        
        # Get all faculty members
        all_faculty = self.faculty_df['Name'].unique().tolist()
        
        # Find free faculty
        free_faculty = [f for f in all_faculty if f not in busy_faculty]
        
        # Get details of busy faculty
        busy_details = []
        for _, row in self.timetable_df[
            (self.timetable_df['Day'].str.contains(day, case=False, na=False)) &
            (self.timetable_df['StartTime'] <= time_24) &
            (self.timetable_df['EndTime'] > time_24)
        ].iterrows():
            busy_details.append({
                "name": row['Faculty'],
                "course": row['Course'],
                "room": row['Room']
            })
        
        return {
            "day": day,
            "time": time,
            "free_faculty": free_faculty,
            "busy_faculty": busy_details
        }
    
    def get_department_summary(self, department: str) -> Dict:
        """Get workload summary for a specific department."""
        dept_faculty = self.faculty_df[
            self.faculty_df['Department'].str.contains(department, case=False, na=False)
        ]
        
        if dept_faculty.empty:
            return {"error": f"Department '{department}' not found"}
        
        result = {
            "department": department,
            "total_faculty": len(dept_faculty),
            "total_hours": dept_faculty['HoursPerWeek'].sum(),
            "faculty_details": []
        }
        
        for _, row in dept_faculty.iterrows():
            result["faculty_details"].append({
                "name": row['Name'],
                "courses": row['Course'],
                "hours_per_week": row['HoursPerWeek']
            })
        
        return result
    
    def search_faculty_by_course(self, course_name: str) -> List[Dict]:
        """Find faculty members teaching a specific course."""
        course_faculty = self.faculty_df[
            self.faculty_df['Course'].str.contains(course_name, case=False, na=False)
        ]
        
        result = []
        for _, row in course_faculty.iterrows():
            result.append({
                "name": row['Name'],
                "department": row['Department'],
                "course": row['Course'],
                "hours_per_week": row['HoursPerWeek']
            })
        
        return result
    
    def get_room_schedule(self, room: str, day: Optional[str] = None) -> Dict:
        """Get schedule for a specific room."""
        room_schedule = self.timetable_df[
            self.timetable_df['Room'].str.contains(room, case=False, na=False)
        ]
        
        if day:
            room_schedule = room_schedule[
                room_schedule['Day'].str.contains(day, case=False, na=False)
            ]
        
        if room_schedule.empty:
            return {"error": f"No schedule found for room '{room}'" + (f" on {day}" if day else "")}
        
        result = {
            "room": room,
            "day": day,
            "sessions": []
        }
        
        for _, row in room_schedule.iterrows():
            result["sessions"].append({
                "day": row['Day'],
                "time": row['Time'],
                "course": row['Course'],
                "faculty": row['Faculty']
            })
        
        return result
    
    def _convert_to_24hour(self, time_str: str) -> str:
        """Convert time string to 24-hour format for comparison."""
        # Handle formats like "2 PM", "2:00 PM", "14:00", etc.
        time_str = time_str.strip().upper()
        
        if 'AM' in time_str or 'PM' in time_str:
            # 12-hour format
            time_part = re.sub(r'[AP]M', '', time_str).strip()
            if ':' not in time_part:
                time_part += ':00'
            
            hour, minute = time_part.split(':')
            hour = int(hour)
            minute = int(minute)
            
            if 'PM' in time_str and hour != 12:
                hour += 12
            elif 'AM' in time_str and hour == 12:
                hour = 0
            
            return f"{hour:02d}:{minute:02d}"
        else:
            # Assume 24-hour format
            if ':' not in time_str:
                time_str += ':00'
            return time_str
    
    def get_all_departments(self) -> List[str]:
        """Get list of all departments."""
        return self.faculty_df['Department'].unique().tolist()
    
    def get_all_faculty(self) -> List[str]:
        """Get list of all faculty members."""
        return self.faculty_df['Name'].unique().tolist()

# Example usage and testing
if __name__ == "__main__":
    # Initialize data loader
    loader = FacultyDataLoader()
    
    print("=== Testing Faculty Data Loader ===")
    
    # Test faculty workload
    print("\n1. Faculty Workload Test:")
    workload = loader.get_faculty_workload("Prof.Sharma")
    print(f"Prof.Sharma's workload: {workload}")
    
    # Test faculty schedule
    print("\n2. Faculty Schedule Test:")
    schedule = loader.get_faculty_schedule("Prof.Sharma", "Monday")
    print(f"Prof.Sharma's Monday schedule: {schedule}")
    
    # Test free faculty
    print("\n3. Free Faculty Test:")
    free_faculty = loader.get_free_faculty("Tuesday", "2 PM")
    print(f"Free faculty on Tuesday at 2 PM: {free_faculty}")
    
    # Test department summary
    print("\n4. Department Summary Test:")
    dept_summary = loader.get_department_summary("CSE")
    print(f"CSE Department Summary: {dept_summary}")
    
    # Test all departments
    print("\n5. All Departments:")
    departments = loader.get_all_departments()
    print(f"Departments: {departments}")
    
    # Test all faculty
    print("\n6. All Faculty:")
    faculty = loader.get_all_faculty()
    print(f"Faculty count: {len(faculty)}")
    print(f"First 5 faculty: {faculty[:5]}")
