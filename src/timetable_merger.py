import pandas as pd
import numpy as np
from config import DAYS_OF_WEEK, TIME_SLOTS


class TimetableMerger:
    def __init__(self):
        self.days = DAYS_OF_WEEK
        self.time_slots = TIME_SLOTS
    
    def normalize_dataframe(self, df):
        """Normalize DataFrame structure for merging"""
        if df.empty:
            return df
        
        # Try to identify day and time columns
        df_normalized = df.copy()
        
        # Convert all column names to string and strip whitespace
        df_normalized.columns = [str(col).strip() for col in df_normalized.columns]
        
        return df_normalized
    
    def merge_timetables(self, dataframes, labels=None):
        """Merge multiple timetables into one consolidated view"""
        if not dataframes:
            return pd.DataFrame()
        
        if labels is None:
            labels = [f"Timetable {i+1}" for i in range(len(dataframes))]
        
        # Normalize all dataframes
        normalized_dfs = [self.normalize_dataframe(df) for df in dataframes]
        
        # Create a master timetable
        master_data = []
        
        for i, (df, label) in enumerate(zip(normalized_dfs, labels)):
            if df.empty:
                continue
            
            # Add source label
            df_copy = df.copy()
            df_copy['Source'] = label
            master_data.append(df_copy)
        
        if not master_data:
            return pd.DataFrame()
        
        # Concatenate all timetables
        merged_df = pd.concat(master_data, ignore_index=True)
        
        return merged_df
    
    def create_structured_timetable(self, dataframes, labels=None):
        """Create a structured timetable with days as columns and times as rows"""
        if not dataframes:
            return pd.DataFrame()
        
        if labels is None:
            labels = [f"Student {i+1}" for i in range(len(dataframes))]
        
        # Initialize structured timetable
        structured = {}
        
        for time_slot in self.time_slots:
            structured[time_slot] = {day: [] for day in self.days}
        
        # Fill in classes from each dataframe
        for df, label in zip(dataframes, labels):
            if df.empty:
                continue
            
            # Try to extract schedule information
            for _, row in df.iterrows():
                # Look for day and time information in the row
                day = None
                time = None
                classes = []
                
                # Extract information from row
                for col, val in row.items():
                    val_str = str(val).strip()
                    
                    # Check if it's a day
                    if any(d.lower() in val_str.lower() for d in self.days):
                        for d in self.days:
                            if d.lower() in val_str.lower():
                                day = d
                                break
                    
                    # Check if it's a time slot
                    if ':' in val_str and '-' in val_str:
                        time = val_str
                    
                    # Otherwise, it might be a class name
                    elif val_str and val_str not in ['', 'nan', 'None']:
                        classes.append(val_str)
                
                # Add to structured timetable if we found valid data
                if day and time and classes:
                    for cls in classes:
                        structured[time][day].append(f"{label}: {cls}")
        
        # Convert to DataFrame
        df_structured = pd.DataFrame(structured).T
        
        # Clean up: join multiple entries with semicolons
        for col in df_structured.columns:
            df_structured[col] = df_structured[col].apply(
                lambda x: '; '.join(x) if isinstance(x, list) and x else ''
            )
        
        return df_structured
    
    def export_merged_timetable(self, merged_df, output_path, format='xlsx'):
        """Export merged timetable to file"""
        if format == 'xlsx':
            merged_df.to_excel(output_path, index=False)
        elif format == 'csv':
            merged_df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path
