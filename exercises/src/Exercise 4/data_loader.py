import os
import re
from datetime import datetime
# וודאו שהתקנתם: pip install python-dateutil
from dateutil import parser 

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.us_dir = os.path.join(base_dir, "US_congressional_speeches_Text_Files")
        self.uk_dir = os.path.join(base_dir, "british_debates_text_files_normalize")

    def normalize_date(self, date_str):
        """
        נרמול תאריכים לפורמט ISO 8601 (YYYY-MM-DD) כפי שנדרש בשלב 2.
        """
        try:
            dt = parser.parse(date_str)
            return dt.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            return "Unknown"

    def load_us_data(self):
        documents = []
        if not os.path.exists(self.us_dir):
            return documents

        for filename in sorted(os.listdir(self.us_dir)):
            if not filename.endswith(".txt"): continue
            
            filepath = os.path.join(self.us_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            sections = content.split("=" * 80)
            for section in sections:
                if not section.strip(): continue
                
                lines = section.strip().split('\n')
                title, raw_date, in_pre = "Unknown Title", "Unknown Date", False
                body_part = []
                
                for line in lines:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("Date:"):
                        raw_date = line.replace("Date:", "").strip()
                    
                    if "<pre>" in line: in_pre = True
                    elif "</pre>" in line: in_pre = False
                    elif in_pre: body_part.append(line)

                text_content = "\n".join(body_part) if body_part else section.strip()
                text_content = re.sub(r'<[^>]+>', '', text_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()

                if text_content:
                    # חילוץ ונירמול תאריך [cite: 17, 20]
                    normalized_timestamp = self.normalize_date(raw_date)
                    documents.append({
                        "source": "US",
                        "filename": filename,
                        "title": title,
                        "date": raw_date,
                        "timestamp": normalized_timestamp, # אחסון מטא-דאטה מובנה [cite: 18]
                        "content": text_content
                    })
        return documents

    def load_uk_data(self):
        documents = []
        if not os.path.exists(self.uk_dir):
            return documents

        for filename in sorted(os.listdir(self.uk_dir)):
            if not filename.endswith(".txt"): continue
            
            filepath = os.path.join(self.uk_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # חילוץ תאריך משם הקובץ [cite: 17]
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            raw_date = date_match.group(1) if date_match else "Unknown Date"
            
            # נירמול התאריך 
            normalized_timestamp = self.normalize_date(raw_date)
            
            text_content = re.sub(r'\s+', ' ', content).strip()
            
            if text_content:
                documents.append({
                    "source": "UK",
                    "filename": filename,
                    "title": f"UK Debate {raw_date}",
                    "date": raw_date,
                    "timestamp": normalized_timestamp, # אחסון מטא-דאטה מובנה [cite: 18]
                    "content": text_content
                })
        return documents

    def load_all(self):
        return self.load_us_data() + self.load_uk_data()