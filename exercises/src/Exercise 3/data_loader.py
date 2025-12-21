import os
import re

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.us_dir = os.path.join(base_dir, "US_congressional_speeches_Text_Files")
        self.uk_dir = os.path.join(base_dir, "british_debates_text_files_normalize")

    def load_us_data(self):
        documents = []
        if not os.path.exists(self.us_dir):
            print(f"Warning: US data directory not found at {self.us_dir}")
            return documents

        for filename in sorted(os.listdir(self.us_dir)):
            if not filename.endswith(".txt"):
                continue
            
            filepath = os.path.join(self.us_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Split by the separator
            sections = content.split("================================================================================")
            
            for section in sections:
                if not section.strip():
                    continue
                
                # Extract metadata if possible (Title, Date, etc are usually at the top of the section)
                # The format seems to be:
                # Title: ...
                # ...
                # Date: ...
                # ...
                # <pre> ... </pre>
                
                lines = section.strip().split('\n')
                title = "Unknown Title"
                date = "Unknown Date"
                text_content = ""
                
                # Simple parsing strategy
                header_part = []
                body_part = []
                in_pre = False
                
                for line in lines:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("Date:"):
                        date = line.replace("Date:", "").strip()
                    
                    if "<pre>" in line:
                        in_pre = True
                        continue
                    if "</pre>" in line:
                        in_pre = False
                        continue
                        
                    if in_pre:
                        body_part.append(line)
                    else:
                        header_part.append(line)

                # If body is empty, maybe the format is different or it's just text
                if not body_part:
                    # Fallback: treat everything after headers as content
                    # Or just use the whole section text if <pre> tags aren't found
                    text_content = section.strip()
                else:
                    text_content = "\n".join(body_part)

                # Clean up text content (remove HTML-like tags if any remain, excessive whitespace)
                text_content = re.sub(r'<[^>]+>', '', text_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()

                if text_content:
                    documents.append({
                        "source": "US",
                        "filename": filename,
                        "title": title,
                        "date": date,
                        "content": text_content
                    })
        
        print(f"Loaded {len(documents)} US documents.")
        return documents

    def load_uk_data(self):
        documents = []
        if not os.path.exists(self.uk_dir):
            print(f"Warning: UK data directory not found at {self.uk_dir}")
            return documents

        for filename in sorted(os.listdir(self.uk_dir)):
            if not filename.endswith(".txt"):
                continue
            
            filepath = os.path.join(self.uk_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # UK files seem to be one long transcript.
            # We'll treat the whole file as one document for now.
            # Metadata can be inferred from filename (e.g., debates2023-06-28.txt)
            
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            date = date_match.group(1) if date_match else "Unknown Date"
            
            # Clean content
            text_content = re.sub(r'\s+', ' ', content).strip()
            
            if text_content:
                documents.append({
                    "source": "UK",
                    "filename": filename,
                    "title": f"UK Debate {date}",
                    "date": date,
                    "content": text_content
                })

        print(f"Loaded {len(documents)} UK documents.")
        return documents

    def load_all(self):
        return self.load_us_data() + self.load_uk_data()
