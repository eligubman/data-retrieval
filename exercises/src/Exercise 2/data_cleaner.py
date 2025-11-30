import os
import re

def clean_text(text):
    # remove HTML/XML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove metadata lines
    text = re.sub(r"^(Title:|Section:|Date:|Volume:|Issue:|Pages:).*$",
                  " ", text, flags=re.MULTILINE)

    # remove long separation lines
    text = re.sub(r"={5,}", " ", text)

    # remove country names/abbreviations (US / UK variants)
    country_patterns = [
        r"\bUSA\b", r"\bU\.S\.A\.\b", r"\bU\.S\.\b",
        r"\bUS\b", r"\bU-K\b", r"\bU\.K\.\b", r"\bUK\b"
    ]
    for p in country_patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)

    # remove HTML entities like &#x27;
    text = re.sub(r"&#x?\w+;", " ", text)

    # keep only letters + spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text)

    # lowercase
    return text.lower().strip()


def clean_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        full_path = os.path.join(input_folder, filename)

        # skip non-files
        if not os.path.isfile(full_path):
            continue

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        cleaned = clean_text(raw)

        out_path = os.path.join(output_folder, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"✓ cleaned: {filename}")


# ---- MAIN ----
if __name__ == "__main__":

    clean_folder("data/US_congressional_speeches_Text_Files/US_congressional_speeches_Text_Files", "clean_data/cleaned_us")
    clean_folder("data/UK_british_debates_text_files_normalize/british_debates_text_files_normalize", "clean_data/cleaned_uk")

    print("\n✔ Done! Cleaned files saved in 'cleaned_us/' and 'cleaned_uk/'")
