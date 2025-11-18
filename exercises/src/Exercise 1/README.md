בס"ד  
אפרים אלגרבלי - 212451074  
אלי גובמן - 213662364  
---

## תיאור כללי
הפרויקט מממש שרשור מלא משלב הורדת הפרוטוקולים מאתר parliament.uk דרך ניקוי, למטיזציה, בניית ייצוגים מבוססי BM25/Okapi ווקטורי Word2Vec, ועד הטמעת מסמכים עם SBERT ו-SimCSE. כל הסקריפטים רצים עם `uv run <script.py>` כדי לשמור על סביבת חבילות סגורה.

## שלבי העיבוד (על פי הסדר הנדרש)
1. **הורדת הנתונים** – `uv run data_scraper.py` מושך את כל קבצי ה-XML החל מ-`debates2023-06-28d.xml` ומאחסן אותם תחת `data/raw_data`.
2. **ניקוי מבני XML** – `uv run data_cleaner.py` יוצר גרסאות נקיות ב-`data/cleaned_data` ובמקביל מפיק טקסטים גולמיים ב-`data/cleaned_text`.
3. **למטיזציה** – `uv run data_lemmatized.py` (לאחר `uv run python -m spacy download en_core_web_sm`) מפיק קבצי למות אל תוך `data/lemmatized_data`.
4. **וקטורים מבוססי BM25/Okapi** – `uv run data_vectoriztion.py` קורא את שני התיקיות ומייצר מטריצות BM25 דלילות (`metrics/tf_idf_vectors/<corpus>/tfidf_sparse_matrix.npz`) יחד עם אובייקט BM25 (`bm25_model.joblib`) וקובץ מונחים.
בעצם עושה מטריצות של emmbeding והוא גם מתחשב בגודל הקובץ עם .הנרמול 
5. **Word2Vec** – `uv run data_wordtovec.py` מאמן מודלים (עם ובלי מילות עצירה) ושומר את הווקטורים תחת `metrics/word_to_vec/...`.
בעצם עוזר לנו למצוא קשר סמנטי בין המילים

6. **SBERT** - ממפה את המסמכים למרחב וקטורי כדי לאפשר חיפוש סמנטי בין משפטים שלמים או מסמכים שלמים

7. **SimCSE** - דרך נוספת לייצוג מסמכים כמו ה SBERT אבל יותר מדויק 
ובנוסף זה עוד מודל לאימות 

6. **הטמעת מסמכים** –
	- `uv run data_sbert.py` (דורש `huggingface_hub login`) מחזיר `metrics/sbert_output`.
	- `uv run data_simcse_embeddings.py` פועל באותה צורה ומגישה fallback אוטומטי לדגם MiniLM.
7. **מדדי חשיבות** –
	- `uv run information_gain.py` מייצא את `information_gain_lemmatized_data.csv`.
	אנחנו מחשבים כל מילה עד כמה היא חשובה ביחס לשאר המילים
	- `uv run sum_culoms_information_gain.py` מסכם את ציוני ה-BM25 לעמודות ושומר את `tfidf_column_sums.csv`.
	בעצם כאן בדקנו כל עמודה ביחס לסכום הכולל של הטבלה 

## טבלאות (20 רשומות ראשונות בלבד)

### Information Gain על קורפוס למוט
| # | מונח | נוכחות | היעדרות | IG |
|---|------|--------|---------|-----|
|1|stephen|473|473|1.000000|
|2|opportunity|473|473|1.000000|
|3|national|473|473|1.000000|
|4|family|473|473|1.000000|
|5|continue|474|472|0.999997|
|6|high|474|472|0.999997|
|7|alex|472|474|0.999997|
|8|future|475|471|0.999987|
|9|deliver|470|476|0.999971|
|10|cross|476|470|0.999971|
|11|raise|476|470|0.999971|
|12|jones|476|470|0.999971|
|13|bring|477|469|0.999948|
|14|point|477|469|0.999948|
|15|johnson|469|477|0.999948|
|16|end|469|477|0.999948|
|17|meet|469|477|0.999948|
|18|secretary|477|469|0.999948|
|19|home|469|477|0.999948|
|20|think|469|477|0.999948|

### סכומי ציוני BM25 (tfidf_column_sums.csv)
| # | מונח | סכום BM25 | חשיבות יחסית |
|---|------|-----------|----------------|
|1|government|90.189686|0.0052774662|
|2|hon|88.197722|0.0051609060|
|3|people|70.678372|0.0041357580|
|4|make|65.962444|0.0038598046|
|5|work|62.481551|0.0036561195|
|6|member|62.333186|0.0036474379|
|7|year|54.999316|0.0032182952|
|8|right|54.269063|0.0031755643|
|9|support|50.374677|0.0029476835|
|10|need|49.243398|0.0028814865|
|11|minister|48.642977|0.0028463528|
|12|say|47.703940|0.0027914049|
|13|friend|47.698197|0.0027910688|
|14|house|45.023666|0.0026345681|
|15|andrew|39.536567|0.0023134895|
|16|time|38.994613|0.0022817770|
|17|new|35.642220|0.0020856111|
|18|know|35.594241|0.0020828036|
|19|country|34.026022|0.0019910390|
|20|community|32.452599|0.0018989698|

## מיקומי קבצים חשובים
- נתונים גולמיים: `data/raw_data`
- נתונים נקיים: `data/cleaned_data`, `data/cleaned_text`
- למות: `data/lemmatized_data`
- מטריצות BM25: `metrics/tf_idf_vectors/<corpus>`
- Word2Vec: `metrics/word_to_vec/<corpus>`
- SBERT / SimCSE: `metrics/sbert_output`, `metrics/simcse_output`
- מידע נוסף: `information_gain_lemmatized_data.csv`, `metrics/tf_idf_vectors/lemmatized_data/tfidf_column_sums.csv`