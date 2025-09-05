# DOI-Miner-Internship
DOI Miner is a Python pipeline I built during my research internship to collect papers by DOI (Crossref/Springer support + local HTML), normalize metadata, extract abstracts/full text, and run analytics—keyword frequencies, TF-IDF, topic modeling, and publication-trend charts. It’s configurable via simple filters (e.g., “2D materials”, “memristor/resistive switching”) and exports clean corpora for downstream ML/LLM tasks.

**Highlights**

Robust DOI harvesters with retry/rate-limit handling

Text preprocessing (spaCy/NLTK), HTML/PDF parsing (BeautifulSoup; optional ChemDataParser)

Analytics: keyword counts, time trends, TF-IDF, LDA topic models

Reproducible runs via run_pipeline.py and YAML configs; clean CSV/JSON exports

Stack: Python, pandas, spaCy, NLTK, scikit-learn, BeautifulSoup.
Note: Respect publisher TOS/licensing when fetching or redistributing content.
