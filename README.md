## Thesis Translation Utilities

This project contains small utilities to translate the PHOENIX-2014-T dataset from German to English.

### Prerequisites
- Python 3.9+
- Google Cloud project with the Cloud Translation API enabled
- A service account key JSON file named `credentials.json`

### Installation
```bash
pip install pandas google-cloud-translate
```

Place `credentials.json` in the project root or set the env var to its full path:
```bash
set GOOGLE_APPLICATION_CREDENTIALS=credentials.json
```

### Files
- `translation.py`: Translates the `orth` and `translation` columns to English (`orth_english`, `translation_english`).

### Usage
1) Translate source CSV (edit input path inside the script as needed):
```bash
python translation.py
```
Output: the translated version of the input file (Excel‑friendly: UTF‑8 BOM and all fields quoted).



### Excel/CSV Notes
- Because translations often contain commas, outputs are saved with all fields quoted and UTF‑8 BOM so Excel imports the full text into single cells.
- If opening in Excel via Data → From Text/CSV, confirm delimiter is comma and encoding is UTF‑8.

