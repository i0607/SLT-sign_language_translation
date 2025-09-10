import pandas as pd
from google.cloud import translate_v2 as translate
import os
import time
import csv

def main():
    # === Setup ===
    # Path to your dataset
    input_file = r'D:\phoenix-2014-T.v3\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\annotations\manual\PHOENIX-2014-T.dev.corpus.csv'

    # Credentials (make sure you have a valid JSON file in the working dir)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'

    # Initialize translator client
    client = translate.Client()

    # Load dataset
    df = pd.read_csv(input_file, delimiter='|', encoding='utf-8')

    # Use the entire dataset (make a copy to avoid SettingWithCopy)
    sample = df.copy()

    # Add new columns for translations
    sample['orth_english'] = ''
    sample['translation_english'] = ''

    # === Translate loop ===
    for idx, row in sample.iterrows():
        orth_text = str(row['orth']) if 'orth' in sample.columns else ''
        trans_text = str(row['translation']) if 'translation' in sample.columns else ''

        # Translate orth
        if orth_text.strip():
            result = client.translate(orth_text, source_language='de', target_language='en')
            translated_orth = result['translatedText']
            sample.at[idx, 'orth_english'] = translated_orth
            print(f"\nORTH_EN: {translated_orth}")
            time.sleep(0.1)  # avoid rate limit

        # Translate translation
        if trans_text.strip():
            result = client.translate(trans_text, source_language='de', target_language='en')
            translated_trans = result['translatedText']
            sample.at[idx, 'translation_english'] = translated_trans
            print(f"TRANS_EN: {translated_trans}")
            
    # Optional: Print final translated texts for the first row in the dataset
    try:
        print("\nFinal translated texts for the first row in the dataset:")
        print(f"ORTH_EN: {sample.iloc[0]['orth_english']}")
        print(f"TRANS_EN: {sample.iloc[0]['translation_english']}")
    except Exception as e:
        print(f"Error printing final texts: {e}")
    time.sleep(0.1)

    # === Save result ===
    # Save with Excel-friendly settings: comma separator, quote all fields, UTF-8 BOM
    sample.to_csv(
        "PHOENIX-2014-T.dev.corpus.csv",
        index=False,
        sep=",",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )
    print("\n✓ Saved translated sample as PHOENIX-2014-T.dev.corpus.csv")

if __name__ == "__main__":
    main()
