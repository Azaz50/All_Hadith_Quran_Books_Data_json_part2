# Islamic Texts Translation

This repository contains JSON data for Hadith and Quran books, along with a script (`script_to_translate_hadith.py`) to translate the english text into other languages (Urdu, Hindi, Bengali) using the `facebook/nllb-200-distilled-600M` model optimized with `ctranslate2`.

## Installation Process

To run the translation script, you need to install the required Python packages and prepare the CTranslate2 model.

### 1. Install Required Python Packages

Install the necessary dependencies via pip:

```bash
pip install ctranslate2 transformers torch tqdm
```

### 2. Prepare the `nllb-200-ct2` Model

The translation script loads an optimized CTranslate2 version of the NLLB-200 model from the `nllb-200-ct2/` directory.

If you don't already have this directory, you can generate it from the Hugging Face model using the `ct2-transformers-converter` tool (which is included when you install `ctranslate2`).

Run the following command to download the model and convert it:

```bash
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-200-ct2 --copy_files tokenizer.json README.md
```

This will create a local folder named `nllb-200-ct2/` containing `model.bin`, `config.json`, and `shared_vocabulary.json`.

*(Note: The `nllb-200-ct2/` directory is added to `.gitignore` so the large model files are not committed to the repository.)*

### 3. Usage

Run the script by passing the name of the JSON file you want to process:

```bash
python3 script_to_translate_hadith.py bukhari.json
```



# Hadith Translation Pipeline

A scalable Python-based translation pipeline for converting Arabic Hadith collections into multilingual JSON datasets.

This project was built to support large-scale Islamic AI systems such as **Hidhaya AI**, enabling authentic Hadith retrieval across multiple languages while preserving original Arabic text and metadata.

---

## Features

* Translate Arabic Hadith collections into multiple languages
* Supports:

  * English
  * Urdu
  * Hindi
  * Bengali
* Preserves original Arabic text
* Maintains Hadith metadata:

  * Book ID
  * Chapter ID
  * Hadith ID
  * Hadith Number (`idInBook`)
  * Narrator
* JSON-based output for easy integration
* Batch processing for large Hadith collections
* Designed for millions of records
* AI-ready dataset generation

---

## Supported Collections

The pipeline can be used for major Hadith books including:

* Sahih al-Bukhari
* Sahih Muslim
* Sunan Abu Dawood
* Jami' al-Tirmidhi
* Sunan al-Nasa'i
* Sunan Ibn Majah
* Musnad Ahmad ibn Hanbal
* Muwatta Imam Malik
* Sunan al-Darimi
* Additional collections can be added easily

---

## Dataset Structure

Example output:

```json
{
  "id": 1,
  "idInBook": 1,
  "chapterId": 1,
  "bookId": 1,
  "arabic": "...",
  "english": {
    "narrator": "...",
    "text": "..."
  },
  "urdu": {
    "narrator": "...",
    "text": "..."
  },
  "hindi": {
    "narrator": "...",
    "text": "..."
  },
  "bengali": {
    "narrator": "...",
    "text": "..."
  }
}
```

---

## Use Cases

* Islamic AI Assistants
* Quran & Hadith Search Engines
* Semantic Search Systems
* Islamic Chatbots
* Knowledge Retrieval Systems
* Educational Applications
* Research Projects

---

## Project Goals

The primary objective of this project is to create a multilingual Hadith dataset that can power advanced Islamic AI systems while maintaining authenticity and traceability.

Key goals:

* Preserve original Arabic sources
* Provide multilingual accessibility
* Enable semantic search
* Support AI-assisted Islamic learning
* Build scalable datasets for future Islamic technologies

---

## Technology Stack

* Python
* JSON Processing
* Batch Translation Pipeline
* Dataset Validation
* Large File Processing

---

## Future Improvements

* Arabic root-word extraction
* Islamic topic tagging
* Semantic indexing
* Vector embeddings
* Translation quality review
* Multi-language search optimization
* Quran dataset integration

---

## Disclaimer

This repository provides translated Hadith datasets for educational and research purposes.

Translations may require scholarly review before being used for formal Islamic rulings (fatwa), academic citation, or religious verdicts.

Always refer to qualified Islamic scholars for authoritative interpretation.

---

## Author

Azaz Mohammad

Built to support the vision of creating a large-scale multilingual Islamic AI ecosystem.

