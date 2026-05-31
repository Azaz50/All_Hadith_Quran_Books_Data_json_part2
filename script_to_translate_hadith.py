import os
import sys
import json
from tqdm import tqdm
import ctranslate2
import transformers

# =========================================================
# CONFIG
# =========================================================

# RUN:
# python3 script.py bukhari.json
# python3 script.py muslim.json

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CT2_MODEL_PATH = "nllb-200-ct2"

# =========================================================
# GPU SETTINGS
# =========================================================

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# =========================================================
# PERFORMANCE SETTINGS
# =========================================================

# GTX 1650 optimized
BATCH_SIZE = 24

# autosave after translations
SAVE_EVERY = 10

# =========================================================
# MODEL LIMITS
# =========================================================

# smaller = faster
MAX_INPUT_TOKENS = 384
MAX_DECODING_LENGTH = 192

# chunk size
MAX_CHARS_PER_CHUNK = 700

# =========================================================
# LANGUAGE CODES
# =========================================================

LANG_CODES = {
    "urdu": "urd_Arab",
    "hindi": "hin_Deva",
    "bengali": "ben_Beng"
}

# =========================================================
# NORMALIZATION
# =========================================================

REPLACEMENTS = {
    "भगवान": "अल्लाह",
    "ईश्वर": "अल्लाह",
    "खुदा": "अल्लाह",
}

def normalize_islamic_terms(text):

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    return text

# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model...")

translator = ctranslate2.Translator(
    CT2_MODEL_PATH,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    inter_threads=1,
    intra_threads=0
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.src_lang = "eng_Latn"

print(f"Model loaded on {DEVICE.upper()}")

# =========================================================
# SPLIT LONG TEXT
# =========================================================

def split_text(text, max_chars=MAX_CHARS_PER_CHUNK):

    text = text.replace("\n", " ")

    words = text.split()

    chunks = []
    current = ""

    for word in words:

        if len(current) + len(word) + 1 <= max_chars:

            current += " " + word

        else:

            chunks.append(current.strip())
            current = word

    if current.strip():

        chunks.append(current.strip())

    return chunks

# =========================================================
# PREPARE TOKENS
# =========================================================

def prepare_tokens(text):

    ids = tokenizer.encode(
        text,
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    return tokenizer.convert_ids_to_tokens(ids)

# =========================================================
# TRANSLATE TOKEN BATCH
# =========================================================

def translate_tokens_batch(
    token_batch,
    target_lang
):

    results = translator.translate_batch(
        token_batch,
        target_prefix=[
            [LANG_CODES[target_lang]]
        ] * len(token_batch),
        beam_size=1,
        max_decoding_length=MAX_DECODING_LENGTH,
        repetition_penalty=1.1,
        return_scores=False
    )

    outputs = []

    for result in results:

        output_tokens = result.hypotheses[0]

        if len(output_tokens) > 0:

            output_tokens = output_tokens[1:]

        output_ids = tokenizer.convert_tokens_to_ids(
            output_tokens
        )

        translated = tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )

        translated = normalize_islamic_terms(
            translated
        )

        outputs.append(
            translated.strip()
        )

    return outputs

# =========================================================
# ENSURE STRUCTURE
# =========================================================

def ensure_translation_object(
    hadith,
    lang
):

    if lang not in hadith:

        hadith[lang] = {
            "source": "NLLB-200",
            "reviewed": False,
            "narrator": "",
            "text": ""
        }

    elif isinstance(hadith[lang], str):

        hadith[lang] = {
            "source": "NLLB-200",
            "reviewed": False,
            "narrator": "",
            "text": hadith[lang]
        }

    else:

        hadith[lang].setdefault(
            "source",
            "NLLB-200"
        )

        hadith[lang].setdefault(
            "reviewed",
            False
        )

        hadith[lang].setdefault(
            "narrator",
            ""
        )

        hadith[lang].setdefault(
            "text",
            ""
        )

# =========================================================
# EMPTY CHECK
# =========================================================

def is_missing(value):

    if value is None:
        return True

    if not isinstance(value, str):
        return True

    if not value.strip():
        return True

    return False

# =========================================================
# SAFE SAVE
# =========================================================

def save_json_atomic(filepath, data):

    temp_path = filepath + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(
        temp_path,
        filepath
    )

# =========================================================
# BUILD TRANSLATION JOBS
# =========================================================

def build_jobs(hadiths):

    jobs = []

    for hadith_index, hadith in enumerate(hadiths):

        english = hadith.get(
            "english",
            {}
        )

        eng_text = english.get(
            "text",
            ""
        ).strip()

        eng_narrator = english.get(
            "narrator",
            ""
        ).strip()

        for lang in LANG_CODES.keys():

            ensure_translation_object(
                hadith,
                lang
            )

            # =========================================
            # TEXT
            # =========================================

            if (
                eng_text
                and
                is_missing(
                    hadith[lang].get(
                        "text",
                        ""
                    )
                )
            ):

                chunks = split_text(
                    eng_text
                )

                for chunk_index, chunk in enumerate(chunks):

                    jobs.append({
                        "hadith_index": hadith_index,
                        "lang": lang,
                        "field": "text",
                        "chunk_index": chunk_index,
                        "total_chunks": len(chunks),
                        "tokens": prepare_tokens(chunk)
                    })

            # =========================================
            # NARRATOR
            # =========================================

            if (
                eng_narrator
                and
                is_missing(
                    hadith[lang].get(
                        "narrator",
                        ""
                    )
                )
            ):

                jobs.append({
                    "hadith_index": hadith_index,
                    "lang": lang,
                    "field": "narrator",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "tokens": prepare_tokens(
                        eng_narrator
                    )
                })

    return jobs

# =========================================================
# PROCESS FILE
# =========================================================

def process_file(filepath):

    print(f"\nProcessing: {filepath}")

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except Exception as e:

        print(f"JSON load error: {e}")
        return

    hadiths = data.get(
        "hadiths",
        []
    )

    if not hadiths:

        print("No hadiths found.")
        return

    # =====================================================
    # BUILD JOBS
    # =====================================================

    print("\nPreparing translation jobs...")

    jobs = build_jobs(hadiths)

    print(f"Total jobs: {len(jobs)}")

    if not jobs:

        print("\nEverything already translated.")
        return

    # =====================================================
    # TEMP STORAGE
    # =====================================================

    chunk_storage = {}

    updated_count = 0

    # =====================================================
    # PROCESS JOBS
    # =====================================================

    for start in tqdm(
        range(0, len(jobs), BATCH_SIZE)
    ):

        batch_jobs = jobs[
            start:start + BATCH_SIZE
        ]

        # ================================================
        # GROUP BY LANGUAGE
        # ================================================

        grouped = {}

        for job in batch_jobs:

            grouped.setdefault(
                job["lang"],
                []
            ).append(job)

        # ================================================
        # PROCESS EACH LANGUAGE
        # ================================================

        for lang, lang_jobs in grouped.items():

            try:

                token_batch = [
                    job["tokens"]
                    for job in lang_jobs
                ]

                translated = (
                    translate_tokens_batch(
                        token_batch,
                        lang
                    )
                )

                for job, output in zip(
                    lang_jobs,
                    translated
                ):

                    key = (
                        job["hadith_index"],
                        job["lang"],
                        job["field"]
                    )

                    chunk_storage.setdefault(
                        key,
                        []
                    )

                    chunk_storage[key].append(
                        (
                            job["chunk_index"],
                            output
                        )
                    )

            except Exception as e:

                print(
                    f"\nBatch error ({lang}): {e}"
                )

        # ================================================
        # WRITE COMPLETED CHUNKS
        # ================================================

        completed_keys = []

        for key, chunk_list in chunk_storage.items():

            hadith_index, lang, field = key

            total_chunks = None

            for job in batch_jobs:

                if (
                    job["hadith_index"] == hadith_index
                    and
                    job["lang"] == lang
                    and
                    job["field"] == field
                ):

                    total_chunks = job[
                        "total_chunks"
                    ]

                    break

            if total_chunks is None:
                continue

            if len(chunk_list) >= total_chunks:

                chunk_list.sort(
                    key=lambda x: x[0]
                )

                final_text = " ".join(
                    chunk[1]
                    for chunk in chunk_list
                )

                hadiths[
                    hadith_index
                ][lang][field] = final_text

                completed_keys.append(key)

                updated_count += 1

        # cleanup
        for key in completed_keys:

            del chunk_storage[key]

        # ================================================
        # AUTO SAVE
        # ================================================

        if (
            updated_count % SAVE_EVERY == 0
            and
            updated_count != 0
        ):

            save_json_atomic(
                filepath,
                data
            )

            print(
                f"\nAuto-saved after "
                f"{updated_count} translations"
            )

    # =====================================================
    # FINAL SAVE
    # =====================================================

    save_json_atomic(
        filepath,
        data
    )

    print(f"\nFinished: {filepath}")

    print(
        f"Updated entries: "
        f"{updated_count}"
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")

        print(
            "python3 script.py bukhari.json"
        )

        sys.exit()

    target_file = sys.argv[1]

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    filepath = os.path.join(
        script_dir,
        target_file
    )

    if not os.path.exists(filepath):

        print(
            f"\nFile not found: {filepath}"
        )

        sys.exit()

    process_file(filepath)

    print("\nAll processing complete.")