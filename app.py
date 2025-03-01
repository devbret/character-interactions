import spacy
import json
import logging
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("character_analysis.log"), logging.StreamHandler()]
)

nlp = spacy.load("en_core_web_sm")

def read_text_file(file_path: Path) -> str:
    if not file_path.is_file():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with file_path.open('r', encoding='utf-8') as file:
            text = file.read()
            logging.info(f"Successfully read file: {file_path}")
            return text
    except Exception as e:
        logging.exception(f"Error reading file {file_path}: {e}")
        raise

def extract_characters(doc) -> List[str]:
    characters = sorted({ent.text for ent in doc.ents if ent.label_ == "PERSON"})

    if not characters:
        logging.warning("No character entities were detected in the text.")
    else:
        logging.info(f"Extracted {len(characters)} unique character(s).")

    return characters

def build_interaction_matrix(doc, characters: List[str]) -> List[List[int]]:
    if not characters:
        logging.warning("No characters found, skipping interaction matrix generation.")
        return []

    char_to_index = {char: i for i, char in enumerate(characters)}
    matrix_size = len(characters)
    interaction_matrix = [[0] * matrix_size for _ in range(matrix_size)]

    for sent in doc.sents:
        sentence_characters = {
            ent.text for ent in sent.ents 
            if ent.label_ == "PERSON" and ent.text in char_to_index
        }

        if len(sentence_characters) > 1:
            for char1 in sentence_characters:
                for char2 in sentence_characters:
                    if char1 != char2:
                        i, j = char_to_index[char1], char_to_index[char2]
                        interaction_matrix[i][j] += 1

    logging.info(f"Built interaction matrix of size {matrix_size}x{matrix_size}.")
    return interaction_matrix

def generate_json_output(characters: List[str], matrix: List[List[int]], output_file: Path) -> None:
    if not characters or not matrix:
        logging.warning("No interaction data to write. Output file will not be created.")
        return

    data = {
        "characters": characters,
        "matrix": matrix
    }

    try:
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Interaction data successfully written to {output_file}")
    except Exception as e:
        logging.exception(f"Error writing JSON file {output_file}: {e}")
        raise

def analyze_text(input_file: Path, output_file: Path) -> None:
    logging.info("Starting text analysis...")
    
    try:
        text = read_text_file(input_file)
        doc = nlp(text)

        characters = extract_characters(doc)
        interaction_matrix = build_interaction_matrix(doc, characters)
        generate_json_output(characters, interaction_matrix, output_file)
    except Exception as e:
        logging.error(f"Failed to analyze text: {e}")
        raise

if __name__ == "__main__":
    input_path = Path("text.txt")
    output_path = Path("character_interactions.json")

    analyze_text(input_path, output_path)
    logging.info("Process completed.")
