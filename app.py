import spacy
import json
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

def read_text_file(file_path: Path) -> str:
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open('r', encoding='utf-8') as file:
        return file.read()

def extract_characters(doc) -> list[str]:
    characters = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
    return sorted(characters)

def build_interaction_matrix(doc, characters: list[str]) -> list[list[int]]:
    char_to_index = {char: i for i, char in enumerate(characters)}

    matrix_size = len(characters)
    interaction_matrix = [[0] * matrix_size for _ in range(matrix_size)]

    for sent in doc.sents:
        sentence_characters = {
            ent.text for ent in sent.ents 
            if ent.label_ == "PERSON" and ent.text in char_to_index
        }
        for char1 in sentence_characters:
            for char2 in sentence_characters:
                if char1 != char2:
                    i = char_to_index[char1]
                    j = char_to_index[char2]
                    interaction_matrix[i][j] += 1

    return interaction_matrix

def generate_json_output(characters: list[str], matrix: list[list[int]], output_file: Path) -> None:
    data = {
        "characters": characters,
        "matrix": matrix
    }
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def analyze_text(input_file: Path, output_file: Path) -> None:
    text = read_text_file(input_file)
    doc = nlp(text)

    characters = extract_characters(doc)
    interaction_matrix = build_interaction_matrix(doc, characters)
    generate_json_output(characters, interaction_matrix, output_file)

if __name__ == "__main__":
    input_path = Path("text.txt")
    output_path = Path("character_interactions.json")

    analyze_text(input_path, output_path)
    print(f"Interaction data written to {output_path}")
