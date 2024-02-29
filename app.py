import spacy
from collections import defaultdict
import json

nlp = spacy.load("en_core_web_sm")

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_characters(doc):
    characters = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
    return sorted(list(characters))

def build_interaction_matrix(doc, characters):
    char_index = {char: i for i, char in enumerate(characters)}
    matrix = [[0 for _ in characters] for _ in characters]

    for sent in doc.sents:
        sent_chars = {ent.text for ent in sent.ents if ent.label_ == "PERSON" and ent.text in char_index}
        for char1 in sent_chars:
            for char2 in sent_chars:
                if char1 != char2:
                    i, j = char_index[char1], char_index[char2]
                    matrix[i][j] += 1

    return matrix

def generate_json(characters, matrix, output_file='character_interactions.json'):
    data = {
        "characters": characters,
        "matrix": matrix
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def analyze_text(file_path):
    text = read_file(file_path)
    doc = nlp(text)

    characters = extract_characters(doc)
    matrix = build_interaction_matrix(doc, characters)

    generate_json(characters, matrix)

analyze_text('text.txt')
