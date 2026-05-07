# Quantifying Character Relationships

![Map of direct character interactions from William Gibson's "Neuromancer".](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/a7542eb2-1c70-42c2-9003-63c3dbecc7aa.png)

Analyzes text files to extract characters, infer relationships using multiple linguistic signals and visualize the resulting network in an interactive D3 graph with filtering and node-limit controls to make large literary networks easier to explore.

## Overview

This application extracts `PERSON` entities, normalizes names, filters obvious false positives and merges aliases carefully. It can also incorporate optional coreference support and preserves richer structure by tracking character presence across sentences, paragraphs and scenes.

The analysis builds multiple forms of relationship evidence, including sliding sentence-window co-mentions, same-paragraph and same-scene presence, dialogue-style proximity, dependency-based interaction cues and coreference-linked presence. The output JSON includes the character list and interaction matrix, as well as detailed `nodes` and `edges` data with metadata such as mentions, weighted degree, evidence breakdowns and confidence scores.

The JavaScript frontend turns `character_interactions.json` into an interactive D3 network graph and supporting ego-network views. It renders characters as nodes and their relationships as weighted links, while providing search, highlight, filtering, k-hop neighborhood controls, fit-to-screen behavior and a radial ego overlay with SVG download support.

To improve readability on dense corpora, the graph limits the default visible set to the top 15 nodes and includes a top-left interface control for increasing or decreasing how many nodes are shown. The UI also includes search and facet behavior making it easier to explore large literary networks without overwhelming the screen.

## Set Up

Below are instructions for installing and running this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/character-interactions.git`

4. Navigate to the repo's directory: `cd character-interactions`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate your virtual environment: `source venv/bin/activate`

7. Install the needed dependencies for running the script: `pip install -r requirements.txt`

8. Place your `.txt` files into the `input` directory for analysis

9. Process the input files: `python3 app.py`

10. Results will be output to the root of this project as `character_interactions.json`

11. Start a local HTTP server: `python3 -m http.server`

12. Open the frontend to explore processed data: `http://localhost:8000`

13. Stop the HTTP server when finished: `Ctrl + C`

14. Exit the virtual environment: `deactivate`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Analyze `.txt` files to identify character mentions and map relationships between them

- Generate structured interaction data showing which characters appear together

- Transform `.json` output into an interactive D3 network graph

- Enable users to visually explore and understand the social structure of a story

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
