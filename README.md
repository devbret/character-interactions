# Quantifying Character Relationships

![Map of direct character interactions from William Gibson's "Neuromancer".](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/653a802f-d491-45db-a3a6-e35952a870c6.png)

Analyzes text files to extract characters, infer relationships using multiple linguistic signals and visualize the resulting network in an interactive D3 graph with filtering and node-limit controls to make large literary networks easier to explore.

## Overview

This application extracts `PERSON` entities, normalizes names, filters obvious false positives and merges aliases carefully. It can also incorporate optional coreference support and preserves richer structure by tracking character presence across sentences, paragraphs and scenes.

The analysis builds multiple forms of relationship evidence, including sliding sentence-window co-mentions, same-paragraph and same-scene presence, dialogue-style proximity, dependency-based interaction cues and coreference-linked presence. The output JSON includes the character list and interaction matrix, as well as detailed `nodes` and `edges` data with metadata such as mentions, weighted degree, evidence breakdowns and confidence scores.

The JavaScript frontend turns `character_interactions.json` into an interactive D3 network graph and supporting ego-network views. It renders characters as nodes and their relationships as weighted links, while providing search, highlight, filtering, k-hop neighborhood controls, fit-to-screen behavior and a radial ego overlay with SVG download support.

To improve readability on dense corpora, the graph limits the default visible set to the top 15 nodes and includes a top-left interface control for increasing or decreasing how many nodes are shown. The UI also includes search and facet behavior, making it easier to explore large literary networks without overwhelming the screen.
