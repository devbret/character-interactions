# Quantifying Character Relationships

![Map of direct character interactions from William Gibson's "Neuromancer".](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/abc09257-64f0-41f3-9fb0-8aeae6d2db01.png)

This app analyzes a body of text to automatically detect, merge and quantify character co-occurrences, then renders an interactive D3 network so you can visually explore relationships.

## Overview

This program analyzes a collection of `.txt` files to automatically detect and map relationships between characters in a text corpus. It uses spaCy to find `PERSON` entities, normalizes their names and filters out rarely mentioned names. It then builds an alias map for merging different forms of the same character using sub-name checks, shared last names and fuzzy string similarity.

As it processes all files, the app tracks which characters appear in which sentences and uses a sliding sentence window to count how often pairs of characters co-occur, producing an interaction matrix that reflects how frequently each pair appears together. Finally, it writes a JSON file containing the list of canonical character names and the symmetric co-occurrence matrix.

The JavaScript frontend takes the `character_interactions.json` produced by the Python script and turns it into an interactive D3-based network visualization of relationships. It loads the list of characters and their co-occurrence matrix, builds nodes and weighted links and then uses a force-directed layout to position them.

The UI supports debounced searching that can either highlight matching characters or filter the graph down to them, while preserving a facet system that also lets you restrict the view by minimum degree and by k-hop distance from a chosen seed character. Clicking a node opens a k-hop control menu, and a separate ego view overlays a radial ego network.
