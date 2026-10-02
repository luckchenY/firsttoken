# SFT Okay statistics

This directory contains streaming statistics from the public NVIDIA Nemotron sources used by M2RL. It samples each published split after a deterministic shuffle; it does not download the full corpora or save raw examples.

- Rows per split: `500`
- Shuffle buffer: `2000`
- Seed: `42`
- Total target rows in M2RL mixture: `14,121,873`

`source_rates.csv` is the main table. `first_words_and_okay_positions.json` gives the most common first words and whether Okay tends to occur at the beginning or later in a response. `examples.json` contains short context snippets only.
