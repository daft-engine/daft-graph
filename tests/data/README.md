# Test data

## wiki-Vote.txt.gz

The Wikipedia who-votes-on-whom network, used by `tests/test_wiki_vote.py` and
`tests/test_motif_dataset.py` as a larger-than-toy regression fixture (7,115
vertices, 103,689 directed edges).

- Source: Stanford Network Analysis Project (SNAP),
  https://snap.stanford.edu/data/wiki-Vote.html
- Reference: J. Leskovec, D. Huttenlocher, J. Kleinberg. "Predicting Positive and
  Negative Links in Online Social Networks." WWW 2010.
- Redistributed here unmodified (gzipped) so the suite runs without network
  access.
