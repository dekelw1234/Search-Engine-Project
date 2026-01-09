export ENGINE_VERSION=BALANCED_2_NO_PR
python search_frontend.py &

python analyze_queries.py


python deep_dive.py
```

---


```
Query: "DNA double helix discovery"
Problem: Relevant docs use "deoxyribonucleic acid" instead of "DNA"
Solution: Query expansion with synonyms
```

```
Query: "Gothic literature Mary Shelley"
Problem: "Gothic" matches architecture articles too
Solution: Context-aware ranking, use all query terms together
```

```
Query: "Ancient Egypt pyramids pharaohs"
Problem: Documents with "Egypt" in title rank too high even if not about pyramids
Solution: Reduce title weight, increase body weight for multi-term queries
```

```
Query: "Industrial Revolution steam engines"
Problem: Misses docs about "locomotives" and "factories"
Solution: Add semantic similarity (word embeddings)



