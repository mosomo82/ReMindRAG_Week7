# ReMindRAG Example Queries — DnD Paladin Dataset

These queries are designed for use with `example/example_data.txt` (DnD 5e Paladin chapter).
Run them via the WebUI at `http://127.0.0.1:5000` or programmatically via `rag_instance.generate_response(query)`.

---

## Simple (Single-Fact Retrieval)

Quick lookups that validate basic RAG retrieval is working.

```
What is the hit dice for a paladin?
What proficiency bonus does a level 5 paladin have?
When does a paladin get Extra Attack?
What spellcasting ability do paladins use?
```

---

## Multi-Hop Reasoning

These require traversing multiple nodes in the knowledge graph — the core strength of ReMindRAG.

```
Which oath would be best for a paladin who wants to protect nature and resist spells?
Compare the 20th-level abilities of Oath of Devotion, Oath of the Ancients, and Oath of Vengeance.
At what level does a paladin first get 3rd-level spell slots, and what oath spells become available
  at that point for Oath of the Ancients?
What Channel Divinity options help against undead specifically, and which oath offers them?
```

---

## Cross-Referencing Auras

Tests the system's ability to connect related features spread across multiple sections.

```
How do the aura features of Oath of Devotion differ from Oath of the Ancients?
Which paladin features improve at 18th level?
What is the maximum range an Aura of Protection can reach, and what does it do?
```

---

## Character Build Questions

Real-world use case: a player building a character using the handbook.

```
What is the recommended quick build for a paladin?
What fighting style pairs best with a paladin who prefers solo combat?
How many spells can a level 7 paladin with 16 Charisma prepare?
```

---

## Oath Comparison

Multi-document synthesis requiring side-by-side comparison of three oath paths.

```
Which oath is most aligned with a character seeking revenge, and what powers do they gain at level 20?
What are the differences in the tenets between Oath of Devotion and Oath of Vengeance?
```

---

## Expected Behavior

| Query Type | Expected Result |
|------------|----------------|
| Simple | Direct answer from a single chunk |
| Multi-hop | Answer synthesizing 2–4 connected knowledge graph nodes |
| Build questions | Calculation using level table + class feature rules |
| Oath comparison | Side-by-side synthesis from three separate oath sections |

> Multi-hop queries produce the most interesting responses and best demonstrate ReMindRAG's
> knowledge graph traversal over a flat RAG baseline.
