# Direct-rating prompt designs

These four designs use the same scenario, response, criteria, models, and
1–10 rating scale. Each design produces one reflection per response/judge pair.
The evaluative wording is identical across designs; the first three designs
replace only the XML transport with externally enforced JSON tool schemas.

## Designs

| Design | Reflection output | Judgment output | Calls for eight criteria |
|---|---|---|---:|
| 1. Combined JSON | One free-form reflection string | All eight ratings together | 2 |
| 2. Structured combined | One reflection entry per criterion | All eight ratings together | 2 |
| 3. Structured isolated | One reflection entry per criterion | One rating call per criterion | 9 |
| 4. Combined XML baseline | Raw reflection text | All eight ratings in XML | 2 |

Designs 1 and 2 deliberately use identical natural-language prompts. Their
only difference is the externally enforced reflection schema. Design 3's
judgment call receives only one criterion and its corresponding reflection.
Design 4 retains the original XML delimiters and rating tags.

## Prompt files

### 1. Combined JSON

- [Reflection prompt](01_combined_json_reflection.md)
- [Judgment prompt](01_combined_json_judgment.md)

Reflection output:

```json
{"reflection": "Criterion-by-criterion reflection text"}
```

Judgment output:

```json
{
  "ratings": [
    {"criterion_id": "criterion_01", "rating": 7}
  ]
}
```

### 2. Structured combined

- [Reflection prompt](02_structured_combined_reflection.md)
- [Judgment prompt](02_structured_combined_judgment.md)

Reflection output:

```json
{
  "reflections": [
    {"criterion_id": "criterion_01", "reflection": "Reflection for this criterion"}
  ]
}
```

Judgment output:

```json
{
  "ratings": [
    {"criterion_id": "criterion_01", "rating": 7}
  ]
}
```

### 3. Structured isolated

- [Reflection prompt](03_structured_isolated_reflection.md)
- [Single-criterion judgment prompt](03_structured_isolated_judgment.md)

The reflection uses Design 2's structured output. Each subsequent judgment
call returns only:

```json
{"rating": 7}
```

The criterion ID remains caller-side bookkeeping and is not shown to the judge
in the isolated judgment.

### 4. Combined XML baseline

- [Reflection prompt](04_combined_xml_reflection.md)
- [Judgment prompt](04_combined_xml_judgment.md)

The judgment returns one XML tag per criterion, exactly as specified inside the
prompt.

## Observed cost

The smoke test used the same 24 response/judge cells for every design: six
response pairs, Claude 4 Sonnet, GPT-4.1, and eight kindness criteria.

| Design | Calls | Actual cost | Cost per completed rating | Relative to Design 1 |
|---|---:|---:|---:|---:|
| 1. Combined JSON | 48 | $0.361825 | $0.015076 | 1.00x |
| 2. Structured combined | 48 | $0.421328 | $0.017555 | 1.16x |
| 3. Structured isolated | 216 | $0.697342 | $0.029056 | 1.93x |
| 4. Combined XML baseline | 48 | $0.373206 | $0.015550 | 1.03x |

These are cost observations, not performance rankings. The small test's
pairwise reference labels are subjective, judge-mismatched in most cells, and
deliberately class-skewed, so they should not be treated as ground truth.
