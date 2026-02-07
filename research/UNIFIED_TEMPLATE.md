# Unified Template for Speculative Decoding Research

This template defines the standard structure for all speculative decoding research markdown files.

---

## Required Sections

All research markdown files should include these sections in order:

### 1. Header
```markdown
# [Title]

**Table of Contents** (optional, for longer documents)

## Overview
[Brief 2-3 paragraph introduction to the topic]

## Key Insight
[One or two key quotes/insights that capture the essence]
```

### 2. Main Content (varies by topic)
```markdown
## Background / Context
[Historical context, why this approach exists]

## Key Methods / Techniques
[Main algorithms, papers, implementations]

## Theoretical Analysis
[Mathematical formulations, equations, analysis]

## Implementation / Code Examples
[Practical code snippets, usage examples]

## Performance / Benchmarks
[Tables, charts, speedup comparisons]
```

### 3. Critical Analysis
```markdown
## Advantages / Benefits
[What works well, supported by data]

## Limitations / Challenges
[What doesn't work, open problems]

## When to Use / When to Avoid
[Practical recommendations, decision criteria]
```

### 4. Related Research
```markdown
## Related Works
[Citations to relevant papers with links]

## Future Directions
[Open questions, research opportunities]
```

### 5. Practical Information
```markdown
## References
[Full citations with links]

## Resources
[GitHub repos, tools, datasets]

## Summary
[Concise takeaway message]
```

### 6. Metadata
```markdown
---

**Last Updated:** [Date]

**Status:** [Complete/In Progress/Needs Update]

**Contributors:** [If applicable]
```

---

## Formatting Guidelines

### Code Blocks
- Use triple backticks with language identifier
- Include descriptive captions
- Comment complex code

```python
# Good
def generate_draft(context):
    """Generate draft tokens"""
    return tokens
```

### Tables
- Include descriptive headers
- Add captions for complex tables
- Use consistent formatting

| Method | Speedup | Year |
|--------|---------|------|
| EAGLE | 3.5x | 2024 |

### Equations
- Use LaTeX-style math notation
- Include variable definitions
- Add brief explanations

```
Speedup = (T_vanilla) / (T_draft + T_verify)

Where:
- T_vanilla: Time for standard decoding
- T_draft: Time to generate drafts
- T_verify: Time for verification
```

### Citations
- Include paper title
- Include authors (for major papers)
- Include year/venue
- Include link

```markdown
**Paper:** "Title Here"
- **Authors:** A. Author, B. Author
- **arXiv:** [1234.56789](https://arxiv.org/abs/1234.56789)
- **Venue:** ICML 2024
```

### Diagrams
- Use ASCII art for simple diagrams
- Use code blocks for more complex ones
- Keep diagrams under 80 characters wide when possible

```
┌─────────┐
│  Model  │
└────┬────┘
     │
     ▼
```

---

## Section-Specific Guidelines

### For Overview Section
- 2-3 paragraphs maximum
- Define the topic clearly
- Mention why it matters
- Preview key insights

### For Key Methods Section
- Organize chronologically or by theme
- Include paper citations for each method
- Provide brief (2-3 sentence) descriptions

### For Implementation Section
- Include working code examples
- Use realistic, tested code when possible
- Add comments explaining key lines
- Provide usage examples

### For References Section
- Use consistent citation format
- Include DOIs/arXiv IDs when available
- Group by type (papers, blogs, repos)
- Verify all links are accessible

---

## Writing Style Guidelines

### Tone
- Technical but accessible
- Objective and evidence-based
- Avoid hype ("revolutionary", "groundbreaking")
- Use precise language

### Voice
- Present tense for established facts
- Past tense for experiments/results
- Future tense for proposed work

### Length Guidelines
- Overview: 2-3 paragraphs
- Method descriptions: 1-3 paragraphs each
- Code examples: 10-30 lines
- Total document length: 1500-3000 words (typical)

---

## Quality Checklist

Before finalizing a markdown file, verify:

- [ ] All paper links are accessible
- [ ] Code examples are syntactically correct
- [ ] Tables have proper headers
- [ ] Equations are properly formatted
- [ ] Citations follow the template
- [ ] Diagrams render correctly
- [ ] Section ordering follows this template
- [ ] Date and status are current
- [ ] No duplicate sections
- [ ] Consistent formatting with other files

---

**Last Updated:** January 2026
**Template Version:** 1.0
