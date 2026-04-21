# Pro Verifier

The Pro verifier is not open source. It provides:

- **Multi-dimensional scoring** across structural, manufacturing, cost, and geometry
- **Anti-gaming detection** — catches reward hacking patterns
- **Monthly updates** as new gaming strategies emerge

## Usage

```python
import os
env = gym.make("prodata/BracketDesign-v0", verifier_mode="pro")
# Requires PRODATA_API_KEY environment variable
```

Sign up at https://prodata.ai
