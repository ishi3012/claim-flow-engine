"""
claimflowengine.routing

This module implements the intelligent routing engine for denied healthcare claims.
It provides functionality to score and route claims based on denial clusters,
payer patterns, team expertise, and simulated resolution outcomes.

Modules:
--------
- `policy_engine`: Computes priority scores and assigns claims to resolution teams.
- `teams`: Defines simulated team profiles, skill mappings, and coverage rules.
- `reward_model`: Placeholder for contextual bandit or RL-style reward-based routing.
- `tests`: Unit and integration tests for the routing logic.

Exposed Functions:
------------------
- `compute_priority_score`: Calculates a score indicating how urgent or complex a claim is.
- `route_claim`: Determines the most appropriate team to handle the claim based on scoring logic.
- `get_team_profiles`: Returns simulated team expertise and availability data.

Intended Use:
-------------
This module is invoked by downstream pipeline stages or API endpoints to determine
claim routing paths that optimize for operational efficiency and resolution time.

Example:
--------
```python
from claimflowengine.routing import route_claim, compute_priority_score

score = compute_priority_score(claim_row)
assigned_team = route_claim(claim_row)
Author:
ClaimFlowEngine Team
"""
