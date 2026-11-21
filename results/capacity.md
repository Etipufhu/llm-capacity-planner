### Weekly brief (PR Radar model, ~740 tokens in, ~145 out)

| GPU | | Max concurrent requests (SLO) | Bound by | Users per GPU | GPUs | $ / month |
|---|---|--:|---|--:|--:|--:|
| RTX 5070 Laptop | measured | 9 | measured | 56 | 18 | n/a |
| L4 | estimate | 7 | first token | 45 | 23 | 13,513 |
| A10G | estimate | 12 | first token | 100 | 10 | 7,344 |
| L40S | estimate | 18 | first token | 155 | 7 | 9,510 |
| A100 40GB | estimate | 31 | first token | 294 | 4 | 8,015 |
| H100 80GB | estimate | 66 | first token | 662 | 2 | 10,045 |

### Chat assistant (~60 tokens in, ~175 out)

| GPU | | Max concurrent requests (SLO) | Bound by | Users per GPU | GPUs | $ / month |
|---|---|--:|---|--:|--:|--:|
| RTX 5070 Laptop | measured | ≥ 32 | not reached | 118 | 9 | n/a |
| L4 | estimate | 25 | first token | 88 | 12 | 7,050 |
| A10G | estimate | 51 | first token | 192 | 6 | 4,406 |
| L40S | estimate | 75 | first token | 286 | 4 | 5,434 |
| A100 40GB | estimate | 136 | first token | 529 | 2 | 4,007 |
| H100 80GB | estimate | 295 | first token | 1162 | 1 | 5,022 |
