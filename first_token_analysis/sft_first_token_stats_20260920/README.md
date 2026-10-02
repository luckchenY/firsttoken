# M2RL SFT first-token statistics

- tokenizer: `/data/chenyang2/models/Qwen3-4B-Base`
- streaming sample: first 1000 rows per source (raw data is not saved)
- `*_raw`: tokenize the field as-is
- `*_semantic`: apply `lstrip()` before tokenization to remove leading whitespace

## math_proofs

- rows: 1000
- assistant messages: 668
- nonempty reasoning_content: 668 (100.00%)

### reasoning_content_semantic

| token | count | fraction |
|---|---:|---:|
| `820:####` | 250 | 37.43% |
| `334:**` | 224 | 33.53% |
| `785:The` | 114 | 17.07% |
| `1654:We` | 44 | 6.59% |
| `16:1` | 27 | 4.04% |
| `5338:First` | 9 | 1.35% |

### content_semantic

| token | count | fraction |
|---|---:|---:|
| `73594:```` | 668 | 100.00% |

## math

- rows: 1000
- assistant messages: 1000
- nonempty reasoning_content: 1000 (100.00%)

### reasoning_content_semantic

| token | count | fraction |
|---|---:|---:|
| `1654:We` | 998 | 99.80% |
| `785:The` | 2 | 0.20% |

### content_semantic

| token | count | fraction |
|---|---:|---:|
| `10061:Let` | 544 | 54.40% |
| `59:\` | 163 | 16.30% |
| `785:The` | 112 | 11.20% |
| `5338:First` | 52 | 5.20% |
| `2461:For` | 36 | 3.60% |
| `334:**` | 36 | 3.60% |
| `1654:We` | 16 | 1.60% |
| `1249:To` | 10 | 1.00% |
| `3830:From` | 10 | 1.00% |
| `12549:Since` | 6 | 0.60% |

## science

- rows: 1000
- assistant messages: 1000
- nonempty reasoning_content: 1000 (100.00%)

### reasoning_content_semantic

| token | count | fraction |
|---|---:|---:|
| `1654:We` | 997 | 99.70% |
| `785:The` | 3 | 0.30% |

### content_semantic

| token | count | fraction |
|---|---:|---:|
| `59:\` | 733 | 73.30% |
| `785:The` | 189 | 18.90% |
| `334:**` | 38 | 3.80% |
| `2461:For` | 14 | 1.40% |
| `24617:Starting` | 5 | 0.50% |
| `16429:Using` | 4 | 0.40% |
| `83725:**(` | 4 | 0.40% |
| `641:In` | 3 | 0.30% |
| `3830:From` | 2 | 0.20% |
| `41961:Within` | 1 | 0.10% |

## code

- rows: 1000
- assistant messages: 1000
- nonempty reasoning_content: 1000 (100.00%)

### reasoning_content_semantic

| token | count | fraction |
|---|---:|---:|
| `1654:We` | 992 | 99.20% |
| `37991:Ðľ` | 8 | 0.80% |

### content_semantic

| token | count | fraction |
|---|---:|---:|
| `1249:To` | 996 | 99.60% |
| `72819:Ð§` | 4 | 0.40% |

## chat

- rows: 1000
- assistant messages: 3339
- nonempty reasoning_content: 818 (24.50%)

### reasoning_content_semantic

| token | count | fraction |
|---|---:|---:|
| `<EMPTY>` | 2521 | 75.50% |
| `1654:We` | 545 | 16.32% |
| `785:The` | 248 | 7.43% |
| `1474:User` | 24 | 0.72% |
| `32313:Okay` | 1 | 0.03% |

### content_semantic

| token | count | fraction |
|---|---:|---:|
| `2610:You` | 648 | 19.41% |
| `77045:Absolutely` | 276 | 8.27% |
| `40:I` | 235 | 7.04% |
| `9454:Yes` | 176 | 5.27% |
| `334:**` | 160 | 4.79% |
| `11908:Oh` | 132 | 3.95% |
| `24765:Ah` | 122 | 3.65% |
| `785:The` | 102 | 3.05% |
| `4792:That` | 85 | 2.55% |
| `2442:<<` | 72 | 2.16% |

## agent

- rows: 1000
- assistant messages: 3495
- nonempty reasoning_content: 0 (0.00%)

### reasoning_content_semantic

| token | count | fraction |
|---|---:|---:|
| `<EMPTY>` | 3495 | 100.00% |

### content_semantic

| token | count | fraction |
|---|---:|---:|
| `<EMPTY>` | 1266 | 36.22% |
| `40:I` | 986 | 28.21% |
| `56389:YOU` | 362 | 10.36% |
| `13060:Thank` | 299 | 8.56% |
| `2610:You` | 76 | 2.17% |
| `7771:Your` | 72 | 2.06% |
| `785:The` | 71 | 2.03% |
| `1249:To` | 42 | 1.20% |
| `9454:Yes` | 40 | 1.14% |
| `28715:Based` | 31 | 0.89% |
