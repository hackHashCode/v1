# Validation sweep (seed 42, train 384 / val 365)

| method | lr | epochs | val pass@1 (365) |
|---|---|---|---|
| jepa | 1e-05 | 1 | 27.4 |
| jepa | 1e-05 | 2 | 40.6 |
| jepa | 1e-05 | 3 | 42.7 |
| jepa | 2e-05 | 1 | 43.0 |
| jepa | 2e-05 | 2 | 40.3 |
| jepa | 2e-05 | 3 | 44.4 **(selected)** |
| jepa | 5e-05 | 1 | 36.2 |
| jepa | 5e-05 | 2 | 13.4 |
| jepa | 5e-05 | 3 | 2.5 |
| jepa | 0.0001 | 1 | 8.0 |
| jepa | 0.0001 | 2 | 0.0 |
| jepa | 0.0001 | 3 | 0.3 |
| jepa | 0.0002 | 1 | 0.0 |
| jepa | 0.0002 | 2 | 0.5 |
| jepa | 0.0002 | 3 | 0.8 |
| precodefull_aux | 1e-05 | 1 | 23.3 |
| precodefull_aux | 1e-05 | 2 | 26.6 |
| precodefull_aux | 1e-05 | 3 | 28.2 |
| precodefull_aux | 2e-05 | 1 | 25.8 |
| precodefull_aux | 2e-05 | 2 | 31.2 |
| precodefull_aux | 2e-05 | 3 | 34.8 |
| precodefull_aux | 5e-05 | 1 | 28.8 |
| precodefull_aux | 5e-05 | 2 | 36.4 |
| precodefull_aux | 5e-05 | 3 | 44.9 **(selected)** |
| precodefull_aux | 0.0001 | 1 | 40.3 |
| precodefull_aux | 0.0001 | 2 | 44.4 |
| precodefull_aux | 0.0001 | 3 | 34.2 |
| precodefull_aux | 0.0002 | 1 | 17.0 |
| precodefull_aux | 0.0002 | 2 | 10.7 |
| precodefull_aux | 0.0002 | 3 | 3.0 |
| sft | 1e-05 | 1 | 31.0 |
| sft | 1e-05 | 2 | 39.2 |
| sft | 1e-05 | 3 | 41.6 |
| sft | 2e-05 | 1 | 41.6 |
| sft | 2e-05 | 2 | 43.8 **(selected)** |
| sft | 2e-05 | 3 | 43.3 |
| sft | 5e-05 | 1 | 41.9 |
| sft | 5e-05 | 2 | 8.5 |
| sft | 5e-05 | 3 | 2.5 |
| sft | 0.0001 | 1 | 1.6 |
| sft | 0.0001 | 2 | 0.5 |
| sft | 0.0001 | 3 | 0.3 |
| sft | 0.0002 | 1 | 0.3 |
| sft | 0.0002 | 2 | 0.0 |
| sft | 0.0002 | 3 | 0.3 |
| sft_loop | 1e-05 | 1 | 28.8 |
| sft_loop | 1e-05 | 2 | 41.6 |
| sft_loop | 1e-05 | 3 | 43.3 |
| sft_loop | 2e-05 | 1 | 40.8 |
| sft_loop | 2e-05 | 2 | 44.4 **(selected)** |
| sft_loop | 2e-05 | 3 | 41.1 |
| sft_loop | 5e-05 | 1 | 40.0 |
| sft_loop | 5e-05 | 2 | 12.9 |
| sft_loop | 5e-05 | 3 | 1.9 |
| sft_loop | 0.0001 | 1 | 3.0 |
| sft_loop | 0.0001 | 2 | 0.0 |
| sft_loop | 0.0001 | 3 | 0.5 |
| sft_loop | 0.0002 | 1 | 0.8 |
| sft_loop | 0.0002 | 2 | 0.0 |
| sft_loop | 0.0002 | 3 | 0.0 |
