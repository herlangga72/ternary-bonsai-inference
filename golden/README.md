# Golden reference logits for M6/M7 validation

Captured by `bonsai-logits` from llama.cpp decode on `Ternary-Bonsai-27B-PQ2_0.gguf`
(n_ctx 4096, batch 512, CPU). These are the target outputs the pure-Rust forward
pass must reproduce.

## Files

| file | prompt |
| --- | --- |
| `prompts/qa.txt` | "What is the capital of France? Answer briefly." |
| `prompts/code.txt` | "Write a Rust function that returns the sum of a slice of u32." |
| `qa.logits.bin` / `code.logits.bin` | logits row after the full prompt prefill |

## Binary layout (little-endian)

```
u32 n_vocab
u32 n_prompt
f32 logits[n_vocab]      # logits of the last prompt token
i32 greedy_id            # argmax of the logits
```

The prompt tokens come from the pure-Rust tokenizer (M3), which matches
`llama_tokenize`. Both prompts use the text-only qwen35 chat template with the
assistant turn seeded by `<think>`.

## Validation approach for M6

1. Run the Rust forward pass over the same prompt tokens.
2. Compare the Rust final logits row to the golden file: greedy id must agree;
   logits should agree within f32 accumulation tolerance (ggml itself is not an
   exact oracle, see ROADMAP M5 note on activation quantization).
