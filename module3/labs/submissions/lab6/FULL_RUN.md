# Full recipe experiment

GPU: NVIDIA RTX 2000 Ada Generation. Two epochs, 2862 training examples, 359 validation-loss examples, 40 generation examples, effective batch 8 (microbatch 1, accumulation 8).

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Ingredient % | Words | Stopped % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 original | 3.73 | 30.84 | 6.86 | 17.40 | 56.25 | 218.97 | 65.00 |
| GPT-2 full FT | 10.55 | 47.48 | 17.82 | 29.70 | 77.05 | 174.78 | 97.50 |
| GPT-2 LoRA | 9.83 | 42.61 | 14.62 | 25.97 | 75.62 | 223.30 | 80.00 |

Full fine-tuning: 12.85 minutes, 2.70 GB peak allocated, validation loss 1.923.

LoRA: 8.58 minutes, 1.10 GB peak allocated, validation loss 2.294.

General-English loss: GPT-2 original: 3.468, GPT-2 full FT: 4.018, GPT-2 LoRA: 3.589.

Source: Jain (2020), Indian Food Recipes Dataset, Mendeley DOI 10.17632/xsphgmmh7b.1, CC BY 4.0; recipes from Archana’s Kitchen. GPT-2: OpenAI, MIT license.

The shared GPU makes timing indicative rather than a clean hardware benchmark. Full fine-tuning improved overlap and stopping most, but generated samples still invent ingredients and omit cooking steps. The notebook contains the case-by-case analysis. General-English loss increased more for full fine-tuning on the four-paragraph probe. These are single-seed results, not evidence of safe recipe generation.
