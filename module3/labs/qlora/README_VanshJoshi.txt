LAB 8 — QLORA DIALOGSUM — VANSH JOSHI
Fresh user-requested required run: 12 November 2026

OPEN FIRST
qlora_dialogsum_lab_VanshJoshi.ipynb
Contains required code, genuine execution outputs, loss figure, Questions 1–8,
the disclosed qualitative six-example assessment and final reflection.
Read the saved outputs; no additional training is needed to review them.

FRESH MEASURED RESULTS
Qwen2.5-1.5B-Instruct / DialogSum / seed 42.
2,000 training dialogues selected; 1,940 retained at <=512 tokens.
60 validation dialogues, identical baseline/tuned prompt and greedy decoding.
NF4 double quantization; fp16; all-linear LoRA r16, alpha32, dropout0.05.
One epoch; microbatch4; accumulation4; LR2e-4; cosine; warmup10;
paged_adamw_8bit; gradient checkpointing. No optional experiments.
GPU: NVIDIA RTX 2000 Ada 16 GB. Training: 11.76 minutes.
Optimizer steps: 122. Peak allocated memory: 6.37 GB.
                     ROUGE-1   ROUGE-2   ROUGE-L   Mean words
Original               31.94      8.33     23.10        33.07
QLoRA                  45.88     19.08     37.42        21.58
Reference mean words: 21.53.
QLoRA strictly improves ROUGE-L on 54/60 dialogues.
Adapter: 18,464,768 parameters, 73.91 MB; frozen base not included.

EXERCISE STATUS
1–6: Required implementation, checks, training/evaluation and answers complete.
7: All 24 scores, six notes, comparison and answer supplied as qualitative work.
   The original personal blind human-rating requirement is not established by
   this substitute. No human participation or instructor acceptance is claimed.
8: Actual adapter saved; size and merge/export explanation complete.
Reflection: Evidence-based recommendation, limits and next experiment included.

VERIFICATION
The fresh notebook ran in one new kernel. Its actual Trainer state and figure
are saved directly. CPU verification recomputed all metrics, checked the exact
validation sample, the adapter's 392 tensors, rating provenance and notebook
structure. It found no saved errors or code stubs. See verification_VanshJoshi.json.
The same-seed rerun is not an independent statistical replication.

ATTRIBUTION AND REPRODUCTION
The earlier completed run is preserved separately. Actual runtime: PyTorch2.10.0+cu128, Transformers4.57.1,
PEFT0.17.1. requirements.txt pins these recorded package versions; versions of
the other supporting packages were not recorded. The optimizer-enum assertion
was adapted for compatibility while retaining the required settings.
Running all cells again deliberately retrains and regenerates predictions.
The assessment fingerprint prevents silently reusing scores if outputs change.

The grading spreadsheet covers Labs1–5, not Lab8. Its general lessons about
the assigned architecture, checks, real outputs and specific written answers
were applied. Lab8 requirements come from:
https://github.com/bk-anupam/generative_ai_shared/blob/master/module3/labs/qlora/qlora_dialogsum_lab.ipynb

FILES
qlora_dialogsum_lab_VanshJoshi.ipynb: executed notebook, answers and reflection.
qlora_dialogsum_evidence_VanshJoshi.zip: measured predictions, metrics, actual
Trainer state, adapter configuration, rating provenance and verification.
The 73.9 MB trained adapter is saved locally and included in the student's
complete download package. Its SHA-256 and tensor count are recorded in the
verification JSON. Large model weights are not included in this Git commit.

ASSESSMENT DISCLOSURE
The six comparisons are preliminary qualitative reviews. This does not establish personal
blind human evaluation or instructor acceptance of that substitute.
