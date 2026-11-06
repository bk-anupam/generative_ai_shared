# Lab: fine-tuning a Transformer for patent phrase matching

**Weight:** 5% of the course grade · **Work:** individual · **Submission deadline:** *6th Nov 2026*

**Contents:** [The problem](#the-problem) · [Data and metric](#data-and-metric) · [Learning objectives](#learning-objectives) · [What you are given](#what-you-are-given) · [Tasks](#tasks) · [Making a late submission](#making-a-late-submission) · [Rules](#rules) · [Deliverables](#deliverables) · [Marking rubric](#marking-rubric) · [References](#references)

## The problem

A patent examiner searching for prior art cannot rely on exact word matches. One inventor writes *"opc drum"*, another *"inorganic photoconductor drum"*, a third *"photoreceptor cylinder"*. Whether two phrases mean the same thing can also depend on the technical field: *"mouse"* means something different in a biology patent than in a computing patent.

The [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) competition (USPTO and Kaggle, 2022) turned this into a supervised task: given an **anchor** phrase, a **target** phrase, and a **context** (the patent's technical field), predict how similar the two phrases are. You will fine-tune a pretrained Transformer for this task, measure what fine-tuning adds, analyse where the model fails, and have your final notebook scored on the competition's hidden test set.

The competition has closed, but it still accepts **late submissions**. A late submission does not appear on the public leaderboard ranking, but Kaggle still shows you its public and private scores. Those scores are the evidence you submit.

## Data and metric

| Column | Meaning | Example |
| --- | --- | --- |
| `id` | Row identifier | `4112d61851461f60` |
| `anchor` | First phrase | `opc drum` |
| `target` | Second phrase | `inorganic photoconductor drum` |
| `context` | CPC (Cooperative Patent Classification) code; its first letter is the section, e.g. G = Physics | `G02` |
| `score` | Similarity rating (training data only): 0, 0.25, 0.5, 0.75, or 1 | — |

- `train.csv` has 36,473 rows with only 733 distinct anchors, so each anchor appears with many targets.
- The labels are imbalanced: 0.5 and 0.25 are the most common scores, and only about 3% of rows score 1.0.
- The visible `test.csv` has only 36 rows. When you submit, Kaggle reruns your notebook with the hidden test set in its place. The leaderboard uses one part of the hidden test set for the **public** score and the rest for the **private** score.
- **Metric:** the Pearson correlation coefficient *r* between your predictions and the true scores. *r* = 1 is perfect agreement, 0 is no linear relationship. Because *r* is unchanged if all predictions are shifted or scaled, it measures whether you put pairs in the right order, not whether you predict exact values.

The data comes from the competition page; read its Data and Rules tabs before you start. Each pair was rated by expert raters, and the dataset was later released under the CC BY 4.0 license (Aslanyan & Wetherbee, 2022; Google Research, 2022).

## Learning objectives

By the end of the project you should be able to:

1. Fine-tune a pretrained Transformer encoder for a sentence-pair regression task, and explain how the input pair, the regression head, and the loss fit together.
2. Design a validation split that does not leak information between training and validation, and use it to compare models fairly.
3. Measure what fine-tuning adds over a pretrained model used without training.
4. Run controlled experiments and draw conclusions that the evidence actually supports.
5. Analyse errors, and explain differences between validation, public-leaderboard, and private-leaderboard scores.

**Prerequisites:** the Module 2 material on Transformers, BERT, and Hugging Face; the Module 3 fine-tuning lectures.

## What you are given

| File | Purpose |
| --- | --- |
| [patent_phrase_starter.ipynb](patent_phrase_starter.ipynb) | Minimal working pipeline: a DeBERTa-v3-small cross-encoder, an anchor-grouped validation split, fine-tuning with the Hugging Face `Trainer`, validation Pearson, and `submission.csv` |
| [download_weights.ipynb](download_weights.ipynb) | Helper that saves a pretrained checkpoint as notebook output, because submitted notebooks run without internet access |

The starter notebook deliberately leaves out everything you are marked on. The baseline, experiments, error analysis, and conclusions are your work.

**Compute.** Use Kaggle notebooks with a T4 or P100 GPU. Fine-tuning the starter model for three epochs should take a few minutes per epoch. That is an estimate: measure your own run. Larger checkpoints such as `deberta-v3-base` are slower and need more GPU memory. Plan your experiments within your weekly Kaggle GPU quota.

## Tasks

Use **the same validation split** (`SEED` and `VAL_FRACTION` unchanged) for every number you report in Tasks 1–3.

### Task 1: Reproduce the fine-tuned baseline

Run the starter notebook unchanged and record its validation Pearson correlation and training time. In two or three sentences, explain why the validation split holds out whole anchors rather than random rows, and what would happen to the validation score if it did not.

### Task 2: A baseline without fine-tuning

Measure how well a pretrained model does **with no training on this data**. Encode the anchor and the target separately with a pretrained sentence-embedding model (for example, a model from the `sentence-transformers` library), and use the cosine similarity of the two embeddings as the prediction. Report its validation Pearson correlation next to Task 1's.

Explain the difference between this **bi-encoder** (each phrase encoded separately, then compared) and the starter's **cross-encoder** (both phrases in one input, so attention can compare them token by token). Use your two numbers to discuss what fine-tuning added.

### Task 3: At least two controlled experiments

Change **one thing at a time** from the starter, and compare each change with the Task 1 baseline. Choose at least two of the following, or propose your own:

- **Context representation:** no context at all; the raw CPC code; the full CPC class title (you would need to find and cite a source for the titles).
- **Checkpoint:** a different pretrained model, for example `deberta-v3-base`, a BERT variant, or a model pretrained on patents. Say how it changes runtime and memory.
- **Training settings:** learning rate, number of epochs, or batch size. Use at most three values of one setting.
- **Task framing:** treat the five scores as classes instead of a regression target, and convert the class probabilities back into a score.
- **Parameter-efficient fine-tuning:** LoRA via the `peft` library. Compare the number of trainable parameters and the validation score with full fine-tuning.

Present the results in one table: the setting changed, validation Pearson, training time, and a one-line interpretation. Differences of about 0.005 may be noise from random seeds. Do not claim one setting is better on a difference that small unless you have rerun it with another seed.

### Task 4: Error analysis

Using `val_predictions.csv` from your best model:

1. Report the Pearson correlation **separately for each CPC section** (A–H). Which sections are hardest? Suggest why.
2. Plot or tabulate the **mean prediction against each true score** (0, 0.25, …, 1). Does the model compress extreme scores towards the middle? Connect this to the label imbalance.
3. Show the **10 validation pairs with the largest absolute error**. Group them into two or three types of failure, for example abbreviations, field-specific meaning, or antonyms that share words. Is the "true" label debatable for any of them?

### Task 5: Late submission and leaderboard scores

1. Choose your **final model**. Choose it by validation score, not by leaderboard score.
2. Make a late submission of the notebook that trains it (see the next section).
3. Report its **public and private leaderboard scores** with a screenshot of your Submissions tab.
4. Compare the validation, public, and private scores. Are they close? Give at least one reason why they could differ, for example a different mix of anchors or sections, test-set size, or choices you made by looking at validation results.

You may submit more than once, but report every submission you made, not only the best one. Choosing a model because it had the best private score would be choosing it using the test set.

## Making a late submission

1. Open the [competition page](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) and accept its rules (**Join Competition** / **Late Submission**).
2. Run [download_weights.ipynb](download_weights.ipynb) once, with internet on, for each checkpoint you use. Add its output to your project notebook with **Add Input → Your Work → Notebooks**.
3. In your project notebook, set **Accelerator → GPU** and **Internet → Off**. Confirm the notebook runs from top to bottom and writes `/kaggle/working/submission.csv`.
4. **Save Version → Save & Run All (Commit)**. When the version finishes, open it and choose **Submit to Competition** (or submit from the competition's **Late Submission** button by choosing the notebook and version).
5. Kaggle reruns the notebook on the hidden test set. This can take as long as your training run. Scores appear under **Submissions** on the competition page, as **Public Score** and **Private Score**.

If the submission fails, check the error on the Submissions tab. The most common causes are internet left on, a model loaded from the Hub instead of from an attached input, a wrong file name or columns, or exceeding the runtime limit. Check the competition's **Code Requirements** for the current limits.

## Rules

- **Individual work.** You may discuss ideas with classmates, but your code, experiments, and report must be your own.
- **Public Kaggle notebooks and solution write-ups** from this competition may be read for ideas. Cite every one you use, and mark any code you adapt from them. Marks are awarded for your own experiments and analysis. A fork of a public high-scoring notebook earns no marks for Tasks 3 and 4.
- **Do not use the test labels.** After the competition, Google released the full rated dataset (48,548 pairs, including the test pairs) as the *Patent Phrase Similarity* dataset. Training on, tuning with, or looking up its test or validation labels, or any copy of them, makes your leaderboard score meaningless and is academic misconduct.
- **No external LLM APIs** for predictions, and no hand-labelling of test rows. Any openly available pretrained checkpoint is allowed if you name it and respect its license.
- Report what you actually observed. Failed experiments are valid results if you report them honestly.

## Deliverables

Submit the following on the course platform by the deadline:

1. **Report (PDF, at most 4 pages, plus an optional appendix):** Tasks 1–5 with tables, figures, and written answers, and a reference list.
2. **Notebook link:** the Kaggle notebook version used for your final late submission, made public or shared with the instructor. It must reproduce your submission.
3. **Leaderboard screenshot:** your Submissions tab showing the public and private scores of every submission you report.

## Marking rubric

The project is marked out of 20, then scaled to its 5% weight. **Leaderboard rank earns no marks.** You are marked on the quality of your evidence and reasoning, not on whether your score beats a classmate's.

| Criterion | Marks | Full marks require |
| --- | --- | --- |
| Working pipeline and late submission | 4 | The notebook reproduces the final submission with internet off; public and private scores shown in the screenshot and stated in the report |
| Task 1 and Task 2: fine-tuned baseline vs. no fine-tuning | 3 | Both scores on the same split; a correct bi-encoder vs. cross-encoder explanation; a sound explanation of the anchor-grouped split |
| Task 3: controlled experiments | 5 | At least two experiments changing one factor each; a clear results table; conclusions that do not go beyond the evidence (noise is acknowledged) |
| Task 4: error analysis | 4 | Per-section and per-score analysis computed correctly; a thoughtful grouping of failure cases supported by examples |
| Task 5: score comparison | 2 | Validation, public, and private scores compared, with a plausible explanation; every submission reported |
| Communication and attribution | 2 | A clear, concise report; labelled tables and figures; every source and adapted code cited |

Partial credit is given within each criterion. An experiment that did not improve the score, but is well designed and honestly interpreted, can earn full marks.

## References

- Aslanyan, G., & Wetherbee, I. (2022). *Patents Phrase to Phrase Semantic Matching Dataset.* arXiv:2208.01171. <https://arxiv.org/abs/2208.01171>
- Google Research (2022, August). *Announcing the Patent Phrase Similarity Dataset.* <https://research.google/blog/announcing-the-patent-phrase-similarity-dataset/>
- Kaggle. *U.S. Patent Phrase to Phrase Matching* competition: Overview, Data, Rules, and Code Requirements tabs. <https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching>
- He, P., Gao, J., & Chen, W. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* arXiv:2111.09543. <https://arxiv.org/abs/2111.09543>
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. arXiv:1908.10084. <https://arxiv.org/abs/1908.10084>
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. <https://arxiv.org/abs/2106.09685>
- Hugging Face documentation: [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [PEFT](https://huggingface.co/docs/peft), [Sentence Transformers](https://www.sbert.net/).
