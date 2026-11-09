# LLM training stages and chat templates

**Problem.** You download `Qwen2.5-0.5B-Instruct` to fine-tune it as a support assistant for a food-delivery app. The model already answers questions politely and stops at the end of its answer. It learned that from earlier rounds of training, done before you downloaded it. Your fine-tuning data has to fit what those rounds built: the same kind of examples, written in the same format. This note explains what those rounds are, how your fine-tuning differs from them, and the format (the **chat template**) that ties them together. It is the background for [Preparing data for fine-tuning](data_preparation.ipynb).

**What you will learn**

1. The four stages that take a language model from raw text to a task-specific assistant, and which of them you do.
2. What instruction tuning changes in a model, and how it differs from task fine-tuning.
3. What a chat template is, using Qwen2.5's ChatML format, and why training and inference must use the same one.

**Prerequisites:** next-token prediction and the cross-entropy loss (Module 2); [Fine-tuning GPT-2](../fine_tuning/gpt2_fine_tuning.ipynb), Section 3 (labels and `-100`) and Section 8 (the end-of-text token).

**Contents**

1. [Four stages of training](#1-four-stages-of-training)
2. [What instruction tuning changes](#2-what-instruction-tuning-changes)
3. [Instruction tuning versus task fine-tuning](#3-instruction-tuning-versus-task-fine-tuning)
4. [Chat templates](#4-chat-templates)
5. [Training and inference must use the same template](#5-training-and-inference-must-use-the-same-template)
6. [Summary](#6-summary)
7. [Check your understanding](#7-check-your-understanding)
8. [References](#references)

## 1. Four stages of training

The chat models we fine-tune have already been through several rounds of training. Each round uses different data and teaches something different.

| Stage | Data | What the model learns | Example |
| --- | --- | --- | --- |
| **1. Pretraining** | Trillions of tokens of raw text (web pages, books, code) | Language, facts and patterns, by predicting the next token | `Qwen2.5-0.5B` (the base model) |
| **2. Instruction tuning** (supervised fine-tuning, SFT) | Many (instruction, response) pairs covering many different tasks, written in the model's chat template | To treat its input as a request, answer it, and then stop | Qwen2.5's SFT: over 1 million examples (Section 2) |
| **3. Preference tuning** (optional; RLHF, DPO) | Pairs of responses to the same prompt, ranked by people or by a reward model | Which of several possible answers people prefer | Qwen2.5's DPO, then online reinforcement learning; most released chat models |
| **4. Task or domain fine-tuning** | Examples of one task: for example, support tickets and the replies agents wrote | One job, in one style, with one company's conventions | The support assistant in [Preparing data for fine-tuning](data_preparation.ipynb) |

RLHF is reinforcement learning from human feedback: a reward model is trained on the human rankings, and the language model is then optimized to score well under it (Ouyang et al., 2022). DPO is direct preference optimization, which learns from the same ranked pairs without a separate reward model or reinforcement-learning loop (Rafailov et al., 2023).

Stages 1 to 3 are usually done by the model's developer, and their result is what you download. `Qwen2.5-0.5B-Instruct`, for example, is not the output of instruction tuning alone: the Qwen2.5 models went through supervised fine-tuning, then DPO, then online reinforcement learning against a reward model (Qwen Team, 2024). **Stage 4 is the one you do.**

## 2. What instruction tuning changes

A pretrained base model only continues text. Give it this prompt:

```text
Summarise this complaint: my biryani arrived cold and an hour late.
```

A base model may continue with another line such as `Summarise this complaint: the driver was rude.`, because it reads the prompt as the start of a document (perhaps a list of exercises) rather than as a request addressed to it.

Instruction tuning fine-tunes the model on many thousands of examples shaped as *request, then answer, then end of turn*. Afterwards, the same prompt gets an answer such as `The customer's biryani was delivered cold and an hour late.`, followed by an end-of-turn token that stops generation. (Both continuations are illustrative of typical behaviour, not recorded outputs.)

The key ingredient is **variety**. FLAN fine-tuned a 137B-parameter model on more than 60 tasks, each phrased as natural-language instructions, and it became much better at task types it had *not* been trained on (Wei et al., 2021). The model learns the general skill of following an instruction, not just the tasks in its training set.

**What the data looks like: Qwen2.5.** Qwen has not released its instruction-tuning data, but the Qwen2.5 technical report describes it (Qwen Team, 2024). The supervised fine-tuning set has more than 1 million examples, trained for two epochs with sequences of up to 32,768 tokens. It covers many areas: long responses, mathematics with step-by-step solutions, code in nearly 40 programming languages, instructions whose result can be checked by code (such as a required length or format), questions about tables, logical reasoning, other languages, and hundreds of different system prompts. Much of it was generated or filtered by other models rather than written by hand. Code answers, for example, were kept only if they passed unit tests in a sandbox, and a final filter kept only responses that every scoring model rated flawless.

Whatever the area, every example is stored as a conversation and rendered in the model's chat template (Section 4) before training. A coding example might look like this (an illustration, not a record from Qwen's data):

```text
<|im_start|>system
You are a senior Python reviewer. Answer with code first, then one sentence.<|im_end|>
<|im_start|>user
Write a function that returns the second-largest distinct number in a list.<|im_end|>
<|im_start|>assistant
def second_largest(xs):
    vals = sorted(set(xs))
    return vals[-2] if len(vals) >= 2 else None
Duplicates are removed first, so [5, 5, 3] gives 3.<|im_end|>
```

The system message sets the behaviour, the user message is the instruction, and the assistant message is the response. As in all fine-tuning in this course, the loss is computed only on the response and the `<|im_end|>` that ends it; the system and user messages get the label `-100` (Section 5). A million such conversations, spread across many kinds of task, is what teaches the model to treat any input as a request, answer it, and stop.

Instruction tuning is also where a model learns its **chat template** (Section 4): the special tokens that mark who is speaking. That is why `Qwen2.5-0.5B-Instruct` already treats them as turn boundaries.

## 3. Instruction tuning versus task fine-tuning

Both are fine-tuning: training an already-trained model further on new data. They differ in purpose and data.

| | Instruction tuning (stage 2) | Task fine-tuning (stage 4) |
| --- | --- | --- |
| Goal | A general assistant that follows any instruction | A model that does one task well, in one domain |
| Data | Broad: many task types, a few to many examples each | Narrow: one task, many examples of it |
| Usual starting point | A base model | An instruction-tuned model |
| Who usually does it | The model's developer | You |
| Main risk | Too little variety, so the model follows only familiar kinds of instruction | Over-specialization: the model can get worse at things outside the task (catastrophic forgetting) |

The **mechanics are the same**. Both use the next-token cross-entropy loss, both give the prompt the label `-100` so that only the response is trained on, and both keep the label of the end-of-turn token so that the model learns to stop (as in Section 8 of the [GPT-2 notebook](../fine_tuning/gpt2_fine_tuning.ipynb)). What differs is the **data**. The training code does not change between the two, so the data decides what the model becomes, which is why [Preparing data for fine-tuning](data_preparation.ipynb) is a whole lecture.

## 4. Chat templates

A language model reads one flat sequence of tokens. A conversation, though, has structure: a system instruction, a customer message, an assistant reply. A **chat template** is the rule that turns a list of messages into that one sequence, using special tokens that mark where each message starts, who is speaking, and where the message ends. Every chat model is instruction-tuned on conversations written in its own template.

**From messages to text.** In code, a conversation is a list of messages, each with a role and its content. For a short illustrative ticket (not one from a real export):

```python
messages = [
    {"role": "system",    "content": "You are a customer-support assistant for a food-delivery app. Be brief, polite, and specific."},
    {"role": "user",      "content": "my biryani arrived cold"},
    {"role": "assistant", "content": "sorry about that. we have refunded rs 250."},
]
```

Qwen2.5 uses the **ChatML** template, which writes every message as `<|im_start|>`, the role, a newline, the content, and `<|im_end|>`. The list above becomes:

```text
<|im_start|>system
You are a customer-support assistant for a food-delivery app. Be brief, polite, and specific.<|im_end|>
<|im_start|>user
my biryani arrived cold<|im_end|>
<|im_start|>assistant
sorry about that. we have refunded rs 250.<|im_end|>
```

- `<|im_start|>` opens a message, and the word after it (`system`, `user` or `assistant`) says who is speaking.
- `<|im_end|>` closes the message. After an assistant reply, it is also the signal to **stop generating**.
- Both are **single special tokens** in Qwen's vocabulary: the tokenizer does not split them into `<`, `|`, `im` and so on. Qwen's instruction-tuned models were trained on conversations in this format, so they already treat these tokens as turn boundaries.

Other model families use other templates, with different special tokens. The idea is the same; the exact text is not.

**Where the template lives.** ChatML is not built into the Hugging Face library. It ships with the model, in its tokenizer files, and the library only applies it:

| Piece | Where it is stored | Provided by |
| --- | --- | --- |
| The special tokens `<\|im_start\|>` (ID 151644) and `<\|im_end\|>` (ID 151645) | The tokenizer's vocabulary (`tokenizer.json`, `tokenizer_config.json`) | The model's developer (Qwen) |
| The rule that writes a list of messages as ChatML | The `chat_template` entry of `tokenizer_config.json` | The model's developer (Qwen) |
| The function that applies the rule, `apply_chat_template` | The `transformers` library | Hugging Face |

`apply_chat_template` is generic: it knows nothing about ChatML. Called on another model's tokenizer, it applies that model's template instead. The format belongs to the model, which is why other tools that load the same model files can produce the same text. Qwen's files also list `<|im_end|>` (ID 151645) as an end-of-sequence token in `generation_config.json`, so generation stops when the model emits it.

**The template string.** The `chat_template` entry is a short program in Jinja, a text-templating language. Below is the part used for ordinary conversations, taken from the `tokenizer_config.json` of `Qwen/Qwen2.5-0.5B-Instruct` (repository revision `7ae5576`, last modified 2024-09-25). The full template also has branches for tool calling, omitted here. For conversations without tools, this excerpt produces the same text as the full template.

```jinja
{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
```

How to read it:

- `{% ... %}` is a statement (`if`, `for`), and `{{ ... }}` writes text to the output. The `-` in `{%-` and `{{-` removes the whitespace before the tag, so the indentation in the template does not appear in the output.
- **Lines 1 to 5: the system message.** The sequence always starts with one. If the first message has the role `system`, its content is used. Otherwise the template inserts Qwen's default, `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.` This is the default system message that Section 5 warns about.
- **Lines 6 to 10: the other messages.** Each user and assistant message is written as `<|im_start|>`, the role, a newline, the content, `<|im_end|>` and a newline. The condition `not loop.first` skips a system message in first position, since lines 1 to 5 already wrote it.
- **Lines 11 to 13: the generation prompt.** With `add_generation_prompt=True`, the template ends with `<|im_start|>assistant` and a newline, so that the model's next tokens are the reply (Section 5).

Applied to the `messages` list above, the template gives exactly the ChatML text shown earlier, plus one newline after the final `<|im_end|>`. To see the whole template, load the tokenizer as in Section 5 and run `print(tok.chat_template)`.

## 5. Training and inference must use the same template

**At inference**, the application renders the conversation only up to the point where the assistant should speak, so the text ends with `<|im_start|>assistant` and a newline:

```text
<|im_start|>system
You are a customer-support assistant for a food-delivery app. Be brief, polite, and specific.<|im_end|>
<|im_start|>user
my biryani arrived cold<|im_end|>
<|im_start|>assistant
```

The model continues that text: it writes a reply and then emits `<|im_end|>`, at which point generation stops. It does this because training showed it the same pattern many times.

**Fine-tuning in a different format**, for example plain `Customer: ... Agent: ...`, causes two problems:

1. The model learns a pattern it will not see when deployed, and the fine-tuning works against the format the model learned during its own instruction tuning.
2. The model never learns to end a reply with `<|im_end|>`, so at inference it may not stop: it keeps writing, for example by inventing the customer's next message.

**Use the tokenizer's template, not your own.** Small details matter: newlines, where the end-of-turn token goes, and whether a default system message is added when you give none (Qwen2.5's template adds one, as the template string in Section 4 and the printed prompt in the [QLoRA lab](../../labs/qlora_dialogsum/qlora_dialogsum_lab.ipynb) show). The template ships with the tokenizer, so `apply_chat_template` produces the model's real format:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")   # downloads the tokenizer (about 12 MB)

# Training example: the whole conversation, including the reply
train_text = tok.apply_chat_template(messages, tokenize=False)

# Inference prompt: everything before the reply, ending with "<|im_start|>assistant" and a newline
prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
```

For conversations with a system message, `train_text` has the ChatML layout shown in Section 4. The data-preparation notebook writes ChatML by hand so that it runs offline, and its optional `USE_REAL_TOKENIZER` cell checks the hand-written string against `apply_chat_template`.

**Which tokens are trained on.** The template decides what the model sees; the labels decide what it is graded on. The rule is to train on exactly the tokens the model must write itself at inference:

| Token(s) | Who writes it at inference | Label |
| --- | --- | --- |
| System and user messages | The application (your input) | `-100` |
| `<\|im_start\|>assistant` and its newline | The application (`add_generation_prompt=True`) | `-100` |
| The reply | **The model** | Kept |
| `<\|im_end\|>` after the reply | **The model** | **Kept** |

Keeping the label of `<|im_end|>` is what teaches the model to stop. The label at each position is the next token, so at the last token of the reply (in the example, the end of `rs 250.`) the model is graded on predicting `<|im_end|>`. Generation ends only when the model emits an end-of-sequence token (Qwen's `generation_config.json` lists `<|im_end|>`) or reaches the `max_new_tokens` limit. A model never trained to emit `<|im_end|>` runs on to the limit, often inventing the customer's next message. An Instruct model already stops, but fine-tuning data that never rewards stopping can weaken that habit. The `<|im_start|>assistant` line needs no label, because the application supplies it and the model never has to predict it.

A common accident: data collators give padding tokens the label `-100`, so if the padding token is the same as the end-of-sequence token (as is usual with GPT-2), the real end-of-sequence token is masked too. Qwen2.5 pads with `<|endoftext|>` (ID 151643), a different token from `<|im_end|>` (ID 151645). After building the labels, print one example and check that the `<|im_end|>` after the reply keeps its ID rather than `-100`.

Section 5 of [Preparing data for fine-tuning](data_preparation.ipynb) and Exercise 2 of the [QLoRA lab](../../labs/qlora_dialogsum/qlora_dialogsum_lab.ipynb) build such examples.

## 6. Summary

| Idea | In one sentence |
| --- | --- |
| Four stages | Pretraining, instruction tuning, (optional) preference tuning, then your task fine-tuning |
| Instruction tuning | Fine-tuning on many varied (instruction, response) pairs, which turns a text continuer into an assistant that answers and stops |
| Task fine-tuning | The same mechanics on narrow data for one task, usually starting from an instruction-tuned model |
| Chat template | The rule that turns a list of messages into one token sequence, with special tokens for turns and roles |
| Same template everywhere | Fine-tune in the template the model will see at inference, generated by `apply_chat_template` |

## 7. Check your understanding

1. A base model, given `Translate to Hindi: good morning`, writes `Translate to Hindi: good night`. Which training stage is missing, and why does the model behave this way?
2. Instruction tuning and task fine-tuning use the same loss and the same prompt masking. What, then, makes their results so different?
3. FLAN's instruction-tuned model improved on task types it was not trained on. Why does the *variety* of instruction-tuning data matter more than the number of examples of any one task?
4. You fine-tune `Qwen2.5-0.5B-Instruct` on support tickets formatted as `Customer: ... Agent: ...`. Describe two things that may go wrong when the model is deployed behind `apply_chat_template`.
5. Why must `<|im_end|>` after the assistant's reply keep its label, while `<|im_start|>assistant` before it gets `-100`?
6. Your company fine-tunes a model so heavily on support tickets that it can no longer summarise a paragraph. What is this effect called, and which stage of training did the fine-tuning partly undo?

## References

- Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Qwen Team (2024). Qwen2.5 technical report. *arXiv preprint*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS 2023*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Wei, J., Bosma, M., Zhao, V. Y., et al. (2021). Finetuned language models are zero-shot learners. *ICLR 2022*. [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
- Hugging Face. Chat templates. *Transformers documentation*. https://huggingface.co/docs/transformers/main/en/chat_templating
- Qwen Team. `Qwen/Qwen2.5-0.5B-Instruct`: `tokenizer_config.json` and `generation_config.json`. *Hugging Face Hub*. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main
