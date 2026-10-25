import math
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bpe_tokenize_word(word, merges, base_vocab, unk_token="[UNK]"):
    tokens = [ch if ch in base_vocab else unk_token for ch in word]
    for (a, b), merged in merges:
        rebuilt = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                rebuilt.append(merged)
                i += 2
            else:
                rebuilt.append(tokens[i])
                i += 1
        tokens = rebuilt
    return tokens


def wordpiece_tokenize_word(word, vocab, unk_token="[UNK]", prefix="##"):
    tokens = []
    start = 0
    while start < len(word):
        end = len(word)
        match = None
        while end > start:
            piece = word[start:end] if start == 0 else prefix + word[start:end]
            if piece in vocab:
                match = piece
                break
            end -= 1
        if match is None:
            return [unk_token]
        tokens.append(match)
        start = end
    return tokens


def subword_stats(tokenizer, texts):
    n_words = n_tokens = n_split = n_unk = 0
    for enc in tokenizer.encode_batch(texts):
        counts = collections.Counter(wid for wid in enc.word_ids if wid is not None)
        n_words += len(counts)
        n_tokens += sum(counts.values())
        n_split += sum(count > 1 for count in counts.values())
        n_unk += sum(token == "[UNK]" for token in enc.tokens)
    return {
        "fertility": n_tokens / n_words if n_words else 0.0,
        "split_rate": n_split / n_words if n_words else 0.0,
        "unk_rate": n_unk / n_tokens if n_tokens else 0.0,
    }


def tokens_per_word(tokenizer, text):
    words = len(text.split())
    return len(tokenizer.tokenize(text)) / words if words else 0.0


def scaled_dot_product_attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


def make_padding_mask(input_ids, pad_id):
    return (input_ids != pad_id)[:, None, None, :]


@torch.no_grad()
def padding_leakage(model, ids, junk_ids, pad_id, use_mask=True):
    mask = make_padding_mask(ids, pad_id) if use_mask else None
    original_output = model(ids, mask)
    changed_output = model(junk_ids, mask)
    real_positions = ids != pad_id
    return float((original_output - changed_output).abs()[real_positions].max().item())


def describe_bert_input(enc, tokenizer):
    ids = enc["input_ids"][0]
    types = enc["token_type_ids"][0]
    attn = enc["attention_mask"][0]
    return {
        "cls_position": int((ids == tokenizer.cls_token_id).nonzero(as_tuple=True)[0][0].item()),
        "sep_positions": (ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0].tolist(),
        "segment_b_tokens": int(((types == 1) & (attn == 1)).sum().item()),
        "pad_tokens": int((attn == 0).sum().item()),
    }


def mask_tokens(input_ids, special_mask, mask_token_id, vocab_size, mlm_prob=0.15, generator=None):
    labels = input_ids.clone()
    masked_ids = input_ids.clone()
    selected = (torch.rand(input_ids.shape, generator=generator, device=input_ids.device) < mlm_prob) & ~special_mask.bool()
    labels[~selected] = -100
    replace_mask = (torch.rand(input_ids.shape, generator=generator, device=input_ids.device) < 0.8) & selected
    masked_ids[replace_mask] = mask_token_id
    remaining = selected & ~replace_mask
    random_mask = (torch.rand(input_ids.shape, generator=generator, device=input_ids.device) < 0.5) & remaining
    random_ids = torch.randint(vocab_size, input_ids.shape, generator=generator, device=input_ids.device)
    masked_ids[random_mask] = random_ids[random_mask]
    return masked_ids, labels


@torch.no_grad()
def _load_bert():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    global bert_tok, bert_mlm
    if "bert_tok" not in globals() or "bert_mlm" not in globals():
        bert_tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        bert_mlm = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased").to(device).eval()
    return bert_tok, bert_mlm


@torch.no_grad()
def top_k_predictions(text, k=5):
    tokenizer, model = _load_bert()
    enc = tokenizer(text, return_tensors="pt").to(device)
    positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(positions) != 1:
        raise ValueError("text must contain exactly one [MASK]")
    probs = torch.softmax(model(**enc).logits[0, positions.item()], dim=-1)
    values, indices = torch.topk(probs, k)
    return [(tokenizer.convert_ids_to_tokens(int(idx)), float(prob)) for idx, prob in zip(indices, values)]


@torch.no_grad()
def pronoun_probs(sentence):
    tokenizer, model = _load_bert()
    enc = tokenizer(sentence, return_tensors="pt").to(device)
    positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(positions) != 1:
        raise ValueError("sentence must contain exactly one [MASK]")
    probs = torch.softmax(model(**enc).logits[0, positions.item()], dim=-1)
    return float(probs[tokenizer.convert_tokens_to_ids("he")]), float(probs[tokenizer.convert_tokens_to_ids("she")])


def run_smoke_tests():
    base_vocab = set("bghnpsu")
    merges = [(("u", "g"), "ug"), (("u", "n"), "un"), (("h", "ug"), "hug")]
    cases = {
        "hugs": ["hug", "s"], "bug": ["b", "ug"], "pun": ["p", "un"],
        "mug": ["[UNK]", "ug"], "thug": ["[UNK]", "hug"], "unhug": ["un", "hug"]
    }
    for word, expected in cases.items():
        assert bpe_tokenize_word(word, merges, base_vocab) == expected
    vocab = {"[UNK]", "b", "h", "p", "u", "##u", "##g", "##n", "##s", "hu", "hug", "##gs", "##un"}
    cases = {
        "hugs": ["hug", "##s"], "bun": ["b", "##un"], "pugs": ["p", "##u", "##gs"],
        "hu": ["hu"], "mug": ["[UNK]"], "hugsx": ["[UNK]"]
    }
    for word, expected in cases.items():
        assert wordpiece_tokenize_word(word, vocab) == expected
    generator = torch.Generator().manual_seed(0)
    B, H, T, d_k = 2, 3, 5, 4
    q, k, v = (torch.randn(B, H, T, d_k, generator=generator) for _ in range(3))
    lengths = torch.tensor([5, 3])
    mask = (torch.arange(T)[None, :] < lengths[:, None])[:, None, None, :]
    out, weights = scaled_dot_product_attention(q, k, v, mask)
    assert out.shape == (B, H, T, d_k)
    assert weights.shape == (B, H, T, T)
    assert torch.allclose(weights.sum(-1), torch.ones(B, H, T))
    assert (weights[1, :, :, 3:] == 0).all()
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), atol=1e-5)
    ids = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 10, 11, 12]])
    padding_mask = make_padding_mask(ids, 0)
    assert padding_mask.shape == (2, 1, 1, 5)
    assert padding_mask.dtype == torch.bool
    assert padding_mask[0, 0, 0].tolist() == [True, True, True, False, False]
    print("Core checks passed.")


QUESTION_ANSWERS = {
    1: "With the new first merge, hug becomes ['hu', 'g'] because hu is merged before ug can be formed. This shows that BPE merge order reflects the learned sequence of operations; choosing longest-first can produce a different segmentation.",
    2: "BPE can preserve recognizable pieces after an unknown character, retaining partial information. WordPiece marks the whole word unknown if any part cannot be segmented. GPT-2 byte-level BPE represents UTF-8 bytes, so every input byte is representable and an unknown token is unnecessary.",
    3: "As vocabulary size increases, fertility and split rate generally decrease because more words or longer subwords can be represented as single tokens. Embedding parameters equal vocabulary size times hidden size, so 16,000 tokens require 16 times as many embedding parameters as 1,000 tokens. Larger vocabularies shorten sequences but increase embedding-table size. Validation text measures performance on held-out text rather than training text.",
    4: "Use the measured table to identify the lowest tokens-per-word tokenizer for each language. The English-only uncased BERT tokenizer may map Devanagari characters to unknown tokens, while GPT-2 byte-level BPE can represent them but often uses many byte tokens. The uncased tokenizer may decode Hindi poorly because its vocabulary does not adequately represent the original script. More tokens consume context-window capacity and can increase token-based API costs for Hindi users.",
    5: "Input IDs: (2, 6); embeddings: (2, 6, 32); q after splitting: (2, 4, 6, 8); attention weights: (2, 4, 6, 6); encoder output: (2, 6, 32). Padded query outputs are harmless when downstream tasks ignore padded positions. BERT attends bidirectionally because masked-token prediction uses left and right context. GPT-2 uses a causal mask to prevent positions from seeing future tokens during next-token prediction, in addition to masking padding.",
    6: "The first [SEP] closes sentence A and the second closes sentence B. Padding positions commonly receive segment ID 0; this does not matter because the attention mask excludes padding from attention and padding positions are not real inputs. For sentence-pair classification, the final [CLS] hidden state is commonly fed to a classification head.",
    7: "Replacing every selected token with [MASK] would create a mismatch because [MASK] is not normally present in downstream text. Leaving some selected tokens unchanged teaches the model to predict tokens that look normal; random replacements help it handle corrupted inputs and avoid relying only on the mask marker. A random replacement can accidentally equal the original token, which is a small sampling effect and does not invalidate the objective.",
    8: "Rank the three position types using the measured accuracies. [MASK] positions are explicit corruption cues, random replacements may be inconsistent with context, and unchanged positions look like ordinary text, so their relative difficulty can differ. Accuracy on unchanged positions is not clean evidence of understanding because the original token is visible and WikiText may overlap with BERT's pretraining data.",
    9: "Compare the two template charts and report whether occupation rankings remain stable. The probe cannot establish real-world bias because it tests few occupations, templates, and pronouns; probabilities depend on sentence wording and context; and masked-token preferences do not directly measure decisions by a deployed system. A stronger experiment would use many balanced templates, more occupations and demographic groups, repeated paraphrases, and statistical tests."
}


if __name__ == "__main__":
    run_smoke_tests()
    print("Device:", device)
    print("Written-answer drafts are available in QUESTION_ANSWERS.")
