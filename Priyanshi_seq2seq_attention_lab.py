import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

SEED = 17
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_MODE = "smoke"                                                     
RUN_FULL_TRAINING = False                                                   
MAX_TRAIN_PAIRS = 12000                                          
OUTPUT_DIR = Path("seq2seq_attention_outputs")
print("PyTorch", torch.__version__, "device", device, "mode", DATA_MODE)


SMOKE_TRAIN = [
    ("das buch ist rot", "the book is red"),
    ("das auto ist blau", "the car is blue"),
    ("das haus ist grün", "the house is green"),
    ("das boot ist gelb", "the boat is yellow"),
    ("ein mann geht", "a man walks"),
    ("eine frau geht", "a woman walks"),
    ("ein mann läuft", "a man runs"),
    ("eine frau läuft", "a woman runs"),
]
SMOKE_VALID = [("das auto ist rot", "the car is red")]
SMOKE_TEST = [("das buch ist blau", "the book is blue")]

def normalized(text):
    return " ".join(text.lower().split())

def remove_seen_sources(pairs, seen):
    kept = []
    for src, tgt in pairs:
        key = normalized(src)
        if key not in seen:
            kept.append((src, tgt))
            seen.add(key)
    return kept

if DATA_MODE == "smoke":
    train_pairs, valid_pairs, test_pairs = SMOKE_TRAIN, SMOKE_VALID, SMOKE_TEST
elif DATA_MODE == "multi30k":
                                                                                 
    from datasets import load_dataset
    raw = load_dataset("bentrevett/multi30k")
    train_pairs = [(row["de"], row["en"]) for row in raw["train"]]
    random.Random(SEED).shuffle(train_pairs)
    if MAX_TRAIN_PAIRS is not None:
        train_pairs = train_pairs[:MAX_TRAIN_PAIRS]
    valid_pairs = [(row["de"], row["en"]) for row in raw["validation"]]
    test_pairs = [(row["de"], row["en"]) for row in raw["test"]]
    seen = {normalized(src) for src, _ in train_pairs}
    valid_pairs = remove_seen_sources(valid_pairs, seen)
    test_pairs = remove_seen_sources(test_pairs, seen)
else:
    raise ValueError("DATA_MODE must be 'smoke' or 'multi30k'")

assert train_pairs and valid_pairs and test_pairs
print("train/validation/test pairs:", len(train_pairs), len(valid_pairs), len(test_pairs))
print("example:", train_pairs[0])


SPECIALS = ["<pad>", "<unk>", "<sos>", "<eos>"]
PAD, UNK, SOS, EOS = range(4)

def tokenize(text):
    return re.findall(r"[^\W_]+|[^\w\s]", text.lower(), flags=re.UNICODE)

def build_vocab(texts, min_freq=2, max_size=6000):
    counts = Counter(tok for text in texts for tok in tokenize(text))
    vocab = {tok: idx for idx, tok in enumerate(SPECIALS)}
    for tok, freq in counts.most_common():
        if freq >= min_freq and len(vocab) < max_size:
            vocab[tok] = len(vocab)
    return vocab

min_freq = 1 if DATA_MODE == "smoke" else 2
src_vocab = build_vocab((src for src, _ in train_pairs), min_freq)
tgt_vocab = build_vocab((tgt for _, tgt in train_pairs), min_freq)
tgt_itos = {idx: tok for tok, idx in tgt_vocab.items()}
print("German/English vocabulary sizes:", len(src_vocab), len(tgt_vocab))

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src, tgt = self.pairs[index]
        src_ids = [SOS] + [src_vocab.get(t, UNK) for t in tokenize(src)] + [EOS]
        tgt_ids = [SOS] + [tgt_vocab.get(t, UNK) for t in tokenize(tgt)] + [EOS]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate(batch):
    srcs, tgts = zip(*batch)
    lengths = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    src = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=PAD)
    tgt = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=PAD)
    return src, lengths, tgt

batch_size = 2 if DATA_MODE == "smoke" else 64
train_loader = DataLoader(PairDataset(train_pairs), batch_size=batch_size, shuffle=True, collate_fn=collate)
valid_loader = DataLoader(PairDataset(valid_pairs), batch_size=batch_size, collate_fn=collate)
src_batch, lengths, tgt_batch = next(iter(train_loader))
print("source", tuple(src_batch.shape), "lengths", tuple(lengths.shape), "target", tuple(tgt_batch.shape))


EMB_DIM = 64 if DATA_MODE == "smoke" else 128
HIDDEN_DIM = 64 if DATA_MODE == "smoke" else 256

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src, lengths):
        embedded = self.embedding(src)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, total_length=src.size(1))
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention=None):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD)
        self.attention = attention
        context_dim = hidden_dim if attention is not None else 0
        self.lstm = nn.LSTM(emb_dim + context_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + context_dim, output_dim)

    def forward(self, previous_token, hidden, cell, encoder_outputs, src_mask):
        embedded = self.embedding(previous_token).unsqueeze(1)           
        weights = None
        if self.attention is not None:
            context, weights = self.attention(hidden[-1], encoder_outputs, src_mask)
            lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=-1)
        else:
            context = None
            lstm_input = embedded
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)
        if context is not None:
            output = torch.cat((output, context), dim=-1)
        logits = self.fc_out(output)
        return logits, hidden, cell, weights

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, attention=None):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, EMB_DIM, HIDDEN_DIM)
        self.decoder = Decoder(tgt_vocab_size, EMB_DIM, HIDDEN_DIM, attention)

    def forward(self, src, lengths, tgt, teacher_forcing_ratio=0.5):
        encoder_outputs, (hidden, cell) = self.encoder(src, lengths)
        src_mask = src.ne(PAD)
        previous = tgt[:, 0]                                         
        all_logits = []
        for t in range(1, tgt.size(1)):
            logits, hidden, cell, _ = self.decoder(previous, hidden, cell, encoder_outputs, src_mask)
            all_logits.append(logits)
            previous = tgt[:, t] if random.random() < teacher_forcing_ratio else logits.argmax(-1)
        return torch.stack(all_logits, dim=1)                 

def make_model(with_attention=False):
    attention = ConcatAttention(HIDDEN_DIM) if with_attention else None
    return Seq2Seq(len(src_vocab), len(tgt_vocab), attention).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
baseline = make_model()
src_d, len_d, tgt_d = (x.to(device) for x in (src_batch, lengths, tgt_batch))
with torch.no_grad():
    demo_logits = baseline(src_d, len_d, tgt_d, teacher_forcing_ratio=1.0)
print("baseline logits", tuple(demo_logits.shape))
assert demo_logits.shape == (len(src_batch), tgt_batch.size(1)-1, len(tgt_vocab))


class ConcatAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.alignment = nn.Linear(2 * hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, encoder_outputs, src_mask):
        source_length = encoder_outputs.size(1)
        repeated_query = query.unsqueeze(1).expand(-1, source_length, -1)
        combined = torch.cat((repeated_query, encoder_outputs), dim=-1)
        energies = self.score(torch.tanh(self.alignment(combined))).squeeze(-1)
        energies = energies.masked_fill(~src_mask, torch.finfo(energies.dtype).min)
        weights = torch.softmax(energies, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

ATTENTION_READY = True

if ATTENTION_READY:
    torch.manual_seed(SEED)
    attention_model = make_model(with_attention=True)
    source_outputs, (hidden, cell) = attention_model.encoder(src_d, len_d)
    mask = src_d.ne(PAD)
    context, weights = attention_model.decoder.attention(hidden[-1], source_outputs, mask)
    assert weights.shape == src_d.shape
    assert context.shape == (src_d.size(0), HIDDEN_DIM)
    assert torch.allclose(weights.sum(dim=1), torch.ones(src_d.size(0), device=device), atol=1e-5)
    assert torch.all(weights[~mask] < 1e-6)
    logits = attention_model(src_d, len_d, tgt_d, teacher_forcing_ratio=1.0)
    loss = criterion(logits.reshape(-1, len(tgt_vocab)), tgt_d[:, 1:].reshape(-1))
    loss.backward()
    assert attention_model.decoder.attention.alignment.weight.grad is not None
    assert attention_model.decoder.attention.score.weight.grad is not None
    assert attention_model.encoder.lstm.weight_ih_l0.grad is not None
    print("attention shapes, mask, and gradient checks passed; loss", round(loss.item(), 3))
else:
    print("Exercise 2 pending: baseline remains runnable; set ATTENTION_READY=True after implementation")


def run_epoch(model, loader, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    for batch_index, (src, lengths, tgt) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        src, lengths, tgt = src.to(device), lengths.to(device), tgt.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(src, lengths, tgt, 0.5 if training else 1.0)
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = gold.ne(PAD).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)

def greedy_translate(model, source_text, max_steps=40):
    model.eval()
    src_ids = [SOS] + [src_vocab.get(t, UNK) for t in tokenize(source_text)] + [EOS]
    src = torch.tensor(src_ids, device=device).unsqueeze(0)
    lengths = torch.tensor([len(src_ids)], device=device)
    generated, rows = [], []
    with torch.no_grad():
        encoder_outputs, (hidden, cell) = model.encoder(src, lengths)
        previous = torch.tensor([SOS], device=device)
        for _ in range(max_steps):
            logits, hidden, cell, weights = model.decoder(
                previous, hidden, cell, encoder_outputs, src.ne(PAD)
            )
            previous = logits.argmax(dim=-1)
            token_id = previous.item()
            if token_id == EOS:
                break
            generated.append(tgt_itos.get(token_id, "<unk>"))
            if weights is not None:
                rows.append(weights[0].cpu())
    attention_map = torch.stack(rows) if rows else None
    return generated, attention_map

smoke_optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
smoke_loss = run_epoch(baseline, train_loader, smoke_optimizer, max_batches=2)
print("baseline two-batch smoke loss:", round(smoke_loss, 3))
print("smoke translation (not a quality result):", greedy_translate(baseline, valid_pairs[0][0])[0])
if ATTENTION_READY:
    smoke_attention_optimizer = torch.optim.Adam(attention_model.parameters(), lr=1e-3)
    print("attention two-batch smoke loss:", round(run_epoch(attention_model, train_loader, smoke_attention_optimizer, max_batches=2), 3))


def train_model(with_attention, epochs=8):
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = make_model(with_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        valid_loss = run_epoch(model, valid_loader)
        history.append((train_loss, valid_loss))
        print(f"{'attention' if with_attention else 'baseline'} epoch {epoch}: train {train_loss:.3f}, valid {valid_loss:.3f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    model.load_state_dict(best_state)
    return model, history

if RUN_FULL_TRAINING:
    assert DATA_MODE == "multi30k", "Use the full dataset for a quality comparison"
    assert ATTENTION_READY, "Implement Exercise 2 first"
    OUTPUT_DIR.mkdir(exist_ok=True)
    baseline, baseline_history = train_model(False)
    attention_model, attention_history = train_model(True)
    torch.save(baseline.state_dict(), OUTPUT_DIR / "baseline.pt")
    torch.save(attention_model.state_dict(), OUTPUT_DIR / "attention.pt")
    print("parameters:", sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in attention_model.parameters()))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, history in [("baseline", baseline_history), ("attention", attention_history)]:
        ax.plot(range(1, len(history)+1), [row[1] for row in history], marker="o", label=name)
    ax.set(xlabel="Epoch", ylabel="Teacher-forced validation cross-entropy", title="Validation loss")
    ax.legend()
    fig.tight_layout()
    plt.show()
else:
    print("Full training skipped. Set both switches deliberately for the week-long comparison.")


def beam_translate(model, source_text, beam_width=3, max_steps=40, length_alpha=0.6):
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    model.eval()
    src_ids = [SOS] + [src_vocab.get(t, UNK) for t in tokenize(source_text)] + [EOS]
    src = torch.tensor(src_ids, device=device).unsqueeze(0)
    lengths = torch.tensor([len(src_ids)], device=device)
    def rank(beam):
        token_ids, log_score, hidden, cell, ended = beam
        generated_length = max(1, len(token_ids) - 1)
        penalty = ((5 + generated_length) / 6) ** length_alpha
        return log_score / penalty
    with torch.no_grad():
        encoder_outputs, (initial_hidden, initial_cell) = model.encoder(src, lengths)
        src_mask = src.ne(PAD)
        beams = [([SOS], 0.0, initial_hidden, initial_cell, False)]
        for _ in range(max_steps):
            candidates = []
            for token_ids, log_score, hidden, cell, ended in beams:
                if ended:
                    candidates.append((token_ids, log_score, hidden, cell, ended))
                    continue
                previous = torch.tensor([token_ids[-1]], device=device)
                logits, next_hidden, next_cell, _ = model.decoder(
                    previous, hidden, cell, encoder_outputs, src_mask
                )
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                count = min(beam_width, log_probs.numel())
                values, indices = torch.topk(log_probs, count)
                for value, index in zip(values.tolist(), indices.tolist()):
                    new_tokens = token_ids + [index]
                    candidates.append((
                        new_tokens,
                        log_score + value,
                        next_hidden.clone(),
                        next_cell.clone(),
                        index == EOS
                    ))
            beams = sorted(candidates, key=rank, reverse=True)[:beam_width]
            if all(beam[4] for beam in beams):
                break
        best = max(beams, key=rank)
        token_ids = best[0][1:]
        if token_ids and token_ids[-1] == EOS:
            token_ids = token_ids[:-1]
        return [tgt_itos.get(token_id, "<unk>") for token_id in token_ids]

BEAM_READY = True

if BEAM_READY:
    for model in [baseline] + ([attention_model] if ATTENTION_READY else []):
        source = valid_pairs[0][0]
        greedy_tokens, _ = greedy_translate(model, source, max_steps=10)
        beam_one = beam_translate(model, source, beam_width=1, max_steps=10)
        assert beam_one == greedy_tokens, (greedy_tokens, beam_one)
    print("Beam width 1 matches greedy on the smoke example.")
else:
    print("Exercise 4 pending: beam search has not been tested.")


import math

TEST_EVAL_LIMIT = 200                                                    

def evaluate_generations(model, pairs, method="greedy"):
    from sacrebleu.metrics import BLEU                                   
    hypotheses, references = [], []
    for source, reference in pairs:
        if method == "greedy":
            tokens, _ = greedy_translate(model, source, max_steps=40)
        elif method == "beam":
            tokens = beam_translate(model, source, beam_width=3, max_steps=40, length_alpha=0.6)
        else:
            raise ValueError("method must be greedy or beam")
        hypotheses.append(" ".join(tokens))
        references.append(" ".join(tokenize(reference)))
    bleu = BLEU().corpus_score(hypotheses, [references]).score
    exact = sum(h == r for h, r in zip(hypotheses, references)) / len(pairs)
    return {"BLEU": bleu, "exact_match": exact}, hypotheses, references

if RUN_FULL_TRAINING:
    assert BEAM_READY, "Complete Exercise 4 before test evaluation"
    evaluation_pairs = test_pairs if TEST_EVAL_LIMIT is None else random.Random(SEED).sample(
        test_pairs, min(TEST_EVAL_LIMIT, len(test_pairs))
    )
    evaluation_loader = DataLoader(PairDataset(evaluation_pairs), batch_size=batch_size, collate_fn=collate)
    results, generated = {}, {}
    for name, model in [("baseline", baseline), ("attention", attention_model)]:
        cross_entropy = run_epoch(model, evaluation_loader)
        perplexity = math.exp(cross_entropy)
        for method in ["greedy", "beam"]:
            scores, hypotheses, references = evaluate_generations(model, evaluation_pairs, method)
            results[(name, method)] = {"perplexity": perplexity, **scores}
            generated[(name, method)] = hypotheses
            print(name, method, results[(name, method)])
    print("Held-out examples evaluated:", len(evaluation_pairs))
    attention_outputs = generated[("attention", "greedy")]
    short_index = min(range(len(evaluation_pairs)), key=lambda i: len(tokenize(evaluation_pairs[i][0])))
    long_index = max(range(len(evaluation_pairs)), key=lambda i: len(tokenize(evaluation_pairs[i][0])))
    failed_index = next((i for i, output in enumerate(attention_outputs) if output != references[i]), None)
    example_indices = dict.fromkeys(i for i in [short_index, long_index, failed_index] if i is not None)
    print("Examples: short, long, and first attention exact-match failure where available")
    for index in example_indices:
        print("\nGerman:", evaluation_pairs[index][0])
        print("English reference:", references[index])
        for name in ["baseline", "attention"]:
            for method in ["greedy", "beam"]:
                print(name, method, generated[(name, method)][index])
else:
    print("Test evaluation waits for the full training run.")


def plot_attention(model, source_text):
    import matplotlib.pyplot as plt
    tokens, matrix = greedy_translate(model, source_text)
    if matrix is None:
        print("No attention map; use the attention model and produce at least one token.")
        return
    source_tokens = ["<sos>"] + tokenize(source_text) + ["<eos>"]
    fig, ax = plt.subplots(figsize=(max(7, len(source_tokens)*0.5), max(3, len(tokens)*0.4)))
    im = ax.imshow(matrix.numpy(), aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(source_tokens)), source_tokens, rotation=60, ha="right")
    ax.set_yticks(range(len(tokens)), tokens)
    ax.set_xlabel("German source position")
    ax.set_ylabel("Generated English step")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    plt.show()

if RUN_FULL_TRAINING:
    successful_index = next((i for i, output in enumerate(attention_outputs) if output == references[i]), None)
    failed_index = next((i for i, output in enumerate(attention_outputs) if output != references[i]), None)
    for label, index in [("success", successful_index), ("failure", failed_index)]:
        if index is not None:
            print(label, "source:", evaluation_pairs[index][0])
            plot_attention(attention_model, evaluation_pairs[index][0])
        else:
            print("No", label, "example found; choose another case or explain this limitation.")
else:
    print("Attention plotting waits for a trained attention model.")
