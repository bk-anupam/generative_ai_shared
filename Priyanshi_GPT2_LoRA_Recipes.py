import os
import re
import time
import math
import random
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from accelerate.utils import extract_model_from_parallel
from peft import LoraConfig, get_peft_model, TaskType
import sacrebleu
from rouge_score import rouge_scorer

SEED = 42
CKPT = "openai-community/gpt2"
MAX_LEN = 768
MAX_NEW = 400
N_EVAL = 40
EPOCHS = 2
BATCH = 8
TRAIN_LIMIT = None
FULL_LR = 5e-5
LORA_LR = 5e-4
GB = 1e9
OUT = Path("/kaggle/working/recipe_lab") if Path("/kaggle/working").is_dir() else Path("recipe_lab_outputs")
DATA_DIR, ADAPTER_DIR = OUT / "data", OUT / "adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_GPU = DEVICE == "cuda"
random.seed(SEED)
torch.manual_seed(SEED)

DATA_URL = "https://data.mendeley.com/public-files/datasets/xsphgmmh7b/files/6fb5ae32-a815-4b8f-9d68-8d512227eac8/file_downloaded"
DATA_SHA256 = "273f8d45f41aca9af0396dda07af137c68c7387a1506992904144b944c4346be"
DEVANAGARI = r"[\u0900-\u097F]"
NOT_INDIAN = {
    "Afghan", "African", "American", "Arab", "Asian", "British", "Burmese", "Cantonese", "Caribbean",
    "Chinese", "Continental", "European", "French", "Fusion", "Greek", "Hunan", "Indonesian",
    "Italian Recipes", "Japanese", "Jewish", "Korean", "Malaysian", "Mediterranean", "Mexican",
    "Middle Eastern", "Nepalese", "Pakistani", "Shandong", "Sichuan", "Sri Lankan", "Thai",
    "Vietnamese", "World Breakfast", "Appetizer", "Brunch", "Dessert", "Dinner", "Lunch", "Side Dish", "Snack"
}
ROUGE_KEYS = {"ROUGE-1": "rouge1", "ROUGE-2": "rouge2", "ROUGE-L": "rougeL"}
scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS.values()), use_stemmer=True)


def clean_text(text):
    text = re.sub(r"\.(?=[A-Z])", ". ", str(text))
    text = re.sub(r",(?=[A-Za-z])", ", ", text)
    return " ".join(text.split())


def clean_name(name):
    return " ".join(re.sub(r"\brecipe\b", "", str(name), flags=re.IGNORECASE).split()).strip(" -")


def dish_key(raw_name):
    name = str(raw_name).lower()
    name = re.split(r" - |\(", name, maxsplit=1)[0]
    name = re.sub(r"\brecipes?\b", " ", name)
    name = re.sub(r"[^a-z ]", " ", name)
    return " ".join(name.split())


def build_example(row, tok, add_bos=False):
    prompt = format_prompt(row)
    prompt_tokens = tok(prompt, add_special_tokens=False)["input_ids"]
    if add_bos:
        prompt_tokens = [tok.bos_token_id] + prompt_tokens
    method_tokens = tok(row["method"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    input_ids = prompt_tokens + method_tokens
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_tokens) + method_tokens,
    }


def text_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    rouge_values = {key: [] for key in ROUGE_KEYS}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for key, rouge_key in ROUGE_KEYS.items():
            rouge_values[key].append(scores[rouge_key].fmeasure * 100)
    return {"BLEU": bleu, **{key: sum(values) / len(values) for key, values in rouge_values.items()}}


def make_args(name, lr):
    return TrainingArguments(
        output_dir=str(OUT / name),
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        fp16=HAS_GPU,
        eval_strategy="epoch",
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
    )


def count_trainable(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def collate(batch, pad_id):
    width = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[i, :length] = torch.tensor(item["input_ids"])
        labels[i, :length] = torch.tensor(item["labels"])
        attention_mask[i, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train(model, args, train_data, eval_data, title, pad_id):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=lambda batch: collate(batch, pad_id),
    )
    if HAS_GPU:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    extract_model_from_parallel(model, keep_fp32_wrapper=False)
    stats = {
        "trainable": count_trainable(model),
        "minutes": (time.perf_counter() - start) / 60,
        "peak_GB": torch.cuda.max_memory_allocated() / GB if HAS_GPU else float("nan"),
    }
    history = trainer.state.log_history
    validation = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
    print(title, stats, "final validation loss:", validation[-1] if validation else None)
    return stats, validation[-1] if validation else float("nan")


def format_prompt(row):
    ingredients = "\n".join(f"- {item}" for item in row["ingredients"])
    return (
        f"Dish: {row['name']}\nCuisine: {row['cuisine']} | Course: {row['course']} | Diet: {row['diet']}\n"
        f"Ingredients:\n{ingredients}\nMethod:\n"
    )


def make_dataset(rows, tok, add_bos=False):
    data = [build_example(row, tok, add_bos) for row in rows]
    return [item for item in data if len(item["input_ids"]) <= MAX_LEN]


@torch.no_grad()
def generate_methods(model, tok, rows, add_bos=False):
    model.eval()
    predictions, stopped = [], []
    for row in rows:
        prompt = format_prompt(row)
        encoded = tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(DEVICE)
        if add_bos:
            input_ids = torch.cat([torch.tensor([[tok.bos_token_id]], device=DEVICE), input_ids], dim=1)
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new_ids = output[0, input_ids.shape[1]:]
        stopped.append(bool(len(new_ids) and new_ids[-1].item() == tok.eos_token_id))
        predictions.append(tok.decode(new_ids, skip_special_tokens=True).strip())
    return predictions, stopped


def ingredient_recall(predictions, rows):
    values = []
    for prediction, row in zip(predictions, rows):
        text = prediction.lower()
        ingredients = [item.lower().strip() for item in row["ingredients"] if item.strip()]
        if ingredients:
            values.append(sum(item in text for item in ingredients) / len(ingredients))
    return 100 * sum(values) / len(values) if values else 0.0


def evaluate(name, predictions, stopped, rows):
    refs = [row["method"] for row in rows]
    metrics = text_metrics(predictions, refs)
    metrics["ingredient_recall"] = ingredient_recall(predictions, rows)
    metrics["eos_rate"] = 100 * sum(stopped) / max(len(stopped), 1)
    print(name, {key: round(value, 2) for key, value in metrics.items()})
    return metrics


@torch.no_grad()
def mean_loss(model, texts, tok):
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for text in texts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        if ids.shape[1] < 2:
            continue
        loss = model(input_ids=ids, labels=ids).loss
        count = ids.shape[1] - 1
        weighted_loss += float(loss.item()) * count
        token_count += count
    return weighted_loss / max(token_count, 1)


def load_recipes():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "IndianFoodDatasetCSV.csv"
    if not csv_path.exists():
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "recipe-lab"})
        with urllib.request.urlopen(request) as response:
            csv_path.write_bytes(response.read())
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset SHA-256 mismatch.")
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw = raw.apply(lambda col: col if pd.api.types.is_numeric_dtype(col) else col.str.replace("\ufeff", "").str.strip())
    english = ~(raw["RecipeName"].str.contains(DEVANAGARI) |
                raw["Ingredients"].fillna("").str.contains(DEVANAGARI) |
                raw["Instructions"].str.contains(DEVANAGARI))
    kept = raw[english & raw["Ingredients"].notna() & ~raw["Cuisine"].isin(NOT_INDIAN)]
    recipes = [{
        "raw_name": row.RecipeName,
        "name": clean_name(row.RecipeName),
        "cuisine": row.Cuisine,
        "course": row.Course,
        "diet": row.Diet,
        "ingredients": [item.strip() for item in row.Ingredients.split(",") if item.strip()],
        "method": clean_text(row.Instructions),
    } for row in kept.itertuples()]
    for row in recipes:
        row["key"] = dish_key(row["raw_name"])
    keys = sorted({row["key"] for row in recipes})
    random.Random(SEED).shuffle(keys)
    n_val = n_test = len(keys) // 10
    split_of = {key: "val" for key in keys[:n_val]}
    split_of.update({key: "test" for key in keys[n_val:n_val + n_test]})
    split_of.update({key: "train" for key in keys[n_val + n_test:]})
    splits = {name: [row for row in recipes if split_of[row["key"]] == name] for name in ("train", "val", "test")}
    assert not ({r["key"] for r in splits["train"]} & {r["key"] for r in splits["val"] + splits["test"]})
    print(f"{len(raw)} raw rows -> {len(recipes)} cleaned recipes", {key: len(value) for key, value in splits.items()})
    return splits


def main():
    print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEVICE)
    OUT.mkdir(parents=True, exist_ok=True)
    splits = load_recipes()
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    tokenizer.pad_token = tokenizer.eos_token
    train_rows = splits["train"][:TRAIN_LIMIT]
    val_rows = splits["val"]
    train_data = make_dataset(train_rows, tokenizer)
    val_data = make_dataset(val_rows, tokenizer)
    eval_rows = random.Random(SEED).sample(val_rows, min(N_EVAL, len(val_rows)))
    print("training examples:", len(train_data), "| validation examples:", len(val_data), "| generation eval:", len(eval_rows))

    base_model = AutoModelForCausalLM.from_pretrained(CKPT).to(DEVICE)
    base_preds, base_stopped = generate_methods(base_model, tokenizer, eval_rows)
    results = [{"model": "GPT-2 original", **evaluate("GPT-2 original", base_preds, base_stopped, eval_rows)}]

    transformers.set_seed(SEED)
    full_model = AutoModelForCausalLM.from_pretrained(CKPT).to(DEVICE)
    full_stats, full_val_loss = train(full_model, make_args("full_finetuning", FULL_LR), train_data, val_data,
                                      "GPT-2 full fine-tuning", tokenizer.pad_token_id)
    full_preds, full_stopped = generate_methods(full_model, tokenizer, eval_rows)
    results.append({"model": "GPT-2 full FT", **evaluate("GPT-2 full FT", full_preds, full_stopped, eval_rows),
                    "val_loss": full_val_loss, **full_stats})

    transformers.set_seed(SEED)
    lora_base = AutoModelForCausalLM.from_pretrained(CKPT).to(DEVICE)
    predicted_lora_params = 12 * (8 * 768 + 2304 * 8)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
        task_type=TaskType.CAUSAL_LM,
        fan_in_fan_out=True,
    )
    lora_model = get_peft_model(lora_base, lora_config)
    assert count_trainable(lora_model) == predicted_lora_params
    lora_stats, lora_val_loss = train(lora_model, make_args("lora", LORA_LR), train_data, val_data,
                                      "GPT-2 LoRA", tokenizer.pad_token_id)
    lora_preds, lora_stopped = generate_methods(lora_model, tokenizer, eval_rows)
    results.append({"model": "GPT-2 LoRA", **evaluate("GPT-2 LoRA", lora_preds, lora_stopped, eval_rows),
                    "val_loss": lora_val_loss, **lora_stats})
    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

    general_texts = [
        "The train to Jaipur left the station twenty minutes late. Most passengers had already found their seats, and a few were reading newspapers while the countryside moved slowly past the windows.",
        "Rainwater that falls on the hills collects in small streams, which join to form rivers. Some of the water soaks into the ground and is stored in aquifers that supply wells during the dry season.",
        "The committee will meet on Monday to discuss the budget for the next financial year. Members are asked to read the report in advance and to send any questions to the secretary by Friday.",
        "In the final over, the batsman needed twelve runs to win. He hit the first ball for four, missed the next two, and then lifted a full toss over long-on for six.",
    ]
    forgetting = {
        "GPT-2 original": mean_loss(base_model, general_texts, tokenizer),
        "GPT-2 full FT": mean_loss(full_model, general_texts, tokenizer),
        "GPT-2 LoRA": mean_loss(lora_model, general_texts, tokenizer),
    }
    print("General English loss/perplexity")
    for name, loss in forgetting.items():
        print(name, round(loss, 3), round(math.exp(loss), 2))

    adapter_path = OUT / "adapter"
    lora_model.save_pretrained(adapter_path)
    merged_model = lora_model.merge_and_unload()
    test_rows = splits["test"][:min(N_EVAL, len(splits["test"]))]
    test_preds, test_stopped = generate_methods(merged_model, tokenizer, test_rows)
    evaluate("Merged LoRA test", test_preds, test_stopped, test_rows)
    print("Saved results to", OUT)


if __name__ == "__main__":
    main()
