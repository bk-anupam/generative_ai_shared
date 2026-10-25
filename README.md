# Generative AI labs

Chirag Duhoon

Course material: [bk-anupam/generative_ai_shared](https://github.com/bk-anupam/generative_ai_shared).

| Lab | Notebook | Saved execution |
| --- | --- | --- |
| 1–4 | Existing submissions in `module1` and `module2` | Preserved without changes |
| 5: Text to BERT | [Notebook](module2/labs/submissions/lab5/text_to_bert_lab_chirag.ipynb) | Complete CPU run; all 11 exercise checks passed |
| 6: GPT-2 recipes | [Notebook](module3/labs/submissions/lab6/gpt2_lora_recipes_lab_chirag.ipynb) | CPU smoke check: 8 training examples, 8 validation examples, 5 generated recipes; full experiment still needs a GPU run |
| 7: Patent matching | [Notebook](module3/labs/submissions/lab7/patent_phrase_matching_chirag.ipynb) | Implementation and targeted checks; competition experiments and late submission not run |
| Optional: U-Net | [Notebook](module1/labs/submissions/optional/unet_pet_segmentation_chirag.ipynb) | Complete GPU run: 10 epochs, 5,912 train / 739 validation / 739 test images; test Dice 0.8576, IoU 0.7652 |

The recipe reduced run checks that its pipeline works and is not a substitute for the full experiment. BERT and U-Net now have complete executions. Outputs, charts and discussion distinguish measured results from pending work.

## Run the remaining experiments

Use a Kaggle GPU notebook. Keep each notebook's working directory separate. Dependencies are listed in [requirements-labs.txt](requirements-labs.txt); restart the session after installing packages if necessary. The versions recorded inside executed notebooks describe the environment used for the saved outputs.

- **Recipes:** enable internet to download the attributed dataset and GPT-2. A GPU session defaults to `RECIPE_RUN_MODE=full`: two epochs, the full filtered training split, and 40 validation recipes. The CPU fallback is explicitly labelled `smoke`. Run all cells and update the saved smoke-run discussion using the new measurements. The optional Llama extension remains disabled and requires separate authorized access to the gated model.
- **Patents:** run [the checkpoint helper](module3/labs/submissions/lab7/download_weights_chirag.ipynb) with internet on. Attach its saved output and the original Kaggle competition data to the project notebook, enable a GPU, and turn internet off. The project runs the starter baseline and two controlled experiments, selects by validation correlation, and writes predictions, CSV analyses and a four-page PDF report. A real late submission is still needed for public/private scores and the screenshot. No scores have been invented. Do not use the later fully labelled dataset.
- **U-Net:** attach Oxford-IIIT Pet images and trimaps. The default uses the full data, width 32 and ten epochs. Locally, set `PET_DATA_ROOT` to the extracted dataset. `PET_SMOKE=1` uses the labelled small subset and width 16. The full GPU run selected epoch 8 by validation Dice; held-out test Dice was 0.8576 and IoU 0.7652 over 739 images. The notebook discusses failures involving occlusion and missed body regions.

## Validation performed

- All five new notebooks passed notebook-schema and Python-syntax validation.
- BERT ran top to bottom, with all exercise assertions passing and charts embedded.
- The recipe CPU check ran top to bottom, including full/LoRA parameter counts, evaluation, forgetting and adapter-merge assertions.
- Patent checks used the real DeBERTa tokenizer to verify pair markers, context removal, float32 regression labels and unlabelled test inputs; a grouped-split check verified anchor isolation. No training scores are claimed.
- U-Net passed output-shape and backward-gradient checks, then trained for ten epochs on the full training split and evaluated its validation-selected checkpoint on all 739 held-out test images.

Downloaded datasets and model weights are excluded from Git. Dataset and model attribution remains in the notebooks. Oxford pet images and derivatives retain the dataset's [CC BY-SA 4.0 attribution requirements](https://www.robots.ox.ac.uk/~vgg/data/pets/).
