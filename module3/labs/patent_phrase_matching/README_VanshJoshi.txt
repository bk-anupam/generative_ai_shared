Lab 7 - US Patent Phrase to Phrase Matching
Vansh Joshi

Completed: baseline, pretrained bi-encoder, two controlled context experiments, error analysis, offline Kaggle run and scored late submission.

Final Kaggle version: https://www.kaggle.com/code/vanshjoshi73/lab-7-patent-phrase-matching-vansh-joshi?scriptVersionId=368715520
Local selection validation Pearson: 0.819542
Kaggle validation Pearson: 0.819209
Public score: 0.7904
Private score: 0.8102

Report: patent_phrase_report_VanshJoshi.pdf (4 pages)
Score screenshot: kaggle_submissions_VanshJoshi.png
Final notebook: final_submission_VanshJoshi.ipynb (executed Kaggle export)
Experiments: patent_phrase_experiments_VanshJoshi.ipynb (executed local runs)
Weight helper: download_weights_VanshJoshi.ipynb (executed Kaggle export)
Evidence archive: patent_phrase_evidence_VanshJoshi.zip

Run the weight helper once with internet on. Attach its saved output and the official competition dataset to the final notebook. Set GPU T4 x2 and Internet Off; the code uses one GPU to retain batch size 32. Save & Run All, then submit its submission.csv through the competition.
The report discloses local versus Kaggle runtime differences and the initial failed compatibility attempt. No hidden labels, external prediction APIs, or leaderboard-guided model selection were used.
Codex assisted with implementation, execution and analysis.
