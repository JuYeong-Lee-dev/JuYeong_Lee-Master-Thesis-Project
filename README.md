# Predicting Crowdfunding Success: Multimodal Deep Learning

Master's thesis project (Tilburg University, MSc Data Science and Society) predicting the success of Kickstarter crowdfunding campaigns using structured, textual, and multimodal deep learning approaches.

## Summary

- Benchmarked **Logistic Regression, MLP, and a Multimodal Transformer** on 245K+ Kickstarter campaigns, combining structured features with **RoBERTa embeddings** (pre-trained vs. fine-tuned).
- Evaluated pre-trained vs. fine-tuned RoBERTa embeddings for campaign titles and blurbs.
- Applied nested cross-validation, Optuna hyperparameter tuning, and category-specific error analysis.
- Best model (**MLP with structured + text fusion**) reached **F1 0.869 / AUC-ROC 0.907**, outperforming the Transformer and showing that added architectural complexity did not improve predictive performance in this setting.

## Background

Crowdfunding has grown from a hobbyist, creator-driven space into a market with real economic weight, accelerated further by AI lowering the barrier to launching a campaign. A successful campaign can generate substantial economic value, while a failed one can derail a creator's entire venture. This growing stake in outcomes is what motivated the topic: predicting crowdfunding success is increasingly a meaningful economic modeling problem, not just an academic exercise.

## Approach

Models were built and compared across three settings: structured features only, text only, and a fusion of both.

- For structured data, Logistic Regression and MLP were evaluated head to head.
- For text, campaign titles and blurbs were embedded using pre-trained versus fine-tuned RoBERTa to test whether task-specific fine-tuning actually mattered.
- The fusion stage combined both modalities and added a Multimodal Transformer to the comparison, with nested cross-validation and Optuna-based hyperparameter tuning throughout, followed by category-specific error analysis to check for systematic bias across campaign types.
- The best result came from MLP with structured data and pre-trained RoBERTa embeddings, not the Transformer. Structured features consistently carried more predictive power than text alone, and adding architectural complexity past a certain point produced no meaningful gain.

## Notebooks

| Notebook | Description |
|---|---|
| `MLP_structured_final.ipynb` | MLP on structured features only |
| `MLP_structured_pretrain_final.ipynb` | MLP on structured + pre-trained RoBERTa embeddings, with nested CV |
| `MLP_structured_finetuned_final.ipynb` | MLP on structured + fine-tuned RoBERTa embeddings, with nested CV |
| `MLP_pretrain_final.ipynb` | MLP on pre-trained RoBERTa text embeddings only |
| `MLP_finetuned_final.ipynb` | MLP on fine-tuned RoBERTa text embeddings only |
| `Earlyfusion_pretrain.ipynb` | Multimodal Transformer (early fusion) with pre-trained RoBERTa embeddings |
| `Earlyfusion_finetune.ipynb` | Multimodal Transformer (early fusion) with fine-tuned RoBERTa embeddings |

## Technology Stack

Python, PyTorch, HuggingFace Transformers, scikit-learn, Optuna
