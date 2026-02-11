Hybrid SHAP-Attention XAI for DistilBERT Spam Detection

A novel explainable AI (XAI) framework combining SHAP values with attention mechanisms for interpretable spam email classification using DistilBERT.

 Overview

This project implements a state-of-the-art spam detection system with multi-level interpretability through a novel **Hybrid SHAP-Attention Framework**. The approach addresses the "Attention is not Explanation" debate by fusing gradient-based attributions (SHAP) with transformer attention weights.

Features

- **Advanced Spam Classification**: Fine-tuned DistilBERT model achieving high accuracy on email spam detection
- **Novel XAI Method**: Hybrid SHAP-Attention framework providing hierarchical explanations
- **Comprehensive Baseline Comparisons**: LIME, Integrated Gradients, and Attention visualizations
- **Multi-layer Analysis**: Token-level importance across all transformer layers
- **Production-Ready Pipeline**: Complete end-to-end workflow from data loading to visualization

 Architecture

### Model Components
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary classification (Spam/Ham)
- **Max Sequence Length**: 128 tokens
- **Output**: Class predictions with attention weights and hidden states

### XAI Methods Implemented
1. **LIME** (Local Interpretable Model-agnostic Explanations)
2. **Integrated Gradients** (Captum-based attributions)
3. **Attention Visualization** (Multi-head attention heatmaps)
4. **Hybrid SHAP-Attention** (Novel contribution)

## Installation

### Requirements
```bash
pip install transformers datasets accelerate
pip install torch torchvision torchaudio
pip install shap lime captum
pip install matplotlib seaborn scikit-learn scipy
pip install bertviz

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Methodology
Phase I: Environment Setup & Preprocessing
Dependency installation and configuration

Dataset loading with automatic column detection

Stratified train/val/test split (72%/8%/20%)

Phase II: Model Training & Evaluation
DistilBERT fine-tuning with AdamW optimizer

Linear warmup scheduler

Comprehensive metrics: accuracy, precision, recall, F1, AUC-ROC

Phase III: XAI Sample Selection
Diverse sample selection strategy

Balanced mix of correct/incorrect predictions

Both spam and ham examples

Phase IV: Baseline XAI Methods
LIME: Local feature importance with 500 perturbed samples

Integrated Gradients: Gradient-based attribution with 50 steps

Attention: Multi-head attention visualization from all layers

Phase V: Hybrid SHAP-Attention Framework
Innovation: Combines token-level SHAP importance with contextual attention influence
