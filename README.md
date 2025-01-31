# Self-Supervised Graph Contrastive Pretraining for Device-level Integrated Circuits
**We propose DICE: Device-level Integrated Circuits Encoder, a pretrained GNN model aimed to solve diverse tasks for both analog and digital integrated circuits.**

---

## Table of Contents  
1. [Installation & Environment Setup](#installation-and-environment-setup)  
   - [Install ngspice (Optional)](#install-ngspice-optional)  
   - [Create & Activate Conda Environment](#create--activate-conda-environment)  
2. [Pretraining](#pretraining)  
   - [Prepare Pretraining Dataset](#prepare-pretraining-dataset)  
   - [Train (Pretraining)](#train-pretraining)  
   - [Plot t-SNE](#plot-t-sne)  
   - [Test (Pretraining)](#test-pretraining)  
3. [Downstream Tasks](#downstream-tasks)  
   - [Prepare Downstream Dataset](#prepare-downstream-dataset)  
   - [Train (Downstream)](#train-downstream)  
   - [Test (Downstream)](#test-downstream)

---

## Installation and Environment Setup

### Install ngspice (Optional)
**ngspice** is only required if you plan to run downstream tasks involving circuit simulation.

```bash
sudo apt-get update
sudo apt-get install ngspice
```

Or install any version from [ngspice official website](http://ngspice.sourceforge.net/).

### Create & Activate Conda Environment
1. **Create** the environment:
   ```bash
   conda create -n anonymous python=3.11
   ```
2. **Activate** the environment:
   ```bash
   conda activate anonymous
   ```
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pretraining

### Prepare Pretraining Dataset
```bash
python pretrain/dataset/save_dataset_pretrain.py
```
Processes data augmentation and saves dataset for pretraining.

### Train (Pretraining)
```bash
python pretrain/encoder/train.py
```
Trains the DICE with graph contrastive learning.

### Plot t-SNE
```bash
python pretrain/encoder/plot/trained_gnn_embeddings/trained_tsne_graph.py
```
Generates a t-SNE visualization of the graph embeddings process by DICE.

### Test (Pretraining)
```bash
python pretrain/encoder/test.py
```
Calculates the cosine similarities between features from the pretrained model.

---

## Downstream Tasks

### Prepare Downstream Dataset
```bash
python downstream_tasks/save_dataset_downstream.py --task_name="circuit_similarity_prediction"
```
Adjust `--task_name` for various downstream tasks.

### Train (Downstream)
```bash
python downstream_tasks/downstream_train.py \
    --task_name="circuit_similarity_prediction" \
    --dice_depth=2 \
    --p_gnn_depth=0 \
    --s_gnn_depth=2
```
- `dice_depth`: Depth of the DICE encoder.  
- `p_gnn_depth` / `s_gnn_depth`: Depths for additional GNNs for the encoder network.

### Test (Downstream)
```bash
python downstream_tasks/downstream_test.py \
    --task_name="circuit_similarity_prediction" \
    --dice_depth=2 \
    --p_gnn_depth=0 \
    --s_gnn_depth=2
```
Evaluates the model on the specified downstream task.

---

All other hyperparameters are in params.json