# DICE: Device-level Integrated Circuits Encoder with Graph Contrastive Pretraining
---
## Abstract
Pretraining models with unsupervised graph representation learning has led to significant advancements in domains such as social network analysis, molecular design, and electronic design automation (EDA). However, prior work in EDA has mainly focused on pretraining models for digital circuits, overlooking analog and mixed-signal circuits. To bridge this gap, we introduce DICE: Device-level Integrated Circuits Encoder—the first graph neural network (GNN) pretrained via self-supervised learning specifically tailored for graph-level prediction tasks in both analog and digital circuits. DICE adopts a simulation-free pretraining approach based on graph contrastive learning, leveraging two novel graph augmentation techniques. Experimental results demonstrate substantial performance improvements across three downstream tasks, highlighting the effectiveness of DICE for both analog and digital circuits.

---

## Table of Contents  
1. [Installation and Environment Setup](#installation-and-environment-setup)  
   - [Install ngspice (Optional)](#install-ngspice-optional)  
   - [Create and Activate Conda Environment](#create-and-activate-conda-environment)  
2. [Pretraining DICE](#pretraining-dice)  
   - [Prepare Pretraining Dataset](#prepare-pretraining-dataset)  
   - [Run Pretraining](#train-pretraining)  
   - [Visualize Embeddings (t-SNE)](#plot-t-sne)  
   - [Evaluate Pretrained Model](#test-pretraining)  
3. [Downstream Tasks](#downstream-tasks)  
   - [Prepare Downstream Dataset](#prepare-downstream-dataset)  
   - [Train on Downstream Task](#train-downstream)  
   - [Test on Downstream Task](#test-downstream)  
4. [Baselines](#baselines)  
   - [Prepare Pretraining Dataset (DeepGen_p)](#prepare-deepgen_p-pretraining-dataset)  
   - [Pretrain with DC voltage prediction task (DeepGen_p)](#pretrain-deepgen_p)  
   - [Test DC voltage prediction task (DeepGen_p)](#test-pretrained-deepgen_p)  
   - [Prepare Downstream Dataset (Baselines)](#prepare-downstream-task-dataset-for-all-baselines)  
   - [Train on Downstream Task (Baselines)](#train-downstream-baseline)  
   - [Test on Downstream Task (Baselines)](#test-downstream-baseline)  
5. [Conclusion](#conclusion)

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
   conda env create -f environment.yml
   ```
2. **Activate** the environment:
   ```bash
   conda activate DICE
   ```

---

## Pretraining

### Prepare Pretraining Dataset
with positive + negative data augmentation (DICE training):
```bash
python pretrain/dataset/save_dataset_pretrain.py
```
with only positive data augmentation (SimSiam, NT-Xent training) :
```bash
python pretrain/dataset/save_dataset_pretrain_pda.py
```
Processes data augmentation and saves dataset for pretraining.

### Train (Pretraining)
with SimSiam loss:
```bash
python pretrain/encoder/train_simsiam.py
```
with NT-Xent loss:
```bash
python pretrain/encoder/train_pda.py
```
with DICE loss:
```bash
python pretrain/encoder/train.py
```
Pretrain the GNNs with graph contrastive learning.

### Plot t-SNE
with initial embeddings:
```bash
python pretrain/encoder/plot/init_embeddings/init_embedding_tsne_graph.py
```
with untrained GNN:
```bash
python pretrain/encoder/plot/untrained_gnn_embeddings/untrained_tsne_graph.py
```
Pretrained with SimSiam loss:
```bash
python pretrain/encoder/plot/trained_gnn_embeddings/trained_tsne_graph_simsiam.py
```
Pretrained with NT-Xent loss:
```bash
python pretrain/encoder/plot/trained_gnn_embeddings/trained_tsne_graph_pda.py
```
Pretrained with DICE loss:
```bash
python pretrain/encoder/plot/trained_gnn_embeddings/trained_tsne_graph.py
```
Generates t-SNE visualizations of the graph embeddings.

### Test (Pretraining)
Pretrained with SimSiam or NT-Xent loss:
```bash
python pretrain/encoder/test_pda_simsiam.py
```
Pretrained with DICE loss:
```bash
python pretrain/encoder/test.py
```
Calculates the cosine similarities between graph-level features from the pretrained models.

---

## Downstream Tasks

### Prepare Downstream Dataset
```bash
python downstream_tasks/save_dataset_downstream.py --task_name="circuit_similarity_prediction"
```
- `--task_name`: downstream tasks (circuit_similarity_prediction, delay_prediction, opamp_metric_prediction).

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

## Baselines

### Prepare DeepGen_p Pretraining Dataset
```bash
python baselines/pretrain_DCvoltages/save_dc_voltage_dataset.py
```

### Pretrain DeepGen_p
```bash
python baselines/pretrain_DCvoltages/DCvoltage_pretrain.py
```
Pretrain DeepGen_p with DC voltage prediction task (node-level prediction task).

### Test Pretrained DeepGen_p
```bash
python baselines/pretrain_DCvoltages/DCvoltage_test.py
```
Test DeepGen_p (node-level prediction task).


### Prepare Downstream Task Dataset for all Baselines
```bash
python baselines/save_dataset_baseline_downstream.py --baseline_name="ParaGraph" --task_name="delay_prediction"
```
- `--baseline_name`: baseline models (ParaGraph, DeepGen).
- `--task_name`: downstream tasks (circuit_similarity_prediction, delay_prediction, opamp_metric_prediction).


### Train Downstream (Baseline)
```bash
python baselines/baseline_downstream_train.py \
    --baseline_name="DeepGen_p" \
    --task_name="opamp_metric_prediction"
```
- `--baseline_name`: baseline models (ParaGraph, DeepGen_u, DeepGen_p).
- `--task_name`: downstream tasks (circuit_similarity_prediction, delay_prediction, opamp_metric_prediction).


### Test Downstream (Baseline)
```bash
python baselines/baseline_downstream_test.py \
    --baseline_name="DeepGen_u" \
    --task_name="opamp_metric_prediction"\
    --trained_model_seeds 0 7
```
- `--baseline_name`: baseline models (ParaGraph, DeepGen_u, DeepGen_p).
- `--task_name`: downstream tasks (circuit_similarity_prediction, delay_prediction, opamp_metric_prediction).
- `--trained_model_seeds`: seeds used for training models.

---

## Conclusion

In this work, we propose DICE, a pretrained GNN model designed for general graph-level prediction tasks in device-level integrated circuits.
Our primary contribution is to highlight the importance of pretraining graph models specifically on device-level circuits, and we introduce the first graph contrastive learning framework to address this challenge.
We argue that device-level representations offer a more general and flexible abstraction for integrated circuits than logic gate-level representations, as they support both analog and digital designs.
Another contribution is the introduction of two novel data augmentation techniques that address the scarcity of circuit data.
Additionally, we propose a new device-level circuit benchmark, where all three tasks require handling multiple circuit topologies.
Experimental results including the ablation studies show that incorporating DICE leads to significant improvements across all three downstream tasks.
We view this work as a step toward building general-purpose models that unify analog and digital circuit design.