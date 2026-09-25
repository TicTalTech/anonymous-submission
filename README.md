# LPPE: Local Predicted Positional Encoding

LPPE learns a structure-only node encoder from positional and structural targets computed on bounded, root-centered personalized PageRank (PPR) subgraphs. A frozen LPPE encoder supplies features to an SGFormer link predictor; on held-out graphs, inference uses observed edges and neither fits to the target graph nor computes the structural targets again; resulting in a zero-shot link predictor.

![LPPE method diagram](LPPE%20Diagram.png)

## Code map

| Location | Purpose |
| --- | --- |
| `src/zero_shot_gfm/data/anygraph.py` | Discover the public archive and load/canonicalize official split matrices. |
| `src/zero_shot_gfm/preprocessing/` | PPR neighborhoods, local structural targets, negative candidates, caches. |
| `src/zero_shot_gfm/models/` | GPS-style LPPE encoder and SGFormer with a symmetric pair decoder. |
| `src/zero_shot_gfm/training/`, `inference/`, `evaluation/` | Training, frozen embeddings, link metrics. |
| `configs/`, `scripts/` | Resolved experiment settings and stage/experiment entry points. |
| `analysis/`, `figures/` | Included aggregate CSVs and plotting scripts. |

Run commands from the repository root after installation. The historical `experiment_1a`, `experiment_1b`, `experiment_2`, `experiment_4`, and `experiment_6` labels remain in filenames because the runners and caches refer to them.

## Download and place the data

The datasets are from the public [AnyGraph code repository](https://github.com/hkuds/anygraph) and [AnyGraph dataset collection](https://huggingface.co/datasets/hkuds/AnyGraph_datasets/tree/main). They are third-party sources, not submission-author accounts. Place the **`zero-shot datasets.zip`** archive inside `lppe-supplement/anygraph_data/`.
Extract it beneath `anygraph_data/`. The loader discovers split files under the extracted `zero-shot datasets/` directory:

```text
lppe-supplement/
  anygraph_data/
    zero-shot datasets.zip
    zero-shot datasets/
      cora/
        trn_mat.pkl
        val_mat.pkl
        tst_mat.pkl
      ...
```

The actual extracted archive may include category subdirectories; leave those intact. `discover_anygraph_datasets` searches one extra directory level and expects the three `.pkl` split matrices in each dataset directory. The public flattened ZIP does not preserve Link1/Link2 folders, so `src/zero_shot_gfm/data/anygraph.py` contains the explicit 15-source Link1 and 18-target Link2 mapping used in the anygraph paper. Configs set `paths.raw_data: anygraph_data`; use the project root as the working directory, or change that field in a derived config.

## Reproduce the paper experiments

The following is the dependency order for the reported top-50 GPU Monte Carlo PPR protocol. These are full-scale runs, with extensive cache storage and GPU time. The commands name the actual scripts, inputs, and output locations; monitor `results/` and `experiments/` for progress and failures. The selected joint config is `configs/experiment_2/joint_small_gpu_mc_50_full_optimized_w2.yaml`. Its semantic settings and seeds are recorded in the YAML and the resolved run metadata.

**1. Prepare the two PPR cache types and training-only structural targets.** The first cache samples at most 100,000 training roots per dataset for LPPE supervision; the second covers every root on the train and train+validation observed graphs for frozen embedding inference. The PSE cache uses the sampled **train graph** only. Commands below use all discovered datasets.

```bash
python scripts/build_ppr_cache.py --config configs/preprocessing/anygraph_gpu_mc_random100k_50.yaml --train-only
python scripts/compute_pse_targets.py --config configs/preprocessing/anygraph_gpu_mc_random100k_50.yaml --ppr-cache-root cache/ppr_gpu_mc_random100k_50 --cache-root cache/pse_gpu_mc_random100k_50_cycle234
python scripts/build_ppr_cache.py --config configs/preprocessing/anygraph_gpu_mc_allroots_50.yaml --reuse-cache-root cache/ppr_gpu_mc_random100k_50
```

**2. Within-dataset capacity and frozen transfer (Section 4.3/Table 4; Section 4.4/Figures 2 and 4).** Train per-dataset LPPE sources, then train/evaluate their link predictors. The two group sweeps must have separate state and model output directories:

```bash
python scripts/train_lppe_capacities.py --output-dir experiments/within_dataset_lppe
python scripts/run_experiment_1a_link_sweep.py --lppe-sweep-state experiments/within_dataset_lppe/tasks.json --dataset-group Link1 --state-dir experiments/experiment_1a_link1_binary_deferred_cuda0 --models-root models/experiment_1a_link1_binary_deferred_cuda0
python scripts/run_experiment_1a_link_sweep.py --lppe-sweep-state experiments/within_dataset_lppe/tasks.json --dataset-group Link2 --state-dir experiments/experiment_1a_link2_binary_deferred_cuda0 --models-root models/experiment_1a_link2_binary_deferred_cuda0
python scripts/run_experiment_1b_transfer.py --source-state experiments/experiment_1a_link1_binary_deferred_cuda0/tasks.json
```

The transfer runner reuses completed Link1 source checkpoints; it does not train on held-out targets. Its built-in protocol is explicitly an operational binary-metric run with ranking evaluation deferred.

**3. Multisource zero-shot scaling (Section 4.5/Figure 3/Table 6).** Make the frozen target candidates, create the Link1 ordering plan, initialize the 30-task state, and run a task key listed in its `tasks.json`. Repeat the last command for all pending task keys; each task trains LPPE and a link model and scores all 18 held-out Link2 datasets.

```bash
python scripts/prepare_multisource_zero_shot.py --data-root anygraph_data --output-dir experiments/multisource_protocol --seed 42
python scripts/run_experiment_2.py --config configs/experiment_2/joint_small_gpu_mc_50_full_optimized_w2.yaml --output-dir experiments/experiment_2
python scripts/run_experiment_2_joint.py --config configs/experiment_2/joint_small_gpu_mc_50_full_optimized_w2.yaml --orderings experiments/experiment_2/orderings.json --output-dir experiments/experiment_2
python scripts/run_experiment_2_task.py --config configs/experiment_2/joint_small_gpu_mc_50_full_optimized_w2.yaml --orderings experiments/experiment_2/orderings.json --output-dir experiments/experiment_2 --task-key "TASK_KEY_FROM_TASKS_JSON"
```

**4. Other reported comparisons.** Random-feature SGFormer and explicit Local PSE use `scripts/train_multisource_random_features_sgformer.py`, `scripts/score_multisource_random_features_sgformer.py`, `scripts/run_experiment_6.py`, and the configurations in `configs/experiment_2/` and `configs/experiment_6/`. The single-graph supervision control uses `scripts/create_experiment_4_fragment_plan.py`, `scripts/run_experiment_4.py`, and `scripts/run_experiment_4_sweep.py` with `configs/experiment_4/citation_classic_root_scaling.yaml`. Inspect these runners' `--help` and their config values before scheduling these substantial additional sweeps. The paper reports timeouts rather than metrics for global PSE and GPSE; completed outputs for them are not supplied.

## Results and verification limits
The `analysis/` CSVs support regeneration of Figures 2–7.



## Citation

```
@article{xia2024anygraph,
  title={AnyGraph: Graph Foundation Model in the Wild},
  author={Xia, Lianghao and Huang, Chao},
  journal={arXiv preprint arXiv:2408.10700},
  year={2024}
}
```