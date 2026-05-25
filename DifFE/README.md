# RoFeX

## Data and Checkpoints
Create the appropiate data directory and checkpoint directory as done in ../data and ../checkpoints.

## Training RoFeX models:

To train a RoFex model on

model-type: lstm
dataset: kitsune

```
python train_modular_changes.py \
       --mode train \
       --dataset kitsune \
       --model-type lstm \
       --pcap-path ../data/kitsune/pcaps/mixed_450k.pcap \
       --csv-path ../data/kitsune/features/afterimage/mixed_450k.csv \
       --hidden-size 128 --num-layers 3 --output-size 100 \
       --batch-size 512--epochs 100 --seq-len 5 --learning-rate 0.001 \
       --weight-decay 1e-3 \
       --device cuda
```

## Extracting RoFeX features:

### Batch mode (recommended):
```
python train_modular_changes.py --mode predict --dataset kitsune   --model-path ../checkpoints/kitsune/lstm_20260505_163134/best_model.pt  --pcap-folder  ../data/kitsune/pcaps/malicious/
```

### Single file mode:
```
python train_modular_changes.py --mode predict --dataset kitsune --model-path ../checkpoints/kitsune/lstm_20260505_163134/best_model.pt --pcap-path ../data/kitsune/pcaps/malicious/Mirai.pcap
```
writes → data/kitsune/features/rofex/malicious/Mirai.csv



## Training an NIDS:

### Train
```
python benchmark.py --dataset x-iot --extractor afterimage --model kitnet --train
```

### Evaluate
```
python benchmark.py --dataset x-iot --extractor afterimage --model kitnet --splits malicious adversarial --max-test-samples 300000 --concurrent
```

###  Compare  Results
```
python compare_results.py --dataset x-iot --model kitnet --split malicious --metrics acc auc
python compare_results.py --dataset x-iot --model kitnet --split adversarial --metrics acc auc
```

## Training Different NIDS on Same Dataset:

### Train
```
python benchmark.py --dataset kitsune --extractor afterimage --model kitnet   --benign-file SSDP_Flood --model-name ssdp_flood --train
```

### Evaluate
```
python benchmark.py --dataset kitsune --extractor afterimage --model kitnet   --model-name ssdp_flood --splits malicious --concurrent
```

### Compare Results
```
python compare_results.py --dataset kitsune --model kitnet_mirai --split adversarial --metrics acc auc
```

### Compare Results across the Dataset:
 
####  Usage: any combo of metrics / models / extractors
```
python eval_table.py --dataset x-iot --split malicious  --metric auc eer
python eval_table.py --dataset x-iot --split adversarial --metric "tpr@fpr=0.001" "fnr@fpr=0.001"
python eval_table.py --dataset x-iot --split adversarial --metric "tpr@fpr=0.01" --models tae kitnet
python eval_table.py --dataset kitsune --split malicious --metric auc      # works once kitsune .npz files exist
```

#### Supported metrics

auc · eer · accuracy · recall · tpr@fpr=X · fnr@fpr=X (X may be 0)

#### Notes
- Δ column = vanilla_avg − defence_avg (raw arithmetic). Sign meaning depends on metric direction: for AUC/TPR/Accuracy a negative Δ means defence wins;
for FNR/EER a positive Δ means defence wins.
- Auto-discovers attacks per dataset (x-iot and kitsune have preset orderings; otherwise alphabetical from the .npz filenames). Override with --attacks
<list>.
- Default model order is TAE → KitNET → AE → IF; override with --models tae kitnet ae if lof etc.
- Cells with only-one-class labels (e.g. Mirai's all-positive test slice) print -- instead of crashing.

-------------------------------------------------------------------------------------------------------
### Both datasets, both extractors work the same way
python benchmark.py --dataset x-iot --extractor rofex --model autoencoder --train
