# Neural Network Training Framework for PCAP Data Analysis

A comprehensive, modular framework for training various neural network models on PCAP (packet capture) data with advanced logging, visualization, and artifact management.

## Features

### 🚀 **Modular Design**
- Clean separation of concerns with dedicated classes for models, training, evaluation, and data management
- Factory pattern for easy model creation and switching
- Configurable hyperparameters through command-line arguments

### 🧠 **Multiple Model Types**
- **RNN**: Vanilla Recurrent Neural Network
- **LSTM**: Long Short-Term Memory
- **BiLSTM**: Bidirectional LSTM
- **GRU**: Gated Recurrent Unit
- **BiGRU**: Bidirectional GRU

### 📊 **Comprehensive Logging & Visualization**
- TensorBoard integration for real-time monitoring
- Automatic plot generation (loss curves, distributions, training summaries)
- Detailed console output with progress bars
- Structured logging to files

### 💾 **Artifact Management**
- Organized directory structure for all outputs
- Model checkpoints (best and final models)
- Training metrics and epoch outputs
- Configuration files for reproducibility
- Automatic timestamping for experiment tracking

### 🔧 **Three Operation Modes**
1. **Training**: Train new models from scratch
2. **Evaluation**: Evaluate trained models on test data
3. **Prediction**: Make predictions on new data

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/subrat/Projects/DifFE
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the datasets module is available:**
   ```bash
   cd code
   python -c "from datasets import PcapDatasetInt_IP_Embedding; print('Dataset import successful')"
   ```

## Usage

### Training Mode

**Basic training with default parameters:**
```bash
python train_modular.py --mode train
```

**Training with custom parameters:**
```bash
python train_modular.py \
    --mode train \
    --model-type lstm \
    --epochs 100 \
    --learning-rate 0.0005 \
    --hidden-size 128 \
    --device cuda
```

**Complete training example:**
```bash
python train_modular.py \
    --mode train \
    --model-type bilstm \
    --pcap-path "../data/benign/new-pcap/weekday.pcap" \
    --csv-path "../data/benign/new-pcap/weekday.csv" \
    --input-size 144 \
    --hidden-size 64 \
    --num-layers 2 \
    --output-size 102 \
    --epochs 60 \
    --batch-size 1 \
    --learning-rate 0.001 \
    --weight-decay 1e-3 \
    --device auto \
    --artifact-dir "../artifacts" \
    --seed 42 \
    --verbose
```

### Evaluation Mode

**Evaluate a trained model:**
```bash
python train_modular.py \
    --mode evaluate \
    --model-path "../artifacts/lstm_20250825_143022/best_model.pt" \
    --pcap-path "../data/test/test.pcap" \
    --csv-path "../data/test/test.csv"
```

### Prediction Mode

**Load model for prediction:**
```bash
python train_modular.py \
    --mode predict \
    --model-path "../artifacts/lstm_20250825_143022/best_model.pt"
```

## Command-Line Arguments

### General Options
- `--mode`: Operation mode (`train`, `evaluate`, `predict`)
- `--device`: Computing device (`cpu`, `cuda`, `auto`)
- `--seed`: Random seed for reproducibility
- `--verbose`: Enable verbose output

### Data Configuration
- `--pcap-path`: Path to PCAP file
- `--csv-path`: Path to CSV features file
- `--artifact-dir`: Directory to save artifacts

### Model Architecture
- `--model-type`: Model type (`rnn`, `lstm`, `bilstm`, `gru`, `bigru`)
- `--input-size`: Input feature dimension
- `--hidden-size`: Hidden layer size
- `--num-layers`: Number of layers
- `--output-size`: Output dimension

### Training Parameters
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate
- `--weight-decay`: Weight decay for regularization

### Evaluation/Prediction
- `--model-path`: Path to trained model
- `--config-path`: Path to model configuration file

## Output Structure

After training, the following directory structure is created:

```
artifacts/
└── {model_type}_{timestamp}/
    ├── models/
    ├── plots/
    │   └── training_plots.png
    ├── logs/
    │   └── training_{timestamp}.log
    ├── epoch_outputs/
    │   ├── epoch_001_outputs.csv
    │   ├── epoch_002_outputs.csv
    │   └── ...
    ├── tensorboard_logs/
    │   └── (TensorBoard log files)
    ├── best_model.pt
    ├── final_model.pt
    ├── training_losses.csv
    └── config.json
```

### File Descriptions

- **`best_model.pt`**: Model checkpoint with the lowest validation loss
- **`final_model.pt`**: Final model state after all epochs
- **`training_losses.csv`**: Loss values for each epoch
- **`config.json`**: Complete training configuration for reproducibility
- **`training_plots.png`**: Comprehensive visualization of training progress
- **`epoch_outputs/`**: Model outputs for each epoch (for analysis)
- **`tensorboard_logs/`**: TensorBoard logs for real-time monitoring

## Monitoring Training

### TensorBoard
Monitor training in real-time:
```bash
tensorboard --logdir artifacts/{model_type}_{timestamp}/tensorboard_logs
```

### Console Output
The framework provides detailed console output including:
- Progress bars for epochs and batches
- Real-time loss updates
- Training statistics
- Artifact save confirmations

## Examples

### 1. Quick Start Training
```bash
# Train an LSTM model with default settings
python train_modular.py --mode train --model-type lstm
```

### 2. High-Performance Training
```bash
# Train with GPU acceleration and larger model
python train_modular.py \
    --mode train \
    --model-type bilstm \
    --hidden-size 256 \
    --num-layers 3 \
    --epochs 100 \
    --device cuda
```

### 3. Experiment Comparison
```bash
# Train multiple models for comparison
for model in rnn lstm gru; do
    python train_modular.py \
        --mode train \
        --model-type $model \
        --epochs 50 \
        --seed 42
done
```

### 4. Model Evaluation
```bash
# Evaluate the best LSTM model
python train_modular.py \
    --mode evaluate \
    --model-path "artifacts/lstm_20250825_143022/best_model.pt" \
    --pcap-path "data/test/test.pcap" \
    --csv-path "data/test/test.csv"
```

### 5. Model Prediction
```bash
# Predict features using trained LSTM Model for your own pcap.
python train_modular_changes.py \
    --mode predict \
    --model-path "/home/kundan/scratch/DifFe/artifacts/lstm_20251104_145912/best_model.pt" \
    --pcap-path "[YOUR-PCAP-PATH]" \
```

## For Training KitNet and Evaluation
1) Put train and test file path in test_new.py
2) Run the following command
```bash
# 
python test_new.py

```
3) Plots get stored to /home/kundan/DifFE/code/artifacts/iitd-iot/plots

## Programmatic Usage

The framework can also be used programmatically:

```python
from train_modular import ModelConfig, ModelFactory, Trainer, Evaluator

# Create configuration
config = ModelConfig()
config.model_type = "lstm"
config.epochs = 50

# Create and train model
model = ModelFactory.create_model("lstm", 144, 64, 2, 102)
trainer = Trainer(model, config, "artifacts/experiment1")
results = trainer.train("data.pcap", "features.csv")

# Evaluate model
evaluator = Evaluator("artifacts/experiment1/best_model.pt")
evaluator.load_model()
eval_results = evaluator.evaluate("test.pcap", "test_features.csv")

# Make predictions
prediction = evaluator.predict_single(packet_data)
```

## Best Practices

### 1. **Data Preparation**
- Ensure PCAP and CSV files are properly aligned
- Verify feature dimensions match model input size
- Check for missing values in CSV data

### 2. **Model Selection**
- Start with simpler models (RNN/GRU) for baseline
- Use bidirectional models for better context capture
- Consider model complexity vs. dataset size

### 3. **Training**
- Monitor training progress with TensorBoard
- Use early stopping if loss plateaus
- Save intermediate checkpoints for long training runs

### 4. **Evaluation**
- Always evaluate on separate test data
- Compare multiple model architectures
- Analyze prediction distributions

### 5. **Reproducibility**
- Set random seeds consistently
- Save all configurations
- Document hyperparameter choices

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Solution: Install requirements with `pip install -r requirements.txt`
   ```

2. **CUDA Out of Memory**
   ```
   Solution: Reduce batch size or use --device cpu
   ```

3. **File Not Found**
   ```
   Solution: Check file paths and ensure data files exist
   ```

4. **Model Loading Errors**
   ```
   Solution: Ensure model architecture matches saved checkpoint
   ```

### Performance Tips

- Use GPU acceleration for large datasets
- Increase batch size if memory allows
- Consider gradient accumulation for large effective batch sizes
- Monitor memory usage during training

## Contributing

To extend the framework:

1. Add new model types in the model classes section
2. Update the ModelFactory to include new models
3. Add any new hyperparameters to ModelConfig
4. Update command-line arguments as needed

## License

This project is part of the DifFE research framework.

## Attack Success Rate
1. Port Scanning
    a. After Image - 0.703
    b. Diffe - 
2. Service Detection
    a. After Image - 0.859
    b. Diffe - 
3. Ack-Dos
    a. After Image - 0.516
    b. Diffe - 

isolation_forest → deep_svdd → dagmm → transformer_ae → lof → ocsvm (slowest last)

