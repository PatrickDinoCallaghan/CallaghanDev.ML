
# CallaghanDev.ML

A machine learning library coded scratch in C#. Very limited use of any outside framework, no use of ML frameworks.

## What's Inside

**Transformer** Decoder-only transformer with multi-head attention, layer normalization, residual connections, and BPE tokenization. Supports four transformer d ata types Such as: text generation, time series regression, symbolic sequences, and time series classification.

**Neural Network** A traditional feedforward neural network with configurable depth/width, six cost functions, five activation functions, L2 regularization, gradient clipping, GPU batch training, auto-tuning, and a polynomial approximation function to approximate the neural network using Chebyshev approximation.

**Acceleration** All matrix operations go through `IAccelerationManager` with implementations for single-threaded CPU, multi-threaded CPU, and GPU/CUDA. GPU acceleration is powered by [ILGPU](https://github.com/m4rs-mt/ILGPU), a high-performance .NET GPU compiler that supports both CUDA and OpenCL backends. No native CUDA toolkit installation required.


## Transformer

### Quick Start

```csharp
// Tokenize
var tokenizer = new BPETokenizer();
tokenizer.Train(corpus, vocabSize: 5000, minFrequency: 2);
int[][] sequences = corpus.Select(t => tokenizer.Encode(t, addSpecialTokens: true)).ToArray();

// Configure
var config = new TransformerConfig {
    DataType = TransformerDataType.Text,
    VocabSize = tokenizer.VocabSize,
    EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 128
};

// Train
var model = new LanguageModel(config);
var trainer = new TransformerTrainer(model, new TrainingConfig { LearningRate = 0.001f, Epochs = 50 });
trainer.Train(sequences);

// Generate
int[] generated = model.Generate(tokenizer.Encode("The cat"), maxNewTokens: 20, temperature: 0.8f);
Console.WriteLine(tokenizer.Decode(generated, skipSpecialTokens: true));
```

### Examples

#### Text Generation

```csharp
string[] corpus = {
    "The cat sat on the mat.",
    "The dog chased the cat.",
    "A bird flew over the house.",
    "The cat and the dog played together.",
    "Birds sing in the morning."
};

var tokenizer = new BPETokenizer();
tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);

int[][] sequences = corpus
    .Select(text => tokenizer.Encode(text, addSpecialTokens: true))
    .Where(seq => seq.Length >= 2)
    .ToArray();

var config = new TransformerConfig
{
    DataType = TransformerDataType.Text,
    VocabSize = tokenizer.VocabSize,
    MaxSequenceLength = 64,
    EmbeddingDim = 64,
    NumHeads = 4,
    NumLayers = 2,
    FeedForwardDim = 128,
    UseDecoderOnly = true,
    AccelerationType = AccelerationType.CPU
};

var model = new LanguageModel(config);

var trainConfig = new TrainingConfig
{
    LearningRate = 0.001f,
    BatchSize = 4,
    Epochs = 50,
    UseGradientClipping = true,
    GradientClipThreshold = 1.0f,
    Verbose = true
};

var trainer = new TransformerTrainer(model, trainConfig);
trainer.Train(sequences);

// Generate
int[] prompt = tokenizer.Encode("The cat", addSpecialTokens: false);
int[] generated = model.Generate(prompt, maxNewTokens: 20, temperature: 0.8f);
string text = tokenizer.Decode(generated, skipSpecialTokens: true);
```

#### Text with Validation Split

```csharp
int splitIdx = (int)(sequences.Length * 0.8);
var trainSeqs = sequences.Take(splitIdx).ToArray();
var valSeqs = sequences.Skip(splitIdx).ToArray();

trainer.Train(trainSeqs, valSeqs);
```

#### Symbolic Sequence - DNA
(NB: I have no background in bio, so I apologise to any biologists in advance for misuse of terminology etc. I used this example as DNA is one of the most well-known examples of symbolic sequence prediction. )
```csharp
// Manual vocabulary: 0=PAD, 1=START, 2=END, 3=UNK, 4=A, 5=T, 6=G, 7=C
var baseToId = new Dictionary<char, int> { {'A',4}, {'T',5}, {'G',6}, {'C',7} };

int[] EncodeDNA(string dna) {
    var tokens = new List<int> { 1 };
    foreach (char c in dna)
    {

        if (baseToId.TryGetValue(c, out int id)) 
        {
            tokens.Add(id);
        }
    }

    tokens.Add(2);

    return tokens.ToArray();
}

string[] dnaData = { "ATGCGATCGATCG", "ATGCCCGATTTAG", "ATGAAAGGGCCCT" };
int[][] sequences = dnaData.Select(d => EncodeDNA(d)).ToArray();

var config = new TransformerConfig
{
    DataType = TransformerDataType.SymbolicSequence,
    VocabSize = 8,
    MaxSequenceLength = 32,
    EmbeddingDim = 32,
    NumHeads = 4,
    NumLayers = 2,
    FeedForwardDim = 64
};

var model = new LanguageModel(config);
var trainer = new TransformerTrainer(model, new TrainingConfig {
    LearningRate = 0.001f, BatchSize = 4, Epochs = 100
});
trainer.Train(sequences);

// Generate from start codon
//A codon is a group of three nucleotides(basic building blocks of DNA and RNA) in DNA/RNA that encodes an amino acid or a control signal. The most famous is the start codon(I didnt know this):
int[] prompt = { 1, 4, 5, 6 }; // START, A, T, G
int[] generated = model.Generate(prompt, maxNewTokens: 10, temperature: 0.7f);
```

#### Symbolic Sequence - MIDI Notes

```csharp
// Token IDs map to MIDI note values or special events
// 0=PAD, 1=START, 2=END, 3+ = note values
var config = new TransformerConfig
{
    DataType = TransformerDataType.SymbolicSequence,
    VocabSize = 131,  // 128 MIDI notes + PAD/START/END
    MaxSequenceLength = 256,
    EmbeddingDim = 64,
    NumHeads = 4,
    NumLayers = 3,
    FeedForwardDim = 128
};
```

#### Time Series Regression - Stock Price Prediction

```csharp
// 5 features per timestep (open, high, low, close, volume)
// Predict 1 output (next close)
int numSamples = 100, seqLen = 20;

var inputs = new float[numSamples][,];   // each: [seqLen, 5]
var targets = new float[numSamples][,];  // each: [seqLen, 1]
// ... populate with your data, normalize to ~[0,1] ...

var config = new TransformerConfig
{
    DataType = TransformerDataType.TimeSeriesRegression,
    InputFeatureDim = 5,
    OutputDim = 1,
    MaxSequenceLength = seqLen,
    EmbeddingDim = 64,
    NumHeads = 4,
    NumLayers = 2,
    FeedForwardDim = 128
};

var model = new LanguageModel(config);
var trainer = new TransformerTrainer(model, new TrainingConfig {
    LearningRate = 0.0005f, BatchSize = 8, Epochs = 30
});

// Train - the trainer slices input[0..N-2] -> target[1..N-1] internally
trainer.TrainContinuous(inputs, regressionTargets: targets);

// Validate
float valLoss = trainer.ValidateContinuous(valInputs, regressionTargets: valTargets);

// Predict next value from a sequence
float[] prediction = model.PredictNext(testInput); // returns float[OutputDim]
```

#### Time Series Regression - Multi-Output (Predict OHLC)

```csharp
var config = new TransformerConfig
{
    DataType = TransformerDataType.TimeSeriesRegression,
    InputFeatureDim = 5,   // open, high, low, close, volume
    OutputDim = 4,          // predict next open, high, low, close
    EmbeddingDim = 128,
    NumHeads = 8,
    NumLayers = 4,
    FeedForwardDim = 256
};

// targets[i] shape: [seqLen, 4]
trainer.TrainContinuous(inputs, regressionTargets: targets);
float[] nextOHLC = model.PredictNext(testInput); // float[4]
```

#### Time Series Regression - Sensor Forecasting

```csharp
// 3 sensor channels, predict next reading for all 3
var config = new TransformerConfig
{
    DataType = TransformerDataType.TimeSeriesRegression,
    InputFeatureDim = 3,
    OutputDim = 3,
    EmbeddingDim = 32,
    NumHeads = 2,
    NumLayers = 2,
    FeedForwardDim = 64
};
```

#### Time Series Classification - Buy/Hold/Sell

```csharp
// 5 features per timestep, 3 classes: 0=Sell, 1=Hold, 2=Buy
var inputs = new float[numSamples][,];    // each: [seqLen, 5]
var classTargets = new int[numSamples][]; // each: int[seqLen] with values 0, 1, or 2

// ... populate data ...

var config = new TransformerConfig
{
    DataType = TransformerDataType.TimeSeriesClassification,
    InputFeatureDim = 5,
    OutputDim = 3,
    MaxSequenceLength = seqLen,
    EmbeddingDim = 64,
    NumHeads = 4,
    NumLayers = 2,
    FeedForwardDim = 128
};

var model = new LanguageModel(config);
var trainer = new TransformerTrainer(model, new TrainingConfig {
    LearningRate = 0.0005f, BatchSize = 8, Epochs = 50
});

trainer.TrainContinuous(inputs, classTargets: classTargets);

// Validate
float valLoss = trainer.ValidateContinuous(valInputs, classTargets: valClassTargets);

// Inference: get logits then softmax
var output = model.Forward(testInput); // [seqLen, 3]
int lastT = output.GetLength(0) - 1;
float[] logits = new float[3];
for (int j = 0; j < 3; j++) logits[j] = output[lastT, j];
// Apply softmax to get probabilities
```

#### Time Series Classification - Anomaly Detection

```csharp
// Binary classification: 0=Normal, 1=Anomaly
var config = new TransformerConfig
{
    DataType = TransformerDataType.TimeSeriesClassification,
    InputFeatureDim = 10,  // 10 sensor channels
    OutputDim = 2,          // normal vs anomaly
    EmbeddingDim = 32,
    NumHeads = 4,
    NumLayers = 2,
    FeedForwardDim = 64
};
```

#### Time Series Classification - Activity Recognition

```csharp
// IMU data (accel x/y/z, gyro x/y/z), classify activity
var config = new TransformerConfig
{
    DataType = TransformerDataType.TimeSeriesClassification,
    InputFeatureDim = 6,   // 3-axis accel + 3-axis gyro
    OutputDim = 5,          // walk, run, sit, stand, climb
    EmbeddingDim = 64,
    NumHeads = 4,
    NumLayers = 3,
    FeedForwardDim = 128
};
```

#### Multi-Threaded CPU

```csharp
var config = new TransformerConfig
{
    AccelerationType = AccelerationType.MultiThreadCPU,
    // ... rest of config
};
```

#### GPU / CUDA

```csharp
var config = new TransformerConfig
{
    AccelerationType = AccelerationType.CUDA,
    AccelerationDeviceId = 0,
    // ... rest of config
};
```

#### Learning Rate Decay

```csharp
var trainConfig = new TrainingConfig
{
    LearningRate = 0.01f,
    Epochs = 30,
    UseLearningRateDecay = true,
    LearningRateDecay = 0.95f  // LR *= 0.95 each epoch
};
```

#### Different FFN Activations

```csharp
var config = new TransformerConfig
{
    FFNActivationType = ActivationType.Relu,       // default
    // or: ActivationType.Leakyrelu
    // or: ActivationType.Tanh
};
```

#### L2 Regularization

```csharp
var config = new TransformerConfig
{
    L2RegulationLamda = 0.01f,
    // ... rest of config
};
```

#### Save / Load

```csharp
// Save feed-forward network weights
model.SaveFeedForwardNetworks("./my_model");

// Load them back
model.LoadFeedForwardNetworks("./my_model");

// Save / load tokenizer
tokenizer.Save("./my_tokenizer");
var loaded = BPETokenizer.Load("./my_tokenizer");
```

---

## Neural Network

### Quick Start

```csharp
var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.mse,
    ActivationDistribution = ActivationDistribution.Normal,
    LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Sigmoid
    }
};

var nn = new NeuralNetwork(parameters);
nn.Train(inputs, expected, learningRate: 0.5f, epochs: 1000);
float[] result = nn.Predict(new float[] { 1, 0 });
```

### Examples

#### XOR Problem

```csharp
float[][] inputs = {
    new float[] { 0, 0 },
    new float[] { 0, 1 },
    new float[] { 1, 0 },
    new float[] { 1, 1 }
};
float[][] expected = {
    new float[] { 0 },
    new float[] { 1 },
    new float[] { 1 },
    new float[] { 0 }
};

var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.mse,
    ActivationDistribution = ActivationDistribution.Normal,
    LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Sigmoid
    }
};

var nn = new NeuralNetwork(parameters);
nn.Train(inputs, expected, learningRate: 0.5f, epochs: 4000);

for (int i = 0; i < inputs.Length; i++)
{
    float pred = nn.Predict(inputs[i])[0];
    Console.WriteLine($"[{inputs[i][0]}, {inputs[i][1]}] -> {pred:F3}");
}
// [0, 0] -> 0.012
// [0, 1] -> 0.987
// [1, 0] -> 0.985
// [1, 1] -> 0.015
```

#### AND Gate

```csharp
float[][] expected = {
    new float[] { 0 },  // 0 AND 0
    new float[] { 0 },  // 0 AND 1
    new float[] { 0 },  // 1 AND 0
    new float[] { 1 }   // 1 AND 1
};

var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.mse,
    ActivationDistribution = ActivationDistribution.Uniform,
    LayerWidths = new List<int> { 2, 4, 4, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.None, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.None
    }
};

var nn = new NeuralNetwork(parameters);
nn.Train(inputs, expected, learningRate: 0.5f, epochs: 1000);
```

#### OR Gate

```csharp
float[][] expected = {
    new float[] { 0 },  // 0 OR 0
    new float[] { 1 },  // 0 OR 1
    new float[] { 1 },  // 1 OR 0
    new float[] { 1 }   // 1 OR 1
};

var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.mse,
    ActivationDistribution = ActivationDistribution.Uniform,
    LayerWidths = new List<int> { 2, 4, 4, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.Tanh, ActivationType.Tanh,
        ActivationType.Tanh, ActivationType.Tanh
    }
};

var nn = new NeuralNetwork(parameters);
nn.Train(inputs, expected, learningRate: 0.01f, epochs: 1000);
```

#### Multi-Output Regression

```csharp
// 3 inputs -> 2 outputs
var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.mse,
    LayerWidths = new List<int> { 3, 16, 16, 2 },
    LayerActivations = new List<ActivationType> {
        ActivationType.None, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.None
    }
};

var nn = new NeuralNetwork(parameters);
nn.Train(inputs, expected, learningRate: 0.01f, epochs: 2000);
float[] prediction = nn.Predict(new float[] { 0.5f, 0.3f, 0.8f }); // returns float[2]
```

#### Binary Classification with Cross-Entropy

```csharp
var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.binaryCrossEntropy,
    LayerWidths = new List<int> { 4, 8, 8, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.None, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.Sigmoid
    }
};

var nn = new NeuralNetwork(parameters);
nn.Train(inputs, expected, learningRate: 0.1f, epochs: 500);
```

#### Multi-Class Classification with Categorical Cross-Entropy

```csharp
// 3 classes, one-hot encoded targets: [1,0,0], [0,1,0], [0,0,1]
var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.categoricalCrossEntropy,
    LayerWidths = new List<int> { 4, 16, 16, 3 },
    LayerActivations = new List<ActivationType> {
        ActivationType.None, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.Sigmoid
    }
};
```

#### Huber Loss (Robust Regression)

```csharp
var parameters = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.huberLoss,
    HuberLossDelta = 1.0f,
    LayerWidths = new List<int> { 3, 16, 16, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.None, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.None
    }
};
```

#### Zero-Weighted MSE (Imbalanced Targets)

```csharp
// Down-weights the loss contribution when the true target is zero
var parameters = new Parameters
{
    CostFunction = CostFunctionType.ZeroWeightedMSE,
    // ... rest of config
};
```

#### L2 Regularization and Gradient Clipping

```csharp
var parameters = new Parameters
{
    L2RegulationLamda = 0.01f,
    GradientClippingThreshold = 1.0f,
    // ... rest of config
};
```

#### GPU Training (Single Sample)

```csharp
var parameters = new Parameters
{
    AccelerationType = AccelerationType.CUDA,
    CostFunction = CostFunctionType.mse,
    ActivationDistribution = ActivationDistribution.Normal,
    LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Sigmoid
    }
};

var nn = new NeuralNetwork(parameters);
nn.Train(inputs, expected, learningRate: 0.1f, epochs: 1000);
```

#### GPU Batch Training

```csharp
// Requires AccelerationType.GPU or AccelerationType.CUDA
var parameters = new Parameters
{
    AccelerationType = AccelerationType.GPU,
    CostFunction = CostFunctionType.mse,
    ActivationDistribution = ActivationDistribution.Normal,
    LayerWidths = new List<int> { 2, 4, 8, 4, 1 },
    LayerActivations = new List<ActivationType> {
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Leakyrelu, ActivationType.Leakyrelu,
        ActivationType.Sigmoid
    }
};

var nn = new NeuralNetwork(parameters);
nn.TrainBatch(inputs, expected, batchSize: 64, learningRate: 0.5f, epochs: 4000);
```

#### Validation Loss

```csharp
float valLoss = nn.EvaluateValidationLoss(valInputs, valExpected);
Console.WriteLine($"Validation MSE: {valLoss:F6}");
```

#### Save / Load

```csharp
// Save
nn.Save("my_model.json");

// Load (specify acceleration type at load time)
var loaded = NeuralNetwork.Load("my_model.json", AccelerationType.CPU);

// Load onto GPU
var loadedGPU = NeuralNetwork.Load("my_model.json", AccelerationType.CUDA, DeviceId: 0);
```

#### Save / Load Parameters Separately

```csharp
parameters.SaveToFile("params.json");
var loadedParams = Parameters.LoadFromFile("params.json");
```

#### Auto-Tuning Network Architecture

```csharp
// Automatically searches for optimal layer widths and depth
var baseParams = new Parameters
{
    AccelerationType = AccelerationType.CPU,
    CostFunction = CostFunctionType.mse,
    ActivationDistribution = ActivationDistribution.Normal,
    LayerWidths = new List<int> { 2, 1 },  // minimal starting point
    LayerActivations = new List<ActivationType> {
        ActivationType.Leakyrelu, ActivationType.Sigmoid
    }
};

ILogger logger = new Logger();
var autoTuner = new NeuralAutoTuner(logger);
autoTuner.SetMaxNeuronWidth(8);
autoTuner.SetMinNeuronWidth(4);

Parameters bestParams = autoTuner.TrainWithAutoTuning(
    inputs, expected,
    learningRate: 0.25f,
    baseParams,
    maxAttempts: 250,
    targetLossThreshold: 0.2f,
    maxChunkTrainingAttempts: 25
);

// Train with the discovered architecture
var nn = new NeuralNetwork(bestParams);
nn.Train(inputs, expected, bestParams.OptimalLearningRate ?? 0.01f, epochs: 4000);
```

#### Polynomial Approximation Export

```csharp
// Extract a symbolic polynomial that approximates the trained network
var polynomialExpressions = PolynomialApproximation.GenerateOutputExpressions(nn.GetData());

// Evaluate the polynomial with specific inputs
var values = new Dictionary<string, FloatingPoint>
{
    ["x0"] = 0, ["x1"] = 1
};
var result = Evaluate.Evaluate(values, polynomialExpressions[0]);
```

#### Accessing FFN Inside Transformer Blocks

```csharp
// The transformer's feed-forward blocks are NeuralNetwork instances
var ffn = model.Blocks[0].FeedForwardNetwork;
var ffnData = ffn.GetData();

float[,] layer1Weights = ffnData.layers[1].Weights;
Console.WriteLine($"FFN parameter count: {ffn.ParameterCount}");
```

#### Silent Training (No Progress Bar)

```csharp
nn.Train(inputs, expected, learningRate: 0.5f, epochs: 1000, silent: true);
```

#### Custom Logger

```csharp
ILogger logger = new Logger(); // or your own ILogger implementation
var nn = new NeuralNetwork(parameters, Logger: logger);
```

---

## Reference

### TransformerConfig Properties

| Property | Type | Description |
|----------|------|-------------|
| `DataType` | `TransformerDataType` | Text, SymbolicSequence, TimeSeriesRegression, or TimeSeriesClassification |
| `VocabSize` | `int` | Token vocabulary size (discrete modes only) |
| `InputFeatureDim` | `int` | Input vector dimension (continuous modes only) |
| `OutputDim` | `int` | Output dimension for regression/classification |
| `MaxSequenceLength` | `int` | Maximum input sequence length |
| `EmbeddingDim` | `int` | Internal representation dimension |
| `NumHeads` | `int` | Number of attention heads |
| `NumLayers` | `int` | Number of transformer blocks |
| `FeedForwardDim` | `int` | Hidden size of FFN within each block |
| `UseDecoderOnly` | `bool` | Use causal (autoregressive) attention mask |
| `FFNActivationType` | `ActivationType` | Activation function in FFN blocks |
| `L2RegulationLamda` | `float` | L2 regularization strength |
| `AccelerationType` | `AccelerationType` | CPU, MultiThreadCPU, GPU, or CUDA |

### TrainingConfig Properties

| Property | Type | Description |
|----------|------|-------------|
| `LearningRate` | `float` | Step size for parameter updates |
| `BatchSize` | `int` | Sequences per gradient accumulation step |
| `Epochs` | `int` | Number of passes through training data |
| `UseGradientClipping` | `bool` | Enable gradient norm clipping |
| `GradientClipThreshold` | `float` | Max gradient norm |
| `UseLearningRateDecay` | `bool` | Multiply LR by decay factor each epoch |
| `LearningRateDecay` | `float` | Decay multiplier (e.g. 0.95) |
| `ValidationInterval` | `int` | Evaluate validation loss every N epochs |
| `Verbose` | `bool` | Print epoch-by-epoch loss |

### Neural Network Parameters

| Property | Type | Description |
|----------|------|-------------|
| `AccelerationType` | `AccelerationType` | CPU, MultiThreadCPU, GPU, or CUDA |
| `AccelerationDeviceId` | `int` | GPU device index (default 0) |
| `CostFunction` | `CostFunctionType` | Loss function to use |
| `ActivationDistribution` | `ActivationDistribution` | Weight initialization distribution (Normal or Uniform) |
| `L2RegulationLamda` | `float` | L2 regularization strength |
| `GradientClippingThreshold` | `float` | Max gradient norm for clipping |
| `HuberLossDelta` | `float` | Delta parameter for Huber loss |
| `LayerWidths` | `List<int>` | Number of neurons per layer (first = input, last = output) |
| `LayerActivations` | `List<ActivationType>` | Activation function per layer |

### Transformer Data Types

| DataType | Input | Output | Loss | Train Method |
|----------|-------|--------|------|-------------|
| `Text` | Token IDs → embedding | Vocab logits | Cross-entropy | `Train(int[][])` |
| `SymbolicSequence` | Token IDs → embedding | Vocab logits | Cross-entropy | `Train(int[][])` |
| `TimeSeriesRegression` | float vectors → linear projection | Continuous values | MSE | `TrainContinuous(inputs, regressionTargets:)` |
| `TimeSeriesClassification` | float vectors → linear projection | Class logits | Cross-entropy | `TrainContinuous(inputs, classTargets:)` |

### Acceleration Types

| Value | Description |
|-------|-------------|
| `CPU` | Single-threaded CPU |
| `MultiThreadCPU` | Multi-threaded CPU |
| `GPU` | GPU via OpenCL |
| `CUDA` | GPU via CUDA |

### Activation Functions

| Value | Description |
|-------|-------------|
| `None` | Identity (pass-through) |
| `Relu` | ReLU |
| `Leakyrelu` | Leaky ReLU (alpha=0.01) |
| `Tanh` | Hyperbolic tangent |
| `Sigmoid` | Sigmoid |

### Cost Functions (Neural Network)

| Value | Use Case |
|-------|----------|
| `mse` | General regression |
| `mae` | Robust regression |
| `binaryCrossEntropy` | Binary classification (sigmoid output) |
| `categoricalCrossEntropy` | Multi-class classification |
| `huberLoss` | Regression with outliers (set `HuberLossDelta`) |
| `ZeroWeightedMSE` | Imbalanced targets where zero is common |

---


### Neural Network Background/Theory

The basic Gradient-based learning technique deployed within this project I try and explain below. 

**Backpropagation**

Backpropagation is a fundamental algorithm used for training artificial
neural networks. It uses the chain rule to compute
gradients of the loss function with respect to each weight in the
network, allowing for efficient updates during training. Here is my attempt at
a breakdown of how backpropagation works (including the role of
the chain rule within it):

**1. Forward Propagation:**

Before starting with backpropagation. Its usefull to first understand 
forward propagation, where inputs pass through the network to generate an
output:

1.  **Input Layer**: The input data is fed into the input layer.

2.  **Hidden Layers**: The data is processed through one or more hidden
    layers, where each neuron performs a weighted sum of its inputs,
    adds a bias, and applies an activation function.

3.  **Output Layer**: The final output is produced after processing
    through the output layer neurons.

Mathematically, for a single neuron 𝑗 in layer *l*:

$$z\_{j}^{l} = \\sum\_{i}^{}w\_{\\text{ij}}^{l}a\_{i}^{(l - 1)} + b\_{j}^{l}$$

*a*<sub>*j*</sub><sup>*l*</sup>= *σ*(*z*<sub>*j*</sub><sup>*l*</sup>)

where:

-   *z*<sub>*j*</sub><sup>*l*</sup> is the weighted sum.

-   *w*<sub>ij</sub><sup>*l*</sup> is the weight connecting neuron 𝑖 in
    layer *l* − 1 to neuron 𝑗 in layer 𝑙.

-   *a*<sub>*i*</sub><sup>(*l*−1)</sup> is the activation of neuron 𝑖 in
    layer *l* − 1.

-   *b*<sub>*j*</sub><sup>*l*</sup> is the bias for neuron 𝑗 in layer
    *l*.

-   *σ* is the activation function.

**2. Loss Calculation:**

The network's output is compared to the actual target values using a
loss function (e.g., Mean Squared Error, Cross-Entropy). The loss
function quantifies the error of the network's prediction.

For instance, with Mean Squared Error (MSE):

$$L = \\frac{1}{2}\\sum\_{k}^{}{(yk - \\widehat{y}k)}^{2}$$

where:

-   yk​ is the actual target value.

-   *ŷk* is the predicted value.

**3. Backpropagation:**

Backpropagation aims to minimize the loss function by adjusting the
weights and biases in the network. It involves the following steps:

1.  **Compute the Gradient of the Loss with Respect to Each Output
    Neuron:** The gradient of the loss *L* with respect to the output
    of neuron *k* in the output layer is given by:

$$\\frac{\\partial L}{\\partial a\_{k}^{L}}$$

For MSE:

$$\\frac{\\partial L}{\\partial a\_{k}^{L}} = \\widehat{y}k - \\text{yk}$$

2.  **Backpropagate the Error:** Using the chain rule, the error is
    propagated backward through the network to compute gradients for
    each weight and bias.

**Chain Rule Application:** For a weight *w*<sub>ij</sub><sup>*l*</sup>,
the chain rule helps compute the gradient of the loss with respect to
this weight by breaking it down into intermediate steps:

$$\\frac{\\partial L}{\\partial w\_{\\text{ij}}^{l}} = \\frac{\\partial L}{\\partial a\_{j}^{l}}\\text{\\!\\!}\*\\frac{\\partial a\_{j}^{l}}{\\partial z\_{j}^{l}}\*\\frac{\\partial z\_{j}^{l}}{\\partial w\_{ij}^{l}}$$

-   **Gradient of the Loss with Respect to the Activation:**
    $\\frac{\\partial L}{\\partial a\_{j}^{l}}$ This term represents the
    gradient of the loss with respect to the activation of neuron *j* in
    layer *l*.

-   **Gradient of the Activation with Respect to the Weighted Sum:**
    $\\frac{\\partial a\_{j}^{l}}{\\partial z\_{j}^{l}} = \\sigma'(z\_{j}^{l}\\mathbf{)}$
    This is the derivative of the activation function.

    1.  For instance, for a sigmoid function
        $\\sigma(z) = \\frac{1}{1 + e^{- z}}$​, the derivative
        is *σ*′(*z*) = *σ*(*z*)(1−*σ*(*z*))

    2.  In this project we use LeakyReLU
       
$$
f(x) = 
\begin{cases} 
x & \text{where } x > 0 \\
cx & \text{where } x < 0 
\end{cases}
$$
The derivative of the LeakyReLU function is:
$$
f'(x) = 
\begin{cases} 
1 & \text{where } x > 0 \\
c & \text{where } x < 0 
\end{cases}
$$

-   **Gradient of the Weighted Sum with Respect to the Weight:**
    $\\frac{\\partial z\_{j}^{l}}{\\partial w\_{\\text{ij}}^{l}} = a\_{i}^{(l - 1)}$
    This term is simply the activation of the neuron from the previous
    layer.

Combining these, the gradient for the weight is:

$$\\frac{\\partial L}{\\partial w\_{\\text{ij}}^{l}} = \\delta\_{j}^{l}\*a\_{i}^{l - 1}$$

where
$\\delta\_{j}^{l} = \\ \\frac{\\partial L}{\\partial a\_{j}^{l}}\*\\sigma'(z\_{j}^{l})$

3.  **Update the Weights and Biases:** Using the computed gradients, the
    weights and biases are updated to minimize the loss. This is
    typically done using gradient descent or a variant (e.g., stochastic
    gradient descent).

$$w\_{\\text{ij}}^{l} \\leftarrow w\_{\\text{ij}}^{l} - \\eta\\frac{\\partial L}{\\partial w\_{\\text{ij}}^{l}}$$

*b*<sub>*j*</sub><sup>*l*</sup> ← *b*<sub>*j*</sub><sup>*l*</sup> − *η***δ*<sub>*j*</sub><sup>*l*</sup>

where η is the learning rate.

**4. Iterate Over Training Data:**

The above steps (forward propagation, loss calculation, backpropagation,
and parameter update) are repeated for multiple epochs over the entire
training dataset until the network's performance converges.
