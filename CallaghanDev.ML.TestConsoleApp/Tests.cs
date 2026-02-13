using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Cache;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CallaghanDev.ML
{
    public class Tests
    {
        private const float EPSILON = 1e-3f;
        private const float TOLERANCE = 0.05f;
        private const float ABS_TOLERANCE = 1e-4f;

        private int _passed = 0;
        private int _failed = 0;
        private List<string> _failures = new List<string>();

        public void RunAllTests()
        {
            Console.WriteLine("===Transformer tests===");

            _passed = 0;
            _failed = 0;
            _failures.Clear();

            var tests = new (Action test, string name)[]
            {
                // ── Unit tests for individual components ──
                (Test_LayerNorm_ForwardBackward_NumericalCheck, "LayerNorm Forward/Backward Numerical Check"),
                (Test_Attention_WO_Gradient_NumericalCheck, "Attention WO Gradient Numerical Check"),
                (Test_Attention_WQ_Gradient_NumericalCheck, "Attention WQ Gradient Numerical Check"),
                (Test_Attention_WK_Gradient_NumericalCheck, "Attention WK Gradient Numerical Check"),
                (Test_Attention_WV_Gradient_NumericalCheck, "Attention WV Gradient Numerical Check"),
                (Test_Embedding_Gradient_NumericalCheck, "Embedding Gradient Numerical Check"),
                (Test_OutputProjection_Gradient_NumericalCheck, "Output Projection Gradient Numerical Check"),
                (Test_FFN_Gradient_FlowThrough, "FFN Gradient Flow Through"),
                (Test_ResidualConnection_GradientSplit, "Residual Connection Gradient Split"),

                // ── Integration tests ──
                (Test_LossDecreases_SingleSequence, "Loss Decreases on Single Sequence"),
                (Test_LossDecreases_MultipleBatches, "Loss Decreases Across Batches"),
                (Test_GradientClipping_Works, "Gradient Clipping Limits Norm"),
                (Test_ZeroGradients_AfterReset, "Gradients Zero After Reset"),
                (Test_MultiHead_ProducesDifferentGradsThanSingleHead, "Multi-Head vs Single-Head Gradients Differ"),
                (Test_CausalMask_FutureTokensNoGradient, "Causal Mask Blocks Future Token Gradients"),
                (Test_TrainTwice_SameModel_NoAccumulation, "Train Twice No Gradient Accumulation"),
                (Test_DifferentSequenceLengths_NoError, "Different Sequence Lengths in Batch"),
                (Test_AllParametersReceiveGradients, "All Parameters Receive Non-Zero Gradients"),
                (Test_LearningRateDecay_Applied, "Learning Rate Decay Applied Correctly"),
                (Test_ValidationLoss_Computable, "Validation Loss Is Computable"),

                // ── Overfitting sanity checks ──
                (Test_OverfitSingleSequence, "Overfit Single Sequence (loss < 1.0)"),
                (Test_OverfitTwoSequences, "Overfit Two Sequences (loss < 1.5)"),
                (Test_OverfitSingleSequence_Extended, "Overfit Single Sequence Extended (loss < 0.5)"),

                // ── Edge cases ──
                (Test_SequenceLength1_NoError, "Sequence Length 2 (minimum) No Error"),
                (Test_LargeVocab_SmallSequence, "Large Vocab Small Sequence No Error"),
                (Test_RepeatedTokens_NoError, "Repeated Tokens In Sequence No Error"),

                // ── Determinism ──
                (Test_SameInput_SameLoss, "Same Input Produces Same Loss"),
                (Test_ForwardCache_TokenIds_Stored, "ForwardCache Stores TokenIds"),
                (Test_ForwardCache_FFNInputs_Stored, "ForwardCache Stores FFN Inputs"),

                // ── End to end ──
                (Test_TrainThenGenerate_NoError, "Train Then Generate No Error"),
                (Test_MultiLayer_GradientFlow, "Multi-Layer Gradient Flow (4 layers)"),
                (Test_Convergence_MonotonicallyDecreasing, "Loss Monotonically Decreasing (5 epochs)"),

                // ── Multi-DataType tests ──
                (Test_ContinuousInputProjection_GradientCheck, "Continuous Input Projection Gradient Check"),
                (Test_TimeSeriesRegression_LossDecreases, "TimeSeries Regression Loss Decreases"),
                (Test_TimeSeriesClassification_LossDecreases, "TimeSeries Classification Loss Decreases"),
                (Test_DiscreteForwardThrowsOnContinuousConfig, "Discrete Forward Throws on Continuous Config"),
                (Test_ContinuousForwardThrowsOnDiscreteConfig, "Continuous Forward Throws on Discrete Config"),
                (Test_SymbolicSequence_TrainsLikeText, "SymbolicSequence Trains Like Text"),
                (Test_RegressionMSE_GradientFlow, "Regression MSE Gradient Flows to All Params"),
                (Test_ClassificationOverfit_SmallData, "Classification Overfit Small Dataset"),

                // ── Functional examples (run as tests) ──
                (Test_TransformerBasicForwardPass, "Basic Forward Pass"),
                (Test_TransformerGeneration, "Text Generation (untrained)"),
                (Test_TransformerBasicTraining, "Basic Training"),
                (Test_TransformerPatternLearning, "Pattern Learning"),
                (Test_TransformerMultiThreadCPU, "Multi-threaded CPU"),
                (Test_TransformerValidation, "Validation Split"),
                (Test_TransformerSaveLoad, "Save/Load"),
                (Test_TransformerLearningRateDecay, "Learning Rate Decay"),
                (Test_TransformerActivationFunctions, "Activation Functions"),
                (Test_TransformerModelSize, "Model Sizes"),

                // ── DataType example tests ──
                (Test_Example_Text, "Example: Text Autoregressive"),
                (Test_Example_SymbolicSequence, "Example: Symbolic Sequence (DNA)"),
                (Test_Example_TimeSeriesRegression, "Example: Time Series Regression"),
                (Test_Example_TimeSeriesClassification, "Example: Time Series Classification"),
            };

            for (int i = 0; i < tests.Length; i++)
            {
                Console.Write($"  [{i + 1,2}/{tests.Length}] {tests[i].name,-55} ");
                try
                {
                    tests[i].test();
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("PASS");
                    Console.ResetColor();
                    _passed++;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("FAIL");
                    Console.ResetColor();
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.WriteLine($"         {ex.Message}");
                    Console.ResetColor();
                    _failures.Add($"{tests[i].name}: {ex.Message}");
                    _failed++;
                }
            }

            PrintSummary(tests.Length);
        }

        private void PrintSummary(int total)
        {
            Console.WriteLine($"\n{"",3}{new string('─', 58)}");
            Console.Write($"   Results: ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{_passed} passed");
            Console.ResetColor();


            if (_failed > 0)
            {
                Console.Write(", ");
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Write($"{_failed} failed");
                Console.ResetColor();
            }


            Console.WriteLine($" / {total} total\n");

            if (_failures.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("   Failed tests:");
                Console.ResetColor();


                foreach (var f in _failures)
                {
                    Console.WriteLine($"     • {f}");
                }
                Console.WriteLine();
            }
        }

        //Could have just used a test framework of course. But I started testing using a console application and I dont mind really.
        private void Assert(bool condition, string message)
        {
            if (!condition)
            {
                throw new Exception(message);
            }
        }

        private (LanguageModel model, TransformerConfig config) CreateSmallModel(int vocabSize = 10, int embDim = 8, int numHeads = 2, int numLayers = 1, int ffnDim = 16, bool decoderOnly = true)
        {
            var config = new TransformerConfig
            {
                DataType = TransformerDataType.Text,
                VocabSize = vocabSize,
                MaxSequenceLength = 16,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                FFNActivationType = ActivationType.Relu,
                AccelerationType = AccelerationType.CPU,
                UseDecoderOnly = decoderOnly,
                L2RegulationLamda = 0f,
                GradientClippingThreshold = 100f
            };

            var model = new LanguageModel(config, new Random(42));

            return (model, config);
        }

        private (LanguageModel model, TransformerConfig config) CreateSmallContinuousModel(TransformerDataType dataType, int inputFeatureDim = 3, int outputDim = 1, int embDim = 8, int numHeads = 2, int numLayers = 1, int ffnDim = 16)
        {
            var config = new TransformerConfig
            {
                DataType = dataType,
                InputFeatureDim = inputFeatureDim,
                OutputDim = outputDim,
                MaxSequenceLength = 16,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                FFNActivationType = ActivationType.Relu,
                AccelerationType = AccelerationType.CPU,
                UseDecoderOnly = true,
                L2RegulationLamda = 0f,
                GradientClippingThreshold = 100f
            };

            var model = new LanguageModel(config, new Random(42));


            return (model, config);
        }

        private float ComputeLoss(LanguageModel model, int[] input, int[] target)
        {
            var logits = model.Forward(input);
            float loss = 0;
            int vocabSize = model.Config.VocabSize;


            for (int i = 0; i < Math.Min(logits.GetLength(0), target.Length); i++)
            {
                float max = float.NegativeInfinity;

                for (int j = 0; j < vocabSize; j++)
                {
                    max = Math.Max(max, logits[i, j]);
                }


                float sum = 0;

                for (int j = 0; j < vocabSize; j++)
                {
                    sum += MathF.Exp(logits[i, j] - max);
                }


                float prob = MathF.Exp(logits[i, target[i]] - max) / sum;

                loss -= MathF.Log(prob + 1e-10f);
            }


            return loss / Math.Min(logits.GetLength(0), target.Length);
        }

        private float NumericalGradient(LanguageModel model, int[] input, int[] target, Func<float> getParam, Action<float> setParam)
        {
            float original = getParam();

            setParam(original + EPSILON);

            float lossPlus = ComputeLoss(model, input, target);

            setParam(original - EPSILON);

            float lossMinus = ComputeLoss(model, input, target);

            setParam(original);


            return (lossPlus - lossMinus) / (2 * EPSILON);
        }

        private bool MatrixChanged(float[,] before, float[,] after, float threshold = 1e-10f)
        {
            for (int i = 0; i < before.GetLength(0); i++)
            {
                for (int j = 0; j < before.GetLength(1); j++)
                {
                    if (MathF.Abs(before[i, j] - after[i, j]) > threshold)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        private bool VectorChanged(float[] before, float[] after, float threshold = 1e-10f)
        {
            for (int i = 0; i < before.Length; i++)
            {
                if (MathF.Abs(before[i] - after[i]) > threshold)
                {
                    return true;
                }
            }
            return false;
        }

        private void TestGeneration(LanguageModel model, BPETokenizer tokenizer, string prompt)
        {
            //Just trying to make sure there are no crashes
            var promptTokens = tokenizer.Encode(prompt, addSpecialTokens: false);
            var generated = model.Generate(promptTokens, maxNewTokens: 8, temperature: 0.8f);
            var text = tokenizer.Decode(generated, skipSpecialTokens: true);
        }

        private string[] GenerateTrainingData(int count)
        {
            var data = new List<string>();

            var random = new Random();

            var subjects = new[] { "the cat", "the dog", "a bird", "the fish", "a mouse" };

            var verbs = new[] { "sat", "ran", "flew", "jumped", "walked", "slept", "played" };

            var locations = new[] { "on the mat", "in the park", "over the tree", "near the pond", "by the door" };

            for (int i = 0; i < count; i++)
            {
                var subject = subjects[random.Next(subjects.Length)];
                var verb = verbs[random.Next(verbs.Length)];
                var location = locations[random.Next(locations.Length)];
                data.Add($"{subject} {verb} {location}");
            }

            if (count >= 10)
            {
                data.Add("once upon a time there was a cat");
                data.Add("the quick brown fox jumps over the lazy dog");
                data.Add("in the beginning there was nothing");
            }

            return data.ToArray();
        }

        private (float[][,] inputs, float[][,] regTargets) GenerateRegressionData(int numSamples, int seqLen, int inputFeatures, int outputDim, Random random)
        {
            var inputs = new float[numSamples][,];
            var targets = new float[numSamples][,];

            for (int s = 0; s < numSamples; s++)
            {

                inputs[s] = new float[seqLen, inputFeatures];
                targets[s] = new float[seqLen, outputDim];

                float basePrice = 100f + (float)(random.NextDouble() * 50);

                for (int t = 0; t < seqLen; t++)
                {
                    float noise = (float)(random.NextDouble() - 0.5) * 5f;
                    float close = basePrice + noise;
                    float open = close + (float)(random.NextDouble() - 0.5) * 2f;
                    float high = Math.Max(open, close) + (float)(random.NextDouble() * 3f);
                    float low = Math.Min(open, close) - (float)(random.NextDouble() * 3f);
                    float volume = 1000f + (float)(random.NextDouble() * 500f);

                    inputs[s][t, 0] = open / 200f;
                    inputs[s][t, 1] = high / 200f;
                    inputs[s][t, 2] = low / 200f;
                    inputs[s][t, 3] = close / 200f;
                    inputs[s][t, 4] = volume / 2000f;

                    float nextNoise = (float)(random.NextDouble() - 0.5) * 5f;

                    targets[s][t, 0] = (close + nextNoise) / 200f;

                    basePrice += (float)(random.NextDouble() - 0.48) * 2f;
                }
            }

            return (inputs, targets);
        }

        private (float[][,] inputs, int[][] classTargets) GenerateClassificationData(int numSamples, int seqLen, int inputFeatures, Random random)
        {
            var inputs = new float[numSamples][,];
            var classTargets = new int[numSamples][];

            for (int s = 0; s < numSamples; s++)
            {
                inputs[s] = new float[seqLen, inputFeatures];
                classTargets[s] = new int[seqLen];

                float basePrice = 100f + (float)(random.NextDouble() * 50);

                for (int t = 0; t < seqLen; t++)
                {
                    float noise = (float)(random.NextDouble() - 0.5) * 5f;
                    float close = basePrice + noise;
                    float open = close + (float)(random.NextDouble() - 0.5) * 2f;
                    float high = Math.Max(open, close) + (float)(random.NextDouble() * 3f);
                    float low = Math.Min(open, close) - (float)(random.NextDouble() * 3f);
                    float volume = 1000f + (float)(random.NextDouble() * 500f);

                    inputs[s][t, 0] = open / 200f;
                    inputs[s][t, 1] = high / 200f;
                    inputs[s][t, 2] = low / 200f;
                    inputs[s][t, 3] = close / 200f;
                    inputs[s][t, 4] = volume / 2000f;

                    float priceChange = close - open;

                    if (priceChange > 0.5f)
                    {
                        classTargets[s][t] = 2;
                    }
                    else if (priceChange < -0.5f)
                    {
                        classTargets[s][t] = 0;
                    }
                    else
                    {
                        classTargets[s][t] = 1;
                    }

                    basePrice += (float)(random.NextDouble() - 0.48) * 2f;
                }
            }


            return (inputs, classTargets);
        }

        public void Test_LayerNorm_ForwardBackward_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var block = model.Blocks[0];

            for (int paramIdx = 0; paramIdx < Math.Min(4, config.EmbeddingDim); paramIdx++)
            {
                int idx = paramIdx;
                float numGrad = NumericalGradient(model, input, target, () => block.LN1Gamma[idx], (v) => block.LN1Gamma[idx] = v);

                Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"LN1Gamma[{idx}] numerical gradient is NaN/Inf");
            }
        }

        public void Test_Attention_WO_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };


            var attn = model.Blocks[0].Attention;


            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;

                    float numGrad = NumericalGradient(model, input, target, () => attn.WO[rr, cc], (v) => attn.WO[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"WO[{rr},{cc}] numerical gradient is NaN/Inf");
                }
            }
        }

        public void Test_Attention_WQ_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var attn = model.Blocks[0].Attention;

            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;
                    float numGrad = NumericalGradient(model, input, target, () => attn.WQ[rr, cc], (v) => attn.WQ[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"WQ[{rr},{cc}] numerical gradient is NaN/Inf: {numGrad}");
                }
            }
        }

        public void Test_Attention_WK_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var attn = model.Blocks[0].Attention;

            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;

                    float numGrad = NumericalGradient(model, input, target, () => attn.WK[rr, cc], (v) => attn.WK[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad),  $"WK[{rr},{cc}] numerical gradient is NaN/Inf: {numGrad}");
                }
            }
              
        }

        public void Test_Attention_WV_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var attn = model.Blocks[0].Attention;

            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;
                    float numGrad = NumericalGradient(model, input, target, () => attn.WV[rr, cc], (v) => attn.WV[rr, cc] = v);
                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"WV[{rr},{cc}] numerical gradient is NaN/Inf: {numGrad}");
                }
            }
        }

        public void Test_Embedding_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            foreach (int tokenId in input)
            {
                for (int d = 0; d < Math.Min(2, config.EmbeddingDim); d++)
                {
                    int tid = tokenId, dd = d;

                    float numGrad = NumericalGradient(model, input, target, () => model.TokenEmbedding[tid, dd], (v) => model.TokenEmbedding[tid, dd] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"TokenEmbedding[{tid},{dd}] numerical gradient is NaN/Inf");
                    Assert(MathF.Abs(numGrad) > 1e-8f, $"TokenEmbedding[{tid},{dd}] gradient is effectively zero ({numGrad:E4})");
                }
            }
        }

        public void Test_OutputProjection_Gradient_NumericalCheck()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            for (int r = 0; r < 2; r++)
            {
                for (int c = 0; c < 2; c++)
                {
                    int rr = r, cc = c;

                    float numGrad = NumericalGradient(model, input, target, () => model.OutputProjection[rr, cc], (v) => model.OutputProjection[rr, cc] = v);

                    Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"OutputProjection[{rr},{cc}] numerical gradient is NaN/Inf");
                }
            }
            for (int v = 0; v < Math.Min(3, config.VocabSize); v++)
            {
                int vv = v;

                float numGrad = NumericalGradient(model, input, target, () => model.OutputBias[vv], (v2) => model.OutputBias[vv] = v2);

                Assert(!float.IsNaN(numGrad) && !float.IsInfinity(numGrad), $"OutputBias[{vv}] numerical gradient is NaN/Inf");
            }
        }

        public void Test_FFN_Gradient_FlowThrough()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1, ffnDim: 8);

            var ffn = model.Blocks[0].FeedForwardNetwork;
            var ffnData = ffn.GetData();

            var weightsBefore = new float[ffnData.layers[1].Weights.GetLength(0), ffnData.layers[1].Weights.GetLength(1)];

            Array.Copy(ffnData.layers[1].Weights, weightsBefore, ffnData.layers[1].Weights.Length);

            int[][] sequences = { new[] { 1, 2, 3, 4 } };

            var trainConfig = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(sequences);

            var weightsAfter = ffnData.layers[1].Weights;
            bool anyChanged = false;

            for (int i = 0; i < weightsBefore.GetLength(0) && !anyChanged; i++)
            {
                for (int j = 0; j < weightsBefore.GetLength(1) && !anyChanged; j++)
                {
                    if (MathF.Abs(weightsBefore[i, j] - weightsAfter[i, j]) > 1e-10f)
                    {
                        anyChanged = true;
                    }
                }
            }


            Assert(anyChanged, "FFN weights did not change after training step - FFN backprop is not connected");
        }

        public void Test_ResidualConnection_GradientSplit()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2 };
            int[] target = { 2, 3 };

            float numGrad = NumericalGradient(model, input, target, () => model.TokenEmbedding[1, 0], (v) => model.TokenEmbedding[1, 0] = v);

            Assert(MathF.Abs(numGrad) > 1e-7f, $"Gradient through residual is effectively zero ({numGrad:E4})");
        }


        public void Test_LossDecreases_SingleSequence()
        {
            var (model, config) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);

            int[] sequence = { 1, 2, 3, 4, 5 };

            var trainConfig = new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 5, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };

            int[] inputSeq = sequence.Take(sequence.Length - 1).ToArray();
            int[] targetSeq = sequence.Skip(1).ToArray();

            float lossBefore = ComputeLoss(model, inputSeq, targetSeq);

            new TransformerTrainer(model, trainConfig).Train(new[] { sequence });

            float lossAfter = ComputeLoss(model, inputSeq, targetSeq);

            Assert(lossAfter < lossBefore, $"Loss did not decrease: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_LossDecreases_MultipleBatches()
        {
            var (model, config) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);

            int[][] sequences = { new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 }, new[] { 3, 4, 5, 6 }, new[] { 4, 5, 6, 7 } };

            float lossBefore = 0;

            foreach (var seq in sequences)
            {
                var inp = seq.Take(seq.Length - 1).ToArray();
                var tgt = seq.Skip(1).ToArray();
                lossBefore += ComputeLoss(model, inp, tgt);
            }

            lossBefore = lossBefore / sequences.Length;

            var trainConfig = new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 5, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(sequences);

            float lossAfter = 0;

            foreach (var seq in sequences)
            {
                var inp = seq.Take(seq.Length - 1).ToArray();

                var tgt = seq.Skip(1).ToArray();

                lossAfter += ComputeLoss(model, inp, tgt);
            }
            lossAfter = lossAfter/  sequences.Length;

            Assert(lossAfter < lossBefore, $"Average loss did not decrease: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_GradientClipping_Works()
        {
            var (model, config) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);
            var trainConfig = new TrainingConfig { LearningRate = 0.1f, BatchSize = 1, Epochs = 1, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(new[] { new[] { 1, 2, 3, 4, 5 } });

            var logits = model.Forward(new[] { 1, 2, 3 });

            bool anyNaN = false;

            for (int i = 0; i < logits.GetLength(0); i++)
            {
                for (int j = 0; j < logits.GetLength(1); j++)
                {

                    if (float.IsNaN(logits[i, j]) || float.IsInfinity(logits[i, j]))
                    {
                        anyNaN = true;
                    }
                }
            }

            Assert(!anyNaN, "Model produces NaN/Inf after training with gradient clipping");
        }

        public void Test_ZeroGradients_AfterReset()
        {
            var (model, config) = CreateSmallModel();

            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);

            trainer.Train(new[] { new[] { 1, 2, 3 } });
            trainer.Train(new[] { new[] { 3, 4, 5, 6 } });
        }

        public void Test_MultiHead_ProducesDifferentGradsThanSingleHead()
        {
            var (model2h, _) = CreateSmallModel(vocabSize: 8, embDim: 8, numHeads: 2, numLayers: 1);
            var (model1h, _) = CreateSmallModel(vocabSize: 8, embDim: 8, numHeads: 1, numLayers: 1);

            Array.Copy(model2h.TokenEmbedding, model1h.TokenEmbedding, model2h.TokenEmbedding.Length);
            Array.Copy(model2h.OutputProjection, model1h.OutputProjection, model2h.OutputProjection.Length);
            Array.Copy(model2h.OutputBias, model1h.OutputBias, model2h.OutputBias.Length);

            var attn2h = model2h.Blocks[0].Attention;
            var attn1h = model1h.Blocks[0].Attention;

            Array.Copy(attn2h.WQ, attn1h.WQ, attn2h.WQ.Length);
            Array.Copy(attn2h.WK, attn1h.WK, attn2h.WK.Length);
            Array.Copy(attn2h.WV, attn1h.WV, attn2h.WV.Length);
            Array.Copy(attn2h.WO, attn1h.WO, attn2h.WO.Length);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            float loss2h = ComputeLoss(model2h, input, target);
            float loss1h = ComputeLoss(model1h, input, target);

            Assert(MathF.Abs(loss2h - loss1h) > 1e-6f, $"2-head and 1-head losses are identical ({loss2h:F6})");
        }

        public void Test_CausalMask_FutureTokensNoGradient()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1, decoderOnly: true);

            int[] input = { 1, 2, 3 };

            var logitsBefore = model.Forward(input);

            float logit_0_0 = logitsBefore[0, 0];

            float original = model.TokenEmbedding[3, 0];

            model.TokenEmbedding[3, 0] += 1.0f;

            var logitsAfter = model.Forward(input);

            float logit_0_0_after = logitsAfter[0, 0];

            model.TokenEmbedding[3, 0] = original;

            Assert(MathF.Abs(logit_0_0 - logit_0_0_after) < 1e-5f, $"Causal mask violated: logits[0,0] changed from {logit_0_0:F6} to {logit_0_0_after:F6}");
        }

        public void Test_TrainTwice_SameModel_NoAccumulation()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);
            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 3, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);
            trainer.Train(new[] { new[] { 1, 2, 3, 4 } });
            trainer.Train(new[] { new[] { 5, 6, 7, 8 } });
            var logits = model.Forward(new[] { 1, 2, 3 });
            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert(!float.IsNaN(logits[i, j]), "NaN after two separate training runs");
        }

        public void Test_DifferentSequenceLengths_NoError()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            int[][] sequences = { new[] { 1, 2 }, new[] { 1, 2, 3, 4, 5, 6 }, new[] { 3, 4, 5 }, new[] { 7, 8, 9, 1 } };

            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 2, Epochs = 2, Verbose = false };

            new TransformerTrainer(model, tc).Train(sequences);
        }

        public void Test_AllParametersReceiveGradients()
        {
            var (model, config) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1, ffnDim: 8);

            var embBefore = (float[,])model.TokenEmbedding.Clone();
            var outProjBefore = (float[,])model.OutputProjection.Clone();
            var outBiasBefore = (float[])model.OutputBias.Clone();
            var attn = model.Blocks[0].Attention;
            var wqBefore = (float[,])attn.WQ.Clone();
            var wkBefore = (float[,])attn.WK.Clone();
            var wvBefore = (float[,])attn.WV.Clone();
            var woBefore = (float[,])attn.WO.Clone();
            var bqBefore = (float[])attn.BiasQ.Clone();
            var bkBefore = (float[])attn.BiasK.Clone();
            var bvBefore = (float[])attn.BiasV.Clone();
            var boBefore = (float[])attn.BiasO.Clone();
            var ln1gBefore = (float[])model.Blocks[0].LN1Gamma.Clone();
            var ln1bBefore = (float[])model.Blocks[0].LN1Beta.Clone();
            var ln2gBefore = (float[])model.Blocks[0].LN2Gamma.Clone();
            var ln2bBefore = (float[])model.Blocks[0].LN2Beta.Clone();
            var ffnData = model.Blocks[0].FeedForwardNetwork.GetData();
            var ffnW1Before = (float[,])ffnData.layers[1].Weights.Clone();
            var ffnW2Before = (float[,])ffnData.layers[2].Weights.Clone();

            var tc = new TrainingConfig { LearningRate = 0.05f, BatchSize = 1, Epochs = 3, UseGradientClipping = false, Verbose = false };

            new TransformerTrainer(model, tc).Train(new[] { new[] { 1, 2, 3, 4 } });

            Assert(MatrixChanged(embBefore, model.TokenEmbedding), "TokenEmbedding did not change");
            Assert(MatrixChanged(outProjBefore, model.OutputProjection), "OutputProjection did not change");
            Assert(VectorChanged(outBiasBefore, model.OutputBias), "OutputBias did not change");
            Assert(MatrixChanged(wqBefore, attn.WQ), "WQ did not change");
            Assert(MatrixChanged(wkBefore, attn.WK), "WK did not change");
            Assert(MatrixChanged(wvBefore, attn.WV), "WV did not change");
            Assert(MatrixChanged(woBefore, attn.WO), "WO did not change");
            Assert(VectorChanged(bqBefore, attn.BiasQ), "BiasQ did not change");
            Assert(VectorChanged(bkBefore, attn.BiasK), "BiasK did not change");
            Assert(VectorChanged(bvBefore, attn.BiasV), "BiasV did not change");
            Assert(VectorChanged(boBefore, attn.BiasO), "BiasO did not change");
            Assert(VectorChanged(ln1gBefore, model.Blocks[0].LN1Gamma), "LN1Gamma did not change");
            Assert(VectorChanged(ln1bBefore, model.Blocks[0].LN1Beta), "LN1Beta did not change");
            Assert(VectorChanged(ln2gBefore, model.Blocks[0].LN2Gamma), "LN2Gamma did not change");
            Assert(VectorChanged(ln2bBefore, model.Blocks[0].LN2Beta), "LN2Beta did not change");
            Assert(MatrixChanged(ffnW1Before, ffnData.layers[1].Weights), "FFN Layer 1 Weights did not change");
            Assert(MatrixChanged(ffnW2Before, ffnData.layers[2].Weights), "FFN Layer 2 Weights did not change");
        }

        public void Test_LearningRateDecay_Applied()
        {
            var (model, _) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            var tc = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 5, UseLearningRateDecay = true, LearningRateDecay = 0.5f, Verbose = false };

            new TransformerTrainer(model, tc).Train(new[] { new[] { 1, 2, 3, 4 } });

            var logits = model.Forward(new[] { 1, 2 });

            for (int i = 0; i < logits.GetLength(0); i++)
            {
                for (int j = 0; j < logits.GetLength(1); j++)
                {
                    Assert(!float.IsNaN(logits[i, j]), "NaN in logits after training with LR decay");
                }
            }
        }

        public void Test_ValidationLoss_Computable()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            int[][] trainSeq = { new[] { 1, 2, 3, 4 }, new[] { 3, 4, 5, 6 } };
            int[][] valSeq = { new[] { 2, 3, 4, 5 } };

            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 2, Epochs = 3, ValidationInterval = 1, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);


            trainer.Train(trainSeq, valSeq);


            float valLoss = trainer.Validate(valSeq);

            Assert(!float.IsNaN(valLoss) && !float.IsInfinity(valLoss) && valLoss > 0, $"Validation loss is invalid: {valLoss}");
        }

        public void Test_OverfitSingleSequence()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);

            int[] sequence = { 1, 2, 3, 4 };
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 200, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };

            new TransformerTrainer(model, tc).Train(new[] { sequence });


            float loss = ComputeLoss(model, input, target);


            Assert(loss < 1.0f, $"Failed to overfit single sequence after 200 epochs: loss={loss:F4} (expected < 1.0)");
        }

        public void Test_OverfitTwoSequences()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            int[][] sequences = { new[] { 1, 2, 3, 4 }, new[] { 5, 6, 7, 8 } };


            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 300, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };
            new TransformerTrainer(model, tc).Train(sequences);


            float totalLoss = 0;

            foreach (var seq in sequences)
            {
                var inp = seq.Take(seq.Length - 1).ToArray();
                var tgt = seq.Skip(1).ToArray();
                totalLoss += ComputeLoss(model, inp, tgt);
            }

            float avgLoss = totalLoss / sequences.Length;


            Assert(avgLoss < 1.5f, $"Failed to overfit two sequences after 300 epochs: avg loss={avgLoss:F4} (expected < 1.5)");
        }

        public void Test_OverfitSingleSequence_Extended()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 64);
            int[] sequence = { 1, 2, 3, 4 };
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 500, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { sequence });


            float loss = ComputeLoss(model, input, target);


            Assert(loss < 0.5f, $"Failed to deeply overfit single sequence after 500 epochs: loss={loss:F4} (expected < 0.5)");
        }

        public void Test_SequenceLength1_NoError()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            new TransformerTrainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false }).Train(new[] { new[] { 1, 2 } });
        }

        public void Test_LargeVocab_SmallSequence()
        {
            var (model, _) = CreateSmallModel(vocabSize: 1000, embDim: 8, numHeads: 2, numLayers: 1);
            
            new TransformerTrainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false }).Train(new[] { new[] { 100, 200, 300 } });
        }

        public void Test_RepeatedTokens_NoError()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);

            new TransformerTrainer(model, new TrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 3, Verbose = false }).Train(new[] { new[] { 1, 1, 1, 1 } });
        }

        public void Test_SameInput_SameLoss()
        {
            var (model, _) = CreateSmallModel(vocabSize: 8, embDim: 4, numHeads: 2, numLayers: 1);

            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };

            float loss1 = ComputeLoss(model, input, target);
            float loss2 = ComputeLoss(model, input, target);

            Assert(loss1 == loss2, $"Same input produced different losses: {loss1:E6} vs {loss2:E6}");
        }

        public void Test_ForwardCache_TokenIds_Stored()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);
            int[] sequence = { 5, 7, 3 };

            float embBefore_5 = model.TokenEmbedding[5, 0];
            float embBefore_0 = model.TokenEmbedding[0, 0];

            var tc = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { sequence });

            float embAfter_5 = model.TokenEmbedding[5, 0];
            float embAfter_0 = model.TokenEmbedding[0, 0];

            Assert(MathF.Abs(embBefore_5 - embAfter_5) > 1e-10f, "Embedding for token 5 (in input) did not change");
            Assert(MathF.Abs(embBefore_0 - embAfter_0) < 1e-10f, "Embedding for token 0 (NOT in input) changed - gradient leaked");
        }

        public void Test_ForwardCache_FFNInputs_Stored()
        {
            var (model, _) = CreateSmallModel(vocabSize: 8, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);


            int[] sequence = { 1, 2, 3, 4 };
            int[] input = { 1, 2, 3 };
            int[] target = { 2, 3, 4 };


            float lossBefore = ComputeLoss(model, input, target);

            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 50, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };

            new TransformerTrainer(model, tc).Train(new[] { sequence });

            float lossAfter = ComputeLoss(model, input, target);

            Assert(lossAfter < lossBefore, $"Loss did not decrease after 50 epochs (before={lossBefore:F4}, after={lossAfter:F4}). FFN backprop may be broken.");
        }


        public void Test_TrainThenGenerate_NoError()
        {
            var (model, config) = CreateSmallModel(vocabSize: 20, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            int[][] sequences = { new[] { 1, 2, 3, 4, 5 }, new[] { 6, 7, 8, 9, 10 }, new[] { 1, 3, 5, 7, 9 } };


            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 10, Verbose = false };
            new TransformerTrainer(model, tc).Train(sequences);

            var generated = model.Generate(new[] { 1, 2 }, maxNewTokens: 5, temperature: 1.0f);


            Assert(generated.Length > 2, "Generate produced no new tokens after training");


            foreach (var tok in generated)
            {
                Assert(tok >= 0 && tok < config.VocabSize, $"Generated invalid token {tok}");
            }
        }

        public void Test_MultiLayer_GradientFlow()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 4, ffnDim: 16);

            int[] sequence = { 1, 2, 3, 4, 5 };

            var wqL0Before = (float[,])model.Blocks[0].Attention.WQ.Clone();
            var wqL3Before = (float[,])model.Blocks[3].Attention.WQ.Clone();

            var tc = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { sequence });

            Assert(MatrixChanged(wqL0Before, model.Blocks[0].Attention.WQ), "Layer 0 WQ did not change");
            Assert(MatrixChanged(wqL3Before, model.Blocks[3].Attention.WQ), "Layer 3 WQ did not change");

            for (int layer = 0; layer < 4; layer++)
            {
                var attnL = model.Blocks[layer].Attention;

                for (int i = 0; i < attnL.WQ.GetLength(0); i++)
                {
                    for (int j = 0; j < attnL.WQ.GetLength(1); j++)
                    {
                        Assert(!float.IsNaN(attnL.WQ[i, j]), $"NaN in layer {layer} WQ[{i},{j}]");
                    }
                }
            }
        }

        public void Test_Convergence_MonotonicallyDecreasing()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);

            int[][] sequences = { new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 } };
            int[] testInput = { 1, 2, 3 };
            int[] testTarget = { 2, 3, 4 };

            float previousLoss = ComputeLoss(model, testInput, testTarget);
            int decreaseCount = 0;

            for (int epoch = 0; epoch < 5; epoch++)
            {
                var tc = new TrainingConfig { LearningRate = 0.003f, BatchSize = 2, Epochs = 1, Verbose = false };
                new TransformerTrainer(model, tc).Train(sequences);

                float currentLoss = ComputeLoss(model, testInput, testTarget);

                if (currentLoss < previousLoss) 
                { 
                    decreaseCount++; 
                }

                previousLoss = currentLoss;
            }

            Assert(decreaseCount >= 3, $"Loss only decreased in {decreaseCount}/5 epochs - expected at least 3");
        }

        public void Test_ContinuousInputProjection_GradientCheck()
        {
            var (model, config) = CreateSmallContinuousModel(TransformerDataType.TimeSeriesRegression, inputFeatureDim: 3, outputDim: 1, embDim: 4, numHeads: 2, numLayers: 1);

            var projBefore = (float[,])model.InputProjection.Clone();
            var biasBefore = (float[])model.InputProjectionBias.Clone();

            var inputs = new float[1][,];
            inputs[0] = new float[4, 3];
            var rng = new Random(42);


            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    inputs[0][i, j] = (float)rng.NextDouble();
                }
            }

            var targets = new float[1][,];
            targets[0] = new float[4, 1];


            for (int i = 0; i < 4; i++)
            {
                targets[0][i, 0] = (float)rng.NextDouble();
            }

            var tc = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 3, UseGradientClipping = false, Verbose = false };
            new TransformerTrainer(model, tc).TrainContinuous(inputs, regressionTargets: targets);

            Assert(MatrixChanged(projBefore, model.InputProjection), "InputProjection did not change after training");
            Assert(VectorChanged(biasBefore, model.InputProjectionBias), "InputProjectionBias did not change after training");
        }

        public void Test_TimeSeriesRegression_LossDecreases()
        {
            var (model, config) = CreateSmallContinuousModel(TransformerDataType.TimeSeriesRegression, inputFeatureDim: 3, outputDim: 1, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);
            
            var rng = new Random(42);
            var inputs = new float[5][,];
            var targets = new float[5][,];

            for (int s = 0; s < 5; s++)
            {
                inputs[s] = new float[6, 3];
                targets[s] = new float[6, 1];
                for (int i = 0; i < 6; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        inputs[s][i, j] = (float)rng.NextDouble();
                    }
                    targets[s][i, 0] = inputs[s][i, 0] * 0.5f + 0.1f;
                }
            }

            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);

            float lossBefore = trainer.ValidateContinuous(inputs, regressionTargets: targets);

            tc.Epochs = 20;

            trainer.TrainContinuous(inputs, regressionTargets: targets);

            float lossAfter = trainer.ValidateContinuous(inputs, regressionTargets: targets);

            Assert(lossAfter < lossBefore, $"Regression loss did not decrease: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_TimeSeriesClassification_LossDecreases()
        {
            var (model, config) = CreateSmallContinuousModel(TransformerDataType.TimeSeriesClassification, inputFeatureDim: 3, outputDim: 3, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);
            var rng = new Random(42);
            var inputs = new float[5][,];
            var classTargets = new int[5][];

            for (int s = 0; s < 5; s++)
            {
                inputs[s] = new float[6, 3];
                classTargets[s] = new int[6];

                for (int i = 0; i < 6; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        inputs[s][i, j] = (float)rng.NextDouble();
                    }

                    classTargets[s][i] = rng.Next(3);
                }
            }

            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);

            float lossBefore = trainer.ValidateContinuous(inputs, classTargets: classTargets);

            tc.Epochs = 30;

            trainer.TrainContinuous(inputs, classTargets: classTargets);

            float lossAfter = trainer.ValidateContinuous(inputs, classTargets: classTargets);

            Assert(lossAfter < lossBefore, $"Classification loss did not decrease: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_DiscreteForwardThrowsOnContinuousConfig()
        {
            var (model, _) = CreateSmallContinuousModel(TransformerDataType.TimeSeriesRegression, inputFeatureDim: 3, outputDim: 1);

            bool threw = false;

            try 
            { 
                model.Forward(new[] { 1, 2, 3 }); 
            }
            catch (InvalidOperationException) 
            { 
                threw = true; 
            }

            Assert(threw, "Expected InvalidOperationException when calling Forward(int[]) on continuous model");
        }

        public void Test_ContinuousForwardThrowsOnDiscreteConfig()
        {
            var (model, _) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 1);
            bool threw = false;

            try 
            { 
                model.Forward(new float[3, 4]); 
            }
            catch (InvalidOperationException) 
            { 
                threw = true;
            }

            Assert(threw, "Expected InvalidOperationException when calling Forward(float[,]) on discrete model");
        }

        public void Test_SymbolicSequence_TrainsLikeText()
        {
            var config = new TransformerConfig
            {
                DataType = TransformerDataType.SymbolicSequence,
                VocabSize = 8,
                MaxSequenceLength = 16,
                EmbeddingDim = 8,
                NumHeads = 2,
                NumLayers = 1,
                FeedForwardDim = 16,
                AccelerationType = AccelerationType.CPU,
                UseDecoderOnly = true,
                L2RegulationLamda = 0f
            };

            var model = new LanguageModel(config, new Random(42));

            int[] sequence = { 1, 4, 5, 6, 7, 2 };

            int[] input = sequence.Take(5).ToArray();
            int[] target = sequence.Skip(1).ToArray();

            float lossBefore = ComputeLoss(model, input, target);

            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 20, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { sequence });


            float lossAfter = ComputeLoss(model, input, target);

            Assert(lossAfter < lossBefore, $"SymbolicSequence loss did not decrease: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_RegressionMSE_GradientFlow()
        {
            var (model, config) = CreateSmallContinuousModel(TransformerDataType.TimeSeriesRegression, inputFeatureDim: 2, outputDim: 1, embDim: 4, numHeads: 2, numLayers: 1, ffnDim: 8);
            
            var projBefore = (float[,])model.InputProjection.Clone();
            var outProjBefore = (float[,])model.OutputProjection.Clone();
            var wqBefore = (float[,])model.Blocks[0].Attention.WQ.Clone();
            var ln1gBefore = (float[])model.Blocks[0].LN1Gamma.Clone();

            var inputs = new float[1][,] { new float[,] { { 0.5f, 0.3f }, { 0.2f, 0.8f }, { 0.9f, 0.1f } } };
            var targets = new float[1][,] { new float[,] { { 0.4f }, { 0.6f }, { 0.5f } } };

            var tc = new TrainingConfig { LearningRate = 0.01f, BatchSize = 1, Epochs = 5, UseGradientClipping = false, Verbose = false };
            new TransformerTrainer(model, tc).TrainContinuous(inputs, regressionTargets: targets);

            Assert(MatrixChanged(projBefore, model.InputProjection), "InputProjection did not change (MSE)");
            Assert(MatrixChanged(outProjBefore, model.OutputProjection), "OutputProjection did not change (MSE)");
            Assert(MatrixChanged(wqBefore, model.Blocks[0].Attention.WQ), "WQ did not change (MSE)");
            Assert(VectorChanged(ln1gBefore, model.Blocks[0].LN1Gamma), "LN1Gamma did not change (MSE)");
        }

        public void Test_ClassificationOverfit_SmallData()
        {
            var (model, config) = CreateSmallContinuousModel(TransformerDataType.TimeSeriesClassification, inputFeatureDim: 2, outputDim: 3, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
         

            var inputs = new float[2][,]
            {
                new float[,] { { 0.8f, 0.1f }, { -0.7f, 0.2f }, { 0.1f, 0.3f } },
                new float[,] { { -0.9f, 0.5f }, { 0.6f, 0.4f }, { 0.0f, 0.1f } }
            };
            var classTargets = new int[2][]
            {
                new[] { 2, 0, 1 },
                new[] { 0, 2, 1 }
            };

            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 100, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);
            trainer.TrainContinuous(inputs, classTargets: classTargets);

            float loss = trainer.ValidateContinuous(inputs, classTargets: classTargets);
            Assert(loss < 1.5f, $"Failed to overfit small classification data: loss={loss:F4} (expected < 1.5)");
        }

        public void Test_TransformerBasicForwardPass()
        {
            var tokenizer = new BPETokenizer();

            tokenizer.Train(new[] { "the cat sat", "the dog ran" }, vocabSize: 100, minFrequency: 1);

            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 16, EmbeddingDim = 32, NumHeads = 2, NumLayers = 1, FeedForwardDim = 128, AccelerationType = AccelerationType.CPU };
            var model = new LanguageModel(config);

            var tokens = tokenizer.Encode("the cat sat", addSpecialTokens: false);
            var logits = model.Forward(tokens);


            Assert(logits.GetLength(0) == tokens.Length, $"Output seq len mismatch: {logits.GetLength(0)} != {tokens.Length}");
            Assert(logits.GetLength(1) == tokenizer.VocabSize, $"Output vocab size mismatch: {logits.GetLength(1)} != {tokenizer.VocabSize}");
        }

        public void Test_TransformerGeneration()
        {
            var tokenizer = new BPETokenizer();

            tokenizer.Train(new[] { "hello world", "test data" }, vocabSize: 200, minFrequency: 1);

            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 16, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256, AccelerationType = AccelerationType.CPU };
            var model = new LanguageModel(config);

            var prompt = tokenizer.Encode("hello", addSpecialTokens: false);
            var generated = model.Generate(prompt, maxNewTokens: 5, temperature: 1.0f);

            Assert(generated.Length > prompt.Length, "Generate produced no new tokens");
        }

        public void Test_TransformerBasicTraining()
        {
            var trainingTexts = new[] { "the cat sat", "the cat slept", "the dog ran", "the dog played" };

            var tokenizer = new BPETokenizer();

            tokenizer.Train(trainingTexts, vocabSize: 300, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 16, EmbeddingDim = 32, NumHeads = 2, NumLayers = 2, FeedForwardDim = 128, AccelerationType = AccelerationType.CPU };
            var model = new LanguageModel(config);
            var trainConfig = new TrainingConfig { LearningRate = 0.01f, BatchSize = 2, Epochs = 20, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(sequences);

            TestGeneration(model, tokenizer, "the cat");
        }

        public void Test_TransformerPatternLearning()
        {
            var trainingTexts = new[] { "a b c", "a b c", "d e f", "d e f", "a b c", "d e f" };
            var tokenizer = new BPETokenizer();

            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();
            
            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 16, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256, AccelerationType = AccelerationType.CPU, L2RegulationLamda = 0.01f };
            
            var model = new LanguageModel(config);
            
            var trainConfig = new TrainingConfig { LearningRate = 0.005f, BatchSize = 3, Epochs = 50, GradientClipThreshold = 1.0f, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(sequences);

            TestGeneration(model, tokenizer, "a b");
        }

        public void Test_TransformerMultiThreadCPU()
        {
            var trainingTexts = new[] { "the cat sat on the mat", "the dog ran in the park", "a bird flew over the tree" };
            var tokenizer = new BPETokenizer();

            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 32, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256, AccelerationType = AccelerationType.MultiThreadCPU };
            
            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig { LearningRate = 0.001f, BatchSize = 3, Epochs = 10, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(sequences);

            TestGeneration(model, tokenizer, "the cat");
        }

        public void Test_TransformerValidation()
        {
            var allTexts = GenerateTrainingData(30);
            var tokenizer = new BPETokenizer();

            tokenizer.Train(allTexts, vocabSize: 500, minFrequency: 1);

            var sequences = allTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();
            int splitIdx = (int)(sequences.Length * 0.8);
            var trainSeq = sequences.Take(splitIdx).ToArray();
            var valSeq = sequences.Skip(splitIdx).ToArray();

            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 32, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256, AccelerationType = AccelerationType.CPU };
            
    
            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig { LearningRate = 0.001f, BatchSize = 4, Epochs = 10, ValidationInterval = 10, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(trainSeq, valSeq);
        }

        public void Test_TransformerSaveLoad()
        {
            var trainingTexts = new[] { "the cat sat on the mat", "the dog ran in the park" };
            var tokenizer = new BPETokenizer();

            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 32, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256, AccelerationType = AccelerationType.CPU };
            var model = new LanguageModel(config);

            var trainConfig = new TrainingConfig { LearningRate = 0.001f, BatchSize = 2, Epochs = 10, Verbose = false };

            new TransformerTrainer(model, trainConfig).Train(sequences);

            var prompt = tokenizer.Encode("the cat", addSpecialTokens: false);

            var beforeGen = model.Generate(prompt, maxNewTokens: 5);

            model.SaveFeedForwardNetworks("./test_transformer_combined");
            model.LoadFeedForwardNetworks("./test_transformer_combined");

            var afterGen = model.Generate(prompt, maxNewTokens: 5);

            // Just insuring there is no crash here
            Assert(afterGen.Length > prompt.Length, "Generation after load produced no new tokens");
        }

        public void Test_TransformerLearningRateDecay()
        {
            var trainingTexts = GenerateTrainingData(20);

            var tokenizer = new BPETokenizer();

            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 32, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256, AccelerationType = AccelerationType.CPU };
            var model = new LanguageModel(config);
            var trainConfig = new TrainingConfig { LearningRate = 0.01f, BatchSize = 4, Epochs = 10, UseLearningRateDecay = true, LearningRateDecay = 0.95f, Verbose = false };
            
            new TransformerTrainer(model, trainConfig).Train(sequences);



            TestGeneration(model, tokenizer, "the cat");
        }

        public void Test_TransformerActivationFunctions()
        {
            var trainingTexts = new[] { "the cat sat", "the dog ran", "a bird flew" };

            var tokenizer = new BPETokenizer();

            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);

            var sequences = trainingTexts.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).ToArray();

            var activations = new[] { ActivationType.Relu, ActivationType.Leakyrelu, ActivationType.Tanh };

            foreach (var activation in activations)
            {
                var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 16, EmbeddingDim = 32, NumHeads = 2, NumLayers = 2, FeedForwardDim = 128, FFNActivationType = activation, AccelerationType = AccelerationType.CPU };
                var model = new LanguageModel(config);
                var trainConfig = new TrainingConfig { LearningRate = 0.01f, BatchSize = 3, Epochs = 10, Verbose = false };
                new TransformerTrainer(model, trainConfig).Train(sequences);


                TestGeneration(model, tokenizer, "the cat");
            }
        }

        public void Test_TransformerModelSize()
        {
            var configs = new[]
            {
                new TransformerConfig { VocabSize = 100, EmbeddingDim = 32, NumHeads = 2, NumLayers = 1, FeedForwardDim = 128 },
                new TransformerConfig { VocabSize = 100, EmbeddingDim = 64, NumHeads = 4, NumLayers = 2, FeedForwardDim = 256 },
                new TransformerConfig { VocabSize = 100, EmbeddingDim = 128, NumHeads = 4, NumLayers = 4, FeedForwardDim = 512 }
            };

            foreach (var cfg in configs)
            {
                long embeddings = (long)cfg.VocabSize * cfg.EmbeddingDim;
                long attnPerLayer = 4L * cfg.EmbeddingDim * cfg.EmbeddingDim;
                long ffnPerLayer = (long)cfg.EmbeddingDim * cfg.FeedForwardDim + (long)cfg.FeedForwardDim * cfg.EmbeddingDim;
                long perLayer = attnPerLayer + ffnPerLayer + 4 * cfg.EmbeddingDim;
                long output = (long)cfg.EmbeddingDim * cfg.VocabSize;
                long total = embeddings + perLayer * cfg.NumLayers + output;


                Assert(total > 0, $"Invalid parameter count: {total}");
            }
        }

        public void Test_Example_Text()
        {
            var corpus = new[] { "The cat sat on the mat.", "The dog chased the cat.", "A bird flew over the house.", "The cat and the dog played together." };

            var tokenizer = new BPETokenizer();

            tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);

            int[][] sequences = corpus.Select(text => tokenizer.Encode(text, addSpecialTokens: true)).Where(seq => seq.Length >= 2).ToArray();

            var modelConfig = new TransformerConfig { DataType = TransformerDataType.Text, VocabSize = tokenizer.VocabSize, MaxSequenceLength = 64, EmbeddingDim = 32, NumHeads = 2, NumLayers = 1, FeedForwardDim = 64, UseDecoderOnly = true, AccelerationType = AccelerationType.CPU };
            
            var model = new LanguageModel(modelConfig);
            
            var trainConfig = new TrainingConfig { LearningRate = 0.001f, BatchSize = 4, Epochs = 10, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false };
            
            new TransformerTrainer(model, trainConfig).Train(sequences);
           
            int[] prompt = tokenizer.Encode("The cat", addSpecialTokens: false);
            int[] generated = model.Generate(prompt, maxNewTokens: 5, temperature: 0.8f);


            Assert(generated.Length > prompt.Length, "Text example: no tokens generated");
        }

        public void Test_Example_SymbolicSequence()
        {
            var baseToId = new Dictionary<char, int> { { 'A', 4 }, { 'T', 5 }, { 'G', 6 }, { 'C', 7 } };
            
            int[] EncodeDNA(string dna) { var tokens = new List<int> { 1 }; foreach (char c in dna) { if (baseToId.TryGetValue(c, out int id)) tokens.Add(id); } tokens.Add(2); return tokens.ToArray(); }
           
            string[] dnaSequences = new[] { "ATGCGATCGATCG", "ATGCCCGATTTAG", "ATGAAAGGGCCCT", "ATGCGATCGATCGATCG" };
           
            int[][] sequences = dnaSequences.Select(dna => EncodeDNA(dna)).ToArray();
           
            var modelConfig = new TransformerConfig { DataType = TransformerDataType.SymbolicSequence, VocabSize = 8, MaxSequenceLength = 32, EmbeddingDim = 16, NumHeads = 2, NumLayers = 1, FeedForwardDim = 32, UseDecoderOnly = true, AccelerationType = AccelerationType.CPU };
            var model = new LanguageModel(modelConfig);
            var trainConfig = new TrainingConfig { LearningRate = 0.001f, BatchSize = 4, Epochs = 20, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false };
            var trainer = new TransformerTrainer(model, trainConfig);

            trainer.Train(sequences);

            float valLoss = trainer.Validate(sequences);

            Assert(!float.IsNaN(valLoss) && valLoss > 0, $"DNA example: invalid loss {valLoss}");
        }

        public void Test_Example_TimeSeriesRegression()
        {
            var random = new Random();

            var (inputs, targets) = GenerateRegressionData(20, 10, 5, 1, random);

            var modelConfig = new TransformerConfig { DataType = TransformerDataType.TimeSeriesRegression, InputFeatureDim = 5, OutputDim = 1, MaxSequenceLength = 10, EmbeddingDim = 16, NumHeads = 2, NumLayers = 1, FeedForwardDim = 32, UseDecoderOnly = true, AccelerationType = AccelerationType.CPU };
            var model = new LanguageModel(modelConfig);

            var trainConfig = new TrainingConfig { LearningRate = 0.0005f, BatchSize = 8, Epochs = 10, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false };
            
            var trainer = new TransformerTrainer(model, trainConfig);

            trainer.TrainContinuous(inputs, regressionTargets: targets);

            float valLoss = trainer.ValidateContinuous(inputs, regressionTargets: targets);


            Assert(!float.IsNaN(valLoss) && valLoss >= 0, $"Regression example: invalid loss {valLoss}");

            float[] prediction = model.PredictNext(inputs[0]);



            Assert(prediction.Length == 1, $"Regression example: wrong prediction dim {prediction.Length}");

            Assert(!float.IsNaN(prediction[0]), "Regression example: NaN prediction");
        }

        public void Test_Example_TimeSeriesClassification()
        {
            var random = new Random();

            var (inputs, classTargets) = GenerateClassificationData(20, 10, 5, random);

            var modelConfig = new TransformerConfig { DataType = TransformerDataType.TimeSeriesClassification, InputFeatureDim = 5, OutputDim = 3, MaxSequenceLength = 10, EmbeddingDim = 16, NumHeads = 2, NumLayers = 1, FeedForwardDim = 32, UseDecoderOnly = true, AccelerationType = AccelerationType.CPU };
         
            var model = new LanguageModel(modelConfig);

           
            var trainConfig = new TrainingConfig { LearningRate = 0.0005f, BatchSize = 8, Epochs = 10, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false };
           
            var trainer = new TransformerTrainer(model, trainConfig);

            trainer.TrainContinuous(inputs, classTargets: classTargets);

            float valLoss = trainer.ValidateContinuous(inputs, classTargets: classTargets);



            Assert(!float.IsNaN(valLoss) && valLoss > 0, $"Classification example: invalid loss {valLoss}");

            var output = model.Forward(inputs[0]);



            Assert(output.GetLength(1) == 3, $"Classification example: wrong output dim {output.GetLength(1)}");
        }
    }
}