using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;

namespace CallaghanDev.ML
{
    public class Tests
    {
        private const float EPSILON = 1e-3f;
        // private const float TOLERANCE = 0.05f;
        // private const float ABS_TOLERANCE = 1e-4f;

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

                // Cross attention multimodal tests (all sorts)
                (Test_CrossAttention_BasicExample, "CrossAttention: Basic Example"),
                (Test_CrossAttention_FrozenTextEncoder, "CrossAttention: Frozen Text Encoder"),
                (Test_CrossAttention_ModelConstruction_NoError, "CrossAttention: Model Construction"),
                (Test_CrossAttention_ForwardPass_OutputShape, "CrossAttention: Forward Output Shape"),
                (Test_CrossAttention_ForwardPass_NoNaN, "CrossAttention: Forward No NaN"),
                (Test_CrossAttention_PredictNext_OutputShape, "CrossAttention: PredictNext Shape"),
                (Test_CrossAttention_ConfidenceHead_SigmoidRange, "CrossAttention: Confidence In [0,1]"),
                (Test_CrossAttention_NoConfidenceHead, "CrossAttention: No Confidence Head"),
                (Test_CrossAttention_LossDecreases, "CrossAttention: Loss Decreases"),
                (Test_CrossAttention_ValidationLoss_Computable, "CrossAttention: Validation Computable"),
                (Test_CrossAttention_DifferentTextLengths_NoError, "CrossAttention: Varying Text Lengths"),
                (Test_CrossAttention_TextLongerThanPrice_NoError, "CrossAttention: Text > Price SeqLen"),
                (Test_CrossAttention_TextShorterThanPrice_NoError, "CrossAttention: Text < Price SeqLen"),
                (Test_CrossAttention_FrozenEncoder_WeightsUnchanged, "CrossAttention: Frozen Weights Unchanged"),
                (Test_CrossAttention_UnfrozenEncoder_WeightsChange, "CrossAttention: Unfrozen Weights Change"),
                (Test_CrossAttention_AllPriceParams_ReceiveGradients, "CrossAttention: Price Params Get Gradients"),
                (Test_CrossAttention_GradientClipping_NoNaN, "CrossAttention: Clipping Prevents NaN"),
                (Test_CrossAttention_DeterministicForward, "CrossAttention: Deterministic Forward"),
                (Test_CrossAttention_LearningRateDecay, "CrossAttention: LR Decay"),
                (Test_CrossAttention_SingleSampleOverfit, "CrossAttention: Single Sample Overfit"),
                (Test_CrossAttention_EmbDimMismatch_Throws, "CrossAttention: EmbDim Mismatch Throws"),
                (Test_CrossAttention_MinimalConfig, "CrossAttention: Minimal (1 layer 1 head)"),


                //Accelleration tests for transformer
                (Test_Std_CPU_TrainGenerate, "Std Transformer CPU: Train+Generate"),
                (Test_Std_MultiThread_TrainGenerate, "Std Transformer MultiThread: Train+Generate"),
                (Test_Std_CPU_LossDecreases, "Std Transformer CPU: Loss Decreases"),
                (Test_Std_MultiThread_LossDecreases, "Std Transformer MultiThread: Loss Decreases"),
                (Test_Std_CPU_Regression, "Std Transformer CPU: Regression"),
                (Test_Std_MultiThread_Regression, "Std Transformer MultiThread: Regression"),

                // Accelleration backend tests for cross attention multimodal
                (Test_CrossAttn_CPU_TrainPredict, "CrossAttention CPU: Train+Predict"),
                (Test_CrossAttn_MultiThread_TrainPredict, "CrossAttention MultiThread: Train+Predict"),
                (Test_CrossAttn_CPU_LossDecreases, "CrossAttention CPU: Loss Decreases"),
                (Test_CrossAttn_MultiThread_LossDecreases, "CrossAttention MultiThread: Loss Decreases"),

                // Cross backend consistancy tests
                (Test_MHA_Forward_CPUvsMultiThread, "MHA Forward: CPU vs MultiThread Consistency"),
                (Test_MHA_Backward_CPUvsMultiThread, "MHA Backward: CPU vs MultiThread Consistency"),
                (Test_CrossAttn_Forward_CPUvsMultiThread, "CrossAttn Forward: CPU vs MultiThread Consistency"),


                // Save and Load all models
                (Test_SaveLoad_Text_ExactWeightsAndForward, "SaveLoad: Text (discrete tokens)"),
                (Test_SaveLoad_SymbolicSequence_ExactWeightsAndForward, "SaveLoad: SymbolicSequence (DNA)"),
                (Test_SaveLoad_TimeSeriesRegression_ExactWeightsAndForward, "SaveLoad: TimeSeries Regression"),
                (Test_SaveLoad_TimeSeriesClassification_ExactWeightsAndForward, "SaveLoad: TimeSeries Classification"),
                (Test_SaveLoad_CrossAttentionMultimodal_ExactWeightsAndForward, "SaveLoad: CrossAttention Multimodal"),
                (Test_SaveLoad_CrossAttention_NoConfidenceHead, "SaveLoad: CrossAttention No Confidence"),
                (Test_SaveLoad_DoubleSaveLoad_RoundTrip, "SaveLoad: Double Round-Trip"),

                // Testing cross attention where text is optional
                (Test_CrossAttention_PriceOnly_ForwardNoError, "CrossAttention: Price-Only Forward"),
                (Test_CrossAttention_PriceOnly_PredictNext, "CrossAttention: Price-Only PredictNext"),
                (Test_CrossAttention_PriceOnly_TrainAndLossDecreases, "CrossAttention: Price-Only Loss Decreases"),
                (Test_CrossAttention_MixedBatch_SomeTextSomeNull, "CrossAttention: Mixed Batch Text+Null"),
                (Test_CrossAttention_EmptyTextArray_TreatedAsNull, "CrossAttention: Empty Text = Null"),
                (Test_CrossAttention_TextVsNoText_DifferentOutputs, "CrossAttention: Text vs NoText Differ"),
                (Test_CrossAttention_PriceOnly_Deterministic, "CrossAttention: Price-Only Deterministic"),
                (Test_CrossAttention_PriceOnly_SingleSampleOverfit, "CrossAttention: Price-Only Overfit"),

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


        private (BPETokenizer tokenizer, int[][] textSeqs, float[][,] priceInputs, float[][,] priceTargets) CreateCrossAttentionTestData(int numSamples = 10, int priceSeqLen = 8, int inputFeatures = 5, int outputDim = 5, int seed = 42)
        {
            var random = new Random(seed);

            string[] corpus = new[]
            {
                "stock price rose sharply today",
                "market crashed due to earnings miss",
                "bullish sentiment on tech sector",
                "bearish outlook for energy stocks",
                "neutral trading day with low volume"
            };
            var tokenizer = new BPETokenizer();

            tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);

            var textSeqs = new int[numSamples][];
            var priceInputs = new float[numSamples][,];
            var priceTargets = new float[numSamples][,];

            for (int s = 0; s < numSamples; s++)
            {

                textSeqs[s] = tokenizer.Encode(corpus[random.Next(corpus.Length)], addSpecialTokens: true);
                priceInputs[s] = new float[priceSeqLen, inputFeatures];
                priceTargets[s] = new float[priceSeqLen, outputDim];
                float basePrice = 100f + (float)(random.NextDouble() * 50);

                for (int t = 0; t < priceSeqLen; t++)
                {
                    for (int f = 0; f < inputFeatures; f++)
                    {
                        priceInputs[s][t, f] = (basePrice + (float)(random.NextDouble() - 0.5) * 10f) / 200f;
                    }
                    for (int f = 0; f < outputDim; f++)
                    {
                        priceTargets[s][t, f] = (basePrice + (float)(random.NextDouble() - 0.5) * 10f) / 200f;
                    }
                    basePrice += (float)(random.NextDouble() - 0.48) * 2f;
                }
            }
            return (tokenizer, textSeqs, priceInputs, priceTargets);
        }

        private (MultimodalTransformerConfig model, TrainingConfig training)
            CreateSmallCrossAttentionConfig(
                int textVocabSize,
                AccelerationType accelType = AccelerationType.CPU,
                int embDim = 16,
                int numHeads = 2,
                int numLayers = 1,
                int ffnDim = 32, int inputFeatures = 5, int outputDim = 5, int priceSeqLen = 10,  bool useConfidence = true, bool freezeTextEncoder = false)
        {
            var model = new MultimodalTransformerConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = textVocabSize,
                    MaxSequenceLength = 32,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    Freeze = freezeTextEncoder
                },

                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = inputFeatures,
                    MaxSequenceLength = priceSeqLen + 2,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim
                },

                Output = new OutputHeadConfig
                {
                    OutputDim = outputDim,
                    UseConfidenceHead = useConfidence
                },

                Runtime = new RuntimeConfig
                {
                    FFNActivationType = ActivationType.Relu,
                    AccelerationType = accelType
                },

                Regularization = new RegularizationConfig
                {
                    L2RegulationLamda = 0f,
                    GradientClippingThreshold = 1.0f
                },

                RequireSharedCrossAttentionEmbeddingDim = true
            };

            var training = TrainingConfig.ForMultimodalModel();

            // Optional: override defaults for "small" config
            training.BatchSize = 4;
            training.Epochs = 10;
            training.ConfidenceLossWeight = useConfidence ? 0.1f : 0f;

            return (model, training);
        }

        private bool MatricesApproxEqual(float[,] a, float[,] b, float tol = 1e-4f)
        {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
            {
                return false;
            }

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {

                    if (MathF.Abs(a[i, j] - b[i, j]) > tol)
                    {
                        return false;
                    }
                }
            }
       
            return true;
        }

        public void Test_CrossAttention_BasicExample()
        {

            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 20, priceSeqLen: 10);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 32, numHeads: 2, numLayers: 2, ffnDim: 64, priceSeqLen: 10);


            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var trainConfig = new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 4, Epochs = 10, UseGradientClipping = true, GradientClipThreshold = 1.0f, ConfidenceLossWeight = 0.1f, Verbose = false };
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, trainConfig);


            trainer.Train(textSeqs, priceInputs, priceTargets);

            float valLoss = trainer.Validate(textSeqs, priceInputs, priceTargets);

            Assert(!float.IsNaN(valLoss) && valLoss >= 0, $"BasicExample validation loss invalid: {valLoss}");

            var (prediction, confidence) = model.PredictNext(textSeqs[0], priceInputs[0]);


            Assert(prediction.Length == 5, $"Prediction dim wrong: {prediction.Length}");
            Assert(!float.IsNaN(confidence) && confidence >= 0f && confidence <= 1f, $"Confidence invalid: {confidence}");
        }

        public void Test_CrossAttention_FrozenTextEncoder()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 10, priceSeqLen: 6, inputFeatures: 3, outputDim: 3);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32,inputFeatures: 3, outputDim: 3, priceSeqLen: 6, useConfidence: false, freezeTextEncoder: true);
            
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            float textEmbBefore = model.TextTokenEmbedding[1, 0];

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 10, Verbose = false });

            trainer.Train(textSeqs, priceInputs, priceTargets);

            float textEmbAfter = model.TextTokenEmbedding[1, 0];
            Assert(textEmbBefore == textEmbAfter, $"Text encoder changed despite being frozen: {textEmbBefore} vs {textEmbAfter}");

            float valLoss = trainer.Validate(textSeqs, priceInputs, priceTargets);
            Assert(!float.IsNaN(valLoss), $"Validation loss NaN after frozen encoder training");
        }

        public void Test_CrossAttention_ModelConstruction_NoError()
        {
            var config = CreateSmallCrossAttentionConfig(textVocabSize: 50);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            Assert(model.TextBlocks.Length == 1, $"Wrong text blocks: {model.TextBlocks.Length}");
            Assert(model.PriceBlocks.Length == 1, $"Wrong price blocks: {model.PriceBlocks.Length}");
            Assert(model.OutputProjection.GetLength(0) == 5, $"Wrong output rows: {model.OutputProjection.GetLength(0)}");
        }

        public void Test_CrossAttention_ForwardPass_OutputShape()
        {
            var (tokenizer, textSeqs, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 8);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 8);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var (predictions, confidence) = model.Forward(textSeqs[0], priceInputs[0]);

            Assert(predictions.GetLength(0) == 8, $"Prediction seqLen: {predictions.GetLength(0)} expected 8");
            Assert(predictions.GetLength(1) == 5, $"Prediction dim: {predictions.GetLength(1)} expected 5");
            Assert(confidence != null && confidence.GetLength(0) == 8, "Confidence shape wrong");
            Assert(confidence.GetLength(1) == 1, $"Confidence cols: {confidence.GetLength(1)} expected 1");
        }

        public void Test_CrossAttention_ForwardPass_NoNaN()
        {
            var (tokenizer, textSeqs, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var (predictions, confidence) = model.Forward(textSeqs[0], priceInputs[0]);

            for (int i = 0; i < predictions.GetLength(0); i++)
            {
                for (int j = 0; j < predictions.GetLength(1); j++)
                {
                    Assert(!float.IsNaN(predictions[i, j]) && !float.IsInfinity(predictions[i, j]), $"NaN/Inf at predictions[{i},{j}]");
                }
            }
            for (int i = 0; i < confidence.GetLength(0); i++)
            {
                Assert(!float.IsNaN(confidence[i, 0]) && !float.IsInfinity(confidence[i, 0]), $"NaN/Inf at confidence[{i}]");
            }
        }

        public void Test_CrossAttention_PredictNext_OutputShape()
        {
            var (tokenizer, textSeqs, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var (prediction, confidence) = model.PredictNext(textSeqs[0], priceInputs[0]);
            Assert(prediction.Length == 5, $"PredictNext dim: {prediction.Length}");
            foreach (var v in prediction)
                Assert(!float.IsNaN(v), "PredictNext has NaN value");
        }

        public void Test_CrossAttention_ConfidenceHead_SigmoidRange()
        {
            var (tokenizer, textSeqs, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 3, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6, useConfidence: true);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            for (int s = 0; s < 3; s++)
            {
                var (_, conf) = model.Forward(textSeqs[s], priceInputs[s]);
                for (int i = 0; i < conf.GetLength(0); i++)
                    Assert(conf[i, 0] >= 0f && conf[i, 0] <= 1f, $"Confidence {conf[i, 0]} outside [0,1]");
            }
        }

        public void Test_CrossAttention_NoConfidenceHead()
        {
            var (tokenizer, textSeqs, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6, useConfidence: false);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var (predictions, confidence) = model.Forward(textSeqs[0], priceInputs[0]);
            Assert(confidence == null, "Confidence should be null when UseConfidenceHead=false");
            Assert(predictions.GetLength(1) == 5, "Predictions dim wrong");
        }

        public void Test_CrossAttention_LossDecreases()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 10, priceSeqLen: 8);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 16, priceSeqLen: 8);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
            float lossBefore = trainer.Validate(textSeqs, priceInputs, priceTargets);
            trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false });
            trainer.Train(textSeqs, priceInputs, priceTargets);
            float lossAfter = trainer.Validate(textSeqs, priceInputs, priceTargets);
            Assert(lossAfter < lossBefore, $"Loss did not decrease: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_CrossAttention_ValidationLoss_Computable()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 5, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { Verbose = false });
            float valLoss = trainer.Validate(textSeqs, priceInputs, priceTargets);
            Assert(!float.IsNaN(valLoss) && !float.IsInfinity(valLoss) && valLoss >= 0, $"Validation loss invalid: {valLoss}");
        }

        public void Test_CrossAttention_DifferentTextLengths_NoError()
        {
            string[] corpus = { "hi", "stock price rose sharply today because of earnings", "bullish", "market crashed due to bad earnings miss report" };

            var tokenizer = new BPETokenizer();

            tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);

            var textSeqs = corpus.Select(t => tokenizer.Encode(t, addSpecialTokens: true)).ToArray();
            var random = new Random(42);

            var priceInputs = new float[4][,];
            var priceTargets = new float[4][,];

            for (int i = 0; i < 4; i++)
            {
                priceInputs[i] = new float[6, 5];
                priceTargets[i] = new float[6, 5];

                for (int t = 0; t < 6; t++)
                {
                    for (int f = 0; f < 5; f++)
                    {
                        { priceInputs[i][t, f] = (float)random.NextDouble(); priceTargets[i][t, f] = (float)random.NextDouble(); }
                    }
                }
            }

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 2, Epochs = 3, Verbose = false });

            trainer.Train(textSeqs, priceInputs, priceTargets);
        }

        public void Test_CrossAttention_TextLongerThanPrice_NoError()
        {
            // Text ~15 tokens, Price 4 timesteps — cross-attention Q(4) x K/V(~15)
            string[] corpus = { "this is a much longer text sequence that has many more tokens than the price sequence" };
            var tokenizer = new BPETokenizer();
            tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);
            var textSeqs = new int[][] { tokenizer.Encode(corpus[0], addSpecialTokens: true) };
            var random = new Random(42);
            var priceInputs = new float[1][,] { new float[4, 5] };
            for (int t = 0; t < 4; t++) for (int f = 0; f < 5; f++) priceInputs[0][t, f] = (float)random.NextDouble();
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var (predictions, _) = model.Forward(textSeqs[0], priceInputs[0]);
            Assert(predictions.GetLength(0) == 4, $"Output should have 4 rows, got {predictions.GetLength(0)}");
        }

        public void Test_CrossAttention_TextShorterThanPrice_NoError()
        {
            string[] corpus = { "hi" };
            var tokenizer = new BPETokenizer();
            tokenizer.Train(corpus, vocabSize: 50, minFrequency: 1);
            var textSeqs = new int[][] { tokenizer.Encode("hi", addSpecialTokens: true) };
            var random = new Random(42);
            var priceInputs = new float[1][,] { new float[10, 5] };
            for (int t = 0; t < 10; t++) for (int f = 0; f < 5; f++) priceInputs[0][t, f] = (float)random.NextDouble();
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 12);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var (predictions, _) = model.Forward(textSeqs[0], priceInputs[0]);
            Assert(predictions.GetLength(0) == 10, $"Output should have 10 rows, got {predictions.GetLength(0)}");
        }

        public void Test_CrossAttention_FrozenEncoder_WeightsUnchanged()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 5, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6, freezeTextEncoder: true);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var embBefore = (float[,])model.TextTokenEmbedding.Clone();
            var wqBefore = (float[,])model.TextBlocks[0].Attention.WQ.Clone();
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false });
            trainer.Train(textSeqs, priceInputs, priceTargets);
            Assert(!MatrixChanged(embBefore, model.TextTokenEmbedding), "Frozen text embedding changed");
            Assert(!MatrixChanged(wqBefore, model.TextBlocks[0].Attention.WQ), "Frozen text WQ changed");
        }

        public void Test_CrossAttention_UnfrozenEncoder_WeightsChange()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 5, priceSeqLen: 6);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6, freezeTextEncoder: false);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var embBefore = (float[,])model.TextTokenEmbedding.Clone();
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, UseGradientClipping = false, Verbose = false });
            
            trainer.Train(textSeqs, priceInputs, priceTargets);
            
            Assert(MatrixChanged(embBefore, model.TextTokenEmbedding), "Unfrozen text embedding did not change");
        }

        public void Test_CrossAttention_AllPriceParams_ReceiveGradients()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 5, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var projBefore = (float[,])model.PriceInputProjection.Clone();
            var outProjBefore = (float[,])model.OutputProjection.Clone();
            var selfWqBefore = (float[,])model.PriceBlocks[0].SelfAttention.WQ.Clone();
            var crossWqBefore = (float[,])model.PriceBlocks[0].CrossAttention.WQ.Clone();

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, UseGradientClipping = false, Verbose = false });
            
            trainer.Train(textSeqs, priceInputs, priceTargets);

            Assert(MatrixChanged(projBefore, model.PriceInputProjection), "PriceInputProjection did not change");
            Assert(MatrixChanged(outProjBefore, model.OutputProjection), "OutputProjection did not change");
            Assert(MatrixChanged(selfWqBefore, model.PriceBlocks[0].SelfAttention.WQ), "Self-attention WQ did not change");
            Assert(MatrixChanged(crossWqBefore, model.PriceBlocks[0].CrossAttention.WQ), "Cross-attention WQ did not change");
        }

        public void Test_CrossAttention_GradientClipping_NoNaN()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 5, priceSeqLen: 6);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.1f, BatchSize = 5, Epochs = 5, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false });
            
            trainer.Train(textSeqs, priceInputs, priceTargets);

            var (predictions, _) = model.Forward(textSeqs[0], priceInputs[0]);

            for (int i = 0; i < predictions.GetLength(0); i++)
            {
                for (int j = 0; j < predictions.GetLength(1); j++)
                {
                    Assert(!float.IsNaN(predictions[i, j]) && !float.IsInfinity(predictions[i, j]), $"NaN/Inf at [{i},{j}] after high-LR training with clipping");
                }
            }
        }

        public void Test_CrossAttention_DeterministicForward()
        {
            var (tokenizer, textSeqs, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var (pred1, _) = model.Forward(textSeqs[0], priceInputs[0]);
            var (pred2, _) = model.Forward(textSeqs[0], priceInputs[0]);

            for (int i = 0; i < pred1.GetLength(0); i++)
            {
                for (int j = 0; j < pred1.GetLength(1); j++)
                {
                    Assert(pred1[i, j] == pred2[i, j], $"Non-deterministic at [{i},{j}]: {pred1[i, j]} vs {pred2[i, j]}");
                }
            }
        }

        public void Test_CrossAttention_LearningRateDecay()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 5, priceSeqLen: 6);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, UseLearningRateDecay = true, LearningRateDecay = 0.5f, Verbose = false });
            
            trainer.Train(textSeqs, priceInputs, priceTargets);

            var (predictions, _) = model.Forward(textSeqs[0], priceInputs[0]);

            for (int i = 0; i < predictions.GetLength(0); i++)
            {
                for (int j = 0; j < predictions.GetLength(1); j++)
                {
                    Assert(!float.IsNaN(predictions[i, j]), $"NaN after LR decay training at [{i},{j}]");
                }
            }
        }

        public void Test_CrossAttention_SingleSampleOverfit()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 32, numHeads: 2, numLayers: 2, ffnDim: 64, priceSeqLen: 6, useConfidence: false);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 100, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false });
            
            float lossBefore = trainer.Validate(textSeqs, priceInputs, priceTargets);



            trainer.Train(textSeqs, priceInputs, priceTargets);



            float lossAfter = trainer.Validate(textSeqs, priceInputs, priceTargets);


            Assert(lossAfter < lossBefore * 0.5f, $"Failed to significantly overfit single sample: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_CrossAttention_EmbDimMismatch_Throws()
        {
            bool threw = false;
            try
            {
                var config = new Config
                {
                    TextVocabSize = 50,
                    TextEmbeddingDim = 16,
                    TextNumHeads = 2,
                    TextNumLayers = 1,
                    TextFeedForwardDim = 32,
                    PriceInputFeatureDim = 5,
                    PriceEmbeddingDim = 32, // Mismatch!
                    PriceNumHeads = 2,
                    PriceNumLayers = 1,
                    PriceFeedForwardDim = 32,
                    OutputDim = 5,
                    AccelerationType = AccelerationType.CPU
                };
                new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            }
            catch (ArgumentException) { threw = true; }
            Assert(threw, "Expected ArgumentException when TextEmbeddingDim != PriceEmbeddingDim");
        }

        public void Test_CrossAttention_MinimalConfig()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 3, priceSeqLen: 4, inputFeatures: 2, outputDim: 2);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 4, numHeads: 1, numLayers: 1, ffnDim: 8, inputFeatures: 2, outputDim: 2, priceSeqLen: 4, useConfidence: false);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 3, Epochs = 5, Verbose = false });

            trainer.Train(textSeqs, priceInputs, priceTargets);

            float valLoss = trainer.Validate(textSeqs, priceInputs, priceTargets);

            Assert(!float.IsNaN(valLoss), $"Minimal config produced NaN loss: {valLoss}");
        }



        private void RunStdTransformerTrainGenerate(AccelerationType accelType)
        {
            var trainingTexts = new[] { "the cat sat on the mat", "the dog ran in the park", "a bird flew over the tree" };
            var tokenizer = new BPETokenizer();
            tokenizer.Train(trainingTexts, vocabSize: 500, minFrequency: 1);
            var sequences = trainingTexts.Select(t => tokenizer.Encode(t, addSpecialTokens: true)).ToArray();
            var config = new TransformerConfig { VocabSize = tokenizer.VocabSize, MaxSequenceLength = 32, EmbeddingDim = 32, NumHeads = 2, NumLayers = 2, FeedForwardDim = 64, AccelerationType = accelType };
            var model = new LanguageModel(config, new Random(42));
            var trainConfig = new TrainingConfig { LearningRate = 0.001f, BatchSize = 3, Epochs = 5, Verbose = false };
            new TransformerTrainer(model, trainConfig).Train(sequences);
            var prompt = tokenizer.Encode("the cat", addSpecialTokens: false);
            var generated = model.Generate(prompt, maxNewTokens: 5, temperature: 1.0f);
            Assert(generated.Length > prompt.Length, $"No tokens generated with {accelType}");
        }

        public void Test_Std_CPU_TrainGenerate() => RunStdTransformerTrainGenerate(AccelerationType.CPU);
        public void Test_Std_MultiThread_TrainGenerate() => RunStdTransformerTrainGenerate(AccelerationType.MultiThreadCPU);

        private void RunStdTransformerLossDecreases(AccelerationType accelType)
        {
            var config = new TransformerConfig { DataType = TransformerDataType.Text, VocabSize = 10, MaxSequenceLength = 16, EmbeddingDim = 8, NumHeads = 2, NumLayers = 1, FeedForwardDim = 16, AccelerationType = accelType, UseDecoderOnly = true, L2RegulationLamda = 0f };
            var model = new LanguageModel(config, new Random(42));
            int[] sequence = { 1, 2, 3, 4, 5 };
            int[] input = { 1, 2, 3, 4 }; int[] target = { 2, 3, 4, 5 };

            // ComputeLoss needs CPU model for logits; compute manually
            var logitsBefore = model.Forward(input);
            float lossBefore = ComputeCELoss(logitsBefore, target, config.VocabSize);

            var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 10, UseGradientClipping = true, GradientClipThreshold = 5.0f, Verbose = false };
            new TransformerTrainer(model, tc).Train(new[] { sequence });

            var logitsAfter = model.Forward(input);
            float lossAfter = ComputeCELoss(logitsAfter, target, config.VocabSize);
            Assert(lossAfter < lossBefore, $"Loss did not decrease with {accelType}: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        private float ComputeCELoss(float[,] logits, int[] target, int vocabSize)
        {
            float loss = 0;
            for (int i = 0; i < Math.Min(logits.GetLength(0), target.Length); i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < vocabSize; j++) max = Math.Max(max, logits[i, j]);
                float sum = 0;
                for (int j = 0; j < vocabSize; j++) sum += MathF.Exp(logits[i, j] - max);
                float prob = MathF.Exp(logits[i, target[i]] - max) / sum;
                loss -= MathF.Log(prob + 1e-10f);
            }
            return loss / Math.Min(logits.GetLength(0), target.Length);
        }

        public void Test_Std_CPU_LossDecreases() => RunStdTransformerLossDecreases(AccelerationType.CPU);
        public void Test_Std_MultiThread_LossDecreases() => RunStdTransformerLossDecreases(AccelerationType.MultiThreadCPU);

        private void RunStdTransformerRegression(AccelerationType accelType)
        {
            var config = new TransformerConfig { DataType = TransformerDataType.TimeSeriesRegression, InputFeatureDim = 3, OutputDim = 1, MaxSequenceLength = 16, EmbeddingDim = 8, NumHeads = 2, NumLayers = 1, FeedForwardDim = 16, AccelerationType = accelType, UseDecoderOnly = true, L2RegulationLamda = 0f };
            var model = new LanguageModel(config, new Random(42));
            var rng = new Random(42);
            var inputs = new float[5][,]; var targets = new float[5][,];
            for (int s = 0; s < 5; s++)
            {
                inputs[s] = new float[6, 3]; targets[s] = new float[6, 1];
                for (int i = 0; i < 6; i++) { for (int j = 0; j < 3; j++) inputs[s][i, j] = (float)rng.NextDouble(); targets[s][i, 0] = inputs[s][i, 0] * 0.5f + 0.1f; }
            }
            var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false };
            var trainer = new TransformerTrainer(model, tc);
            float lossBefore = trainer.ValidateContinuous(inputs, regressionTargets: targets);
            tc.Epochs = 20;
            trainer.TrainContinuous(inputs, regressionTargets: targets);
            float lossAfter = trainer.ValidateContinuous(inputs, regressionTargets: targets);
            Assert(lossAfter < lossBefore, $"Regression loss did not decrease with {accelType}: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_Std_CPU_Regression() => RunStdTransformerRegression(AccelerationType.CPU);
        public void Test_Std_MultiThread_Regression() => RunStdTransformerRegression(AccelerationType.MultiThreadCPU);

        private void RunCrossAttnTrainPredict(AccelerationType accelType)
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 10, priceSeqLen: 6);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, accelType: accelType, priceSeqLen: 6);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));
            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 5, Verbose = false });

            trainer.Train(textSeqs, priceInputs, priceTargets);

            var (prediction, confidence) = model.PredictNext(textSeqs[0], priceInputs[0]);

            Assert(prediction.Length == 5, $"Wrong prediction dim with {accelType}: {prediction.Length}");

            foreach (var v in prediction)
            {
                Assert(!float.IsNaN(v), $"NaN in prediction with {accelType}");
            }

            Assert(!float.IsNaN(confidence) && confidence >= 0f && confidence <= 1f, $"Invalid confidence with {accelType}: {confidence}");
        }

        public void Test_CrossAttn_CPU_TrainPredict() => RunCrossAttnTrainPredict(AccelerationType.CPU);
        public void Test_CrossAttn_MultiThread_TrainPredict() => RunCrossAttnTrainPredict(AccelerationType.MultiThreadCPU);

        private void RunCrossAttnLossDecreases(AccelerationType accelType)
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 10, priceSeqLen: 6);

            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, accelType: accelType, priceSeqLen: 6);

            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });

            float lossBefore = trainer.Validate(textSeqs, priceInputs, priceTargets);

            trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false });

            trainer.Train(textSeqs, priceInputs, priceTargets);

            float lossAfter = trainer.Validate(textSeqs, priceInputs, priceTargets);

            Assert(lossAfter < lossBefore, $"CrossAttn loss did not decrease with {accelType}: before={lossBefore:F6}, after={lossAfter:F6}");
        }

        public void Test_CrossAttn_CPU_LossDecreases() => RunCrossAttnLossDecreases(AccelerationType.CPU);
        public void Test_CrossAttn_MultiThread_LossDecreases() => RunCrossAttnLossDecreases(AccelerationType.MultiThreadCPU);

        public void Test_MHA_Forward_CPUvsMultiThread()
        {

            var cpu = new AccelerationCPU();
            var mt = new AccelerationMutliThreadCPU();

            var rng = new Random(42);
            int seqLen = 6, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);

            var Q = new float[seqLen, embDim];
            var K = new float[seqLen, embDim];
            var V = new float[seqLen, embDim];

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embDim; j++)
                {
                    Q[i, j] = (float)(rng.NextDouble() - 0.5); K[i, j] = (float)(rng.NextDouble() - 0.5); V[i, j] = (float)(rng.NextDouble() - 0.5);
                }
            }

            var resultCPU = cpu.MultiHeadAttentionForward(Q, K, V, numHeads, scale, null);
            var resultMT = mt.MultiHeadAttentionForward(Q, K, V, numHeads, scale, null);
            Assert(MatricesApproxEqual(resultCPU, resultMT, 1e-5f), "MHA Forward: CPU vs MultiThread results differ");
        }

        public void Test_MHA_Backward_CPUvsMultiThread()
        {
            var cpu = new AccelerationCPU();
            var mt = new AccelerationMutliThreadCPU();
            var rng = new Random(42);
            int seqLen = 6, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = new float[seqLen, embDim];
            var K = new float[seqLen, embDim];
            var V = new float[seqLen, embDim];
            var dConcat = new float[seqLen, embDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embDim; j++)
                { Q[i, j] = (float)(rng.NextDouble() - 0.5); K[i, j] = (float)(rng.NextDouble() - 0.5); V[i, j] = (float)(rng.NextDouble() - 0.5); dConcat[i, j] = (float)(rng.NextDouble() - 0.5); }

            var (dQ_cpu, dK_cpu, dV_cpu) = cpu.MultiHeadAttentionBackward(Q, K, V, dConcat, numHeads, scale, true);
            var (dQ_mt, dK_mt, dV_mt) = mt.MultiHeadAttentionBackward(Q, K, V, dConcat, numHeads, scale, true);
            Assert(MatricesApproxEqual(dQ_cpu, dQ_mt, 1e-5f), "MHA Backward dQ: CPU vs MultiThread differ");
            Assert(MatricesApproxEqual(dK_cpu, dK_mt, 1e-5f), "MHA Backward dK: CPU vs MultiThread differ");
            Assert(MatricesApproxEqual(dV_cpu, dV_mt, 1e-5f), "MHA Backward dV: CPU vs MultiThread differ");
        }

        public void Test_CrossAttn_Forward_CPUvsMultiThread()
        {
            var cpu = new AccelerationCPU();
            var mt = new AccelerationMutliThreadCPU();

            var rng = new Random(42);

            int seqLenQ = 5, seqLenK = 8, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);

            var Q = new float[seqLenQ, embDim];
            var K = new float[seqLenK, embDim];
            var V = new float[seqLenK, embDim];


            for (int i = 0; i < seqLenQ; i++)
            {
                for (int j = 0; j < embDim; j++)
                {
                    Q[i, j] = (float)(rng.NextDouble() - 0.5);
                }
            }
            for (int i = 0; i < seqLenK; i++)
            {
                for (int j = 0; j < embDim; j++)
                {
                    K[i, j] = (float)(rng.NextDouble() - 0.5); V[i, j] = (float)(rng.NextDouble() - 0.5);
                }
            }

            var resultCPU = cpu.MultiHeadAttentionForward(Q, K, V, numHeads, scale, null);
            var resultMT = mt.MultiHeadAttentionForward(Q, K, V, numHeads, scale, null);

            Assert(resultCPU.GetLength(0) == seqLenQ, $"CPU output seqLen wrong: {resultCPU.GetLength(0)}");

            Assert(resultMT.GetLength(0) == seqLenQ, $"MT output seqLen wrong: {resultMT.GetLength(0)}");

            Assert(MatricesApproxEqual(resultCPU, resultMT, 1e-5f), "CrossAttn Forward: CPU vs MultiThread results differ");
        }

        #region Save/Load

        private string GetTempDir(string name)
        {
            var dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "transformer_tests", name + "_" + Guid.NewGuid().ToString("N").Substring(0, 8));
            System.IO.Directory.CreateDirectory(dir);
            return dir;
        }

        private void CleanupDir(string dir)
        {
            try { if (System.IO.Directory.Exists(dir)) System.IO.Directory.Delete(dir, true); } catch { }
        }

        private void AssertMatricesEqual(float[,] a, float[,] b, string name, float tol = 0f)
        {
            Assert(a.GetLength(0) == b.GetLength(0) && a.GetLength(1) == b.GetLength(1),
                $"{name} shape mismatch: [{a.GetLength(0)},{a.GetLength(1)}] vs [{b.GetLength(0)},{b.GetLength(1)}]");
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    Assert(MathF.Abs(a[i, j] - b[i, j]) <= tol,
                        $"{name}[{i},{j}] mismatch: {a[i, j]} vs {b[i, j]}");
        }

        private void AssertVectorsEqual(float[] a, float[] b, string name, float tol = 0f)
        {
            Assert(a.Length == b.Length, $"{name} length mismatch: {a.Length} vs {b.Length}");
            for (int i = 0; i < a.Length; i++)
                Assert(MathF.Abs(a[i] - b[i]) <= tol,
                    $"{name}[{i}] mismatch: {a[i]} vs {b[i]}");
        }

        private void AssertAttentionEqual(MultiHeadAttention a, MultiHeadAttention b, string prefix, float tol = 0f)
        {
            AssertMatricesEqual(a.WQ, b.WQ, $"{prefix}.WQ", tol);
            AssertMatricesEqual(a.WK, b.WK, $"{prefix}.WK", tol);
            AssertMatricesEqual(a.WV, b.WV, $"{prefix}.WV", tol);
            AssertMatricesEqual(a.WO, b.WO, $"{prefix}.WO", tol);
            AssertVectorsEqual(a.BiasQ, b.BiasQ, $"{prefix}.BiasQ", tol);
            AssertVectorsEqual(a.BiasK, b.BiasK, $"{prefix}.BiasK", tol);
            AssertVectorsEqual(a.BiasV, b.BiasV, $"{prefix}.BiasV", tol);
            AssertVectorsEqual(a.BiasO, b.BiasO, $"{prefix}.BiasO", tol);
        }

        private void AssertLanguageModelsEqual(LanguageModel original, LanguageModel loaded)
        {
            // Config
            Assert(original.Config.DataType == loaded.Config.DataType, "DataType mismatch");
            Assert(original.Config.VocabSize == loaded.Config.VocabSize, "VocabSize mismatch");
            Assert(original.Config.EmbeddingDim == loaded.Config.EmbeddingDim, "EmbeddingDim mismatch");
            Assert(original.Config.NumHeads == loaded.Config.NumHeads, "NumHeads mismatch");
            Assert(original.Config.NumLayers == loaded.Config.NumLayers, "NumLayers mismatch");
            Assert(original.Config.FeedForwardDim == loaded.Config.FeedForwardDim, "FeedForwardDim mismatch");
            Assert(original.Config.UseDecoderOnly == loaded.Config.UseDecoderOnly, "UseDecoderOnly mismatch");
            Assert(original.Config.InputFeatureDim == loaded.Config.InputFeatureDim, "InputFeatureDim mismatch");
            Assert(original.Config.OutputDim == loaded.Config.OutputDim, "OutputDim mismatch");

            // Input layer
            if (original.Config.UsesDiscreteTokens)
            {
                AssertMatricesEqual(original.TokenEmbedding, loaded.TokenEmbedding, "TokenEmbedding");
            }
            else
            {
                AssertMatricesEqual(original.InputProjection, loaded.InputProjection, "InputProjection");
                AssertVectorsEqual(original.InputProjectionBias, loaded.InputProjectionBias, "InputProjectionBias");
            }

            // Blocks
            for (int layer = 0; layer < original.Config.NumLayers; layer++)
            {
                var origBlock = original.Blocks[layer];
                var loadBlock = loaded.Blocks[layer];

                AssertAttentionEqual(origBlock.Attention, loadBlock.Attention, $"Block[{layer}].Attention");
                AssertVectorsEqual(origBlock.LN1Gamma, loadBlock.LN1Gamma, $"Block[{layer}].LN1Gamma");
                AssertVectorsEqual(origBlock.LN1Beta, loadBlock.LN1Beta, $"Block[{layer}].LN1Beta");
                AssertVectorsEqual(origBlock.LN2Gamma, loadBlock.LN2Gamma, $"Block[{layer}].LN2Gamma");
                AssertVectorsEqual(origBlock.LN2Beta, loadBlock.LN2Beta, $"Block[{layer}].LN2Beta");
            }

            // Output head
            AssertMatricesEqual(original.OutputProjection, loaded.OutputProjection, "OutputProjection");
            AssertVectorsEqual(original.OutputBias, loaded.OutputBias, "OutputBias");
        }

        public void Test_SaveLoad_Text_ExactWeightsAndForward()
        {
            var dir = GetTempDir("sl_text");
            try
            {
                // Create and train
                var (model, config) = CreateSmallModel(vocabSize: 12, embDim: 8, numHeads: 2, numLayers: 2, ffnDim: 16);
                int[][] sequences = { new[] { 1, 2, 3, 4, 5 }, new[] { 3, 4, 5, 6, 7 } };
                var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 10, Verbose = false };
                new TransformerTrainer(model, tc).Train(sequences);

                // Forward before save
                int[] testInput = { 1, 2, 3, 4 };
                var logitsBefore = model.Forward(testInput);

                // Save + Load
                model.Save(dir);
                var loaded = LanguageModel.Load(dir);

                // Exact weight comparison
                AssertLanguageModelsEqual(model, loaded);

                // Forward after load must be bit-identical
                var logitsAfter = loaded.Forward(testInput);
                AssertMatricesEqual(logitsBefore, logitsAfter, "Forward output after load");

                // Generate should work
                var generated = loaded.Generate(new[] { 1, 2 }, maxNewTokens: 5, temperature: 1.0f);
                Assert(generated.Length > 2, "Loaded text model failed to generate");

                // Continue training on loaded model — loss should decrease
                int[] input = { 1, 2, 3, 4 };
                int[] target = { 2, 3, 4, 5 };
                float lossBefore = ComputeLoss(loaded, input, target);
                new TransformerTrainer(loaded, new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 20, Verbose = false }).Train(sequences);
                float lossAfter = ComputeLoss(loaded, input, target);
                Assert(lossAfter < lossBefore, $"Text: loss did not decrease after retraining loaded model: {lossBefore:F6} -> {lossAfter:F6}");
            }
            finally { CleanupDir(dir); }
        }

        public void Test_SaveLoad_SymbolicSequence_ExactWeightsAndForward()
        {
            var dir = GetTempDir("sl_symbolic");
            try
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

                // DNA-like sequences: A=4, T=5, G=6, C=7, BOS=1, EOS=2
                int[][] sequences = { new[] { 1, 4, 5, 6, 7, 2 }, new[] { 1, 6, 7, 4, 5, 2 } };
                var tc = new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 15, Verbose = false };
                new TransformerTrainer(model, tc).Train(sequences);

                int[] testInput = { 1, 4, 5, 6 };
                var logitsBefore = model.Forward(testInput);

                model.Save(dir);
                var loaded = LanguageModel.Load(dir);

                // Config check
                Assert(loaded.Config.DataType == TransformerDataType.SymbolicSequence, "DataType not SymbolicSequence");
                Assert(loaded.Config.UsesDiscreteTokens, "UsesDiscreteTokens should be true");

                AssertLanguageModelsEqual(model, loaded);

                var logitsAfter = loaded.Forward(testInput);
                AssertMatricesEqual(logitsBefore, logitsAfter, "Symbolic forward output");

                // Generate works
                var generated = loaded.Generate(new[] { 1, 4 }, maxNewTokens: 4, temperature: 1.0f);
                Assert(generated.Length > 2, "Loaded symbolic model failed to generate");

                // Continue training
                float lossBefore = ComputeLoss(loaded, new[] { 1, 4, 5 }, new[] { 4, 5, 6 });
                new TransformerTrainer(loaded, new TrainingConfig { LearningRate = 0.005f, BatchSize = 2, Epochs = 30, Verbose = false }).Train(sequences);
                float lossAfterRetrain = ComputeLoss(loaded, new[] { 1, 4, 5 }, new[] { 4, 5, 6 });
                Assert(lossAfterRetrain < lossBefore, $"Symbolic: loss did not decrease after retraining: {lossBefore:F6} -> {lossAfterRetrain:F6}");
            }
            finally { CleanupDir(dir); }
        }

        public void Test_SaveLoad_TimeSeriesRegression_ExactWeightsAndForward()
        {
            var dir = GetTempDir("sl_regression");
            try
            {
                var (model, config) = CreateSmallContinuousModel(
                    TransformerDataType.TimeSeriesRegression,
                    inputFeatureDim: 3, outputDim: 1,
                    embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);

                var rng = new Random(42);
                var inputs = new float[5][,];
                var targets = new float[5][,];
                for (int s = 0; s < 5; s++)
                {
                    inputs[s] = new float[6, 3];
                    targets[s] = new float[6, 1];
                    for (int i = 0; i < 6; i++)
                    {
                        for (int j = 0; j < 3; j++) inputs[s][i, j] = (float)rng.NextDouble();
                        targets[s][i, 0] = inputs[s][i, 0] * 0.5f + 0.1f;
                    }
                }

                var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false };
                new TransformerTrainer(model, tc).TrainContinuous(inputs, regressionTargets: targets);

                var forwardBefore = model.Forward(inputs[0]);

                model.Save(dir);
                var loaded = LanguageModel.Load(dir);

                // Config check
                Assert(loaded.Config.DataType == TransformerDataType.TimeSeriesRegression, "DataType not TimeSeriesRegression");
                Assert(!loaded.Config.UsesDiscreteTokens, "UsesDiscreteTokens should be false");
                Assert(loaded.Config.InputFeatureDim == 3, $"InputFeatureDim: {loaded.Config.InputFeatureDim}");
                Assert(loaded.Config.OutputDim == 1, $"OutputDim: {loaded.Config.OutputDim}");

                AssertLanguageModelsEqual(model, loaded);

                var forwardAfter = loaded.Forward(inputs[0]);
                AssertMatricesEqual(forwardBefore, forwardAfter, "Regression forward output");

                // PredictNext works
                float[] prediction = loaded.PredictNext(inputs[0]);
                Assert(prediction.Length == 1, $"PredictNext dim: {prediction.Length}");
                Assert(!float.IsNaN(prediction[0]), "PredictNext NaN");

                // Continue training
                var trainer = new TransformerTrainer(loaded, new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
                float lossBefore = trainer.ValidateContinuous(inputs, regressionTargets: targets);
                trainer = new TransformerTrainer(loaded, new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 30, Verbose = false });
                trainer.TrainContinuous(inputs, regressionTargets: targets);
                float lossAfterRetrain = trainer.ValidateContinuous(inputs, regressionTargets: targets);
                Assert(lossAfterRetrain < lossBefore, $"Regression: loss did not decrease after retraining: {lossBefore:F6} -> {lossAfterRetrain:F6}");
            }
            finally { CleanupDir(dir); }
        }

        public void Test_SaveLoad_TimeSeriesClassification_ExactWeightsAndForward()
        {
            var dir = GetTempDir("sl_classification");
            try
            {
                var (model, config) = CreateSmallContinuousModel(
                    TransformerDataType.TimeSeriesClassification,
                    inputFeatureDim: 3, outputDim: 3,
                    embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);

                var rng = new Random(42);
                var inputs = new float[5][,];
                var classTargets = new int[5][];
                for (int s = 0; s < 5; s++)
                {
                    inputs[s] = new float[6, 3];
                    classTargets[s] = new int[6];
                    for (int i = 0; i < 6; i++)
                    {
                        for (int j = 0; j < 3; j++) inputs[s][i, j] = (float)rng.NextDouble();
                        classTargets[s][i] = rng.Next(3);
                    }
                }

                var tc = new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false };
                new TransformerTrainer(model, tc).TrainContinuous(inputs, classTargets: classTargets);

                var forwardBefore = model.Forward(inputs[0]);

                model.Save(dir);
                var loaded = LanguageModel.Load(dir);

                // Config check
                Assert(loaded.Config.DataType == TransformerDataType.TimeSeriesClassification, "DataType not TimeSeriesClassification");
                Assert(!loaded.Config.UsesDiscreteTokens, "UsesDiscreteTokens should be false");
                Assert(loaded.Config.OutputDim == 3, $"OutputDim: {loaded.Config.OutputDim}");

                AssertLanguageModelsEqual(model, loaded);

                var forwardAfter = loaded.Forward(inputs[0]);
                AssertMatricesEqual(forwardBefore, forwardAfter, "Classification forward output");

                // Continue training
                var trainer = new TransformerTrainer(loaded, new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
                float lossBefore = trainer.ValidateContinuous(inputs, classTargets: classTargets);
                trainer = new TransformerTrainer(loaded, new TrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 30, Verbose = false });
                trainer.TrainContinuous(inputs, classTargets: classTargets);
                float lossAfterRetrain = trainer.ValidateContinuous(inputs, classTargets: classTargets);
                Assert(lossAfterRetrain < lossBefore, $"Classification: loss did not decrease after retraining: {lossBefore:F6} -> {lossAfterRetrain:F6}");
            }
            finally { CleanupDir(dir); }
        }

        public void Test_SaveLoad_CrossAttentionMultimodal_ExactWeightsAndForward()
        {
            var dir = GetTempDir("sl_crossattn");
            try
            {
                var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 10, priceSeqLen: 6);

                var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32, priceSeqLen: 6, useConfidence: true, freezeTextEncoder: false);

                var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

                var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 10, Verbose = false });

                trainer.Train(textSeqs, priceInputs, priceTargets);

                var (predBefore, confBefore) = model.Forward(textSeqs[0], priceInputs[0]);

                model.Save(dir);

                var loaded = Transformers.CrossAttentionMultimodal.Model.Load(dir);

                Assert(loaded.Config.Text.VocabSize == config.TextVocabSize, "TextVocabSize mismatch");
                Assert(loaded.Config.Text.EmbeddingDim == config.TextEmbeddingDim, "TextEmbeddingDim mismatch");
                Assert(loaded.Config.Text.NumLayers == config.TextNumLayers, "TextNumLayers mismatch");
                Assert(loaded.Config.Price.InputFeatureDim == config.PriceInputFeatureDim, "PriceInputFeatureDim mismatch");
                Assert(loaded.Config.Price.EmbeddingDim == config.PriceEmbeddingDim, "PriceEmbeddingDim mismatch");
                Assert(loaded.Config.Price.NumLayers == config.PriceNumLayers, "PriceNumLayers mismatch");
                Assert(loaded.Config.Output.OutputDim == config.OutputDim, "OutputDim mismatch");
                Assert(loaded.Config.Output.UseConfidenceHead == config.UseConfidenceHead, "UseConfidenceHead mismatch");

                AssertMatricesEqual(model.TextTokenEmbedding, loaded.TextTokenEmbedding, "TextTokenEmbedding");

                for (int layer = 0; layer < config.TextNumLayers; layer++)
                {
                    AssertAttentionEqual(model.TextBlocks[layer].Attention, loaded.TextBlocks[layer].Attention, $"TextBlock[{layer}].Attention");
                    AssertVectorsEqual(model.TextBlocks[layer].LN1Gamma, loaded.TextBlocks[layer].LN1Gamma, $"TextBlock[{layer}].LN1Gamma");
                    AssertVectorsEqual(model.TextBlocks[layer].LN1Beta, loaded.TextBlocks[layer].LN1Beta, $"TextBlock[{layer}].LN1Beta");
                    AssertVectorsEqual(model.TextBlocks[layer].LN2Gamma, loaded.TextBlocks[layer].LN2Gamma, $"TextBlock[{layer}].LN2Gamma");
                    AssertVectorsEqual(model.TextBlocks[layer].LN2Beta, loaded.TextBlocks[layer].LN2Beta, $"TextBlock[{layer}].LN2Beta");
                }

                AssertMatricesEqual(model.PriceInputProjection, loaded.PriceInputProjection, "PriceInputProjection");
                AssertVectorsEqual(model.PriceInputProjectionBias, loaded.PriceInputProjectionBias, "PriceInputProjectionBias");

                for (int layer = 0; layer < config.PriceNumLayers; layer++)
                {
                    AssertAttentionEqual(model.PriceBlocks[layer].SelfAttention, loaded.PriceBlocks[layer].SelfAttention, $"PriceBlock[{layer}].SelfAttention");
                    AssertVectorsEqual(model.PriceBlocks[layer].LNSelfGamma, loaded.PriceBlocks[layer].LNSelfGamma, $"PriceBlock[{layer}].LNSelfGamma");
                    AssertVectorsEqual(model.PriceBlocks[layer].LNSelfBeta, loaded.PriceBlocks[layer].LNSelfBeta, $"PriceBlock[{layer}].LNSelfBeta");
                    AssertAttentionEqual(model.PriceBlocks[layer].CrossAttention, loaded.PriceBlocks[layer].CrossAttention, $"PriceBlock[{layer}].CrossAttention");
                    AssertVectorsEqual(model.PriceBlocks[layer].LNCrossGamma, loaded.PriceBlocks[layer].LNCrossGamma, $"PriceBlock[{layer}].LNCrossGamma");
                    AssertVectorsEqual(model.PriceBlocks[layer].LNCrossBeta, loaded.PriceBlocks[layer].LNCrossBeta, $"PriceBlock[{layer}].LNCrossBeta");
                    AssertVectorsEqual(model.PriceBlocks[layer].LNFFNGamma, loaded.PriceBlocks[layer].LNFFNGamma, $"PriceBlock[{layer}].LNFFNGamma");
                    AssertVectorsEqual(model.PriceBlocks[layer].LNFFNBeta, loaded.PriceBlocks[layer].LNFFNBeta, $"PriceBlock[{layer}].LNFFNBeta");
                }

                AssertMatricesEqual(model.OutputProjection, loaded.OutputProjection, "OutputProjection");
                AssertVectorsEqual(model.OutputBias, loaded.OutputBias, "OutputBias");
                AssertMatricesEqual(model.ConfidenceProjection, loaded.ConfidenceProjection, "ConfidenceProjection");
                AssertVectorsEqual(model.ConfidenceBias, loaded.ConfidenceBias, "ConfidenceBias");

                var (predAfter, confAfter) = loaded.Forward(textSeqs[0], priceInputs[0]);
                AssertMatricesEqual(predBefore, predAfter, "CrossAttn prediction output");
                AssertMatricesEqual(confBefore, confAfter, "CrossAttn confidence output");

                var (pred, conf) = loaded.PredictNext(textSeqs[0], priceInputs[0]);

                Assert(pred.Length == config.OutputDim, $"PredictNext dim: {pred.Length}");

                Assert(!float.IsNaN(conf) && conf >= 0f && conf <= 1f, $"PredictNext confidence: {conf}");

                var loadedTrainer = new Transformers.CrossAttentionMultimodal.Trainer(loaded, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });

                float lossBefore = loadedTrainer.Validate(textSeqs, priceInputs, priceTargets);

                loadedTrainer = new Transformers.CrossAttentionMultimodal.Trainer(loaded, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 20, Verbose = false });

                loadedTrainer.Train(textSeqs, priceInputs, priceTargets);

                float lossAfterRetrain = loadedTrainer.Validate(textSeqs, priceInputs, priceTargets);

                Assert(lossAfterRetrain < lossBefore, $"CrossAttn: loss did not decrease after retraining: {lossBefore:F6} -> {lossAfterRetrain:F6}");
            }
            finally 
            { 
                CleanupDir(dir);
            }
        }

        public void Test_SaveLoad_CrossAttention_NoConfidenceHead()
        {
            var dir = GetTempDir("sl_crossattn_noconf");
            try
            {
                var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 5, priceSeqLen: 6);
                var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, priceSeqLen: 6, useConfidence: false);

                var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

                var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 5, Verbose = false });

                trainer.Train(textSeqs, priceInputs, priceTargets);

                var (predBefore, confBefore) = model.Forward(textSeqs[0], priceInputs[0]);
                Assert(confBefore == null, "Confidence should be null when UseConfidenceHead=false");

                model.Save(dir);

                var loaded = Transformers.CrossAttentionMultimodal.Model.Load(dir);

                Assert(!loaded.Config.Output.UseConfidenceHead, "Loaded UseConfidenceHead should be false");

                var (predAfter, confAfter) = loaded.Forward(textSeqs[0], priceInputs[0]);

                Assert(confAfter == null, "Loaded confidence should be null");
                AssertMatricesEqual(predBefore, predAfter, "NoConfidence prediction output");
            }
            finally 
            { 
                CleanupDir(dir);
            }
        }

        public void Test_SaveLoad_DoubleSaveLoad_RoundTrip()
        {
            var dir1 = GetTempDir("sl_roundtrip1");
            var dir2 = GetTempDir("sl_roundtrip2");
            try
            {
                var (model, config) = CreateSmallModel(vocabSize: 10, embDim: 8, numHeads: 2, numLayers: 2, ffnDim: 16);
                int[][] sequences = { new[] { 1, 2, 3, 4, 5 } };
                new TransformerTrainer(model, new TrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 10, Verbose = false }).Train(sequences);

                int[] testInput = { 1, 2, 3 };
                var logitsOriginal = model.Forward(testInput);

                // Save -> Load -> Save -> Load
                model.Save(dir1);
                var loaded1 = LanguageModel.Load(dir1);
                loaded1.Save(dir2);
                var loaded2 = LanguageModel.Load(dir2);

                var logitsFinal = loaded2.Forward(testInput);
                AssertMatricesEqual(logitsOriginal, logitsFinal, "Double round-trip forward output");
                AssertLanguageModelsEqual(model, loaded2);
            }
            finally { CleanupDir(dir1); CleanupDir(dir2); }
        }

        #endregion

        public void Test_CrossAttention_PriceOnly_ForwardNoError()
        {
            var (tokenizer, _, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            // Forward with null text
            var (predictions, confidence) = model.Forward(null, priceInputs[0]);
            Assert(predictions.GetLength(0) == 6, $"PriceOnly forward: wrong seq len {predictions.GetLength(0)}");
            Assert(predictions.GetLength(1) == 5, $"PriceOnly forward: wrong output dim {predictions.GetLength(1)}");
            for (int i = 0; i < predictions.GetLength(0); i++)
                for (int j = 0; j < predictions.GetLength(1); j++)
                    Assert(!float.IsNaN(predictions[i, j]), $"NaN at predictions[{i},{j}]");
        }

        public void Test_CrossAttention_PriceOnly_PredictNext()
        {
            var (tokenizer, _, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var (prediction, confidence) = model.PredictNext(null, priceInputs[0]);
            Assert(prediction.Length == 5, $"PriceOnly PredictNext dim: {prediction.Length}");
            Assert(!float.IsNaN(confidence) && confidence >= 0f && confidence <= 1f, $"Confidence: {confidence}");
        }

        public void Test_CrossAttention_PriceOnly_TrainAndLossDecreases()
        {
            var (tokenizer, _, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 10, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            // All null text
            var nullTextSeqs = new int[10][];
            // nullTextSeqs[i] remains null

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model,
                new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
            float lossBefore = trainer.Validate(nullTextSeqs, priceInputs, priceTargets);

            trainer = new Transformers.CrossAttentionMultimodal.Trainer(model,
                new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false });
            trainer.Train(nullTextSeqs, priceInputs, priceTargets);

            float lossAfter = trainer.Validate(nullTextSeqs, priceInputs, priceTargets);
            Assert(lossAfter < lossBefore, $"PriceOnly: loss did not decrease: {lossBefore:F6} -> {lossAfter:F6}");
        }

        public void Test_CrossAttention_MixedBatch_SomeTextSomeNull()
        {
            var (tokenizer, textSeqs, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 10, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            // Set half the text sequences to null
            var mixedTextSeqs = (int[][])textSeqs.Clone();
            for (int i = 0; i < mixedTextSeqs.Length; i += 2)
                mixedTextSeqs[i] = null;

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model,
                new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
            float lossBefore = trainer.Validate(mixedTextSeqs, priceInputs, priceTargets);

            trainer = new Transformers.CrossAttentionMultimodal.Trainer(model,
                new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false });
            trainer.Train(mixedTextSeqs, priceInputs, priceTargets);

            float lossAfter = trainer.Validate(mixedTextSeqs, priceInputs, priceTargets);
            Assert(lossAfter < lossBefore, $"MixedBatch: loss did not decrease: {lossBefore:F6} -> {lossAfter:F6}");
        }

        public void Test_CrossAttention_EmptyTextArray_TreatedAsNull()
        {
            var (tokenizer, _, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            // Empty array should be treated same as null
            var (predictions, _) = model.Forward(new int[0], priceInputs[0]);
            Assert(predictions.GetLength(0) == 6, "Empty text array should work like null");
        }

        public void Test_CrossAttention_TextVsNoText_DifferentOutputs()
        {
            var (tokenizer, textSeqs, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var (predWithText, _) = model.Forward(textSeqs[0], priceInputs[0]);
            var (predNoText, _) = model.Forward(null, priceInputs[0]);

            // Outputs should differ because cross-attention adds information
            bool anyDiff = false;
            for (int i = 0; i < predWithText.GetLength(0) && !anyDiff; i++)
                for (int j = 0; j < predWithText.GetLength(1) && !anyDiff; j++)
                    if (MathF.Abs(predWithText[i, j] - predNoText[i, j]) > 1e-6f)
                        anyDiff = true;

            Assert(anyDiff, "Forward with text and without text produced identical outputs — cross-attention has no effect");
        }

        public void Test_CrossAttention_PriceOnly_Deterministic()
        {
            var (tokenizer, _, priceInputs, _) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var (pred1, _) = model.Forward(null, priceInputs[0]);
            var (pred2, _) = model.Forward(null, priceInputs[0]);

            for (int i = 0; i < pred1.GetLength(0); i++)
                for (int j = 0; j < pred1.GetLength(1); j++)
                    Assert(pred1[i, j] == pred2[i, j], $"PriceOnly non-deterministic at [{i},{j}]");
        }

        public void Test_CrossAttention_PriceOnly_SingleSampleOverfit()
        {
            var (tokenizer, _, priceInputs, priceTargets) = CreateCrossAttentionTestData(numSamples: 1, priceSeqLen: 6);
            var config = CreateSmallCrossAttentionConfig(tokenizer.VocabSize + 2, embDim: 32, numHeads: 2, numLayers: 2,
                ffnDim: 64, priceSeqLen: 6, useConfidence: false);
            var model = new Transformers.CrossAttentionMultimodal.Model(config, new Random(42));

            var nullTextSeqs = new int[1][];  // null text

            var trainer = new Transformers.CrossAttentionMultimodal.Trainer(model,
                new MultimodalTrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 1, Verbose = false });
            float lossBefore = trainer.Validate(nullTextSeqs, priceInputs, priceTargets);

            trainer = new Transformers.CrossAttentionMultimodal.Trainer(model,
                new MultimodalTrainingConfig { LearningRate = 0.005f, BatchSize = 1, Epochs = 100, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false });
            trainer.Train(nullTextSeqs, priceInputs, priceTargets);

            float lossAfter = trainer.Validate(nullTextSeqs, priceInputs, priceTargets);
            Assert(lossAfter < lossBefore * 0.5f, $"PriceOnly overfit failed: {lossBefore:F6} -> {lossAfter:F6}");
        }





























    }
}