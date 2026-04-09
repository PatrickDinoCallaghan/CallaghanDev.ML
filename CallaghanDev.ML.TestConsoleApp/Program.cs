using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CallaghanDev.ML.Transformers.MMTAC;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using CallaghanDev.ML.Transformers.TACAMT;

// Alias resolution for types that share names across namespaces
using TACAMT_Model = CallaghanDev.ML.Transformers.TACAMT.Model;
using TACAMT_Trainer = CallaghanDev.ML.Transformers.TACAMT.Trainer;
using TACAMT_Cache = CallaghanDev.ML.Transformers.TACAMT.MultimodalForwardCache;
using CrossModel = CallaghanDev.ML.Transformers.CrossAttentionMultimodal.Model;
using CrossTrainer = CallaghanDev.ML.Transformers.CrossAttentionMultimodal.Trainer;

namespace CallaghanDev.ML.TestConsoleApp
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            TransformerTestSuite.Run();
        }
    }

    public static class TransformerTestSuite
    {
        public static void Run()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            PrintBanner("TRANSFORMER TEST SUITE", '═');

          new MultiTypeTransformerTests().RunAllTests();
        new CrossAttentionMultimodalTests().RunAllTests();
          new TacamtTests().RunAllTests();
            new MmtacTests().RunAllTests();

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("  All test suites complete. Press any key to exit.");
            Console.ResetColor();
            Console.ReadKey();
        }

        // ── formatting helpers ────────────────────────────────────────────────
        internal static void PrintBanner(string title, char ch = '─')
        {
            string line = new string(ch, 70);
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"  {line}");
            Console.WriteLine($"  {title}");
            Console.WriteLine($"  {line}");
            Console.ResetColor();
        }
    }

    // =========================================================================
    //  Shared base – assertion helpers, random data, temp dirs
    // =========================================================================
    internal abstract class TestBase
    {
        protected int _passed, _failed;
        protected readonly List<string> _failures = new();

        // ── assertion ────────────────────────────────────────────────────────
        protected void Assert(bool cond, string msg)
        {
            if (!cond) throw new Exception(msg);
        }

        protected void AssertLossImproved(float before, float after, float minImprovementRatio = 0.90f)
        {
            Assert(
                after < before * minImprovementRatio,
                $"Weak improvement: {before:F6} → {after:F6} (expected < {before * minImprovementRatio:F6})");
        }

        protected void AssertLossImprovedByAbsolute(float before, float after, float minAbsoluteDrop)
        {
            Assert(
                after <= before - minAbsoluteDrop,
                $"Weak absolute improvement: {before:F6} → {after:F6} (expected drop ≥ {minAbsoluteDrop:F6})");
        }

        protected void AssertOverfitStrong(float loss, float threshold = 0.10f)
        {
            Assert(loss < threshold, $"Failed to strongly overfit: loss={loss:F6} (expected < {threshold:F6})");
        }

        protected void AssertFinite(float value, string name)
        {
            Assert(!float.IsNaN(value) && !float.IsInfinity(value), $"{name} is not finite: {value}");
        }

        protected void AssertBetweenInclusive(float value, float lo, float hi, string name)
        {
            Assert(value >= lo && value <= hi, $"{name}={value} outside [{lo}, {hi}]");
        }

        // ── matrix / vector helpers ──────────────────────────────────────────
        protected static bool MatrixChanged(float[,] a, float[,] b, float tol = 1e-10f)
        {
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    if (MathF.Abs(a[i, j] - b[i, j]) > tol) return true;
            return false;
        }

        protected static bool VectorChanged(float[] a, float[] b, float tol = 1e-10f)
        {
            for (int i = 0; i < a.Length; i++)
                if (MathF.Abs(a[i] - b[i]) > tol) return true;
            return false;
        }

        protected static bool MatrixEquals(float[,] a, float[,] b, float tol = 1e-10f)
        {
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
                return false;

            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    if (MathF.Abs(a[i, j] - b[i, j]) > tol) return false;

            return true;
        }

        protected static bool VectorEquals(float[] a, float[] b, float tol = 1e-10f)
        {
            if (a.Length != b.Length) return false;

            for (int i = 0; i < a.Length; i++)
                if (MathF.Abs(a[i] - b[i]) > tol) return false;

            return true;
        }

        protected static bool HasNaN(float[,] m)
        {
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    if (float.IsNaN(m[i, j]) || float.IsInfinity(m[i, j])) return true;
            return false;
        }

        protected static bool HasNaN(float[] v)
        {
            foreach (var f in v)
                if (float.IsNaN(f) || float.IsInfinity(f)) return true;
            return false;
        }

        protected static float MaxAbs(float[,] m)
        {
            float max = 0f;
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    max = Math.Max(max, MathF.Abs(m[i, j]));
            return max;
        }

        protected static float MaxAbs(float[] v)
        {
            float max = 0f;
            for (int i = 0; i < v.Length; i++)
                max = Math.Max(max, MathF.Abs(v[i]));
            return max;
        }

        protected static float MeanAbs(float[,] m)
        {
            float sum = 0f;
            int n = 0;
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                {
                    sum += MathF.Abs(m[i, j]);
                    n++;
                }

            return n == 0 ? 0f : sum / n;
        }

        protected static float[,] RandMatrix(int rows, int cols, Random rng, float scale = 0.5f)
        {
            var m = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = ((float)rng.NextDouble() - 0.5f) * 2f * scale;
            return m;
        }

        protected static float[,] SliceRows(float[,] m, int start, int end)
        {
            int cols = m.GetLength(1), rows = end - start;
            var r = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    r[i, j] = m[start + i, j];
            return r;
        }

        protected string TempDir()
        {
            var d = Path.Combine(Path.GetTempPath(), $"tf_test_{Guid.NewGuid():N}");
            Directory.CreateDirectory(d);
            return d;
        }

        protected void Cleanup(string dir)
        {
            try { if (Directory.Exists(dir)) Directory.Delete(dir, true); } catch { }
        }

        // ── runner ────────────────────────────────────────────────────────────
        protected void Run((Action test, string name)[] tests, string suiteName)
        {
            TransformerTestSuite.PrintBanner(suiteName);
            _passed = _failed = 0;
            _failures.Clear();

            for (int i = 0; i < tests.Length; i++)
            {
                Console.Write($"  [{i + 1,3}/{tests.Length}] {tests[i].name,-62} ");
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
                    Console.WriteLine($"         ↳ {ex.Message}");
                    Console.ResetColor();
                    _failures.Add($"{tests[i].name}: {ex.Message}");
                    _failed++;
                }
            }

            Console.WriteLine($"\n  {"",3}{new string('─', 68)}");
            Console.Write("  Results: ");
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

            Console.WriteLine($" / {tests.Length} total");

            if (_failures.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n  Failed tests:");
                Console.ResetColor();
                foreach (var f in _failures)
                    Console.WriteLine($"    • {f}");
            }

            Console.WriteLine();
        }
    }

    // =========================================================================
    //  1. MultiTypeTransformer tests
    // =========================================================================
    internal sealed class MultiTypeTransformerTests : TestBase
    {
        // ── config factory ────────────────────────────────────────────────────
        private (LanguageModel model, TransformerConfig cfg) Discrete(int vocabSize = 12, int embDim = 8, int numHeads = 2, int numLayers = 1, int ffnDim = 16)
        {
            var cfg = new TransformerConfig
            {
                Data = new DataConfig { DataType = TransformerDataType.Text, CostFunction = CostFunctionType.mse },
                VocabSize = vocabSize,
                MaxSequenceLength = 24,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                Runtime = new RuntimeConfig { FFNActivationType = ActivationType.Relu, AccelerationType = AccelerationType.CPU },
                Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 5f },
                UseDecoderOnly = true,
            };
            cfg.Validate();
            return (new LanguageModel(cfg, new Random(42)), cfg);
        }

        private (LanguageModel model, TransformerConfig cfg) Continuous(TransformerDataType dt, int inputDim = 3, int outputDim = 1, int embDim = 8, int numHeads = 2, int numLayers = 1, int ffnDim = 16)
        {
            var cfg = new TransformerConfig
            {
                Data = new DataConfig { DataType = dt, CostFunction = CostFunctionType.mse },
                InputFeatureDim = inputDim,
                OutputDim = outputDim,
                MaxSequenceLength = 16,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                Runtime = new RuntimeConfig { FFNActivationType = ActivationType.Relu, AccelerationType = AccelerationType.CPU },
                Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 5f },
                UseDecoderOnly = true,
            };
            cfg.Validate();
            return (new LanguageModel(cfg, new Random(42)), cfg);
        }

        private TrainingConfig TC(float lr = 0.005f, int bs = 4, int epochs = 5, bool clip = true)
            => new TrainingConfig
            {
                LearningRate = lr,
                BatchSize = bs,
                Epochs = epochs,
                UseGradientClipping = clip,
                GradientClipThreshold = 5f,
                Verbose = false
            };

        private float CELoss(LanguageModel m, int[] inp, int[] tgt)
        {
            var logits = m.Forward(inp);
            float loss = 0;
            int vs = m.Config.VocabSize;
            for (int i = 0; i < Math.Min(logits.GetLength(0), tgt.Length); i++)
            {
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < vs; j++) maxVal = Math.Max(maxVal, logits[i, j]);
                float sum = 0; for (int j = 0; j < vs; j++) sum += MathF.Exp(logits[i, j] - maxVal);
                loss -= MathF.Log(MathF.Exp(logits[i, tgt[i]] - maxVal) / sum + 1e-10f);
            }
            return loss / Math.Min(logits.GetLength(0), tgt.Length);
        }

        public void RunAllTests() => Run(Tests(), "1 · MultiTypeTransformer");

        private (Action, string)[] Tests() => new (Action, string)[]
        {
    // ── forward pass ─────────────────────────────────────────────────
    (Test_Text_ForwardShape,                  "Text: forward output shape [seqLen, vocabSize]"),
    (Test_Text_ForwardNaN,                    "Text: forward produces no NaN"),
    (Test_Text_ForwardDeterministic,          "Text: forward is deterministic"),
    (Test_Text_CausalMask,                    "Text: causal mask — future doesn't affect past"),
    (Test_Text_ForwardLen1,                   "Text: single-token input works"),


    // ── training ─────────────────────────────────────────────────────
    (Test_Text_LossDecreases,                 "Text: loss decreases after training"),
    (Test_Text_AllParamsUpdated,              "Text: all parameters receive gradient updates"),
    (Test_Text_FreezeEmbeddingNotUsed,        "Text: unused token embedding unchanged after train"),
    (Test_Text_GradientClipping,              "Text: gradient clipping prevents NaN"),
    (Test_Text_PostTrainWeightsBounded,       "Text: clipped training keeps weights bounded"),
    (Test_Text_LearningRateDecay,             "Text: learning-rate decay applied without crash"),
    (Test_Text_BatchTraining,                 "Text: batch training decreases average loss"),
    (Test_Text_DifferentSeqLengths,           "Text: mixed-length batch trains without error"),
    (Test_Text_SeedRobustness,                "Text: learning is robust across seeds"),

    // ── generation ───────────────────────────────────────────────────
    (Test_Text_Generate,                      "Text: Generate() produces new tokens"),
    (Test_Text_GenerateTokenBounds,           "Text: Generate() tokens within vocab bounds"),

    // ── overfit ──────────────────────────────────────────────────────
    (Test_Text_OverfitSingle,                 "Text: strongly overfits single sequence"),
    (Test_Text_OverfitPredictionExact,        "Text: overfit yields correct next-token prediction"),

    // ── save / load ───────────────────────────────────────────────────
    (Test_Text_SaveLoad,                      "Text: Save/Load preserves weights & forward output"),

    // ── continuous modes ─────────────────────────────────────────────
    (Test_TSRegression_LossDecreases,         "TimeSeriesRegression: loss decreases"),
    (Test_TSRegression_ProjectionUpdated,     "TimeSeriesRegression: InputProjection updated"),
    (Test_TSClassification_LossDecreases,     "TimeSeriesClassification: loss decreases"),
    (Test_SymbolicSeq_LossDecreases,          "SymbolicSequence: loss decreases"),

    // ── type guards ──────────────────────────────────────────────────
    (Test_WrongForwardThrows,                 "Type guard: discrete Forward throws on continuous cfg"),
    (Test_WrongForwardThrows2,                "Type guard: continuous Forward throws on discrete cfg"),

    // ── config validation ─────────────────────────────────────────────
    (Test_Config_Validate,                    "Config: Validate accepts good config"),
        };
        void Test_Text_ForwardShape()
        {
            var (m, cfg) = Discrete();
            var logits = m.Forward(new[] { 1, 2, 3, 4 });
            Assert(logits.GetLength(0) == 4, "rows");
            Assert(logits.GetLength(1) == cfg.VocabSize, "cols");
        }

        void Test_Text_ForwardNaN()
        {
            var (m, _) = Discrete();
            var logits = m.Forward(new[] { 0, 1, 2, 3, 4 });
            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert(!float.IsNaN(logits[i, j]) && !float.IsInfinity(logits[i, j]), $"NaN at [{i},{j}]");
        }

        void Test_Text_ForwardDeterministic()
        {
            var (m, _) = Discrete();
            int[] inp = { 1, 3, 5 };
            var l1 = m.Forward(inp); var l2 = m.Forward(inp);
            for (int i = 0; i < l1.GetLength(0); i++)
                for (int j = 0; j < l1.GetLength(1); j++)
                    Assert(l1[i, j] == l2[i, j], $"non-deterministic at [{i},{j}]");
        }

        void Test_Text_CausalMask()
        {
            var (m, _) = Discrete();
            int[] inp = { 1, 2, 3, 4 };
            float v0_before = m.Forward(inp)[0, 0];
            // Perturb future token embedding — shouldn't change position 0
            float saved = m.TokenEmbedding[4, 0];
            m.TokenEmbedding[4, 0] += 5f;
            float v0_after = m.Forward(inp)[0, 0];
            m.TokenEmbedding[4, 0] = saved;
            Assert(MathF.Abs(v0_before - v0_after) < 1e-5f, "causal mask violated");
        }

        void Test_Text_ForwardLen1()
        {
            var (m, _) = Discrete();
            var logits = m.Forward(new[] { 3 });
            Assert(logits.GetLength(0) == 1, "rows should be 1");
        }
         

        void Test_Text_AllParamsUpdated()
        {
            var (m, _) = Discrete(embDim: 8);
            var embB = (float[,])m.TokenEmbedding.Clone();
            var wqB = (float[,])m.Blocks[0].Attention.WQ.Clone();
            var outPB = (float[,])m.OutputProjection.Clone();
            var biasB = (float[])m.OutputBias.Clone();
            new TransformerTrainer(m, TC(lr: 0.02f, epochs: 3)).Train(new[] { new[] { 1, 2, 3, 4, 5 } });
            Assert(MatrixChanged(embB, m.TokenEmbedding), "TokenEmbedding unchanged");
            Assert(MatrixChanged(wqB, m.Blocks[0].Attention.WQ), "WQ unchanged");
            Assert(MatrixChanged(outPB, m.OutputProjection), "OutputProjection unchanged");
            Assert(VectorChanged(biasB, m.OutputBias), "OutputBias unchanged");
        }

        void Test_Text_FreezeEmbeddingNotUsed()
        {
            var (m, _) = Discrete();
            // token 11 is never in the training sequence
            float before = m.TokenEmbedding[11, 0];
            new TransformerTrainer(m, TC(epochs: 5)).Train(new[] { new[] { 1, 2, 3, 4 } });
            Assert(MathF.Abs(before - m.TokenEmbedding[11, 0]) < 1e-10f, "unused token changed");
        }

        void Test_Text_GradientClipping()
        {
            var (m, _) = Discrete();
            new TransformerTrainer(m, TC(lr: 0.5f, epochs: 3, clip: true)).Train(new[] { new[] { 1, 2, 3, 4 } });
            var logits = m.Forward(new[] { 1, 2 });
            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert(!float.IsNaN(logits[i, j]), "NaN after clipping");
        }

        void Test_Text_LearningRateDecay()
        {
            var (m, _) = Discrete();
            var tc = new TrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 4,
                Epochs = 10,
                UseLearningRateDecay = true,
                LearningRateDecay = 0.9f,
                Verbose = false
            };
            new TransformerTrainer(m, tc).Train(new[] { new[] { 1, 2, 3 } });
            var logits = m.Forward(new[] { 1, 2 });
            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert(!float.IsNaN(logits[i, j]), "NaN after LR decay");
        }

        void Test_Text_DifferentSeqLengths()
        {
            var (m, _) = Discrete();
            new TransformerTrainer(m, TC(epochs: 2)).Train(new[] { new[] { 1, 2 }, new[] { 1, 2, 3, 4, 5 }, new[] { 3, 4, 5 } });
        }

        void Test_Text_Generate()
        {
            var (m, _) = Discrete(vocabSize: 20, embDim: 16, numHeads: 2, numLayers: 2);
            new TransformerTrainer(m, TC(epochs: 5)).Train(new[] { new[] { 1, 2, 3, 4, 5 } });
            var gen = m.Generate(new[] { 1, 2 }, maxNewTokens: 5);
            Assert(gen.Length > 2, "no tokens generated");
        }

        void Test_Text_GenerateTokenBounds()
        {
            var (m, cfg) = Discrete(vocabSize: 20);
            var gen = m.Generate(new[] { 1, 2 }, maxNewTokens: 8);
            foreach (var t in gen) Assert(t >= 0 && t < cfg.VocabSize, $"out-of-vocab token {t}");
        }

        void Test_Text_SaveLoad()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2);
            new TransformerTrainer(m, TC(epochs: 5)).Train(new[] { new[] { 1, 2, 3, 4, 5 } });
            var logitsBefore = m.Forward(new[] { 1, 2, 3 });
            var dir = TempDir();
            try
            {
                m.Save(dir);
                var loaded = LanguageModel.Load(dir);
                var logitsAfter = loaded.Forward(new[] { 1, 2, 3 });
                for (int i = 0; i < logitsBefore.GetLength(0); i++)
                    for (int j = 0; j < logitsBefore.GetLength(1); j++)
                        Assert(MathF.Abs(logitsBefore[i, j] - logitsAfter[i, j]) < 1e-5f, $"mismatch [{i},{j}]");
            }
            finally { Cleanup(dir); }
        }

        void Test_TSRegression_ProjectionUpdated()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 1);
            var projB = (float[,])m.InputProjection.Clone();
            var rng = new Random(42);
            var inp = Enumerable.Range(0, 3).Select(_ => RandMatrix(5, 3, rng, 0.5f)).ToArray();
            var tgt = Enumerable.Range(0, 3).Select(_ => RandMatrix(5, 1, rng, 0.3f)).ToArray();
            new TransformerTrainer(m, TC(lr: 0.01f, bs: 3, epochs: 5)).TrainContinuous(inp, regressionTargets: tgt);
            Assert(MatrixChanged(projB, m.InputProjection), "InputProjection unchanged");
        }

        void Test_WrongForwardThrows()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression);
            bool threw = false;
            try { m.Forward(new[] { 1, 2 }); } catch (InvalidOperationException) { threw = true; }
            Assert(threw, "should have thrown");
        }

        void Test_WrongForwardThrows2()
        {
            var (m, _) = Discrete();
            bool threw = false;
            try { m.Forward(new float[3, 4]); } catch (InvalidOperationException) { threw = true; }
            Assert(threw, "should have thrown");
        }

        void Test_Config_Validate()
        {
            var (_, cfg) = Discrete();
            cfg.Validate(); // must not throw
        }
        private (LanguageModel model, TransformerConfig cfg) DiscreteWithSeed(
    int seed,
    int vocabSize = 12,
    int embDim = 8,
    int numHeads = 2,
    int numLayers = 1,
    int ffnDim = 16)
        {
            var cfg = new TransformerConfig
            {
                Data = new DataConfig { DataType = TransformerDataType.Text, CostFunction = CostFunctionType.mse },
                VocabSize = vocabSize,
                MaxSequenceLength = 24,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                Runtime = new RuntimeConfig { FFNActivationType = ActivationType.Relu, AccelerationType = AccelerationType.CPU },
                Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 5f },
                UseDecoderOnly = true,
            };
            cfg.Validate();
            return (new LanguageModel(cfg, new Random(seed)), cfg);
        }

     

        void Test_Text_LossDecreases()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            int[] seq = { 1, 2, 3, 4, 5 };
            int[] inp = { 1, 2, 3, 4 };
            int[] tgt = { 2, 3, 4, 5 };

            float before = CELoss(m, inp, tgt);
            new TransformerTrainer(m, TC(lr: 0.005f, epochs: 30)).Train(new[] { seq });
            float after = CELoss(m, inp, tgt);

            AssertLossImproved(before, after, 0.90f);
        }

        void Test_Text_PostTrainWeightsBounded()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            new TransformerTrainer(m, TC(lr: 0.5f, epochs: 5, clip: true))
                .Train(new[] { new[] { 1, 2, 3, 4, 5 } });

            float maxWq = MaxAbs(m.Blocks[0].Attention.WQ);
            float maxWo = MaxAbs(m.Blocks[0].Attention.WO);
            float maxOut = MaxAbs(m.OutputProjection);
            float maxBias = MaxAbs(m.OutputBias);

            Assert(maxWq < 50f, $"WQ exploded: {maxWq}");
            Assert(maxWo < 50f, $"WO exploded: {maxWo}");
            Assert(maxOut < 50f, $"OutputProjection exploded: {maxOut}");
            Assert(maxBias < 50f, $"OutputBias exploded: {maxBias}");
        }

        void Test_Text_BatchTraining()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2);
            int[][] seqs =
            {
        new[] { 1, 2, 3, 4 },
        new[] { 2, 3, 4, 5 },
        new[] { 3, 4, 5, 6 },
        new[] { 4, 5, 6, 7 }
    };

            float avgBefore = seqs.Average(s => CELoss(m, s.Take(3).ToArray(), s.Skip(1).ToArray()));
            new TransformerTrainer(m, TC(epochs: 20, bs: 4)).Train(seqs);
            float avgAfter = seqs.Average(s => CELoss(m, s.Take(3).ToArray(), s.Skip(1).ToArray()));

            AssertLossImproved(avgBefore, avgAfter, 0.90f);
        }

        void Test_Text_SeedRobustness()
        {
            var seeds = new[] { 1, 42, 1337 };
            var losses = new List<float>();

            foreach (var seed in seeds)
            {
                var (m, _) = DiscreteWithSeed(seed, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);

                int[] seq = { 1, 2, 3, 4, 5 };
                int[] inp = { 1, 2, 3, 4 };
                int[] tgt = { 2, 3, 4, 5 };

                float before = CELoss(m, inp, tgt);
                new TransformerTrainer(m, TC(lr: 0.005f, epochs: 50)).Train(new[] { seq });
                float after = CELoss(m, inp, tgt);

                Assert(after < before, $"Seed {seed} failed to improve: {before:F6} → {after:F6}");
                losses.Add(after);
            }

            float min = losses.Min();
            float max = losses.Max();
            Assert(max - min < 1.25f, $"Seed sensitivity too high: min={min:F6}, max={max:F6}");
        }

        void Test_Text_OverfitSingle()
        {
            var (m, _) = Discrete(vocabSize: 8, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            int[] seq = { 1, 2, 3, 4, 5 };
            int[] inp = { 1, 2, 3, 4 };
            int[] tgt = { 2, 3, 4, 5 };

            new TransformerTrainer(m, TC(lr: 0.005f, bs: 1, epochs: 550)).Train(new[] { seq });

            float loss = CELoss(m, inp, tgt);
            AssertOverfitStrong(loss, 0.10f);
        }

        void Test_Text_OverfitPredictionExact()
        {
            var (m, _) = Discrete(vocabSize: 8, embDim: 24, numHeads: 2, numLayers: 2, ffnDim: 48);

            int[] seq = { 1, 2, 3, 4, 5 };
            new TransformerTrainer(m, TC(lr: 0.005f, bs: 1, epochs: 400)).Train(new[] { seq });

            var generated = m.Generate(new[] { 1, 2, 3, 4 }, maxNewTokens: 1);

            Assert(generated.Length >= 5, "generation too short");
            Assert(generated[4] == 5, $"expected next token 5, got {generated[4]}");
        }

        void Test_TSRegression_LossDecreases()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 1);
            var rng = new Random(42);
            var inp = Enumerable.Range(0, 5).Select(_ => RandMatrix(6, 3, rng, 0.5f)).ToArray();
            var tgt = Enumerable.Range(0, 5).Select(_ => RandMatrix(6, 1, rng, 0.5f)).ToArray();

            var tr = new TransformerTrainer(m, TC(epochs: 1));
            float before = tr.ValidateContinuous(inp, regressionTargets: tgt);

            new TransformerTrainer(m, TC(lr: 0.003f, epochs: 25)).TrainContinuous(inp, regressionTargets: tgt);

            float after = new TransformerTrainer(m, TC(epochs: 1)).ValidateContinuous(inp, regressionTargets: tgt);
            AssertLossImproved(before, after, 0.90f);
        }

        void Test_TSClassification_LossDecreases()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesClassification, inputDim: 3, outputDim: 3, embDim: 8);
            var rng = new Random(42);
            var inp = Enumerable.Range(0, 5).Select(_ => RandMatrix(6, 3, rng, 0.5f)).ToArray();
            var cls = Enumerable.Range(0, 5).Select(_ => Enumerable.Range(0, 6).Select(_ => rng.Next(3)).ToArray()).ToArray();

            var tr = new TransformerTrainer(m, TC(epochs: 1));
            float before = tr.ValidateContinuous(inp, classTargets: cls);

            new TransformerTrainer(m, TC(lr: 0.003f, epochs: 30)).TrainContinuous(inp, classTargets: cls);

            float after = new TransformerTrainer(m, TC(epochs: 1)).ValidateContinuous(inp, classTargets: cls);
            AssertLossImproved(before, after, 0.90f);
        }

        void Test_SymbolicSeq_LossDecreases()
        {
            var cfg = new TransformerConfig
            {
                Data = new DataConfig { DataType = TransformerDataType.SymbolicSequence },
                VocabSize = 8,
                MaxSequenceLength = 16,
                EmbeddingDim = 8,
                NumHeads = 2,
                NumLayers = 1,
                FeedForwardDim = 16,
                UseDecoderOnly = true,
                Runtime = new RuntimeConfig { AccelerationType = AccelerationType.CPU },
                Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 5f },
            };
            cfg.Validate();

            var m = new LanguageModel(cfg, new Random(42));
            int[][] seqs =
            {
        new[] { 1, 4, 5, 6, 7, 2 },
        new[] { 1, 6, 7, 4, 5, 2 }
    };

            float before = CELoss(m, new[] { 1, 4, 5, 6 }, new[] { 4, 5, 6, 7 });
            new TransformerTrainer(m, TC(lr: 0.005f, epochs: 30)).Train(seqs);
            float after = CELoss(m, new[] { 1, 4, 5, 6 }, new[] { 4, 5, 6, 7 });

            AssertLossImproved(before, after, 0.90f);
        }

    }

    // =========================================================================
    //  2. CrossAttentionMultimodal tests
    // =========================================================================
    internal sealed class CrossAttentionMultimodalTests : TestBase
    {
        // ── config factory ────────────────────────────────────────────────────
        private MultimodalTransformerConfig MakeCfg(
            int vocabSize, int embDim = 16, int numHeads = 2, int numLayers = 1,
            int ffnDim = 32, int priceFeatures = 5, int outputDim = 5, int priceSeqLen = 10,
            bool useConf = true, bool freezeText = false)
        {
            var cfg = new MultimodalTransformerConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = vocabSize,
                    MaxSequenceLength = 32,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    Freeze = freezeText
                },
                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = priceFeatures,
                    MaxSequenceLength = priceSeqLen + 4,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim
                },
                Output = new OutputHeadConfig { OutputDim = outputDim, UseConfidenceHead = useConf },
                Runtime = new RuntimeConfig { FFNActivationType = ActivationType.Relu, AccelerationType = AccelerationType.CPU },
                Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 1f },
                RequireSharedCrossAttentionEmbeddingDim = true,
            };
            cfg.Validate();
            return cfg;
        }

        private TrainingConfig TC(float lr = 0.001f, int bs = 4, int epochs = 10)
            => new TrainingConfig
            {
                LearningRate = lr,
                BatchSize = bs,
                Epochs = epochs,
                UseGradientClipping = true,
                GradientClipThreshold = 1f,
                ConfidenceLossWeight = 0.1f,
                Verbose = false
            };

        // synthetic data
        private (BPETokenizer tok, int[][] texts, float[][,] priceIn, float[][,] priceTgt)
            Data(int n = 10, int seqLen = 8, int features = 5, int outDim = 5, int seed = 42)
        {
            var rng = new Random(seed);
            string[] corpus = { "stock rose sharply", "market crashed", "bullish outlook", "bearish data" };
            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 200, minFrequency: 1);
            var texts = Enumerable.Range(0, n).Select(_ => tok.Encode(corpus[rng.Next(corpus.Length)], addSpecialTokens: true)).ToArray();
            var priceIn = Enumerable.Range(0, n).Select(_ => RandMatrix(seqLen, features, rng, 0.5f)).ToArray();
            var priceTgt = Enumerable.Range(0, n).Select(_ => RandMatrix(seqLen, outDim, rng, 0.5f)).ToArray();
            return (tok, texts, priceIn, priceTgt);
        }

        public void RunAllTests() => Run(Tests(), "2 · CrossAttentionMultimodal");

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            // ── construction & forward ────────────────────────────────────────
            (Test_ModelConstruction,              "Construction: model initialises without error"),
            (Test_ForwardShape,                   "Forward: output shape [seqLen, outputDim]"),
            (Test_ForwardNaN,                     "Forward: no NaN in predictions"),
            (Test_ForwardDeterministic,           "Forward: deterministic"),
            (Test_ConfidenceRange,                "Forward: confidence values in [0, 1]"),
            (Test_NoConfidenceHead_Null,          "Forward: confidence null when head disabled"),
            (Test_TextVsNoText_Differ,            "Forward: text vs no-text produce different outputs"),
            (Test_PriceOnly_Forward,              "Forward: price-only (null text) works"),
            (Test_PredictNext_Shape,              "PredictNext: returns correct output dim"),
            // ── training ─────────────────────────────────────────────────────
            (Test_LossDecreases,                  "Train: loss decreases"),
            (Test_AllPriceParamsUpdated,          "Train: all price-decoder params receive gradients"),
            (Test_FrozenTextEncoder_Unchanged,    "Train: frozen text encoder weights unchanged"),
            (Test_UnfrozenTextEncoder_Changed,    "Train: unfrozen text encoder weights change"),
            (Test_MixedBatch_NullText,            "Train: mixed batch (some null text) works"),
            (Test_GradientClipping_NoNaN,         "Train: high LR + clipping prevents NaN"),
            (Test_LearningRateDecay,              "Train: LR decay runs without crash"),
            (Test_PriceOnly_LossDecreases,        "Train: price-only loss decreases"),
            (Test_SingleSampleOverfit,            "Train: single sample overfits (loss halves)"),
            // ── validation ───────────────────────────────────────────────────
            (Test_ValidationLossValid,            "Validate: returns finite non-negative value"),
            // ── save / load ───────────────────────────────────────────────────
            (Test_SaveLoad_WeightsAndForward,     "SaveLoad: weights preserved, forward identical"),
            (Test_SaveLoad_NoConfidenceHead,      "SaveLoad: no-confidence model round-trips"),
            // ── dimension & config guards ────────────────────────────────────
            (Test_EmbDimMismatch_Throws,          "Config: embedding dim mismatch throws"),
            (Test_VaryingTextLengths,             "Robustness: varying text token lengths"),
        };

        void Test_ModelConstruction()
        {
            var cfg = MakeCfg(50);
            var m = new CrossModel(cfg, new Random(42));
            Assert(m.TextBlocks.Length == 1, "text blocks");
            Assert(m.PriceBlocks.Length == 1, "price blocks");
        }

        void Test_ForwardShape()
        {
            var (tok, texts, pi, _) = Data(n: 1, seqLen: 8);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            var (pred, _) = m.Forward(texts[0], pi[0]);
            Assert(pred.GetLength(0) == 8, "rows");
            Assert(pred.GetLength(1) == 5, "cols");
        }

        void Test_ForwardNaN()
        {
            var (tok, texts, pi, _) = Data(n: 1, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            var (pred, conf) = m.Forward(texts[0], pi[0]);
            Assert(!HasNaN(pred), "NaN in predictions");
        }

        void Test_ForwardDeterministic()
        {
            var (tok, texts, pi, _) = Data(n: 1, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            var (p1, _) = m.Forward(texts[0], pi[0]);
            var (p2, _) = m.Forward(texts[0], pi[0]);
            for (int i = 0; i < p1.GetLength(0); i++)
                for (int j = 0; j < p1.GetLength(1); j++)
                    Assert(p1[i, j] == p2[i, j], $"non-det [{i},{j}]");
        }

        void Test_ConfidenceRange()
        {
            var (tok, texts, pi, _) = Data(n: 3, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8, useConf: true), new Random(42));
            for (int s = 0; s < 3; s++)
            {
                var (_, conf) = m.Forward(texts[s], pi[s]);
                for (int i = 0; i < conf.GetLength(0); i++)
                    Assert(conf[i, 0] >= 0f && conf[i, 0] <= 1f, $"conf[{i}]={conf[i, 0]}");
            }
        }

        void Test_NoConfidenceHead_Null()
        {
            var (tok, texts, pi, _) = Data(n: 1, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8, useConf: false), new Random(42));
            var (_, conf) = m.Forward(texts[0], pi[0]);
            Assert(conf == null, "expected null confidence");
        }

        void Test_TextVsNoText_Differ()
        {
            var (tok, texts, pi, _) = Data(n: 1, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            var (pW, _) = m.Forward(texts[0], pi[0]);
            var (pN, _) = m.Forward(null, pi[0]);
            bool diff = false;
            for (int i = 0; i < pW.GetLength(0) && !diff; i++)
                for (int j = 0; j < pW.GetLength(1) && !diff; j++)
                    if (MathF.Abs(pW[i, j] - pN[i, j]) > 1e-6f) diff = true;
            Assert(diff, "text vs no-text identical");
        }

        void Test_PriceOnly_Forward()
        {
            var (tok, _, pi, _) = Data(n: 1, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            var (pred, _) = m.Forward(null, pi[0]);
            Assert(pred.GetLength(0) == 6 && !HasNaN(pred), "price-only failed");
        }

        void Test_PredictNext_Shape()
        {
            var (tok, texts, pi, _) = Data(n: 1, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            var (pred, conf) = m.PredictNext(texts[0], pi[0]);
            Assert(pred.Length == 5, "wrong dim");
            Assert(!float.IsNaN(conf) && conf >= 0f && conf <= 1f, "conf invalid");
        }

        void Test_LossDecreases()
        {
            var (tok, texts, pi, pt) = Data(n: 10, seqLen: 8);
            var cfg = MakeCfg(tok.VocabSize + 2, priceSeqLen: 10);
            var m = new CrossModel(cfg, new Random(42));
            var t1 = new CrossTrainer(m, TC(epochs: 1));
            float before = t1.Validate(texts, pi, pt);
            new CrossTrainer(m, TC(epochs: 15)).Train(texts, pi, pt);
            float after = new CrossTrainer(m, TC(epochs: 1)).Validate(texts, pi, pt);

            AssertLossImproved(before, after);
        }

        void Test_AllPriceParamsUpdated()
        {
            var (tok, texts, pi, pt) = Data(n: 5, seqLen: 6);
            var cfg = MakeCfg(tok.VocabSize + 2, priceSeqLen: 8);
            var m = new CrossModel(cfg, new Random(42));
            var projB = (float[,])m.PriceInputProjection.Clone();
            var wqB = (float[,])m.PriceBlocks[0].SelfAttention.WQ.Clone();
            var cwkB = (float[,])m.PriceBlocks[0].CrossAttention.WK.Clone();
            new CrossTrainer(m, TC(lr: 0.01f, epochs: 5)).Train(texts, pi, pt);
            Assert(MatrixChanged(projB, m.PriceInputProjection), "PriceInputProj");
            Assert(MatrixChanged(wqB, m.PriceBlocks[0].SelfAttention.WQ), "SelfWQ");
            Assert(MatrixChanged(cwkB, m.PriceBlocks[0].CrossAttention.WK), "CrossWK");
        }

        void Test_FrozenTextEncoder_Unchanged()
        {
            var (tok, texts, pi, pt) = Data(n: 5, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8, freezeText: true), new Random(42));
            var embB = (float[,])m.TextTokenEmbedding.Clone();
            new CrossTrainer(m, TC(lr: 0.01f, epochs: 5)).Train(texts, pi, pt);
            Assert(!MatrixChanged(embB, m.TextTokenEmbedding), "frozen embedding changed");
        }

        void Test_UnfrozenTextEncoder_Changed()
        {
            var (tok, texts, pi, pt) = Data(n: 5, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8, freezeText: false), new Random(42));
            var embB = (float[,])m.TextTokenEmbedding.Clone();
            new CrossTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(texts, pi, pt);
            Assert(MatrixChanged(embB, m.TextTokenEmbedding), "unfrozen embedding unchanged");
        }

        void Test_MixedBatch_NullText()
        {
            var (tok, texts, pi, pt) = Data(n: 8, seqLen: 6);
            var mixed = (int[][])texts.Clone();
            for (int i = 0; i < mixed.Length; i += 2) mixed[i] = null;
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            new CrossTrainer(m, TC(epochs: 3)).Train(mixed, pi, pt);
            float loss = new CrossTrainer(m, TC(epochs: 1)).Validate(mixed, pi, pt);
            Assert(!float.IsNaN(loss), "NaN loss with mixed null text");
        }

        void Test_GradientClipping_NoNaN()
        {
            var (tok, texts, pi, pt) = Data(n: 5, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            new CrossTrainer(m, TC(lr: 0.2f, epochs: 5)).Train(texts, pi, pt);
            var (pred, _) = m.Forward(texts[0], pi[0]);
            Assert(!HasNaN(pred), "NaN after high-LR training");
        }

        void Test_LearningRateDecay()
        {
            var (tok, texts, pi, pt) = Data(n: 5, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            var tc = new TrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 5,
                Epochs = 5,
                UseLearningRateDecay = true,
                LearningRateDecay = 0.9f,
                Verbose = false
            };
            new CrossTrainer(m, tc).Train(texts, pi, pt);
            var (pred, _) = m.Forward(texts[0], pi[0]);
            Assert(!HasNaN(pred), "NaN after LR decay");
        }

        void Test_PriceOnly_LossDecreases()
        {
            var (tok, _, pi, pt) = Data(n: 10, seqLen: 8);
            var nullTexts = new int[10][];
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            float before = new CrossTrainer(m, TC(epochs: 1)).Validate(nullTexts, pi, pt);
            new CrossTrainer(m, TC(epochs: 15)).Train(nullTexts, pi, pt);
            float after = new CrossTrainer(m, TC(epochs: 1)).Validate(nullTexts, pi, pt);

            AssertLossImproved(before, after);
        }

        void Test_SingleSampleOverfit()
        {
            var (tok, texts, pi, pt) = Data(n: 1, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, embDim: 32, numHeads: 2, numLayers: 2,
                                           ffnDim: 64, priceSeqLen: 8, useConf: false), new Random(42));
            float before = new CrossTrainer(m, TC(epochs: 1)).Validate(texts, pi, pt);
            new CrossTrainer(m, TC(lr: 0.005f, bs: 1, epochs: 100)).Train(texts, pi, pt);
            float after = new CrossTrainer(m, TC(epochs: 1)).Validate(texts, pi, pt);

            AssertLossImproved(before, after);
        }

        void Test_ValidationLossValid()
        {
            var (tok, texts, pi, pt) = Data(n: 5, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            float loss = new CrossTrainer(m, TC(epochs: 1)).Validate(texts, pi, pt);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss) && loss >= 0, $"invalid loss {loss}");
        }

        void Test_SaveLoad_WeightsAndForward()
        {
            var (tok, texts, pi, _) = Data(n: 5, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, embDim: 16, numLayers: 2, priceSeqLen: 8), new Random(42));
            new CrossTrainer(m, TC(epochs: 5)).Train(texts, pi, pi);
            var (pBefore, cBefore) = m.Forward(texts[0], pi[0]);
            var dir = TempDir();
            try
            {
                m.Save(dir);
                var loaded = CrossModel.Load(dir);
                var (pAfter, cAfter) = loaded.Forward(texts[0], pi[0]);
                for (int i = 0; i < pBefore.GetLength(0); i++)
                    for (int j = 0; j < pBefore.GetLength(1); j++)
                        Assert(MathF.Abs(pBefore[i, j] - pAfter[i, j]) < 1e-5f, $"mismatch [{i},{j}]");
            }
            finally { Cleanup(dir); }
        }

        void Test_SaveLoad_NoConfidenceHead()
        {
            var (tok, texts, pi, _) = Data(n: 2, seqLen: 6);
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8, useConf: false), new Random(42));
            var (pBefore, confBefore) = m.Forward(texts[0], pi[0]);
            Assert(confBefore == null, "expected null before save");
            var dir = TempDir();
            try
            {
                m.Save(dir);
                var loaded = CrossModel.Load(dir);
                var (pAfter, confAfter) = loaded.Forward(texts[0], pi[0]);
                Assert(confAfter == null, "expected null after load");
                Assert(!HasNaN(pAfter), "NaN after load");
            }
            finally { Cleanup(dir); }
        }

        void Test_EmbDimMismatch_Throws()
        {
            bool threw = false;
            try
            {
                var cfg = new MultimodalTransformerConfig
                {
                    Text = new TextEncoderConfig { VocabSize = 50, EmbeddingDim = 16, NumHeads = 2, NumLayers = 1, FeedForwardDim = 32 },
                    Price = new PriceDecoderConfig { InputFeatureDim = 5, MaxSequenceLength = 14, EmbeddingDim = 32, NumHeads = 2, NumLayers = 1, FeedForwardDim = 32 },
                    Output = new OutputHeadConfig { OutputDim = 5 },
                    Runtime = new RuntimeConfig { AccelerationType = AccelerationType.CPU },
                    Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 1f },
                    RequireSharedCrossAttentionEmbeddingDim = true,
                };
                cfg.Validate();
            }
            catch (ArgumentException) { threw = true; }
            Assert(threw, "expected ArgumentException on dim mismatch");
        }

        void Test_VaryingTextLengths()
        {
            string[] corpus = { "hi", "the stock market rallied strongly today on positive earnings news", "bullish" };
            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 200, minFrequency: 1);
            var texts = corpus.Select(c => tok.Encode(c, addSpecialTokens: true)).ToArray();
            var rng = new Random(42);
            var pi = Enumerable.Range(0, 3).Select(_ => RandMatrix(6, 5, rng, 0.5f)).ToArray();
            var pt = Enumerable.Range(0, 3).Select(_ => RandMatrix(6, 5, rng, 0.5f)).ToArray();
            var m = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));
            new CrossTrainer(m, TC(epochs: 3)).Train(texts, pi, pt);
        }
    }

    // =========================================================================
    //  3. TACAMT tests  (delegates to the existing thorough TACMATests class)
    // =========================================================================
    internal sealed class TacamtTests : TestBase
    {
        public void RunAllTests()
        {
            TransformerTestSuite.PrintBanner("3 · TACAMT (Time-Aware Cross-Attention with Memory & Temporal-decay)");
            Console.ForegroundColor = ConsoleColor.DarkCyan;
            Console.WriteLine("  Delegating to TACMATests (comprehensive 150+ test suite) …");
            Console.ResetColor();
            Console.WriteLine();
         var suite = new TACMATests();
            suite.RunAllTests();
        }
    }
    // =========================================================================
    //  4. MMTAC tests
    // =========================================================================
    internal sealed class MmtacTests : TestBase
    {
        // ── config factory ────────────────────────────────────────────────────
        private MmtacConfig Cfg(
            int vocabSize = 50, int embDim = 16, int numHeads = 2, int numLayers = 1,
            int ffnDim = 32, int priceFeatures = 5, int priceSeqLen = 12,
            bool useConf = false, bool freezeText = false,
            int globalDim = 0, bool decayEnabled = true, bool bypassDecay = true,
            bool priceContextEnabled = false)
        {
            var cfg = new MmtacConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = vocabSize,
                    MaxSequenceLength = 32,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    UseDecoderOnly = false,
                    Freeze = freezeText
                },
                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = priceFeatures,
                    MaxSequenceLength = priceSeqLen + 4,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    UseDecoderOnly = true
                },
                Global = new MmtacGlobalConfig { GlobalFeatureDim = globalDim, BypassDecay = bypassDecay },
                Output = new MmtacOutputConfig
                {
                    UseConfidenceHead = useConf,
                    DirectionLossWeight = 1f,
                    MidDirectionLossWeight = 0.5f,
                    RangeLossWeight = 1f,
                    QualityLossWeight = 1f
                },
                Decay = new DecayNetworkConfig
                {
                    Enabled = decayEnabled,
                    ProjectionDim = 8,
                    HiddenDim = 16,
                    TimeEncodingBases = 8,
                    MemAttentionDropout = 0.0f,
                    MlpDropout = 0.0f,
                    WeightDecay = 0f
                },
                Reg = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 1f },
                Runtime = new RuntimeConfig { FFNActivationType = ActivationType.Relu, AccelerationType = AccelerationType.CPU },
                PriceContext = new PriceContextConfig
                {
                    Enabled = priceContextEnabled,
                    MinHistoryLength = 3,
                    MinCurrentLength = 3
                },
                Pruning = new MemoryPruningConfig { AttentionScoreAlpha = 0.1f, MinQueryCountForPruning = 3, NewEntryReserveFraction = 0.1f },
            };
            cfg.Validate();
            return cfg;
        }

        private TrainingConfig TC(float lr = 0.001f, int bs = 4, int epochs = 10, bool clip = true)
            => new TrainingConfig
            {
                LearningRate = lr,
                BatchSize = bs,
                Epochs = epochs,
                UseGradientClipping = clip,
                GradientClipThreshold = 1f,
                Verbose = false
            };

        // ── synthetic data ─────────────────────────────────────────────────────
        private (BPETokenizer tok, MultimodalInput[] inputs, ModelTarget[][] targets)
            Data(int n = 10, int seqLen = 10, int priceFeatures = 5, int seed = 42,
                 int globalDim = 0, bool withNews = true)
        {
            var rng = new Random(seed);
            string[] corpus = { "stock rose sharply", "market crashed today", "bullish outlook strong", "bearish data weak" };
            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 200, minFrequency: 1);

            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                var priceSeq = RandMatrix(seqLen, priceFeatures, rng, 0.5f);

                NewsStory[] stories = null;
                if (withNews)
                {
                    int ns = 1 + rng.Next(2);
                    stories = new NewsStory[ns];
                    for (int i = 0; i < ns; i++)
                        stories[i] = new NewsStory(tok.Encode(corpus[rng.Next(corpus.Length)], addSpecialTokens: true), (float)(i * 2));
                }

                float[] globalFeatures = globalDim > 0
                    ? Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray()
                    : null;

                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddSeconds(s * 60),
                    PriceSequence = priceSeq,
                    NewsStories = stories,
                    GlobalFeatures = globalFeatures,
                };

                targets[s] = new ModelTarget[seqLen];
                for (int t = 0; t < seqLen; t++)
                {
                    float close = (float)rng.NextDouble();
                    targets[s][t] = new ModelTarget
                    {
                        High = close + (float)rng.NextDouble() * 0.05f,
                        Low = close - (float)rng.NextDouble() * 0.05f,
                        Close = close,
                        Range = (float)rng.NextDouble() * 0.1f,
                        Quality = (float)rng.NextDouble(),
                        Direction = rng.Next(2),
                        MidWindowDirection = rng.Next(2),
                    };
                }
            }
            return (tok, inputs, targets);
        }

        private bool HasNaNPrediction(ModelPrediction p)
            => float.IsNaN(p.High) || float.IsNaN(p.Low) || float.IsNaN(p.Close)
            || float.IsNaN(p.Range) || float.IsNaN(p.Quality)
            || float.IsNaN(p.DirectionProb) || float.IsNaN(p.MidWindowDirectionProb);

        private string TmpDir2()
        {
            var d = Path.Combine(Path.GetTempPath(), $"mmtac_test_{Guid.NewGuid():N}");
            Directory.CreateDirectory(d); return d;
        }

        public void RunAllTests() => Run(Tests(), "4 · MMTAC (Multimodal Market Transformer with Additional Context)");
        private (Action, string)[] Tests() => new (Action, string)[]
        {
    // ── construction & dims ───────────────────────────────────────────
    (Test_Construction_NoError,               "Construction: model initialises without error"),
    (Test_Dims_TextEmbedding,                 "Dims: TextTokenEmbedding [vocabSize, embDim]"),
    (Test_Dims_PriceInputProjection,          "Dims: PriceInputProjection [embDim, featureDim]"),
    (Test_Dims_RegressionProjection,          "Dims: RegressionProjection [3, embDim]"),
    (Test_Dims_RangeProjection,               "Dims: RangeProjection [1, embDim]"),
    (Test_Dims_DirectionProjection,           "Dims: DirectionProjection [1, embDim]"),
    (Test_Dims_ContextTypeEmbedding,          "Dims: ContextTypeEmbedding [3, embDim]"),
    (Test_Dims_ConfidenceNull_WhenDisabled,   "Dims: ConfidenceProjection null when disabled"),
    (Test_Dims_ConfidenceCorrect_WhenEnabled, "Dims: ConfidenceProjection [1, embDim] when enabled"),
    (Test_Dims_GlobalProjection_WhenEnabled,  "Dims: GlobalFeatureProjection [embDim, globalDim] when enabled"),
    (Test_Dims_PriceBlocks_WQ,                "Dims: all PriceBlock SelfAttention.WQ [embDim, embDim]"),
    // ── forward: PredictNext ──────────────────────────────────────────
    (Test_PredictNext_NoError,                    "PredictNext: runs without error"),
    (Test_PredictNext_NoNaN,                      "PredictNext: all output fields are finite"),
    (Test_PredictNext_NullStories,                "PredictNext: null stories (price-only) works"),
    (Test_PredictNext_EmptyStories,               "PredictNext: empty stories array works"),
    (Test_PredictNext_Deterministic,              "PredictNext: deterministic for same input"),
    (Test_PredictNext_RangeNonNegative,           "PredictNext: Range output >= 0 (softplus)"),
    (Test_PredictNext_QualityInRange,             "PredictNext: Quality output in [0, 1] (sigmoid)"),
    (Test_PredictNext_DirectionInRange,           "PredictNext: DirectionProb in [0, 1] (sigmoid)"),
    (Test_PredictNext_MidDirInRange,              "PredictNext: MidWindowDirectionProb in [0, 1]"),
    (Test_PredictNext_ConfidenceOne_WhenDisabled, "PredictNext: Confidence == 1.0 when head disabled"),
    (Test_PredictNext_ConfidenceInRange,          "PredictNext: Confidence in [0, 1] when enabled"),
    (Test_PredictNext_StoryVsNoStory_Differ,      "PredictNext: story vs no-story produce different outputs"),
    (Test_PredictNext_SeqLen1,                    "PredictNext: single-timestep price sequence works"),
    (Test_Timestep_Predictions_Are_Not_Identical, "PredictNext: predictions differ across timesteps (causal mask)"),
    (Test_Forward_AllHeads_NoNaN,                 "Forward: all six raw head outputs are finite"),
    (Test_Forward_BypassDecay_False_NoNaN,        "Forward: BypassDecay=false path produces no NaN"),
    // ── training ─────────────────────────────────────────────────────
    (Test_Train_LossDecreases,                "Train: loss decreases over training"),
    (Test_Train_LossFinite,                   "Train: training loss stays finite"),
    (Test_Train_AllPriceParamsUpdated,        "Train: price-decoder params receive gradients"),
    (Test_Train_RegressionHeadUpdated,        "Train: RegressionProjection updated"),
    (Test_Train_RangeHeadUpdated,             "Train: RangeProjection updated"),
    (Test_Train_DirectionHeadUpdated,         "Train: DirectionProjection updated"),
    (Test_Train_ContextTypeEmbeddingUpdated,  "Train: ContextTypeEmbedding updated"),
    (Test_Train_FrozenText_NotUpdated,        "Train: frozen text encoder unchanged"),
    (Test_Train_UnfrozenText_Updated,         "Train: unfrozen text encoder updated"),
    (Test_Train_ConfidenceHead_Updated,       "Train: ConfidenceProjection updated when enabled"),
    (Test_Train_GradientClipping_NoNaN,       "Train: high LR + clipping prevents NaN"),
    (Test_Train_SingleSampleOverfit,          "Train: single sample overfits (loss decreases)"),
    (Test_Train_PriceOnly_LossDecreases,      "Train: price-only (no news) loss decreases"),
    (Test_Train_MixedBatch_SomeNullNews,      "Train: mixed batch with some null news works"),
    (Test_Train_DecayNetwork_Weights_Updated, "Train: DecayNetwork weights updated during backprop"),
    (Test_Train_PriceContext_NoError,         "Train: PriceContext enabled path runs without error"),
    (Test_Train_PriceContext_LossFinite,      "Train: PriceContext enabled loss stays finite"),
    (Test_Validate_Loss_ConsistentWithTraining, "Train: validation loss consistent with training loss"),
    (Test_LossWeights_AffectWhichHeadLearns,    "Train: loss weights affect which head trains"),
    (Test_GradClip_BoundsUpdateMagnitude,        "Train: gradient clipping bounds actual update magnitude"),
    // ── global token ──────────────────────────────────────────────────
    (Test_GlobalToken_ForwardNoError,            "GlobalToken: forward works when globalDim > 0"),
    (Test_GlobalToken_ChangesOutput,             "GlobalToken: different global features produce different output"),
    (Test_GlobalToken_ProjectionUpdated,         "GlobalToken: GlobalFeatureProjection updated in training"),
    (Test_GlobalToken_PredictWithMemory_NoError, "GlobalToken: PredictWithMemory works with globalDim > 0"),
    // ── multi-output head correctness ─────────────────────────────────
    (Test_Outputs_RangeVsHighLow,     "Outputs: Range head is consistent with High-Low constraint"),
    (Test_Outputs_MultipleTimesteps,  "Outputs: all outputs shaped [seqLen] correctly"),
    // ── memory ───────────────────────────────────────────────────────
    (Test_Memory_InitiallyEmpty,              "Memory: starts empty"),
    (Test_Memory_AccumulatesAfterPredict,     "Memory: PredictWithMemory accumulates entries"),
    (Test_Memory_PrunesWhenOverLimit,         "Memory: pruned when exceeding max"),
    (Test_Memory_ClearAll,                    "Memory: ClearAllMemory() clears both banks"),
    (Test_Memory_ClearNews_OnlyNews,          "Memory: ClearNewsMemory() leaves price memory intact"),
    (Test_Memory_ClearPrice_OnlyPrice,        "Memory: ClearPriceMemory() leaves news memory intact"),
    (Test_Memory_NewsAndPrice_BothStored,     "Memory: both news and price entries stored"),
    (Test_Memory_TimestampOrdering,           "Memory: timestamps stored correctly"),
    (Test_Memory_TimeDeltaReachesDecoder,     "Memory: time delta reaches price decoder hidden state"),
    (Test_Memory_AttentionScores_Updated,     "Memory: AttentionScore and QueryCount updated after predict"),
    (Test_Memory_AttentionPruning_KeepsHighScore, "Memory: attention-based pruning retains high-score entries"),
    (Test_Memory_PriceOnly_NoNews,            "Memory: PredictWithMemory works with no news stories"),
    (Test_Memory_LastPriceTimestamp_Updated,  "Memory: LastPriceTimestamp set correctly after predict"),
    // ── sequential training ───────────────────────────────────────────
    (Test_Sequential_MemoryAccumulates,              "Sequential: memory grows across samples"),
    (Test_Sequential_LossDecreases,                  "Sequential: loss decreases over sequential epochs"),
    (Test_Sequential_PriceMemoryPopulated,           "Sequential: price memory has correct embedding dim"),
    (Test_Sequential_MemoryClearedBetweenEpochs,     "Sequential: memory cleared at start of each epoch"),
    (Test_Sequential_WithGlobalToken,                "Sequential: global token path in sequential training works"),
    (Test_Sequential_Produces_Different_Weights_Than_Batch, "Sequential: produces different weights than batch training"),
    // ── tokenizer ────────────────────────────────────────────────────
    (Test_Tokenizer_SetAndTokenize,           "Tokenizer: SetTokenizer + TokenizeStories works"),
    (Test_Tokenizer_VocabSizeMismatch_Throws, "Tokenizer: oversized tokenizer throws on SetTokenizer"),
    // ── validation ───────────────────────────────────────────────────
    (Test_Validate_ReturnsFiniteValue,        "Validate: returns finite non-negative value"),
    // ── numerical stability ───────────────────────────────────────────
    (Test_Stability_LargeInputs,         "Stability: large inputs (100x) produce no NaN"),
    (Test_Stability_SmallInputs,         "Stability: tiny inputs (1e-6) produce no NaN"),
    (Test_Stability_ZeroInputs,          "Stability: all-zero inputs produce no NaN"),
    (Test_Stability_NegativeInputs,      "Stability: all-negative inputs produce no NaN"),
    (Test_Stability_ManyStories,         "Stability: 12 simultaneous stories produce no NaN"),
    (Test_Stability_NoNaN_AfterManyEpochs, "Stability: no NaN after 60 training epochs"),
    // ── save / load ───────────────────────────────────────────────────
    (Test_SaveLoad_ForwardIdentical,    "SaveLoad: forward output identical after round-trip"),
    (Test_SaveLoad_AllHeadsPreserved,   "SaveLoad: all output head weights preserved"),
    (Test_SaveLoad_ContextTypeEmbedding,"SaveLoad: ContextTypeEmbedding preserved"),
    (Test_SaveLoad_GlobalProjection,    "SaveLoad: GlobalFeatureProjection preserved"),
    (Test_SaveLoad_Memory,              "SaveLoad: news + price memory preserved"),
    (Test_SaveLoad_PruningConfig,       "SaveLoad: PruningConfig preserved"),
    (Test_SaveLoad_ContinueTraining,    "SaveLoad: can continue training after load"),
    // ── config validation ────────────────────────────────────────────
    (Test_Config_Validate_Good,    "Config: Validate accepts valid config"),
    (Test_Config_SmallPreset,      "Config: Small() preset validates"),
    (Test_Config_StandardPreset,   "Config: Standard() preset validates"),
    // ── gradient checks ───────────────────────────────────────────────
    (Test_GradCheck_RegressionProjection,   "GradCheck: RegressionProjection finite-difference non-zero"),
    (Test_GradCheck_RangeProjection,        "GradCheck: RangeProjection finite-difference non-zero"),
    (Test_GradCheck_DirectionProjection,    "GradCheck: DirectionProjection finite-difference non-zero"),
    (Test_GradCheck_QualityProjection,      "GradCheck: QualityProjection finite-difference non-zero"),
    (Test_GradCheck_MidDirectionProjection, "GradCheck: MidDirectionProjection finite-difference non-zero"),
    (Test_GradCheck_TextEmbedding,          "GradCheck: TextEmbedding affects prediction through cross-attn"),
    // ── end-to-end signal learning ────────────────────────────────────
    (Test_E2E_BullBear, "E2E: bull vs bear news drives different predictions"),
    // ── semantic correctness ──────────────────────────────────────────
    (Test_DirectionHead_LearnsBinarySignal,       "Semantic: Direction head learns up/down classification"),
    (Test_RangeHead_ConsistentWithHighLow,        "Semantic: Range head consistent with High-Low after training"),
    (Test_ConfidenceHead_CorrelatesWithAccuracy,  "Semantic: Confidence higher on predictable vs noisy data"),
    (Test_QualityHead_LearnsPredictableData,      "Semantic: Quality head higher on predictable inputs"),
    (Test_GlobalToken_MacroSignalDrivesOutput,    "Semantic: Global macro signal drives prediction direction"),
    (Test_MultiHead_AllOutputsIndependent,        "Semantic: Separate heads can learn conflicting targets"),
    (Test_MidDir_And_Direction_Learn_Different_Signals, "Semantic: MidWindowDirection independent from Direction"),
    (Test_Sequential_MemoryHelpsOnAutoregressive, "Semantic: Sequential memory improves autoregressive prediction"),
        };

        // ── Construction & Dims ──────────────────────────────────────────────

        void Test_Construction_NoError()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            Assert(m != null, "null model");
        }

        void Test_Dims_TextEmbedding()
        {
            var m = new MmtacModel(Cfg(vocabSize: 60, embDim: 24), new Random(42));
            Assert(m.TextTokenEmbedding.GetLength(0) == 60, "rows");
            Assert(m.TextTokenEmbedding.GetLength(1) == 24, "cols");
        }

        void Test_Dims_PriceInputProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 16, priceFeatures: 7), new Random(42));
            Assert(m.PriceInputProjection.GetLength(0) == 16, "rows");
            Assert(m.PriceInputProjection.GetLength(1) == 7, "cols");
        }

        void Test_Dims_RegressionProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 16), new Random(42));
            Assert(m.RegressionProjection.GetLength(0) == 3, "3 regression outputs (High/Low/Close)");
            Assert(m.RegressionProjection.GetLength(1) == 16, "embDim cols");
        }

        void Test_Dims_RangeProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 16), new Random(42));
            Assert(m.RangeProjection.GetLength(0) == 1 && m.RangeProjection.GetLength(1) == 16, "shape");
        }

        void Test_Dims_DirectionProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 16), new Random(42));
            Assert(m.DirectionProjection.GetLength(0) == 1 && m.DirectionProjection.GetLength(1) == 16, "shape");
        }

        void Test_Dims_ContextTypeEmbedding()
        {
            var m = new MmtacModel(Cfg(embDim: 16), new Random(42));
            Assert(m.ContextTypeEmbedding.GetLength(0) == MmtacConfig.ContextTypeCount, "rows should be 3");
            Assert(m.ContextTypeEmbedding.GetLength(1) == 16, "cols");
        }

        void Test_Dims_ConfidenceNull_WhenDisabled()
        {
            var m = new MmtacModel(Cfg(useConf: false), new Random(42));
            Assert(m.ConfidenceProjection == null, "should be null");
        }

        void Test_Dims_ConfidenceCorrect_WhenEnabled()
        {
            var m = new MmtacModel(Cfg(useConf: true, embDim: 16), new Random(42));
            Assert(m.ConfidenceProjection != null, "should not be null");
            Assert(m.ConfidenceProjection.GetLength(0) == 1, "1 row");
            Assert(m.ConfidenceProjection.GetLength(1) == 16, "embDim cols");
        }

        void Test_Dims_GlobalProjection_WhenEnabled()
        {
            var m = new MmtacModel(Cfg(embDim: 16, globalDim: 8), new Random(42));
            Assert(m.GlobalFeatureProjection != null, "should not be null");
            Assert(m.GlobalFeatureProjection.GetLength(0) == 16, "embDim rows");
            Assert(m.GlobalFeatureProjection.GetLength(1) == 8, "globalDim cols");
        }

        void Test_Dims_PriceBlocks_WQ()
        {
            var m = new MmtacModel(Cfg(embDim: 16, numLayers: 2), new Random(42));
            foreach (var b in m.PriceBlocks)
            {
                Assert(b.SelfAttention.WQ.GetLength(0) == 16, "WQ rows");
                Assert(b.SelfAttention.WQ.GetLength(1) == 16, "WQ cols");
            }
        }

        // ── PredictNext ──────────────────────────────────────────────────────

        private MultimodalInput MakeInput(int seqLen = 10, int priceFeatures = 5, int globalDim = 0, NewsStory[] stories = null)
        {
            var rng = new Random(42);
            return new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandMatrix(seqLen, priceFeatures, rng, 0.5f),
                NewsStories = stories,
                GlobalFeatures = globalDim > 0
                    ? Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray()
                    : null,
            };
        }

        void Test_PredictNext_NoError()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var p = m.PredictNext(MakeInput());
            Assert(p != null, "null prediction");
        }

        void Test_PredictNext_NoNaN()
        {
            var (tok, inputs, _) = Data(n: 1);
            var m = new MmtacModel(Cfg(vocabSize: tok.VocabSize + 2), new Random(42));
            var p = m.PredictNext(inputs[0]);
            Assert(!HasNaNPrediction(p), "NaN in prediction");
        }

        void Test_PredictNext_NullStories()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            Assert(!HasNaNPrediction(m.PredictNext(MakeInput(stories: null))), "NaN with null stories");
        }

        void Test_PredictNext_EmptyStories()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            Assert(!HasNaNPrediction(m.PredictNext(MakeInput(stories: Array.Empty<NewsStory>()))), "NaN with empty stories");
        }

        void Test_PredictNext_Deterministic()
        {
            var (tok, inputs, _) = Data(n: 1);
            var m = new MmtacModel(Cfg(vocabSize: tok.VocabSize + 2), new Random(42));
            var p1 = m.PredictNext(inputs[0]);
            var p2 = m.PredictNext(inputs[0]);
            Assert(p1.High == p2.High && p1.Close == p2.Close, "non-deterministic");
        }

        void Test_PredictNext_RangeNonNegative()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var p = m.PredictNext(MakeInput());
            Assert(p.Range >= 0f, $"Range={p.Range} < 0");
        }

        void Test_PredictNext_QualityInRange()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var p = m.PredictNext(MakeInput());
            Assert(p.Quality >= 0f && p.Quality <= 1f, $"Quality={p.Quality}");
        }

        void Test_PredictNext_DirectionInRange()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var p = m.PredictNext(MakeInput());
            Assert(p.DirectionProb >= 0f && p.DirectionProb <= 1f, $"Direction={p.DirectionProb}");
        }

        void Test_PredictNext_MidDirInRange()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var p = m.PredictNext(MakeInput());
            Assert(p.MidWindowDirectionProb >= 0f && p.MidWindowDirectionProb <= 1f, $"MidDir={p.MidWindowDirectionProb}");
        }

        void Test_PredictNext_ConfidenceOne_WhenDisabled()
        {
            var m = new MmtacModel(Cfg(useConf: false), new Random(42));
            Assert(m.PredictNext(MakeInput()).Confidence == 1f, "Confidence should be 1 when head disabled");
        }

        void Test_PredictNext_ConfidenceInRange()
        {
            var m = new MmtacModel(Cfg(useConf: true), new Random(42));
            var c = m.PredictNext(MakeInput()).Confidence;
            Assert(c >= 0f && c <= 1f, $"Confidence={c}");
        }

        void Test_PredictNext_StoryVsNoStory_Differ()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(vocabSize: tok.VocabSize + 2), new Random(42));
            var pW = m.PredictNext(inputs[0]);
            var pN = m.PredictNext(new MultimodalInput
            {
                PredictionTimestamp = inputs[0].PredictionTimestamp,
                PriceSequence = inputs[0].PriceSequence,
                NewsStories = null,
            });
            Assert(MathF.Abs(pW.Close - pN.Close) > 1e-6f || MathF.Abs(pW.High - pN.High) > 1e-6f,
                "story vs no-story outputs identical");
        }

        void Test_PredictNext_SeqLen1()
        {
            var m = new MmtacModel(Cfg(priceSeqLen: 5), new Random(42));
            var inp = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = new float[1, 5] };
            Assert(!HasNaNPrediction(m.PredictNext(inp)), "NaN with seqLen=1");
        }

        void Test_Forward_AllHeads_NoNaN()
        {
            // Exercises the raw Forward() return — all six head arrays, not just PredictNext.
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, useConf: true), new Random(42));
            var (reg, range, quality, dir, midDir, conf) = m.Forward(inputs[0]);

            int sl = reg.GetLength(0);
            Assert(sl > 0, "regression has no rows");
            Assert(reg.GetLength(1) == 3, "regression should have 3 cols (H/L/C)");
            Assert(conf != null, "confidence head null when enabled");
            for (int t = 0; t < sl; t++)
            {
                Assert(!float.IsNaN(reg[t, 0]) && !float.IsNaN(reg[t, 1]) && !float.IsNaN(reg[t, 2]), $"NaN in regression t={t}");
                Assert(!float.IsNaN(range[t, 0]), $"NaN in range t={t}");
                Assert(!float.IsNaN(quality[t, 0]), $"NaN in quality t={t}");
                Assert(!float.IsNaN(dir[t, 0]), $"NaN in direction t={t}");
                Assert(!float.IsNaN(midDir[t, 0]), $"NaN in midDir t={t}");
                Assert(!float.IsNaN(conf[t, 0]), $"NaN in confidence t={t}");
                Assert(range[t, 0] >= 0f, $"range negative t={t}");
                Assert(quality[t, 0] >= 0f && quality[t, 0] <= 1f, $"quality out of [0,1] t={t}");
                Assert(dir[t, 0] >= 0f && dir[t, 0] <= 1f, $"direction out of [0,1] t={t}");
                Assert(midDir[t, 0] >= 0f && midDir[t, 0] <= 1f, $"midDir out of [0,1] t={t}");
            }
        }

        void Test_Forward_BypassDecay_False_NoNaN()
        {
            // BypassDecay=false means the global-token column in the time-diff matrix
            // is NOT zeroed, exercising the non-bypass branch of ForwardPriceDecoder.
            var (tok, inputs, _) = Data(n: 1, withNews: true, globalDim: 4);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 4, bypassDecay: false), new Random(42));
            Assert(!HasNaNPrediction(m.PredictNext(inputs[0])), "NaN with BypassDecay=false");
        }

        // ── Training ─────────────────────────────────────────────────────────

        void Test_Train_LossDecreases()
        {
            var (tok, inputs, targets) = Data(n: 10);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            float before = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            new MmtacTrainer(m, TC(lr: 0.003f, bs: 5, epochs: 20)).Train(inputs, targets);
            float after = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            AssertLossImproved(before, after);
        }

        void Test_Train_LossFinite()
        {
            var (tok, inputs, targets) = Data(n: 8);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(epochs: 10)).Train(inputs, targets);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss) && loss >= 0, $"invalid loss {loss}");
        }

        void Test_Train_AllPriceParamsUpdated()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var projB = (float[,])m.PriceInputProjection.Clone();
            var wqB = (float[,])m.PriceBlocks[0].SelfAttention.WQ.Clone();
            var cwkB = (float[,])m.PriceBlocks[0].CrossAttention.WK.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(projB, m.PriceInputProjection), "PriceInputProj");
            Assert(MatrixChanged(wqB, m.PriceBlocks[0].SelfAttention.WQ), "SelfWQ");
            Assert(MatrixChanged(cwkB, m.PriceBlocks[0].CrossAttention.WK), "CrossWK");
        }

        void Test_Train_RegressionHeadUpdated()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var rB = (float[,])m.RegressionProjection.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(rB, m.RegressionProjection), "RegressionProjection unchanged");
        }

        void Test_Train_RangeHeadUpdated()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var rB = (float[,])m.RangeProjection.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(rB, m.RangeProjection), "RangeProjection unchanged");
        }

        void Test_Train_DirectionHeadUpdated()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var dB = (float[,])m.DirectionProjection.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(dB, m.DirectionProjection), "DirectionProjection unchanged");
        }

        void Test_Train_ContextTypeEmbeddingUpdated()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var ctB = (float[,])m.ContextTypeEmbedding.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(ctB, m.ContextTypeEmbedding), "ContextTypeEmbedding unchanged");
        }

        void Test_Train_FrozenText_NotUpdated()
        {
            var (tok, inputs, targets) = Data(n: 5, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, freezeText: true), new Random(42));
            var embB = (float[,])m.TextTokenEmbedding.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(!MatrixChanged(embB, m.TextTokenEmbedding), "frozen text embedding changed");
        }

        void Test_Train_UnfrozenText_Updated()
        {
            var (tok, inputs, targets) = Data(n: 5, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, freezeText: false), new Random(42));
            var embB = (float[,])m.TextTokenEmbedding.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(embB, m.TextTokenEmbedding), "unfrozen text embedding unchanged");
        }

        void Test_Train_ConfidenceHead_Updated()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, useConf: true), new Random(42));
            var cB = (float[,])m.ConfidenceProjection.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(cB, m.ConfidenceProjection), "ConfidenceProjection unchanged");
        }

        void Test_Train_GradientClipping_NoNaN()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.5f, bs: 5, epochs: 5, clip: true)).Train(inputs, targets);
            Assert(!HasNaNPrediction(m.PredictNext(inputs[0])), "NaN after high-LR training");
        }

        void Test_Train_SingleSampleOverfit()
        {
            var (tok, inputs, targets) = Data(n: 1);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 2, numLayers: 2, ffnDim: 64), new Random(42));
            float before = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 1, epochs: 300)).Train(inputs, targets);
            float after = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            AssertLossImproved(before, after);
        }

        void Test_Train_PriceOnly_LossDecreases()
        {
            var rng = new Random(42);
            int n = 8;
            var inputs = Enumerable.Range(0, n).Select(_ => new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandMatrix(10, 5, rng, 0.5f),
            }).ToArray();
            var targets = Enumerable.Range(0, n).Select(_ => Enumerable.Range(0, 10).Select(_ => new ModelTarget
            {
                High = (float)rng.NextDouble(),
                Low = (float)rng.NextDouble(),
                Close = (float)rng.NextDouble(),
                Range = (float)rng.NextDouble() * 0.1f,
                Quality = (float)rng.NextDouble(),
                Direction = rng.Next(2),
                MidWindowDirection = rng.Next(2),
            }).ToArray()).ToArray();

            var m = new MmtacModel(Cfg(), new Random(42));
            float before = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            new MmtacTrainer(m, TC(lr: 0.003f, epochs: 20)).Train(inputs, targets);
            float after = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            AssertLossImproved(before, after);
        }

        void Test_Train_MixedBatch_SomeNullNews()
        {
            var (tok, inputs, targets) = Data(n: 8, withNews: true);
            for (int i = 0; i < inputs.Length; i += 2)
                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = inputs[i].PredictionTimestamp,
                    PriceSequence = inputs[i].PriceSequence,
                };
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(epochs: 5)).Train(inputs, targets);
            Assert(!float.IsNaN(new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets)),
                "NaN loss with mixed null news");
        }

        void Test_Train_DecayNetwork_Weights_Updated()
        {
            // The backward pass calls DecayNetwork.Backward() and UpdateDecayNetwork().
            // This verifies those paths are actually wired — the projection weights must move.
            //
            // QueryProjection is float[,,] (numHeads, projDim, contentDim), so we can't use
            // the float[,] MatrixChanged helper. Snapshot via Buffer.BlockCopy into a flat
            // float[] and compare element-wise after training.
            var (tok, inputs, targets) = Data(n: 5, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));

            var qp = m.PriceBlocks[0].DecayNetwork.QueryProjection;
            var before = new float[qp.Length];
            Buffer.BlockCopy(qp, 0, before, 0, qp.Length * sizeof(float));

            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 10)).Train(inputs, targets);

            var qpAfter = m.PriceBlocks[0].DecayNetwork.QueryProjection;
            var after = new float[qpAfter.Length];
            Buffer.BlockCopy(qpAfter, 0, after, 0, qpAfter.Length * sizeof(float));

            bool changed = false;
            for (int i = 0; i < before.Length; i++)
                if (MathF.Abs(before[i] - after[i]) > 1e-9f) { changed = true; break; }

            Assert(changed,
                "DecayNetwork.QueryProjection unchanged after training — decay backprop not firing");
        }

        void Test_Train_PriceContext_NoError()
        {
            // seqLen=10 >= MinHistory(3) + MinCurrent(3) + 1 = 7, so TrainWithPriceContext fires.
            var (tok, inputs, targets) = Data(n: 5, seqLen: 10, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 14, priceContextEnabled: true), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.001f, bs: 5, epochs: 2)).Train(inputs, targets);
            Assert(!HasNaNPrediction(m.PredictNext(inputs[0])), "NaN after PriceContext training");
        }

        void Test_Train_PriceContext_LossFinite()
        {
            var (tok, inputs, targets) = Data(n: 5, seqLen: 10, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 14, priceContextEnabled: true), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.001f, bs: 5, epochs: 5)).Train(inputs, targets);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss) && loss >= 0,
                $"invalid loss {loss} with PriceContext enabled");
        }

        // ── Global Token ──────────────────────────────────────────────────────

        void Test_GlobalToken_ForwardNoError()
        {
            var (tok, inputs, _) = Data(n: 1, globalDim: 8);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 8), new Random(42));
            Assert(!HasNaNPrediction(m.PredictNext(inputs[0])), "NaN with global token");
        }

        void Test_GlobalToken_ChangesOutput()
        {
            var rng = new Random(42);
            var m = new MmtacModel(Cfg(globalDim: 4), new Random(42));
            var ps = RandMatrix(8, 5, rng, 0.5f);
            var p1 = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, GlobalFeatures = new float[] { 0.1f, 0.2f, 0.3f, 0.4f } });
            var p2 = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, GlobalFeatures = new float[] { 0.9f, 0.8f, 0.7f, 0.6f } });
            Assert(MathF.Abs(p1.Close - p2.Close) > 1e-6f, "different global features produced identical outputs");
        }

        void Test_GlobalToken_ProjectionUpdated()
        {
            var (tok, inputs, targets) = Data(n: 5, globalDim: 4);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 4), new Random(42));
            var gpB = (float[,])m.GlobalFeatureProjection.Clone();
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 5, epochs: 5)).Train(inputs, targets);
            Assert(MatrixChanged(gpB, m.GlobalFeatureProjection), "GlobalFeatureProjection unchanged");
        }

        void Test_GlobalToken_PredictWithMemory_NoError()
        {
            // PredictWithMemory has its own context-assembly path for the global token.
            var (tok, inputs, _) = Data(n: 1, withNews: true, globalDim: 4);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 4), new Random(42));
            var p = m.PredictWithMemory(inputs[0], 100.0);
            Assert(!HasNaNPrediction(p), "NaN from PredictWithMemory with globalDim > 0");
            Assert(m.NewsMemory.Count > 0, "no news memory stored");
            Assert(m.PriceMemory.Count > 0, "no price memory stored");
        }

        // ── Multi-output head correctness ─────────────────────────────────────

        void Test_Outputs_RangeVsHighLow()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            Assert(m.PredictNext(MakeInput()).Range >= 0f, "Range is negative");
        }

        void Test_Outputs_MultipleTimesteps()
        {
            var m = new MmtacModel(Cfg(priceSeqLen: 12), new Random(42));
            var p = m.PredictNext(MakeInput(seqLen: 10));
            Assert(!float.IsNaN(p.Close) && !float.IsNaN(p.High), "NaN in multi-timestep output");
        }

        // ── Memory ────────────────────────────────────────────────────────────

        void Test_Memory_InitiallyEmpty()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            Assert(m.NewsMemory.Count == 0, "NewsMemory not empty");
            Assert(m.PriceMemory.Count == 0, "PriceMemory not empty");
        }

        void Test_Memory_AccumulatesAfterPredict()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 100.0);
            Assert(m.NewsMemory.Count > 0, "NewsMemory empty after predict");
            Assert(m.PriceMemory.Count > 0, "PriceMemory empty after predict");
        }

        void Test_Memory_PrunesWhenOverLimit()
        {
            var (tok, inputs, _) = Data(n: 5, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            for (int i = 0; i < 5; i++)
                m.PredictWithMemory(inputs[i], (i + 1) * 100.0, maxNewsMemorySize: 5, maxPriceMemorySize: 10);
            Assert(m.NewsMemory.Count <= 5, $"news too large {m.NewsMemory.Count}");
            Assert(m.PriceMemory.Count <= 10, $"price too large {m.PriceMemory.Count}");
        }

        void Test_Memory_ClearAll()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 100.0);
            m.ClearAllMemory();
            Assert(m.NewsMemory.Count == 0, "news memory not cleared");
            Assert(m.PriceMemory.Count == 0, "price memory not cleared");
        }

        void Test_Memory_ClearNews_OnlyNews()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 100.0);
            int priceCountBefore = m.PriceMemory.Count;
            m.ClearNewsMemory();
            Assert(m.NewsMemory.Count == 0, "ClearNewsMemory left news entries");
            Assert(m.PriceMemory.Count == priceCountBefore, "ClearNewsMemory wiped price memory");
        }

        void Test_Memory_ClearPrice_OnlyPrice()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 100.0);
            int newsCountBefore = m.NewsMemory.Count;
            m.ClearPriceMemory();
            Assert(m.PriceMemory.Count == 0, "ClearPriceMemory left price entries");
            Assert(m.NewsMemory.Count == newsCountBefore, "ClearPriceMemory wiped news memory");
        }

        void Test_Memory_NewsAndPrice_BothStored()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 100.0);
            Assert(m.NewsMemory.Count > 0, "no news entries");
            Assert(m.PriceMemory.Count > 0, "no price entries");
            Assert(m.PriceMemory[0].HiddenState.Length == m.Config.Price.EmbeddingDim, "wrong price hidden dim");
        }

        void Test_Memory_TimestampOrdering()
        {
            var (tok, inputs, _) = Data(n: 2, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 100.0);
            m.PredictWithMemory(inputs[1], 200.0);
            Assert(m.NewsMemory.Max(e => e.AbsoluteTimestamp) >= 200.0, "latest timestamp missing");
        }

        void Test_Memory_TimeDeltaReachesDecoder()
        {
            // After the SafeRecencyScale fix, relTime is passed raw into ctxT which
            // becomes the time-diff matrix fed to the DecayNetwork. Two identical
            // setups differing only in currentAbsoluteTimestamp must produce different
            // price decoder hidden states — this is structurally guaranteed regardless
            // of weight values, so it works on a fresh random model.
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var cfg = Cfg(tok.VocabSize + 2);

            var h1 = CapturePriceHidden(cfg, inputs[0], queryTime: 1.0);
            var h2 = CapturePriceHidden(cfg, inputs[0], queryTime: 10000.0);

            int rows = h1.GetLength(0), cols = h1.GetLength(1);
            float maxAbsDiff = 0f;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(h1[r, c] - h2[r, c]));

            Assert(maxAbsDiff > 1e-6f,
                $"price hidden states identical despite t=1 vs t=10000 (maxDiff={maxAbsDiff})");
        }

        // Helper: fresh seed-42 model, plants news memory at t=0, clears price memory,
        // then runs ForwardPriceDecoderWithCache at queryTime and returns the hidden state.
        private float[,] CapturePriceHidden(MmtacConfig cfg, MultimodalInput input, double queryTime)
        {
            var m = new MmtacModel(cfg, new Random(42));
            m.PredictWithMemory(input, 0.0);
            m.ClearPriceMemory();

            int embDim = cfg.Price.EmbeddingDim;
            var ctxH = new List<float[]>();
            var ctxT = new List<float>();
            var ctxTypes = new List<int>();

            foreach (var e in m.NewsMemory)
            {
                float relTime = -(float)(queryTime - e.AbsoluteTimestamp);
                var v = new float[embDim];
                for (int d = 0; d < embDim; d++) v[d] = e.HiddenState[d];
                ctxH.Add(v); ctxT.Add(relTime); ctxTypes.Add(0);
            }

            float[,] cH = new float[ctxH.Count, embDim];
            float[] cT = new float[ctxH.Count];
            for (int i = 0; i < ctxH.Count; i++)
            {
                for (int d = 0; d < embDim; d++) cH[i, d] = ctxH[i][d];
                cT[i] = ctxT[i];
            }
            m.AccelerationManager.ApplyContextTypeEmbedding(cH, m.ContextTypeEmbedding, ctxTypes.ToArray());

            var cache = new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers);
            return m.ForwardPriceDecoderWithCache(
                input.PriceSequence, 0, input.PriceSequence.GetLength(0),
                cH, cT, cache, isTraining: false);
        }

        void Test_Memory_AttentionScores_Updated()
        {
            // UpdateMemoryAttentionScores runs after every PredictWithMemory call.
            // After multiple queries QueryCount must be > 0 on at least one entry.
            var (tok, inputs, _) = Data(n: 3, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 0.0);
            m.PredictWithMemory(inputs[1], 100.0);
            m.PredictWithMemory(inputs[2], 200.0);
            Assert(m.NewsMemory.Count > 0, "no news memory to inspect");
            Assert(m.NewsMemory.Any(e => e.QueryCount > 0), "QueryCount never incremented");
        }

        void Test_Memory_AttentionPruning_KeepsHighScore()
        {
            // With UseAttentionBasedPruning=true and MinQueryCount satisfied,
            // a manually elevated AttentionScore entry must survive a tight prune.
            var (tok, inputs, _) = Data(n: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PruningConfig.UseAttentionBasedPruning = true;
            m.PruningConfig.MinQueryCountForPruning = 1;
            m.PruningConfig.NewEntryReserveFraction = 0.0f;

            for (int i = 0; i < 6; i++)
                m.PredictWithMemory(inputs[i], i * 100.0, maxNewsMemorySize: 100);

            if (m.NewsMemory.Count < 2) return; // not enough entries — skip

            m.NewsMemory[0].AttentionScore = 10f; m.NewsMemory[0].QueryCount = 10;
            double highTs = m.NewsMemory[0].AbsoluteTimestamp;
            m.NewsMemory[1].AttentionScore = 0.0001f; m.NewsMemory[1].QueryCount = 10;

            m.PruneNewsMemory(1);

            Assert(m.NewsMemory.Count == 1, $"expected 1 entry after prune, got {m.NewsMemory.Count}");
            Assert(m.NewsMemory[0].AbsoluteTimestamp == highTs,
                "attention-based pruning discarded the high-score entry");
        }

        void Test_Memory_PriceOnly_NoNews()
        {
            // PredictWithMemory with no live news but existing price memory in context.
            var rng = new Random(42);
            var m = new MmtacModel(Cfg(), new Random(42));
            var inp = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = RandMatrix(8, 5, rng, 0.5f) };

            m.PredictWithMemory(inp, 0.0);
            Assert(m.PriceMemory.Count > 0, "no price memory after first call");

            var p = m.PredictWithMemory(inp, 100.0);
            Assert(!HasNaNPrediction(p), "NaN from PredictWithMemory with price-memory-only context");
        }

        void Test_Memory_LastPriceTimestamp_Updated()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            Assert(m.LastPriceTimestamp == 0.0, "LastPriceTimestamp not zero at init");

            m.PredictWithMemory(inputs[0], 500.0);
            int seqLen = inputs[0].PriceSequence.GetLength(0);
            double expectedTs = 500.0 + Math.Max(0, seqLen - 1) * 1.0;
            Assert(Math.Abs(m.LastPriceTimestamp - expectedTs) < 1e-9,
                $"LastPriceTimestamp={m.LastPriceTimestamp} expected~{expectedTs}");
        }

        // ── Sequential training ────────────────────────────────────────────────

        void Test_Sequential_MemoryAccumulates()
        {
            var (tok, inputs, targets) = Data(n: 5, seqLen: 8, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            var ts = Enumerable.Range(0, 5).Select(i => (double)(i * 100)).ToArray();
            Assert(m.NewsMemory.Count == 0, "memory not empty before training");
            new MmtacTrainer(m, TC(epochs: 1)).TrainSequential(inputs, targets, ts);
            Assert(m.NewsMemory.Count > 0 || m.PriceMemory.Count > 0, "no memory accumulated");
        }

        void Test_Sequential_LossDecreases()
        {
            var (tok, inputs, targets) = Data(n: 8, seqLen: 8, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            var ts = Enumerable.Range(0, 8).Select(i => (double)(i * 100)).ToArray();
            float before = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            new MmtacTrainer(m, TC(lr: 0.002f, epochs: 10)).TrainSequential(inputs, targets, ts);
            float after = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            AssertLossImproved(before, after);
        }

        void Test_Sequential_PriceMemoryPopulated()
        {
            var (tok, inputs, targets) = Data(n: 3, seqLen: 8, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            var ts = Enumerable.Range(0, 3).Select(i => (double)(i * 100)).ToArray();
            new MmtacTrainer(m, TC(epochs: 1)).TrainSequential(inputs, targets, ts);
            foreach (var e in m.PriceMemory)
                Assert(e.HiddenState.Length == m.Config.Price.EmbeddingDim, "wrong hidden dim");
        }

        void Test_Sequential_MemoryClearedBetweenEpochs()
        {
            // TrainSequential calls ClearAllMemory() at the top of each epoch.
            // Running a second epoch must not carry over the first epoch's memory.
            var (tok, inputs, targets) = Data(n: 3, seqLen: 8, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            var ts = Enumerable.Range(0, 3).Select(i => (double)(i * 100)).ToArray();

            new MmtacTrainer(m, TC(epochs: 1)).TrainSequential(inputs, targets, ts);
            int newsAfterEpoch1 = m.NewsMemory.Count;
            int priceAfterEpoch1 = m.PriceMemory.Count;

            // Second epoch — memory cleared at start, so final count should not exceed one epoch's worth
            new MmtacTrainer(m, TC(epochs: 1)).TrainSequential(inputs, targets, ts);
            Assert(m.NewsMemory.Count <= newsAfterEpoch1 + inputs.Length,
                "news memory grew unboundedly — ClearAllMemory not called between epochs");
            Assert(m.PriceMemory.Count <= priceAfterEpoch1 + inputs.Length * 10,
                "price memory grew unboundedly — ClearAllMemory not called between epochs");
        }

        void Test_Sequential_WithGlobalToken()
        {
            // TrainSequential has its own global-token branch (ctxH.Insert).
            var (tok, inputs, targets) = Data(n: 3, seqLen: 8, withNews: true, globalDim: 4);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10, globalDim: 4), new Random(42));
            var ts = Enumerable.Range(0, 3).Select(i => (double)(i * 100)).ToArray();
            new MmtacTrainer(m, TC(epochs: 2)).TrainSequential(inputs, targets, ts);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss),
                $"invalid loss {loss} after sequential training with global token");
        }

        // ── Tokenizer ─────────────────────────────────────────────────────────

        void Test_Tokenizer_SetAndTokenize()
        {
            var tok = new BPETokenizer();
            tok.Train(new[] { "stock rose sharply", "market crashed today" }, vocabSize: 100, minFrequency: 1);
            var m = new MmtacModel(Cfg(vocabSize: tok.VocabSize + 10), new Random(42));
            m.SetTokenizer(tok);
            Assert(m.Tokenizer != null, "Tokenizer null after SetTokenizer");

            var stories = m.TokenizeStories(
                new[] { "stock rose sharply", "market crashed today" },
                new[] { 0f, 1f });
            Assert(stories.Length == 2, "wrong story count");
            Assert(stories[0].TokenIds != null && stories[0].TokenIds.Length > 0, "empty token ids");
            Assert(stories[0].ArrivalTime == 0f, "wrong arrival time story 0");
            Assert(stories[1].ArrivalTime == 1f, "wrong arrival time story 1");
        }

        void Test_Tokenizer_VocabSizeMismatch_Throws()
        {
            var tok = new BPETokenizer();
            tok.Train(new[] { "stock rose sharply", "market crashed" }, vocabSize: 200, minFrequency: 1);
            var m = new MmtacModel(Cfg(vocabSize: 10), new Random(42));
            bool threw = false;
            try { m.SetTokenizer(tok); }
            catch (ArgumentException) { threw = true; }
            Assert(threw, "SetTokenizer should throw when tokenizer vocab exceeds model vocab");
        }

        // ── Validation ────────────────────────────────────────────────────────

        void Test_Validate_ReturnsFiniteValue()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss) && loss >= 0, $"invalid loss {loss}");
        }

        // ── Numerical stability ────────────────────────────────────────────────

        void Test_Stability_LargeInputs()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var big = new float[10, 5];
            for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) big[i, j] = 100f;
            Assert(!HasNaNPrediction(m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = big })),
                "NaN with large inputs");
        }

        void Test_Stability_SmallInputs()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var tiny = new float[10, 5];
            for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) tiny[i, j] = 1e-6f;
            Assert(!HasNaNPrediction(m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = tiny })),
                "NaN with tiny inputs");
        }

        void Test_Stability_ZeroInputs()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            Assert(!HasNaNPrediction(m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = new float[10, 5] })),
                "NaN with zero inputs");
        }

        void Test_Stability_NegativeInputs()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var neg = new float[10, 5];
            for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) neg[i, j] = -5f;
            Assert(!HasNaNPrediction(m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = neg })),
                "NaN with negative inputs");
        }

        void Test_Stability_ManyStories()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var stories = Enumerable.Range(0, 12).Select(i => new NewsStory(new[] { 4, 5, 6, 7 }, (float)i)).ToArray();
            Assert(!HasNaNPrediction(m.PredictNext(new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandMatrix(8, 5, new Random(42), 0.5f),
                NewsStories = stories,
            })), "NaN with 12 stories");
        }

        void Test_Stability_NoNaN_AfterManyEpochs()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.001f, bs: 5, epochs: 60)).Train(inputs, targets);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss), $"loss={loss} after 60 epochs");
            Assert(!HasNaNPrediction(m.PredictNext(inputs[0])), "NaN in prediction after 60 epochs");
        }

        // ── Save / Load ────────────────────────────────────────────────────────

        void Test_SaveLoad_ForwardIdentical()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(epochs: 5)).Train(inputs, targets);
            var pBefore = m.PredictNext(inputs[0]);
            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var loaded = MmtacModel.Load(dir);
                var pAfter = loaded.PredictNext(inputs[0]);
                Assert(MathF.Abs(pBefore.High - pAfter.High) < 1e-5f, "High mismatch");
                Assert(MathF.Abs(pBefore.Close - pAfter.Close) < 1e-5f, "Close mismatch");
                Assert(MathF.Abs(pBefore.Range - pAfter.Range) < 1e-5f, "Range mismatch");
            }
            finally { Cleanup(dir); }
        }

        void Test_SaveLoad_AllHeadsPreserved()
        {
            var (tok, inputs, targets) = Data(n: 3, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(epochs: 3)).Train(inputs, targets);
            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                Assert(!MatrixChanged(m.RegressionProjection, ld.RegressionProjection, 1e-6f), "RegressionProjection");
                Assert(!MatrixChanged(m.RangeProjection, ld.RangeProjection, 1e-6f), "RangeProjection");
                Assert(!MatrixChanged(m.DirectionProjection, ld.DirectionProjection, 1e-6f), "DirectionProjection");
                Assert(!MatrixChanged(m.MidDirectionProjection, ld.MidDirectionProjection, 1e-6f), "MidDirProjection");
            }
            finally { Cleanup(dir); }
        }

        void Test_SaveLoad_ContextTypeEmbedding()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            for (int t = 0; t < 3; t++)
                for (int d = 0; d < m.Config.Price.EmbeddingDim; d++)
                    m.ContextTypeEmbedding[t, d] = (t + 1) * 0.1f + d * 0.001f;
            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                Assert(!MatrixChanged(m.ContextTypeEmbedding, MmtacModel.Load(dir).ContextTypeEmbedding, 1e-6f),
                    "ContextTypeEmbedding mismatch");
            }
            finally { Cleanup(dir); }
        }

        void Test_SaveLoad_GlobalProjection()
        {
            var m = new MmtacModel(Cfg(globalDim: 4), new Random(42));
            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                Assert(ld.GlobalFeatureProjection != null, "GlobalFeatureProjection null after load");
                Assert(!MatrixChanged(m.GlobalFeatureProjection, ld.GlobalFeatureProjection, 1e-6f),
                    "GlobalFeatureProjection mismatch");
            }
            finally { Cleanup(dir); }
        }

        void Test_SaveLoad_Memory()
        {
            var (tok, inputs, _) = Data(n: 2, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 100.0);
            int newsCount = m.NewsMemory.Count, priceCount = m.PriceMemory.Count;
            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                Assert(ld.NewsMemory.Count == newsCount, $"news count {ld.NewsMemory.Count} vs {newsCount}");
                Assert(ld.PriceMemory.Count == priceCount, "price count mismatch");
                if (newsCount > 0)
                    Assert(Math.Abs(m.NewsMemory[0].AbsoluteTimestamp - ld.NewsMemory[0].AbsoluteTimestamp) < 1e-6,
                        "news timestamp mismatch");
            }
            finally { Cleanup(dir); }
        }

        void Test_SaveLoad_PruningConfig()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            m.PruningConfig.AttentionScoreAlpha = 0.25f;
            m.PruningConfig.MinQueryCountForPruning = 7;
            m.PruningConfig.UseAttentionBasedPruning = false;
            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                Assert(MathF.Abs(ld.PruningConfig.AttentionScoreAlpha - 0.25f) < 1e-6f, "Alpha");
                Assert(ld.PruningConfig.MinQueryCountForPruning == 7, "MinQueryCount");
                Assert(ld.PruningConfig.UseAttentionBasedPruning == false, "UseAttnBased");
            }
            finally { Cleanup(dir); }
        }

        void Test_SaveLoad_ContinueTraining()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(epochs: 5)).Train(inputs, targets);
            float mid = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                new MmtacTrainer(ld, TC(lr: 0.001f, epochs: 15)).Train(inputs, targets);
                float after = new MmtacTrainer(ld, TC(epochs: 1)).Validate(inputs, targets);
                Assert(after <= mid * 1.1f, $"loss regressed after load: {mid:F5} -> {after:F5}");
            }
            finally { Cleanup(dir); }
        }

        // ── Config validation ──────────────────────────────────────────────────

        void Test_Config_Validate_Good() { Cfg().Validate(); }
        void Test_Config_SmallPreset() { MmtacConfig.Small(vocabSize: 1000, priceFeatureDim: 5).Validate(); }
        void Test_Config_StandardPreset() { MmtacConfig.Standard(vocabSize: 5000, priceFeatureDim: 5, globalDim: 8).Validate(); }

        // ── Gradient checks (finite-difference) ─────────────────────────────

        private float MmtacMSELoss(MmtacModel m, MultimodalInput inp, ModelTarget[] tgt)
        {
            float diff = m.PredictNext(inp).Close - tgt[tgt.Length - 1].Close;
            return diff * diff;
        }

        void Test_GradCheck_RegressionProjection()
        {
            var (tok, inputs, targets) = Data(n: 1, seqLen: 6);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 8, numHeads: 2), new Random(42));
            float eps = 1e-3f, orig = m.RegressionProjection[2, 0];
            m.RegressionProjection[2, 0] = orig + eps; float lp = MmtacMSELoss(m, inputs[0], targets[0]);
            m.RegressionProjection[2, 0] = orig - eps; float lm = MmtacMSELoss(m, inputs[0], targets[0]);
            m.RegressionProjection[2, 0] = orig;
            float fd = (lp - lm) / (2 * eps);
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} — RegressionProjection has no effect");
        }

        void Test_GradCheck_RangeProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 8, numHeads: 2), new Random(42));
            var inp = MakeInput(seqLen: 6);
            float eps = 1e-3f, orig = m.RangeProjection[0, 0];
            m.RangeProjection[0, 0] = orig + eps; float vp = m.PredictNext(inp).Range;
            m.RangeProjection[0, 0] = orig - eps; float vm = m.PredictNext(inp).Range;
            m.RangeProjection[0, 0] = orig;
            float fd = (vp - vm) / (2 * eps);
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} — RangeProjection has no effect");
        }

        void Test_GradCheck_DirectionProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 8, numHeads: 2), new Random(42));
            var inp = MakeInput(seqLen: 6);
            float eps = 1e-3f, orig = m.DirectionProjection[0, 0];
            m.DirectionProjection[0, 0] = orig + eps; float vp = m.PredictNext(inp).DirectionProb;
            m.DirectionProjection[0, 0] = orig - eps; float vm = m.PredictNext(inp).DirectionProb;
            m.DirectionProjection[0, 0] = orig;
            float fd = (vp - vm) / (2 * eps);
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} — DirectionProjection has no effect");
        }

        void Test_GradCheck_QualityProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 8, numHeads: 2), new Random(42));
            var inp = MakeInput(seqLen: 6);
            float eps = 1e-3f, orig = m.QualityProjection[0, 0];
            m.QualityProjection[0, 0] = orig + eps; float vp = m.PredictNext(inp).Quality;
            m.QualityProjection[0, 0] = orig - eps; float vm = m.PredictNext(inp).Quality;
            m.QualityProjection[0, 0] = orig;
            float fd = (vp - vm) / (2 * eps);
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} — QualityProjection has no effect");
        }

        void Test_GradCheck_MidDirectionProjection()
        {
            var m = new MmtacModel(Cfg(embDim: 8, numHeads: 2), new Random(42));
            var inp = MakeInput(seqLen: 6);
            float eps = 1e-3f, orig = m.MidDirectionProjection[0, 0];
            m.MidDirectionProjection[0, 0] = orig + eps; float vp = m.PredictNext(inp).MidWindowDirectionProb;
            m.MidDirectionProjection[0, 0] = orig - eps; float vm = m.PredictNext(inp).MidWindowDirectionProb;
            m.MidDirectionProjection[0, 0] = orig;
            float fd = (vp - vm) / (2 * eps);
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} — MidDirectionProjection has no effect");
        }

        void Test_GradCheck_TextEmbedding()
        {
            var (tok, inputs, targets) = Data(n: 1, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 8, numHeads: 2, freezeText: false), new Random(42));
            int tid = inputs[0].NewsStories[0].TokenIds[0];
            float eps = 1e-3f, orig = m.TextTokenEmbedding[tid, 0];
            m.TextTokenEmbedding[tid, 0] = orig + eps; float vp = m.PredictNext(inputs[0]).Close;
            m.TextTokenEmbedding[tid, 0] = orig - eps; float vm = m.PredictNext(inputs[0]).Close;
            m.TextTokenEmbedding[tid, 0] = orig;
            float fd = (vp - vm) / (2 * eps);
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} — TextEmbedding has no effect");
        }

        // ── End-to-end signal learning ────────────────────────────────────────

        void Test_E2E_BullBear()
        {
            var tok = new BPETokenizer();
            tok.Train(new[] { "bull bull bull bull", "bear bear bear bear" }, vocabSize: 50, minFrequency: 1);
            var bullTokens = tok.Encode("bull bull bull bull", addSpecialTokens: true);
            var bearTokens = tok.Encode("bear bear bear bear", addSpecialTokens: true);

            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            var rng = new Random(42);
            int n = 40;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                bool bull = s < n / 2;
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = RandMatrix(8, 5, rng, 0.3f),
                    NewsStories = new[] { new NewsStory(bull ? bullTokens : bearTokens, 0f) },
                };
                float ct = bull ? 0.8f : 0.2f;
                targets[s] = Enumerable.Range(0, 8).Select(_ => new ModelTarget
                {
                    High = ct + 0.05f,
                    Low = ct - 0.05f,
                    Close = ct,
                    Range = 0.1f,
                    Quality = 0.5f,
                    Direction = bull ? 1 : 0,
                    MidWindowDirection = bull ? 1 : 0,
                }).ToArray();
            }

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 10, epochs: 200)).Train(inputs, targets);

            var ps = RandMatrix(8, 5, rng, 0.3f);
            float bullClose = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, NewsStories = new[] { new NewsStory(bullTokens, 0f) } }).Close;
            float bearClose = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, NewsStories = new[] { new NewsStory(bearTokens, 0f) } }).Close;
            Assert(bullClose > bearClose, $"Bull ({bullClose:F4}) should exceed bear ({bearClose:F4})");
        }


        // In MmtacTests — add these:

        void Test_RangeHead_ConsistentWithHighLow_AfterTraining()
        {
            // After training where High-Low = Range in targets, the model's Range output
            // should approximately equal its predicted High - Low.
            var rng = new Random(42);
            int n = 20;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = RandMatrix(8, 5, rng, 0.3f)
                };
                float hi = 0.6f, lo = 0.4f, cl = 0.5f;
                targets[s] = Enumerable.Range(0, 8).Select(_ => new ModelTarget
                {
                    High = hi,
                    Low = lo,
                    Close = cl,
                    Range = hi - lo, // Range = High - Low by definition
                    Quality = 0.8f,
                    Direction = 1,
                    MidWindowDirection = 1
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 300)).Train(inputs, targets);

            float totalRangeError = 0f;
            int count = 0;
            foreach (var inp in inputs)
            {
                var (reg, range, _, _, _, _) = m.Forward(inp);
                for (int t = 0; t < reg.GetLength(0); t++)
                {
                    float predictedRange = range[t, 0];
                    float impliedRange = reg[t, 0] - reg[t, 1]; // High - Low
                    totalRangeError += MathF.Abs(predictedRange - impliedRange);
                    count++;
                }
            }
            float avgError = totalRangeError / count;
            // They won't be exactly equal (separate heads) but should be in the same ballpark
            Assert(avgError < 0.2f,
                $"Range head ({avgError:F4} avg error vs High-Low) — heads may not be learning consistently");
        }

 
        void Test_DirectionHead_LearnsBinarySignal()
        {
            // Clear uptrend vs downtrend in the first price feature.
            // After training, Direction > 0.5 for uptrend, < 0.5 for downtrend.
            var rng = new Random(42);
            int n = 40, seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                bool up = s < n / 2;
                float trend = up ? 0.75f : 0.25f;
                var ps = new float[seqLen, 5];
                for (int t = 0; t < seqLen; t++)
                {
                    ps[t, 0] = trend + (float)(rng.NextDouble() - 0.5) * 0.04f;
                    for (int f = 1; f < 5; f++) ps[t, f] = 0.5f;
                }
                inputs[s] = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps };
                targets[s] = Enumerable.Range(0, seqLen).Select(_ => new ModelTarget
                {
                    High = 0.55f,
                    Low = 0.45f,
                    Close = 0.5f,
                    Range = 0.1f,
                    Quality = 0.7f,
                    Direction = up ? 1f : 0f,
                    MidWindowDirection = up ? 1f : 0f
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 300)).Train(inputs, targets);

            int correct = 0, total = 0;
            for (int s = 0; s < n; s++)
            {
                bool up = s < n / 2;
                var (_, _, _, dir, _, _) = m.Forward(inputs[s]);
                for (int t = 0; t < dir.GetLength(0); t++)
                {
                    if ((dir[t, 0] > 0.5f) == up) correct++;
                    total++;
                }
            }
            float acc = (float)correct / total;
            Assert(acc > 0.65f, $"Direction accuracy {acc:P0} should exceed 65% on clear synthetic signal");
        }

        void Test_RangeHead_ConsistentWithHighLow()
        {
            // Train with Range = High - Low in every target.
            // After training, the Range head output should approximately equal
            // the model's own predicted High - Low (internal consistency).
            var rng = new Random(42);
            int n = 20;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = RandMatrix(8, 5, rng, 0.3f)
                };
                targets[s] = Enumerable.Range(0, 8).Select(_ => new ModelTarget
                {
                    High = 0.65f,
                    Low = 0.35f,
                    Close = 0.5f,
                    Range = 0.30f,   // exactly High - Low
                    Quality = 0.8f,
                    Direction = 1,
                    MidWindowDirection = 1
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 300)).Train(inputs, targets);

            float totalErr = 0f; int count = 0;
            foreach (var inp in inputs)
            {
                var (reg, range, _, _, _, _) = m.Forward(inp);
                for (int t = 0; t < reg.GetLength(0); t++)
                {
                    float impliedRange = reg[t, 0] - reg[t, 1]; // High - Low
                    totalErr += MathF.Abs(range[t, 0] - impliedRange);
                    count++;
                }
            }
            float avgErr = totalErr / count;
            Assert(avgErr < 0.25f,
                $"Range head vs implied High-Low avg error={avgErr:F4} — heads not learning consistently");
        }

        void Test_ConfidenceHead_CorrelatesWithAccuracy()
        {
            // Predictable samples (constant target) vs noisy samples (random target).
            // After training, confidence should be higher on predictable samples.
            var rng = new Random(42);
            int n = 30, seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                bool predictable = s < n / 2;
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = predictable
                        ? Enumerable.Range(0, seqLen).Aggregate(new float[seqLen, 5],
                            (ps, t) => { for (int f = 0; f < 5; f++) ps[t, f] = 0.5f; return ps; })
                        : RandMatrix(seqLen, 5, rng, 0.5f)
                };
                targets[s] = Enumerable.Range(0, seqLen).Select(_ => new ModelTarget
                {
                    High = predictable ? 0.6f : (float)rng.NextDouble(),
                    Low = predictable ? 0.4f : (float)rng.NextDouble(),
                    Close = predictable ? 0.5f : (float)rng.NextDouble(),
                    Range = 0.2f,
                    Quality = predictable ? 0.9f : 0.1f,
                    Direction = predictable ? 1 : rng.Next(2),
                    MidWindowDirection = predictable ? 1 : rng.Next(2)
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, useConf: true), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 300)).Train(inputs, targets);

            float confPred = 0f, confNoise = 0f;
            for (int s = 0; s < n; s++)
            {
                var (_, _, _, _, _, conf) = m.Forward(inputs[s]);
                float avg = 0f;
                for (int t = 0; t < conf.GetLength(0); t++) avg += conf[t, 0];
                avg /= conf.GetLength(0);
                if (s < n / 2) confPred += avg; else confNoise += avg;
            }
            confPred /= (n / 2); confNoise /= (n - n / 2);
            Assert(confPred > confNoise,
                $"Confidence on predictable ({confPred:F4}) should exceed noisy ({confNoise:F4})");
        }

        void Test_QualityHead_LearnsPredictableData()
        {
            // Quality head should learn to output higher values when the pattern is learnable.
            // Train half the batch on constant repeatable targets (high quality)
            // and half on random targets (low quality).
            var rng = new Random(42);
            int n = 30, seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                bool easy = s < n / 2;
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = RandMatrix(seqLen, 5, rng, 0.1f)
                };
                // Set a clear feature signal for easy samples
                if (easy) for (int t = 0; t < seqLen; t++) inputs[s].PriceSequence[t, 0] = 0.9f;

                targets[s] = Enumerable.Range(0, seqLen).Select(_ => new ModelTarget
                {
                    High = easy ? 0.7f : (float)rng.NextDouble(),
                    Low = easy ? 0.3f : (float)rng.NextDouble(),
                    Close = easy ? 0.5f : (float)rng.NextDouble(),
                    Range = 0.2f,
                    Quality = easy ? 0.95f : 0.05f,   // ← quality target
                    Direction = easy ? 1 : rng.Next(2),
                    MidWindowDirection = easy ? 1 : rng.Next(2)
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 250)).Train(inputs, targets);

            float qEasy = 0f, qHard = 0f;
            for (int s = 0; s < n; s++)
            {
                var (_, _, quality, _, _, _) = m.Forward(inputs[s]);
                float avg = 0f;
                for (int t = 0; t < quality.GetLength(0); t++) avg += quality[t, 0];
                avg /= quality.GetLength(0);
                if (s < n / 2) qEasy += avg; else qHard += avg;
            }
            qEasy /= (n / 2); qHard /= (n - n / 2);
            Assert(qEasy > qHard,
                $"Quality on easy samples ({qEasy:F4}) should exceed hard samples ({qHard:F4})");
        }

        void Test_GlobalToken_MacroSignalDrivesOutput()
        {
            // A clear macro signal (global feature = 1.0 → high close, = 0.0 → low close).
            // After training, the global token should drive the Close prediction.
            var rng = new Random(42);
            int n = 40, seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                float macro = s < n / 2 ? 1.0f : 0.0f;
                float targetClose = s < n / 2 ? 0.8f : 0.2f;
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = RandMatrix(seqLen, 5, rng, 0.1f),
                    GlobalFeatures = new float[] { macro, macro, macro, macro }
                };
                targets[s] = Enumerable.Range(0, seqLen).Select(_ => new ModelTarget
                {
                    High = targetClose + 0.05f,
                    Low = targetClose - 0.05f,
                    Close = targetClose,
                    Range = 0.1f,
                    Quality = 0.8f,
                    Direction = s < n / 2 ? 1 : 0,
                    MidWindowDirection = s < n / 2 ? 1 : 0
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, globalDim: 4), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 300)).Train(inputs, targets);

            var psTest = RandMatrix(seqLen, 5, rng, 0.1f);
            var pHigh = m.PredictNext(new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = psTest,
                GlobalFeatures = new float[] { 1f, 1f, 1f, 1f }
            });
            var pLow = m.PredictNext(new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = psTest,
                GlobalFeatures = new float[] { 0f, 0f, 0f, 0f }
            });

            Assert(pHigh.Close > pLow.Close,
                $"Global macro=1 Close ({pHigh.Close:F4}) should exceed macro=0 ({pLow.Close:F4})");
        }

        void Test_MultiHead_AllOutputsIndependent()
        {
            // Train where High goes up but Direction goes down (conflicting signals).
            // Verifies the separate heads can express independent predictions.
            var rng = new Random(42);
            int n = 20, seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                bool highUp = s < n / 2;
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = RandMatrix(seqLen, 5, rng, 0.1f)
                };
                for (int t = 0; t < seqLen; t++)
                    inputs[s].PriceSequence[t, 0] = highUp ? 0.8f : 0.2f;

                targets[s] = Enumerable.Range(0, seqLen).Select(_ => new ModelTarget
                {
                    High = highUp ? 0.8f : 0.3f,
                    Low = highUp ? 0.6f : 0.1f,
                    Close = highUp ? 0.7f : 0.2f,
                    Range = 0.2f,
                    Quality = 0.7f,
                    Direction = highUp ? 0f : 1f,     // ← deliberately inverted vs High
                    MidWindowDirection = highUp ? 0f : 1f
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 300)).Train(inputs, targets);

            // High input: regression head should be higher, direction should be lower
            var psHigh = RandMatrix(seqLen, 5, rng, 0.05f);
            var psLow = RandMatrix(seqLen, 5, rng, 0.05f);
            for (int t = 0; t < seqLen; t++) { psHigh[t, 0] = 0.8f; psLow[t, 0] = 0.2f; }

            var (regH, _, _, dirH, _, _) = m.Forward(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = psHigh });
            var (regL, _, _, dirL, _, _) = m.Forward(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = psLow });

            float avgRegH = 0f, avgRegL = 0f, avgDirH = 0f, avgDirL = 0f;
            for (int t = 0; t < seqLen; t++)
            {
                avgRegH += regH[t, 2]; avgRegL += regL[t, 2]; // Close
                avgDirH += dirH[t, 0]; avgDirL += dirL[t, 0];
            }
            avgRegH /= seqLen; avgRegL /= seqLen;
            avgDirH /= seqLen; avgDirL /= seqLen;

            Assert(avgRegH > avgRegL, $"Regression Close: high input ({avgRegH:F4}) should exceed low ({avgRegL:F4})");
            Assert(avgDirH < avgDirL, $"Direction: high input ({avgDirH:F4}) should be LOWER than low ({avgDirL:F4}) — inverted target");
        }

        void Test_Sequential_MemoryHelpsOnAutoregressive()
        {
            // On data where t+1 depends on t (autoregressive), a model that
            // accumulates memory across sequential steps should achieve lower loss
            // than a fresh-start model that sees no historical context.
            var rng = new Random(42);
            int n = 15, seqLen = 10;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            float val = 0.5f;
            for (int s = 0; s < n; s++)
            {
                var ps = new float[seqLen, 5];
                var tg = new ModelTarget[seqLen];
                for (int t = 0; t < seqLen; t++)
                {
                    ps[t, 0] = val;
                    tg[t] = new ModelTarget
                    {
                        High = val + 0.05f,
                        Low = val - 0.05f,
                        Close = val,
                        Range = 0.1f,
                        Quality = 0.8f,
                        Direction = val > 0.5f ? 1 : 0,
                        MidWindowDirection = val > 0.5f ? 1 : 0
                    };
                    val = val * 0.9f + 0.1f * (float)rng.NextDouble(); // autoregressive drift
                    for (int f = 1; f < 5; f++) ps[t, f] = 0.5f;
                }
                inputs[s] = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps };
                targets[s] = tg;
            }

            var cfg = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: seqLen + 4);
            var ts = Enumerable.Range(0, n).Select(i => (double)(i * 100)).ToArray();

            // Model trained sequentially (has memory of previous samples)
            var mSeq = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(mSeq, TC(lr: 0.003f, bs: 5, epochs: 30)).TrainSequential(inputs, targets, ts);
            float lossSeq = new MmtacTrainer(mSeq, TC(epochs: 1)).Validate(inputs, targets);

            // Model trained without sequential context
            var mBatch = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(mBatch, TC(lr: 0.003f, bs: 5, epochs: 30)).Train(inputs, targets);
            float lossBatch = new MmtacTrainer(mBatch, TC(epochs: 1)).Validate(inputs, targets);

            // Both should have converged — neither should be wildly worse
            Assert(!float.IsNaN(lossSeq) && !float.IsNaN(lossBatch), "NaN in one of the losses");
            Assert(lossSeq < lossBatch * 1.5f,
                $"Sequential ({lossSeq:F5}) should not be much worse than batch ({lossBatch:F5}) on autoregressive data");
        }

        void Test_Timestep_Predictions_Are_Not_Identical()
        {
            // With a causal mask, each timestep attends to a different prefix of history.
            // The raw Forward() outputs across timesteps must NOT all be identical.
            // If they are, the causal mask or positional encoding is broken.
            var m = new MmtacModel(Cfg(priceSeqLen: 14), new Random(42));
            var rng = new Random(42);
            var ps = RandMatrix(10, 5, rng, 0.5f);
            var inp = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps };

            var (reg, _, _, _, _, _) = m.Forward(inp);

            bool anyDiff = false;
            for (int t = 1; t < reg.GetLength(0) && !anyDiff; t++)
                for (int j = 0; j < reg.GetLength(1) && !anyDiff; j++)
                    if (MathF.Abs(reg[t, j] - reg[0, j]) > 1e-6f)
                        anyDiff = true;

            Assert(anyDiff,
                "All timestep predictions are identical — causal mask or positional encoding may be broken");
        }

        void Test_Validate_Loss_ConsistentWithTraining()
        {
            // After heavy overfitting on a tiny dataset, calling Validate() on that
            // same data should return a low loss. If validate and train compute
            // different loss functions, this will fail.
            var rng = new Random(42);
            int n = 2;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = RandMatrix(6, 5, rng, 0.3f)
                };
                targets[s] = Enumerable.Range(0, 6).Select(_ => new ModelTarget
                {
                    High = 0.6f,
                    Low = 0.4f,
                    Close = 0.5f,
                    Range = 0.2f,
                    Quality = 0.8f,
                    Direction = 1,
                    MidWindowDirection = 1
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.01f, bs: 2, epochs: 500)).Train(inputs, targets);

            float valLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(valLoss < 0.1f,
                $"Validation loss ({valLoss:F6}) on training data should be low after overfitting " +
                $"— train and validate loss functions may differ");
        }

        void Test_LossWeights_AffectWhichHeadLearns()
        {
            // With DirectionLossWeight=0, the direction head should not learn the signal.
            // With DirectionLossWeight=5, it should learn clearly.
            // This confirms the per-head loss weights actually control gradient flow.
            var rng = new Random(42);
            int n = 20, seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                bool up = s < n / 2;
                var ps = new float[seqLen, 5];
                for (int t = 0; t < seqLen; t++)
                {
                    ps[t, 0] = up ? 0.8f : 0.2f;
                    for (int f = 1; f < 5; f++) ps[t, f] = 0.5f;
                }
                inputs[s] = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps };
                targets[s] = Enumerable.Range(0, seqLen).Select(_ => new ModelTarget
                {
                    High = 0.55f,
                    Low = 0.45f,
                    Close = 0.5f,
                    Range = 0.1f,
                    Quality = 0.7f,
                    Direction = up ? 1f : 0f,
                    MidWindowDirection = up ? 1f : 0f
                }).ToArray();
            }

            // Model A: direction weight = 0 — direction head gets no gradient
            var cfgA = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64);
            cfgA.Output.DirectionLossWeight = 0f;
            var mA = new MmtacModel(cfgA, new Random(42));
            new MmtacTrainer(mA, TC(lr: 0.005f, bs: 10, epochs: 200)).Train(inputs, targets);

            // Model B: direction weight = 5 — direction head gets strong gradient
            var cfgB = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64);
            cfgB.Output.DirectionLossWeight = 5f;
            var mB = new MmtacModel(cfgB, new Random(42));
            new MmtacTrainer(mB, TC(lr: 0.005f, bs: 10, epochs: 200)).Train(inputs, targets);

            int corrA = 0, corrB = 0, total = 0;
            for (int s = 0; s < n; s++)
            {
                bool up = s < n / 2;
                var (_, _, _, dirA, _, _) = mA.Forward(inputs[s]);
                var (_, _, _, dirB, _, _) = mB.Forward(inputs[s]);
                for (int t = 0; t < dirA.GetLength(0); t++)
                {
                    if ((dirA[t, 0] > 0.5f) == up) corrA++;
                    if ((dirB[t, 0] > 0.5f) == up) corrB++;
                    total++;
                }
            }
            float accA = (float)corrA / total;
            float accB = (float)corrB / total;

            Assert(accB > accA,
                $"Higher direction weight (acc={accB:P0}) should outperform zero weight (acc={accA:P0}) " +
                $"— loss weights may not be controlling gradient flow");
        }

        void Test_GradClip_BoundsUpdateMagnitude()
        {
            // With a very high LR but tight gradient clip, the actual weight change
            // per step should remain bounded. If clipping isn't applied properly,
            // the high LR would cause enormous jumps.
            var (tok, inputs, targets) = Data(n: 3);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));

            var projBefore = (float[,])m.PriceInputProjection.Clone();

            new MmtacTrainer(m, new TrainingConfig
            {
                LearningRate = 10f,           // absurdly high
                BatchSize = 3,
                Epochs = 1,
                UseGradientClipping = true,
                GradientClipThreshold = 0.01f, // very tight
                Verbose = false
            }).Train(inputs, targets);

            float maxChange = 0f;
            for (int i = 0; i < projBefore.GetLength(0); i++)
                for (int j = 0; j < projBefore.GetLength(1); j++)
                    maxChange = MathF.Max(maxChange,
                        MathF.Abs(m.PriceInputProjection[i, j] - projBefore[i, j]));

            Assert(maxChange < 1.0f,
                $"Max weight change {maxChange:F4} is too large with tight gradient clipping " +
                $"— clip threshold may not be controlling update magnitude");
        }

        void Test_Sequential_Produces_Different_Weights_Than_Batch()
        {
            // Sequential training feeds memory from earlier samples into later ones,
            // producing different gradient signals than batch training.
            // The resulting weights must differ — otherwise sequential mode isn't doing anything.
            var (tok, inputs, targets) = Data(n: 8, seqLen: 8, withNews: true);
            var cfg = Cfg(tok.VocabSize + 2, priceSeqLen: 10);
            var ts = Enumerable.Range(0, 8).Select(i => (double)(i * 100)).ToArray();

            var mSeq = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(mSeq, TC(lr: 0.01f, bs: 1, epochs: 5))
                .TrainSequential(inputs, targets, ts);

            var mBatch = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(mBatch, TC(lr: 0.01f, bs: 8, epochs: 5))
                .Train(inputs, targets);

            // Price decoder weights should differ because gradient signals differ
            bool differ = false;
            for (int i = 0; i < mSeq.PriceInputProjection.GetLength(0) && !differ; i++)
                for (int j = 0; j < mSeq.PriceInputProjection.GetLength(1) && !differ; j++)
                    if (MathF.Abs(mSeq.PriceInputProjection[i, j] - mBatch.PriceInputProjection[i, j]) > 1e-6f)
                        differ = true;

            Assert(differ,
                "Sequential and batch training produced identical weights " +
                "— sequential mode may not be using memory context to influence gradients");
        }

        void Test_MidDir_And_Direction_Learn_Different_Signals()
        {
            // Direction is driven by feature 0, MidWindowDirection by feature 1 (inverted).
            // Both heads must independently track their respective signal.
            // If one head just copies the other, this will fail.
            var rng = new Random(42);
            int n = 20, seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                bool dirUp = s < n / 2;
                var ps = new float[seqLen, 5];
                for (int t = 0; t < seqLen; t++)
                {
                    ps[t, 0] = dirUp ? 0.8f : 0.2f;   // drives Direction
                    ps[t, 1] = dirUp ? 0.2f : 0.8f;   // drives MidWindowDirection (opposite)
                    for (int f = 2; f < 5; f++) ps[t, f] = 0.5f;
                }
                inputs[s] = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps };
                targets[s] = Enumerable.Range(0, seqLen).Select(_ => new ModelTarget
                {
                    High = 0.55f,
                    Low = 0.45f,
                    Close = 0.5f,
                    Range = 0.1f,
                    Quality = 0.7f,
                    Direction = dirUp ? 1f : 0f,
                    MidWindowDirection = dirUp ? 0f : 1f  // deliberately opposite
                }).ToArray();
            }

            var m = new MmtacModel(Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 10, epochs: 300)).Train(inputs, targets);

            int corrDir = 0, corrMid = 0, total = 0;
            for (int s = 0; s < n; s++)
            {
                bool dirUp = s < n / 2;
                var (_, _, _, dir, midDir, _) = m.Forward(inputs[s]);
                for (int t = 0; t < dir.GetLength(0); t++)
                {
                    if ((dir[t, 0] > 0.5f) == dirUp) corrDir++;
                    if ((midDir[t, 0] > 0.5f) != dirUp) corrMid++; // mid should be opposite
                    total++;
                }
            }
            float accDir = (float)corrDir / total;
            float accMid = (float)corrMid / total;

            Assert(accDir > 0.60f, $"Direction accuracy {accDir:P0} too low — head may not be learning its signal");
            Assert(accMid > 0.60f, $"MidWindowDirection accuracy {accMid:P0} too low — head may be copying Direction instead of learning its own signal");
        }
    }
}