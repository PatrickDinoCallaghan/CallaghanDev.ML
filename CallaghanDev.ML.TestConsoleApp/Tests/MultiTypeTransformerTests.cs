using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;


namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class MultiTypeTransformerTests : TestBase
    {
        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * MultiTypeTransformer");
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            //  forward pass 
            (Test_Text_ForwardShape,                  "Text: forward output shape [seqLen, vocabSize]"),
            (Test_Text_ForwardNaN,                    "Text: forward produces no NaN"),
            (Test_Text_ForwardDeterministic,          "Text: forward is deterministic"),
            (Test_Text_CausalMask,                    "Text: causal mask - future doesn't affect past"),
            (Test_Text_ForwardLen1,                   "Text: single-token input works"),


            //  training 
            (Test_Text_LossDecreases,                 "Text: loss decreases after training"),
            (Test_Text_AllParamsUpdated,              "Text: all parameters receive gradient updates"),
            (Test_Text_FreezeEmbeddingNotUsed,        "Text: unused token embedding unchanged after train"),
            (Test_Text_GradientClipping,              "Text: gradient clipping prevents NaN"),
            (Test_Text_PostTrainWeightsBounded,       "Text: clipped training keeps weights bounded"),
            (Test_Text_LearningRateDecay,             "Text: learning-rate decay applied without crash"),
            (Test_Text_BatchTraining,                 "Text: batch training decreases average loss"),
            (Test_Text_DifferentSeqLengths,           "Text: mixed-length batch trains without error"),
            (Test_Text_SeedRobustness,                "Text: learning is robust across seeds"),

            //  generation 
            (Test_Text_Generate,                      "Text: Generate() produces new tokens"),
            (Test_Text_GenerateTokenBounds,           "Text: Generate() tokens within vocab bounds"),

            //  overfit 
            (Test_Text_OverfitSingle,                 "Text: strongly overfits single sequence"),
            (Test_Text_OverfitPredictionExact,        "Text: overfit yields correct next-token prediction"),

            //  save / load 
            (Test_Text_SaveLoad,                      "Text: Save/Load preserves weights & forward output"),

            //  continuous modes 
            (Test_TSRegression_LossDecreases,         "TimeSeriesRegression: loss decreases"),
            (Test_TSRegression_ProjectionUpdated,     "TimeSeriesRegression: InputProjection updated"),
            (Test_TSClassification_LossDecreases,     "TimeSeriesClassification: loss decreases"),
            (Test_SymbolicSeq_LossDecreases,          "SymbolicSequence: loss decreases"),

            //  type guards 
            (Test_WrongForwardThrows,                 "Type guard: discrete Forward throws on continuous cfg"),
            (Test_WrongForwardThrows2,                "Type guard: continuous Forward throws on discrete cfg"),

            //  config validation 
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
            // Perturb future token embedding - shouldn't change position 0
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

        #region Helpers
        private (LanguageModel model, TransformerConfig cfg) DiscreteWithSeed(int seed, int vocabSize = 12, int embDim = 8, int numHeads = 2, int numLayers = 1, int ffnDim = 16)
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
        { 
            return new TrainingConfig
            {
                LearningRate = lr,
                BatchSize = bs,
                Epochs = epochs,
                UseGradientClipping = clip,
                GradientClipThreshold = 5f,
                Verbose = false
            };
        }

        private float CELoss(LanguageModel m, int[] inp, int[] tgt)
        {
            var logits = m.Forward(inp);
            float loss = 0;
            int vs = m.Config.VocabSize;

            for (int i = 0; i < Math.Min(logits.GetLength(0), tgt.Length); i++)
            {
                float maxVal = float.NegativeInfinity;
                for (int j = 0; j < vs; j++)
                {
                    maxVal = Math.Max(maxVal, logits[i, j]);
                }
                float sum = 0; 
                
                for (int j = 0; j < vs; j++)
                {
                    sum += MathF.Exp(logits[i, j] - maxVal);
                }
                loss -= MathF.Log(MathF.Exp(logits[i, tgt[i]] - maxVal) / sum + 1e-10f);
            }
            return loss / Math.Min(logits.GetLength(0), tgt.Length);
        }

        #endregion
    }

}
