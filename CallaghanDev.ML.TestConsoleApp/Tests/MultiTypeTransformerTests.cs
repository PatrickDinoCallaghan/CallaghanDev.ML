using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MultiTypeTransformer;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
            // Forward pass: discrete
            (Test_Text_ForwardShape,                         "Text: forward output shape [seqLen, vocabSize]"),
            (Test_Text_ForwardNoNaNOrInfinity,               "Text: forward produces finite logits"),
            (Test_Text_ForwardDeterministic,                 "Text: forward is deterministic"),
            (Test_Text_ForwardLen1,                          "Text: single-token input works"),
            (Test_Text_CausalMaskFutureTokenDoesNotAffectPast,"Text: causal mask blocks future token content"),
            (Test_Text_CausalMaskFutureEmbeddingDoesNotAffectPastPositions,"Text: causal mask blocks future embedding changes"),
            (Test_Text_ForwardRejectsBadToken,               "Text: invalid token id is rejected"),
            (Test_Text_ForwardRejectsTooLongSequence,        "Text: sequence longer than MaxSequenceLength is rejected"),

            // Generation
            (Test_Text_GenerateAddsTokens,                   "Text: Generate() produces new tokens"),
            (Test_Text_GenerateTokenBounds,                  "Text: Generate() tokens stay inside vocabulary"),
            (Test_Text_GenerateZeroNewTokensReturnsPrompt,   "Text: Generate(0) returns the prompt unchanged"),
            (Test_Text_GenerateRespectsMaxSequenceLength,    "Text: Generate() respects MaxSequenceLength"),
            (Test_Text_GenerateRejectsBadTemperature,        "Text: Generate() rejects non-positive temperature"),

            // Training: discrete
            (Test_Text_LossDecreases,                        "Text: loss decreases after training"),
            (Test_Text_BatchTrainingLossDecreases,           "Text: batch training decreases average loss"),
            (Test_Text_DifferentSeqLengthsTrain,             "Text: mixed-length batch trains without error"),
            (Test_Text_InvalidAndShortSequencesDoNotPoisonTraining,"Text: invalid/short sequences do not poison valid training"),
            (Test_Text_UsedTokenEmbeddingUpdated,            "Text: used token embeddings update"),
            (Test_Text_UnusedTokenEmbeddingUnchanged,        "Text: unused token embedding remains unchanged"),
            (Test_Text_AttentionParametersUpdated,           "Text: attention parameters update"),
            (Test_Text_OutputProjectionAndBiasUpdated,       "Text: output projection and bias update"),
            (Test_Text_LayerNormParametersUpdated,           "Text: layer norm parameters update"),
            (Test_Text_GradientClippingPreventsNaN,          "Text: gradient clipping prevents NaN"),
            (Test_Text_PostTrainWeightsBounded,              "Text: clipped training keeps weights bounded"),
            (Test_Text_LearningRateDecayNoCrash,             "Text: learning-rate decay trains without crash"),
            (Test_Text_SeedRobustness,                       "Text: learning is robust across seeds"),
            (Test_Text_OverfitSingleSequence,                "Text: strongly overfits a single sequence"),
            (Test_Text_OverfitNextTokenArgmax,               "Text: overfit gives correct argmax next-token prediction"),

            // Save/load
            (Test_Text_SaveLoadBeforeTraining,               "Text: Save/Load preserves untrained forward output"),
            (Test_Text_SaveLoadAfterTraining,                "Text: Save/Load preserves trained forward output"),
            (Test_Continuous_SaveLoadRegression,             "Continuous: Save/Load preserves regression forward output"),

            // Continuous modes
            (Test_TSRegression_ForwardShape,                 "TimeSeriesRegression: forward shape"),
            (Test_TSRegression_ForwardNoNaNOrInfinity,       "TimeSeriesRegression: forward produces finite values"),
            (Test_TSRegression_PredictNextShape,             "TimeSeriesRegression: PredictNext shape"),
            (Test_TSRegression_LossDecreases,                "TimeSeriesRegression: loss decreases"),
            (Test_TSRegression_InputProjectionUpdated,       "TimeSeriesRegression: input projection updates"),
            (Test_TSClassification_ForwardShape,             "TimeSeriesClassification: forward shape"),
            (Test_TSClassification_LossDecreases,            "TimeSeriesClassification: loss decreases"),
            (Test_SymbolicSeq_LossDecreases,                 "SymbolicSequence: loss decreases"),

            // Type guards
            (Test_DiscreteForwardThrowsOnContinuousModel,     "Type guard: Forward(int[]) throws on continuous model"),
            (Test_ContinuousForwardThrowsOnDiscreteModel,     "Type guard: Forward(float[,]) throws on discrete model"),
            (Test_GenerateThrowsOnContinuousModel,            "Type guard: Generate() throws on continuous model"),
            (Test_PredictNextThrowsOnDiscreteModel,           "Type guard: PredictNext() throws on discrete model"),
            (Test_TrainThrowsOnContinuousModel,               "Type guard: Train() throws on continuous model"),
            (Test_TrainContinuousThrowsOnDiscreteModel,       "Type guard: TrainContinuous() throws on discrete model"),

            // Config guards
            (Test_Config_ValidateAcceptsGoodConfig,           "Config: Validate accepts good config"),
            (Test_Config_RejectsEmbeddingNotDivisibleByHeads, "Config: rejects EmbeddingDim not divisible by NumHeads"),
            (Test_Config_RejectsOddRoPEHeadDim,               "Config: rejects odd per-head dim for RoPE"),
            (Test_Config_RejectsInvalidContinuousClassificationOutputDim,"Config: rejects continuous classification OutputDim < 2"),

            // Attention and numerical primitives
            (Test_MHA_ConstructorRejectsOddHeadDim,           "MHA: constructor rejects odd per-head dim"),
            (Test_MHA_ForwardShape,                          "MHA: forward shape"),
            (Test_MHA_MaskBlocksFutureToken,                  "MHA: explicit mask blocks future token"),
            (Test_CPU_SoftmaxMaskedRows,                      "CPU: masked softmax rows sum to 1 over valid entries"),
            (Test_CPU_SoftmaxAllMaskedRowIsZero,              "CPU: all-masked softmax row is zero"),
            (Test_CPU_MatrixMultiplyTransposeValues,          "CPU: MatrixMultiplyTranspose values are correct"),
            (Test_CPU_LayerNormForwardStats,                  "CPU: LayerNorm output has expected row stats"),
            (Test_CPU_LayerNormBackwardFinite,                "CPU: LayerNorm backward produces finite gradients"),
        };

        #region Forward discrete

        private void Test_Text_ForwardShape()
        {
            var (m, cfg) = Discrete();
            var logits = m.Forward(new[] { 1, 2, 3, 4 });
            Assert(logits.GetLength(0) == 4, "rows");
            Assert(logits.GetLength(1) == cfg.VocabSize, "cols");
        }

        private void Test_Text_ForwardNoNaNOrInfinity()
        {
            var (m, _) = Discrete();
            AssertFinite(m.Forward(new[] { 0, 1, 2, 3, 4 }), "logits");
        }

        private void Test_Text_ForwardDeterministic()
        {
            var (m, _) = Discrete();
            int[] input = { 1, 3, 5 };
            var a = m.Forward(input);
            var b = m.Forward(input);
            AssertMatrixClose(a, b, 0f, "forward should be exactly deterministic on CPU");
        }

        private void Test_Text_ForwardLen1()
        {
            var (m, cfg) = Discrete();
            var logits = m.Forward(new[] { 3 });
            Assert(logits.GetLength(0) == 1, "rows should be 1");
            Assert(logits.GetLength(1) == cfg.VocabSize, "cols should match vocab");
            AssertFinite(logits, "single-token logits");
        }

        private void Test_Text_CausalMaskFutureTokenDoesNotAffectPast()
        {
            var (m, _) = Discrete(vocabSize: 16, embDim: 16, numHeads: 2, numLayers: 2);

            int[] a = { 1, 2, 3, 4, 5 };
            int[] b = { 1, 2, 3, 4, 9 };

            var logitsA = m.Forward(a);
            var logitsB = m.Forward(b);

            for (int pos = 0; pos < 4; pos++)
                for (int j = 0; j < logitsA.GetLength(1); j++)
                    Assert(MathF.Abs(logitsA[pos, j] - logitsB[pos, j]) < 1e-5f, $"future token affected pos={pos}, col={j}");
        }

        private void Test_Text_CausalMaskFutureEmbeddingDoesNotAffectPastPositions()
        {
            var (m, _) = Discrete(vocabSize: 16, embDim: 16, numHeads: 2, numLayers: 2);
            int[] input = { 1, 2, 3, 4, 5 };
            var before = m.Forward(input);

            var saved = GetRow(m.TokenEmbedding, 5);
            for (int j = 0; j < m.Config.EmbeddingDim; j++)
                m.TokenEmbedding[5, j] += 10f * (j + 1);

            var after = m.Forward(input);
            SetRow(m.TokenEmbedding, 5, saved);

            for (int pos = 0; pos < input.Length - 1; pos++)
                for (int j = 0; j < before.GetLength(1); j++)
                    Assert(MathF.Abs(before[pos, j] - after[pos, j]) < 1e-5f, $"future embedding affected pos={pos}, col={j}");
        }

        private void Test_Text_ForwardRejectsBadToken()
        {
            var (m, _) = Discrete(vocabSize: 8);
            AssertThrows<ArgumentOutOfRangeException>(() => m.Forward(new[] { 1, 99 }), "bad token should throw");
            AssertThrows<ArgumentOutOfRangeException>(() => m.Forward(new[] { -1, 2 }), "negative token should throw");
        }

        private void Test_Text_ForwardRejectsTooLongSequence()
        {
            var (m, _) = Discrete(vocabSize: 8, maxSeq: 4);
            AssertThrows<ArgumentException>(() => m.Forward(new[] { 1, 2, 3, 4, 5 }), "too-long sequence should throw");
        }

        #endregion

        #region Generation

        private void Test_Text_GenerateAddsTokens()
        {
            var (m, _) = Discrete(vocabSize: 20, embDim: 16, numHeads: 2, numLayers: 2);
            var generated = m.Generate(new[] { 1, 2 }, maxNewTokens: 5);
            Assert(generated.Length > 2, "no tokens generated");
        }

        private void Test_Text_GenerateTokenBounds()
        {
            var (m, cfg) = Discrete(vocabSize: 20);
            var generated = m.Generate(new[] { 1, 2 }, maxNewTokens: 8);
            foreach (int token in generated)
                Assert(token >= 0 && token < cfg.VocabSize, $"out-of-vocab token {token}");
        }

        private void Test_Text_GenerateZeroNewTokensReturnsPrompt()
        {
            var (m, _) = Discrete();
            int[] prompt = { 1, 2, 3 };
            var generated = m.Generate(prompt, maxNewTokens: 0);
            AssertSequenceEqual(prompt, generated, "Generate(0)");
        }

        private void Test_Text_GenerateRespectsMaxSequenceLength()
        {
            var (m, cfg) = Discrete(vocabSize: 12, maxSeq: 6);
            int[] prompt = { 1, 2, 3, 4, 5, 6 };
            var generated = m.Generate(prompt, maxNewTokens: 5);
            Assert(generated.Length == cfg.MaxSequenceLength, $"expected cap at {cfg.MaxSequenceLength}, got {generated.Length}");
        }

        private void Test_Text_GenerateRejectsBadTemperature()
        {
            var (m, _) = Discrete();
            AssertThrows<ArgumentOutOfRangeException>(() => m.Generate(new[] { 1 }, 1, 0f), "zero temperature should throw");
            AssertThrows<ArgumentOutOfRangeException>(() => m.Generate(new[] { 1 }, 1, -1f), "negative temperature should throw");
        }

        #endregion

        #region Training discrete

        private void Test_Text_LossDecreases()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            int[] seq = { 1, 2, 3, 4, 5 };
            float before = CrossEntropyLoss(m, new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 });

            new TransformerTrainer(m, TrainCfg(lr: 0.005f, epochs: 35, bs: 1)).Train(new[] { seq });

            float after = CrossEntropyLoss(m, new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 });
            AssertLocalLossImproved(before, after, 0.90f);
        }

        private void Test_Text_BatchTrainingLossDecreases()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2);
            int[][] seqs =
            {
                new[] { 1, 2, 3, 4 },
                new[] { 2, 3, 4, 5 },
                new[] { 3, 4, 5, 6 },
                new[] { 4, 5, 6, 7 }
            };

            float before = AverageDiscreteLoss(m, seqs);
            new TransformerTrainer(m, TrainCfg(epochs: 25, bs: 4)).Train(seqs);
            float after = AverageDiscreteLoss(m, seqs);

            AssertLocalLossImproved(before, after, 0.90f);
        }

        private void Test_Text_DifferentSeqLengthsTrain()
        {
            var (m, _) = Discrete();
            new TransformerTrainer(m, TrainCfg(epochs: 3, bs: 3)).Train(new[]
            {
                new[] { 1, 2 },
                new[] { 1, 2, 3, 4, 5 },
                new[] { 3, 4, 5 }
            });
            AssertFinite(m.Forward(new[] { 1, 2, 3 }), "post-train logits");
        }

        private void Test_Text_InvalidAndShortSequencesDoNotPoisonTraining()
        {
            var (m, _) = Discrete(vocabSize: 8, embDim: 16, numHeads: 2, numLayers: 1);
            float before = CrossEntropyLoss(m, new[] { 1, 2, 3 }, new[] { 2, 3, 4 });

            new TransformerTrainer(m, TrainCfg(lr: 0.005f, epochs: 10, bs: 4)).Train(new[]
            {
                Array.Empty<int>(),
                new[] { 1 },
                new[] { 99, 2, 3 },
                new[] { 1, 2, 3, 4 }
            });

            float after = CrossEntropyLoss(m, new[] { 1, 2, 3 }, new[] { 2, 3, 4 });
            Assert(float.IsFinite(after), "loss must remain finite");
            Assert(after < before, $"valid sample did not learn: {before:F6} -> {after:F6}");
        }

        private void Test_Text_UsedTokenEmbeddingUpdated()
        {
            var (m, _) = Discrete(embDim: 8);
            var before = GetRow(m.TokenEmbedding, 1);
            new TransformerTrainer(m, TrainCfg(lr: 0.02f, epochs: 4, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4 } });
            var after = GetRow(m.TokenEmbedding, 1);
            Assert(LocalVectorChanged(before, after), "used token embedding did not change");
        }

        private void Test_Text_UnusedTokenEmbeddingUnchanged()
        {
            var (m, _) = Discrete(vocabSize: 12);
            var before = GetRow(m.TokenEmbedding, 11);
            new TransformerTrainer(m, TrainCfg(epochs: 5, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4 } });
            var after = GetRow(m.TokenEmbedding, 11);
            AssertVectorClose(before, after, 1e-10f, "unused token should not change");
        }

        private void Test_Text_AttentionParametersUpdated()
        {
            var (m, _) = Discrete(embDim: 8, numHeads: 2, numLayers: 1);
            var beforeWQ = (float[,])m.Blocks[0].Attention.WQ.Clone();
            var beforeWK = (float[,])m.Blocks[0].Attention.WK.Clone();
            var beforeWV = (float[,])m.Blocks[0].Attention.WV.Clone();
            var beforeWO = (float[,])m.Blocks[0].Attention.WO.Clone();

            new TransformerTrainer(m, TrainCfg(lr: 0.02f, epochs: 4, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4, 5 } });

            bool changed = LocalMatrixChanged(beforeWQ, m.Blocks[0].Attention.WQ)
                        || LocalMatrixChanged(beforeWK, m.Blocks[0].Attention.WK)
                        || LocalMatrixChanged(beforeWV, m.Blocks[0].Attention.WV)
                        || LocalMatrixChanged(beforeWO, m.Blocks[0].Attention.WO);
            Assert(changed, "no attention matrix changed");
        }

        private void Test_Text_OutputProjectionAndBiasUpdated()
        {
            var (m, _) = Discrete(embDim: 8);
            var beforeW = (float[,])m.OutputProjection.Clone();
            var beforeB = (float[])m.OutputBias.Clone();
            new TransformerTrainer(m, TrainCfg(lr: 0.02f, epochs: 3, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4 } });
            Assert(LocalMatrixChanged(beforeW, m.OutputProjection), "OutputProjection unchanged");
            Assert(LocalVectorChanged(beforeB, m.OutputBias), "OutputBias unchanged");
        }

        private void Test_Text_LayerNormParametersUpdated()
        {
            var (m, _) = Discrete(embDim: 8);
            var beforeGamma = (float[])m.Blocks[0].LN1Gamma.Clone();
            var beforeBeta = (float[])m.Blocks[0].LN1Beta.Clone();
            new TransformerTrainer(m, TrainCfg(lr: 0.02f, epochs: 4, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4, 5 } });
            Assert(LocalVectorChanged(beforeGamma, m.Blocks[0].LN1Gamma) || LocalVectorChanged(beforeBeta, m.Blocks[0].LN1Beta), "LN1 params unchanged");
        }

        private void Test_Text_GradientClippingPreventsNaN()
        {
            var (m, _) = Discrete();
            new TransformerTrainer(m, TrainCfg(lr: 0.5f, epochs: 4, clip: true, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4 } });
            AssertFinite(m.Forward(new[] { 1, 2 }), "logits after clipped training");
        }

        private void Test_Text_PostTrainWeightsBounded()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            new TransformerTrainer(m, TrainCfg(lr: 0.5f, epochs: 5, clip: true, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4, 5 } });

            Assert(LocalMaxAbs(m.Blocks[0].Attention.WQ) < 50f, "WQ exploded");
            Assert(LocalMaxAbs(m.Blocks[0].Attention.WO) < 50f, "WO exploded");
            Assert(LocalMaxAbs(m.OutputProjection) < 50f, "OutputProjection exploded");
            Assert(LocalMaxAbs(m.OutputBias) < 50f, "OutputBias exploded");
        }

        private void Test_Text_LearningRateDecayNoCrash()
        {
            var (m, _) = Discrete();
            var cfg = TrainCfg(lr: 0.01f, bs: 1, epochs: 10);
            cfg.UseLearningRateDecay = true;
            cfg.LearningRateDecay = 0.9f;
            new TransformerTrainer(m, cfg).Train(new[] { new[] { 1, 2, 3 } });
            AssertFinite(m.Forward(new[] { 1, 2 }), "logits after LR decay");
        }

        private void Test_Text_SeedRobustness()
        {
            int[] seeds = { 1, 42, 1337 };
            var losses = new List<float>();

            foreach (int seed in seeds)
            {
                var (m, _) = DiscreteWithSeed(seed, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
                int[] seq = { 1, 2, 3, 4, 5 };

                float before = CrossEntropyLoss(m, new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 });
                new TransformerTrainer(m, TrainCfg(lr: 0.005f, epochs: 55, bs: 1)).Train(new[] { seq });
                float after = CrossEntropyLoss(m, new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 });

                Assert(after < before, $"seed {seed} failed to improve: {before:F6} -> {after:F6}");
                losses.Add(after);
            }

            Assert(losses.Max() - losses.Min() < 1.50f, $"seed sensitivity too high: min={losses.Min():F6}, max={losses.Max():F6}");
        }

        private void Test_Text_OverfitSingleSequence()
        {
            var (m, _) = Discrete(vocabSize: 8, embDim: 16, numHeads: 2, numLayers: 2, ffnDim: 32);
            int[] seq = { 1, 2, 3, 4, 5 };
            new TransformerTrainer(m, TrainCfg(lr: 0.005f, bs: 1, epochs: 550)).Train(new[] { seq });

            float loss = CrossEntropyLoss(m, new[] { 1, 2, 3, 4 }, new[] { 2, 3, 4, 5 });
            Assert(loss < 0.10f, $"overfit loss too high: {loss:F6}");
        }

        private void Test_Text_OverfitNextTokenArgmax()
        {
            var (m, _) = Discrete(vocabSize: 8, embDim: 24, numHeads: 2, numLayers: 2, ffnDim: 48);
            int[] seq = { 1, 2, 3, 4, 5 };
            new TransformerTrainer(m, TrainCfg(lr: 0.005f, bs: 1, epochs: 450)).Train(new[] { seq });

            var logits = m.Forward(new[] { 1, 2, 3, 4 });
            int predicted = ArgmaxRow(logits, logits.GetLength(0) - 1);
            Assert(predicted == 5, $"expected next token 5, got {predicted}");
        }

        #endregion

        #region Save/load

        private void Test_Text_SaveLoadBeforeTraining()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2);
            AssertSaveLoadPreservesDiscrete(m, new[] { 1, 2, 3 });
        }

        private void Test_Text_SaveLoadAfterTraining()
        {
            var (m, _) = Discrete(embDim: 16, numHeads: 2, numLayers: 2);
            new TransformerTrainer(m, TrainCfg(epochs: 5, bs: 1)).Train(new[] { new[] { 1, 2, 3, 4, 5 } });
            AssertSaveLoadPreservesDiscrete(m, new[] { 1, 2, 3 });
        }

        private void Test_Continuous_SaveLoadRegression()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 1, embDim: 8);
            var rng = new Random(123);
            var input = LocalRandMatrix(5, 3, rng, 0.5f);
            var before = m.Forward(input);

            string dir = MakeTempDir();
            try
            {
                m.Save(dir);
                var loaded = LanguageModel.Load(dir);
                var after = loaded.Forward(input);
                AssertMatrixClose(before, after, 1e-5f, "continuous Save/Load mismatch");
            }
            finally
            {
                CleanupTempDir(dir);
            }
        }

        #endregion

        #region Continuous

        private void Test_TSRegression_ForwardShape()
        {
            var (m, cfg) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 2);
            var output = m.Forward(new float[4, 3]);
            Assert(output.GetLength(0) == 4, "rows");
            Assert(output.GetLength(1) == cfg.OutputDim, "cols");
        }

        private void Test_TSRegression_ForwardNoNaNOrInfinity()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 1);
            var input = LocalRandMatrix(6, 3, new Random(42), 0.5f);
            AssertFinite(m.Forward(input), "regression forward");
        }

        private void Test_TSRegression_PredictNextShape()
        {
            var (m, cfg) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 2);
            var next = m.PredictNext(LocalRandMatrix(5, 3, new Random(42), 0.5f));
            Assert(next.Length == cfg.OutputDim, "PredictNext length");
            AssertFinite(next, "PredictNext");
        }

        private void Test_TSRegression_LossDecreases()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 1, embDim: 12, numHeads: 2, numLayers: 1, ffnDim: 24);
            BuildRegressionDataset(out var inputs, out var targets, count: 8, seqLen: 7, inputDim: 3);

            var before = new TransformerTrainer(m, TrainCfg(epochs: 1)).ValidateContinuous(inputs, regressionTargets: targets);
            new TransformerTrainer(m, TrainCfg(lr: 0.004f, epochs: 45, bs: 4)).TrainContinuous(inputs, regressionTargets: targets);
            var after = new TransformerTrainer(m, TrainCfg(epochs: 1)).ValidateContinuous(inputs, regressionTargets: targets);

            AssertLocalLossImproved(before, after, 0.90f);
        }

        private void Test_TSRegression_InputProjectionUpdated()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression, inputDim: 3, outputDim: 1);
            BuildRegressionDataset(out var inputs, out var targets, count: 4, seqLen: 6, inputDim: 3);
            var before = (float[,])m.InputProjection.Clone();
            new TransformerTrainer(m, TrainCfg(lr: 0.005f, bs: 2, epochs: 8)).TrainContinuous(inputs, regressionTargets: targets);
            Assert(LocalMatrixChanged(before, m.InputProjection), "InputProjection unchanged");
        }

        private void Test_TSClassification_ForwardShape()
        {
            var (m, cfg) = Continuous(TransformerDataType.TimeSeriesClassification, inputDim: 3, outputDim: 3, embDim: 8);
            var output = m.Forward(LocalRandMatrix(5, 3, new Random(42), 0.5f));
            Assert(output.GetLength(0) == 5, "rows");
            Assert(output.GetLength(1) == cfg.OutputDim, "class cols");
            AssertFinite(output, "classification logits");
        }

        private void Test_TSClassification_LossDecreases()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesClassification, inputDim: 3, outputDim: 3, embDim: 12, numHeads: 2, numLayers: 1, ffnDim: 24);
            BuildClassificationDataset(out var inputs, out var classes, count: 8, seqLen: 7, inputDim: 3, classCount: 3);

            var before = new TransformerTrainer(m, TrainCfg(epochs: 1)).ValidateContinuous(inputs, classTargets: classes);
            new TransformerTrainer(m, TrainCfg(lr: 0.004f, epochs: 55, bs: 4)).TrainContinuous(inputs, classTargets: classes);
            var after = new TransformerTrainer(m, TrainCfg(epochs: 1)).ValidateContinuous(inputs, classTargets: classes);

            AssertLocalLossImproved(before, after, 0.95f);
        }

        private void Test_SymbolicSeq_LossDecreases()
        {
            var cfg = NewDiscreteConfig(TransformerDataType.SymbolicSequence, vocabSize: 8, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);
            var m = new LanguageModel(cfg, new Random(42));
            int[][] seqs =
            {
                new[] { 1, 4, 5, 6, 7, 2 },
                new[] { 1, 6, 7, 4, 5, 2 }
            };

            float before = CrossEntropyLoss(m, new[] { 1, 4, 5, 6 }, new[] { 4, 5, 6, 7 });
            new TransformerTrainer(m, TrainCfg(lr: 0.005f, epochs: 35, bs: 2)).Train(seqs);
            float after = CrossEntropyLoss(m, new[] { 1, 4, 5, 6 }, new[] { 4, 5, 6, 7 });

            AssertLocalLossImproved(before, after, 0.90f);
        }

        #endregion

        #region Type/config guards

        private void Test_DiscreteForwardThrowsOnContinuousModel()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression);
            AssertThrows<InvalidOperationException>(() => m.Forward(new[] { 1, 2 }), "Forward(int[]) should reject continuous model");
        }

        private void Test_ContinuousForwardThrowsOnDiscreteModel()
        {
            var (m, _) = Discrete();
            AssertThrows<InvalidOperationException>(() => m.Forward(new float[3, 4]), "Forward(float[,]) should reject discrete model");
        }

        private void Test_GenerateThrowsOnContinuousModel()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression);
            AssertThrows<InvalidOperationException>(() => m.Generate(new[] { 1 }, 1), "Generate should reject continuous model");
        }

        private void Test_PredictNextThrowsOnDiscreteModel()
        {
            var (m, _) = Discrete();
            AssertThrows<InvalidOperationException>(() => m.PredictNext(new float[2, 3]), "PredictNext should reject discrete model");
        }

        private void Test_TrainThrowsOnContinuousModel()
        {
            var (m, _) = Continuous(TransformerDataType.TimeSeriesRegression);
            AssertThrows<InvalidOperationException>(() => new TransformerTrainer(m, TrainCfg()).Train(new[] { new[] { 1, 2 } }), "Train should reject continuous model");
        }

        private void Test_TrainContinuousThrowsOnDiscreteModel()
        {
            var (m, _) = Discrete();
            AssertThrows<InvalidOperationException>(() => new TransformerTrainer(m, TrainCfg()).TrainContinuous(new[] { new float[3, 2] }, regressionTargets: new[] { new float[3, 1] }), "TrainContinuous should reject discrete model");
        }

        private void Test_Config_ValidateAcceptsGoodConfig()
        {
            var (_, cfg) = Discrete();
            cfg.Validate();
        }

        private void Test_Config_RejectsEmbeddingNotDivisibleByHeads()
        {
            var cfg = NewDiscreteConfig(TransformerDataType.Text, vocabSize: 8, embDim: 10, numHeads: 3, numLayers: 1, ffnDim: 16);
            AssertThrows<ArgumentException>(() => cfg.Validate(), "EmbeddingDim not divisible by NumHeads should throw");
        }

        private void Test_Config_RejectsOddRoPEHeadDim()
        {
            var cfg = NewDiscreteConfig(TransformerDataType.Text, vocabSize: 8, embDim: 6, numHeads: 2, numLayers: 1, ffnDim: 12);
            AssertThrows<ArgumentException>(() => cfg.Validate(), "odd per-head dim should throw");
        }

        private void Test_Config_RejectsInvalidContinuousClassificationOutputDim()
        {
            var cfg = NewContinuousConfig(TransformerDataType.TimeSeriesClassification, inputDim: 3, outputDim: 1, embDim: 8, numHeads: 2, numLayers: 1, ffnDim: 16);
            AssertThrows<ArgumentException>(() => cfg.Validate(), "classification OutputDim < 2 should throw");
        }

        #endregion

        #region Attention / CPU primitive tests

        private void Test_MHA_ConstructorRejectsOddHeadDim()
        {
            AssertThrows<ArgumentException>(() => new MultiHeadAttention(embeddingDim: 6, numHeads: 2, accel: new AccelerationCPU(), random: new Random(1)), "odd head dim should throw");
        }

        private void Test_MHA_ForwardShape()
        {
            var mha = new MultiHeadAttention(embeddingDim: 8, numHeads: 2, accel: new AccelerationCPU(), random: new Random(1));
            var output = mha.Forward(LocalRandMatrix(4, 8, new Random(2), 0.2f));
            Assert(output.GetLength(0) == 4, "rows");
            Assert(output.GetLength(1) == 8, "cols");
            AssertFinite(output, "mha output");
        }

        private void Test_MHA_MaskBlocksFutureToken()
        {
            var mha = new MultiHeadAttention(embeddingDim: 8, numHeads: 2, accel: new AccelerationCPU(), random: new Random(1));
            var inputA = LocalRandMatrix(4, 8, new Random(2), 0.2f);
            var inputB = (float[,])inputA.Clone();
            for (int j = 0; j < 8; j++)
                inputB[3, j] += 100f + j;

            var mask = new AccelerationCPU().CreateCausalMask(4);
            var outA = mha.Forward(inputA, mask);
            var outB = mha.Forward(inputB, mask);

            for (int pos = 0; pos < 3; pos++)
                for (int j = 0; j < 8; j++)
                    Assert(MathF.Abs(outA[pos, j] - outB[pos, j]) < 1e-4f, $"masked future affected pos={pos}, col={j}");
        }

        private void Test_CPU_SoftmaxMaskedRows()
        {
            var acc = new AccelerationCPU();
            float[,] values =
            {
                { 1f, 2f, 3f },
                { -1f, 0f, 1f }
            };
            bool[,] mask =
            {
                { true, false, true },
                { false, true, true }
            };

            var softmax = acc.Softmax(values, mask);
            Assert(MathF.Abs(softmax[0, 1]) < 1e-7f, "masked entry [0,1] not zero");
            Assert(MathF.Abs(softmax[1, 0]) < 1e-7f, "masked entry [1,0] not zero");
            Assert(MathF.Abs((softmax[0, 0] + softmax[0, 2]) - 1f) < 1e-6f, "row 0 sum");
            Assert(MathF.Abs((softmax[1, 1] + softmax[1, 2]) - 1f) < 1e-6f, "row 1 sum");
        }

        private void Test_CPU_SoftmaxAllMaskedRowIsZero()
        {
            var acc = new AccelerationCPU();
            var softmax = acc.Softmax(new float[,] { { 1f, 2f, 3f } }, new bool[,] { { false, false, false } });
            Assert(MathF.Abs(softmax[0, 0]) < 1e-7f, "entry 0");
            Assert(MathF.Abs(softmax[0, 1]) < 1e-7f, "entry 1");
            Assert(MathF.Abs(softmax[0, 2]) < 1e-7f, "entry 2");
        }

        private void Test_CPU_MatrixMultiplyTransposeValues()
        {
            var acc = new AccelerationCPU();
            float[,] a = { { 1f, 2f }, { 3f, 4f } };
            float[,] b = { { 5f, 6f }, { 7f, 8f }, { 9f, 10f } };
            var c = acc.MatrixMultiplyTranspose(a, b);
            Assert(c.GetLength(0) == 2 && c.GetLength(1) == 3, "shape");
            Assert(MathF.Abs(c[0, 0] - 17f) < 1e-6f, "c00");
            Assert(MathF.Abs(c[0, 1] - 23f) < 1e-6f, "c01");
            Assert(MathF.Abs(c[1, 2] - 67f) < 1e-6f, "c12");
        }

        private void Test_CPU_LayerNormForwardStats()
        {
            var acc = new AccelerationCPU();
            float[,] input =
            {
                { 1f, 2f, 3f, 4f },
                { -1f, 0f, 1f, 2f }
            };
            var gamma = Enumerable.Repeat(1f, 4).ToArray();
            var beta = new float[4];
            var output = acc.LayerNorm(input, gamma, beta);

            for (int i = 0; i < output.GetLength(0); i++)
            {
                float mean = 0f;
                for (int j = 0; j < 4; j++) mean += output[i, j];
                mean /= 4f;

                float variance = 0f;
                for (int j = 0; j < 4; j++) variance += (output[i, j] - mean) * (output[i, j] - mean);
                variance /= 4f;

                Assert(MathF.Abs(mean) < 1e-5f, $"mean row {i}");
                Assert(MathF.Abs(variance - 1f) < 1e-3f, $"variance row {i}: {variance}");
            }
        }

        private void Test_CPU_LayerNormBackwardFinite()
        {
            var acc = new AccelerationCPU();
            float[,] input =
            {
                { 1f, 2f, 3f, 4f },
                { -1f, 0f, 1f, 2f }
            };
            var gamma = Enumerable.Repeat(1f, 4).ToArray();
            var beta = new float[4];
            var (output, mean, variance, normalized) = acc.LayerNormForward(input, gamma, beta);
            var dOut = new float[2, 4];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    dOut[i, j] = 0.1f + i + j;

            var (dInput, dGamma, dBeta) = acc.LayerNormBackward(dOut, normalized, gamma, input, mean, variance);
            AssertFinite(output, "LayerNormForward output");
            AssertFinite(dInput, "LayerNormBackward dInput");
            AssertFinite(dGamma, "LayerNormBackward dGamma");
            AssertFinite(dBeta, "LayerNormBackward dBeta");
        }

        #endregion

        #region Helpers

        private (LanguageModel model, TransformerConfig cfg) Discrete(
            int vocabSize = 12,
            int embDim = 8,
            int numHeads = 2,
            int numLayers = 1,
            int ffnDim = 16,
            int maxSeq = 24)
        {
            var cfg = NewDiscreteConfig(TransformerDataType.Text, vocabSize, embDim, numHeads, numLayers, ffnDim, maxSeq);
            cfg.Validate();
            return (new LanguageModel(cfg, new Random(42)), cfg);
        }

        private (LanguageModel model, TransformerConfig cfg) DiscreteWithSeed(
            int seed,
            int vocabSize = 12,
            int embDim = 8,
            int numHeads = 2,
            int numLayers = 1,
            int ffnDim = 16,
            int maxSeq = 24)
        {
            var cfg = NewDiscreteConfig(TransformerDataType.Text, vocabSize, embDim, numHeads, numLayers, ffnDim, maxSeq);
            cfg.Validate();
            return (new LanguageModel(cfg, new Random(seed)), cfg);
        }

        private (LanguageModel model, TransformerConfig cfg) Continuous(
            TransformerDataType dataType,
            int inputDim = 3,
            int outputDim = 1,
            int embDim = 8,
            int numHeads = 2,
            int numLayers = 1,
            int ffnDim = 16,
            int maxSeq = 16)
        {
            var cfg = NewContinuousConfig(dataType, inputDim, outputDim, embDim, numHeads, numLayers, ffnDim, maxSeq);
            cfg.Validate();
            return (new LanguageModel(cfg, new Random(42)), cfg);
        }

        private TransformerConfig NewDiscreteConfig(
            TransformerDataType dataType,
            int vocabSize,
            int embDim,
            int numHeads,
            int numLayers,
            int ffnDim,
            int maxSeq = 24)
        {
            return new TransformerConfig
            {
                Data = new DataConfig { DataType = dataType, CostFunction = CostFunctionType.mse },
                VocabSize = vocabSize,
                MaxSequenceLength = maxSeq,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                UseDecoderOnly = true,
                Runtime = new RuntimeConfig { FFNActivationType = ActivationType.Relu, AccelerationType = AccelerationType.CPU, AccelerationDeviceId = 0 },
                Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 5f }
            };
        }

        private TransformerConfig NewContinuousConfig(
            TransformerDataType dataType,
            int inputDim,
            int outputDim,
            int embDim,
            int numHeads,
            int numLayers,
            int ffnDim,
            int maxSeq = 16)
        {
            return new TransformerConfig
            {
                Data = new DataConfig { DataType = dataType, CostFunction = CostFunctionType.mse },
                InputFeatureDim = inputDim,
                OutputDim = outputDim,
                MaxSequenceLength = maxSeq,
                EmbeddingDim = embDim,
                NumHeads = numHeads,
                NumLayers = numLayers,
                FeedForwardDim = ffnDim,
                UseDecoderOnly = true,
                Runtime = new RuntimeConfig { FFNActivationType = ActivationType.Relu, AccelerationType = AccelerationType.CPU, AccelerationDeviceId = 0 },
                Regularization = new RegularizationConfig { L2RegulationLamda = 0f, GradientClippingThreshold = 5f }
            };
        }

        private TrainingConfig TrainCfg(float lr = 0.005f, int bs = 4, int epochs = 5, bool clip = true)
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

        private float CrossEntropyLoss(LanguageModel model, int[] input, int[] target)
        {
            var logits = model.Forward(input);
            int vocabSize = model.Config.VocabSize;
            int len = Math.Min(logits.GetLength(0), target.Length);
            float loss = 0f;

            for (int i = 0; i < len; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < vocabSize; j++) max = Math.Max(max, logits[i, j]);

                float sum = 0f;
                for (int j = 0; j < vocabSize; j++) sum += MathF.Exp(logits[i, j] - max);

                float prob = MathF.Exp(logits[i, target[i]] - max) / sum;
                loss -= MathF.Log(prob + 1e-10f);
            }

            return loss / Math.Max(1, len);
        }

        private float AverageDiscreteLoss(LanguageModel model, int[][] sequences)
        {
            return sequences.Average(seq => CrossEntropyLoss(model, seq.Take(seq.Length - 1).ToArray(), seq.Skip(1).ToArray()));
        }

        private void AssertSaveLoadPreservesDiscrete(LanguageModel model, int[] input)
        {
            var before = model.Forward(input);
            string dir = MakeTempDir();
            try
            {
                model.Save(dir);
                var loaded = LanguageModel.Load(dir);
                var after = loaded.Forward(input);
                AssertMatrixClose(before, after, 1e-5f, "Save/Load mismatch");
            }
            finally
            {
                CleanupTempDir(dir);
            }
        }

        private void BuildRegressionDataset(out float[][,] inputs, out float[][,] targets, int count, int seqLen, int inputDim)
        {
            var rng = new Random(42);
            inputs = new float[count][,];
            targets = new float[count][,];

            for (int s = 0; s < count; s++)
            {
                inputs[s] = LocalRandMatrix(seqLen, inputDim, rng, 1.0f);
                targets[s] = new float[seqLen, 1];

                for (int t = 1; t < seqLen; t++)
                {
                    targets[s][t, 0] = 0.65f * inputs[s][t - 1, 0]
                                     - 0.25f * inputs[s][t - 1, 1]
                                     + 0.10f * inputs[s][t - 1, 2];
                }
            }
        }

        private void BuildClassificationDataset(out float[][,] inputs, out int[][] classes, int count, int seqLen, int inputDim, int classCount)
        {
            var rng = new Random(123);
            inputs = new float[count][,];
            classes = new int[count][];

            for (int s = 0; s < count; s++)
            {
                inputs[s] = LocalRandMatrix(seqLen, inputDim, rng, 1.0f);
                classes[s] = new int[seqLen];

                for (int t = 1; t < seqLen; t++)
                {
                    float score = inputs[s][t - 1, 0] + 0.5f * inputs[s][t - 1, 1] - 0.25f * inputs[s][t - 1, 2];
                    classes[s][t] = score < -0.25f ? 0 : score > 0.25f ? 2 : 1;
                    if (classes[s][t] >= classCount) classes[s][t] = classCount - 1;
                }
            }
        }

        private float[,] LocalRandMatrix(int rows, int cols, Random rng, float scale)
        {
            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (rng.NextSingle() * 2f - 1f) * scale;
            return matrix;
        }

        private void AssertLocalLossImproved(float before, float after, float maxRatio)
        {
            Assert(float.IsFinite(before) && float.IsFinite(after), $"loss not finite: {before} -> {after}");
            Assert(after < before * maxRatio, $"loss did not improve enough: {before:F6} -> {after:F6}");
        }

        private void AssertThrows<T>(Action action, string message) where T : Exception
        {
            bool threw = false;
            try
            {
                action();
            }
            catch (T)
            {
                threw = true;
            }

            Assert(threw, message);
        }

        private void AssertFinite(float[,] matrix, string name)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    Assert(float.IsFinite(matrix[i, j]), $"{name}[{i},{j}] is {matrix[i, j]}");
        }

        private void AssertFinite(float[] vector, string name)
        {
            for (int i = 0; i < vector.Length; i++)
                Assert(float.IsFinite(vector[i]), $"{name}[{i}] is {vector[i]}");
        }

        private void AssertMatrixClose(float[,] a, float[,] b, float tolerance, string message)
        {
            Assert(a.GetLength(0) == b.GetLength(0), $"{message}: row mismatch");
            Assert(a.GetLength(1) == b.GetLength(1), $"{message}: col mismatch");
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    Assert(MathF.Abs(a[i, j] - b[i, j]) <= tolerance, $"{message} at [{i},{j}]: {a[i, j]} vs {b[i, j]}");
        }

        private void AssertVectorClose(float[] a, float[] b, float tolerance, string message)
        {
            Assert(a.Length == b.Length, $"{message}: length mismatch");
            for (int i = 0; i < a.Length; i++)
                Assert(MathF.Abs(a[i] - b[i]) <= tolerance, $"{message} at [{i}]: {a[i]} vs {b[i]}");
        }

        private void AssertSequenceEqual(int[] expected, int[] actual, string message)
        {
            Assert(expected.Length == actual.Length, $"{message}: length mismatch");
            for (int i = 0; i < expected.Length; i++)
                Assert(expected[i] == actual[i], $"{message}: mismatch at {i}");
        }

        private bool LocalMatrixChanged(float[,] before, float[,] after, float tolerance = 1e-7f)
        {
            for (int i = 0; i < before.GetLength(0); i++)
                for (int j = 0; j < before.GetLength(1); j++)
                    if (MathF.Abs(before[i, j] - after[i, j]) > tolerance)
                        return true;
            return false;
        }

        private bool LocalVectorChanged(float[] before, float[] after, float tolerance = 1e-7f)
        {
            for (int i = 0; i < before.Length; i++)
                if (MathF.Abs(before[i] - after[i]) > tolerance)
                    return true;
            return false;
        }

        private float[] GetRow(float[,] matrix, int row)
        {
            var result = new float[matrix.GetLength(1)];
            for (int j = 0; j < result.Length; j++)
                result[j] = matrix[row, j];
            return result;
        }

        private void SetRow(float[,] matrix, int row, float[] values)
        {
            for (int j = 0; j < values.Length; j++)
                matrix[row, j] = values[j];
        }

        private int ArgmaxRow(float[,] matrix, int row)
        {
            int best = 0;
            float bestValue = matrix[row, 0];
            for (int j = 1; j < matrix.GetLength(1); j++)
            {
                if (matrix[row, j] > bestValue)
                {
                    bestValue = matrix[row, j];
                    best = j;
                }
            }
            return best;
        }

        private float LocalMaxAbs(float[,] matrix)
        {
            float max = 0f;
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    max = Math.Max(max, MathF.Abs(matrix[i, j]));
            return max;
        }

        private float LocalMaxAbs(float[] vector)
        {
            float max = 0f;
            for (int i = 0; i < vector.Length; i++)
                max = Math.Max(max, MathF.Abs(vector[i]));
            return max;
        }

        private string MakeTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "mtt_tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        private void CleanupTempDir(string dir)
        {
            try
            {
                if (Directory.Exists(dir))
                    Directory.Delete(dir, recursive: true);
            }
            catch
            {
                // Test cleanup should not hide the real test failure.
            }
        }

        #endregion
    }
}
