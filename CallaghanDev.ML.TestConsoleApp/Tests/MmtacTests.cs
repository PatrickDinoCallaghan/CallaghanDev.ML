using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MMTAC;
using CallaghanDev.ML.Transformers.TACAMT;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    /// <summary>
    /// Correctness-focused tests for MMTAC.
    ///
    /// This suite deliberately avoids brittle assertions like "this random model must learn a
    /// difficult semantic task to a fixed threshold after N epochs".  Those are optimization
    /// benchmarks, not correctness tests.  The learning tests below use deterministic synthetic
    /// tasks and assert either finite training, parameter movement, or clear loss improvement.
    /// </summary>
    internal sealed class MmtacTests : TestBase
    {
        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * MMTAC (Multimodal Market Transformer with Additional Context)");
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            // High-confidence functional tests
            (Test_HighConfidence_OutputHeadFiniteDifferenceUpdateDirection, "Grad: one-step output-head updates follow finite-difference loss direction"),
            (Test_HighConfidence_TargetAlignment_TrainSimpleLearnsNextRowTargets, "Train: simple path learns next-row target alignment"),
            (Test_HighConfidence_TargetAlignment_PriceContextLearnsNextRowTargets, "Train: price-context path learns next-row target alignment"),
            (Test_HighConfidence_MultimodalHeldOutAblation, "E2E: multimodal held-out improves and ablations hurt"),
            (Test_HighConfidence_SequentialMemoryCarryTask, "E2E: sequential rolling memory improves carry task"),

            // Construction / configuration
            (Test_Config_ValidPresets, "Config: presets validate"),
            (Test_Construction_CoreShapes, "Construction: core tensor shapes are correct"),
            (Test_Construction_OptionalHeadAndGlobalShapes, "Construction: confidence/global optional shapes are correct"),
            (Test_Construction_RuntimePruningConfigCopied, "Construction: runtime pruning config is copied from config"),

            // Forward contracts
            (Test_Forward_PredictNextFiniteAndConstrained_ConfidenceDisabled, "Forward: PredictNext finite and constrained, confidence disabled"),
            (Test_Forward_PredictNextFiniteAndConstrained_ConfidenceEnabled, "Forward: PredictNext finite and constrained, confidence enabled"),
            (Test_Forward_NullAndEmptyStoriesWork, "Forward: null and empty stories work"),
            (Test_Forward_SeqLenOneWorks, "Forward: single timestep works"),
            (Test_Forward_AllHeadShapesMatchSequenceLength, "Forward: all head shapes match sequence length"),
            (Test_Forward_DeterministicAndDoesNotMutateMemory, "Forward: deterministic and does not mutate memory"),
            (Test_Forward_StoryAndGlobalChangePrediction, "Forward: story/global context changes prediction"),
            (Test_Forward_WithCacheMatchesForward_AllHeads, "ForwardWithCache: matches Forward for all heads"),
            (Test_Forward_Causal_NoFutureLeakage_PriceOnly, "Causal: future price rows do not affect previous outputs"),
            (Test_Forward_Causal_NoFutureLeakage_WithNewsAndGlobal, "Causal: future price rows do not affect previous outputs with context"),
            (Test_Forward_WrongPriceFeatureDimThrowsArgumentException, "Forward: wrong price feature dimension throws ArgumentException"),
            (Test_Forward_WrongGlobalFeatureDimThrowsArgumentException, "Forward: wrong global feature dimension throws ArgumentException"),
            (Test_Forward_GlobalFeatureExtraDimThrowsArgumentException, "Forward: extra global features throw ArgumentException"),

            // Decay/context wiring
            (Test_Decay_TimeValueReachesDecoder, "Decay: changing context time changes decoder hidden state"),
            (Test_Decay_ContextContentReachesDecoder, "Decay: changing context content changes decoder output"),
            (Test_Decay_DisabledIgnoresContextTimes, "Decay: disabled decay ignores context times"),

            // Tokenizer
            (Test_Tokenizer_SetTokenizeAndOversizedThrows, "Tokenizer: SetTokenizer/TokenizeStories and oversized tokenizer guard"),
            (Test_Tokenizer_SaveLoadRoundTrip, "Tokenizer: save/load round-trip preserves tokenization"),

            // Public validation / argument contracts
            (Test_PublicMethods_LengthMismatchThrow, "Validation: public methods throw on mismatched array lengths"),
            (Test_PublicMethods_ZeroTimeUnitsThrow, "Validation: zero timeUnitsPerPosition throws"),
            (Test_Validate_ReturnsFiniteNonNegative, "Validate: returns finite non-negative values"),
            (Test_Validate_DoesNotMutateWeights, "Validate: does not mutate weights"),
            (Test_ValidateSequential_DoesNotMutateWeights, "ValidateSequential: does not mutate weights"),
            (Test_ValidateSequential_OrderInvariant, "ValidateSequential: timestamp order invariant"),
            (Test_ValidateSequential_MatchesReferenceLoop, "ValidateSequential: matches no-commit/commit reference"),
            (Test_ValidateSequential_ClearsMemoryOnExit, "ValidateSequential: clears rolling memory on exit"),

            // Training mechanics
            (Test_Train_LossDecreases_OnDeterministicPriceData, "Train: deterministic price loss decreases"),
            (Test_Train_SingleSampleOverfit_LossDecreases, "Train: single sample overfit decreases loss"),
            (Test_Train_MixedNullNewsFinite, "Train: mixed null/non-null news batch is finite"),
            (Test_Train_ParameterUpdateCoverage, "Train: key parameter groups update"),
            (Test_Train_FrozenTextUnchanged_UnfrozenTextChanges, "Train: text freeze is respected"),
            (Test_Train_GlobalProjectionUpdated, "Train: global projection updates"),
            (Test_Train_ConfidenceHeadUpdated, "Train: confidence head updates when enabled"),
            (Test_Train_LossWeightsControlDirectionHeadGradient, "Train: direction loss weight controls direction-head gradient"),
            (Test_Train_ConfidenceLossWeightZeroDisablesConfidenceGradient, "Train: confidence loss weight 0 disables confidence-head gradient"),
            (Test_Train_GradientClippingBoundsUpdate, "Train: gradient clipping bounds update magnitude"),
            (Test_Train_HighLearningRateWithClippingStaysFinite, "Train: high LR with clipping stays finite"),
            (Test_Train_PriceContextPathRunsFinite, "Train: price-context branch runs and stays finite"),
            (Test_Train_DecayNetworkUpdatedWithContext, "Train: decay network receives gradients when context exists"),
            (Test_Train_ZeroLearningRateDoesNotChangeWeights, "Train: zero learning rate does not change weights"),

            // Semantic but deterministic smoke tests
            (Test_Signal_PriceHeldOutLossImproves, "Signal: held-out deterministic price mapping improves"),
            (Test_Signal_NewsSeparationAfterTraining, "Signal: trained news sentiment separates predictions"),
            (Test_Signal_GlobalSeparationAfterTraining, "Signal: trained global signal separates predictions"),
            (Test_Signal_DirectionAndMidDirectionCanLearnOppositeLabels, "Signal: Direction and MidDirection can learn opposite labels"),
            (Test_Signal_RangeHeadLearnsRangeTarget, "Signal: Range head learns range target"),

            // Memory
            (Test_Memory_InitiallyEmptyAndClearMethods, "Memory: starts empty and clear methods work"),
            (Test_Memory_PredictWithMemoryExactGrowthAndTimestamps, "Memory: PredictWithMemory exact growth and timestamps"),
            (Test_Memory_CustomTimeUnitsScaleTimestamps, "Memory: custom timeUnitsPerPosition scales timestamps"),
            (Test_Memory_ZeroBudgetsKeepBanksEmpty, "Memory: zero budgets keep banks empty"),
            (Test_Memory_PrunesToLimits, "Memory: pruning respects max sizes"),
            (Test_Memory_AttentionPruningKeepsHighScore_NewsAndPrice, "Memory: attention pruning keeps high-score entries"),
            (Test_Memory_AttentionScoresUpdated, "Memory: attention scores/query counts update"),
            (Test_Memory_GlobalTokenNotPersistedAsMemory, "Memory: global token affects output but is not persisted"),
            (Test_Memory_PriceOnlyWorksWithExistingPriceMemory, "Memory: price-only rolling memory works"),
            (Test_Memory_NullStoryEntryDoesNotCrashPredictNext, "Memory: null story entries do not crash PredictNext"),
            (Test_Memory_NullStoryEntryDoesNotCrashPredictWithMemory, "Memory: null story entries do not crash PredictWithMemory"),

            // Sequential training / rolling memory
            (Test_Sequential_CommitsFullObservedSampleAndScaledTimestamps, "Sequential: commits full observed sample with scaled timestamps"),
            (Test_Sequential_SkippedInvalidSampleDoesNotCommit, "Sequential: skipped invalid sample does not commit memory"),
            (Test_Sequential_MemoryClearedBetweenEpochs, "Sequential: memory cleared at start of each epoch"),
            (Test_Sequential_TimestampShuffleInvariantAtZeroLearningRate, "Sequential: timestamp shuffle invariant at zero learning rate"),
            (Test_Sequential_ZeroLearningRateDoesNotChangeWeights, "Sequential: zero learning rate does not change weights"),
            (Test_Sequential_WithGlobalTokenRunsFinite, "Sequential: global token path runs finite"),
            (Test_Sequential_MemoryAccumulates, "Sequential: memory accumulates across samples"),

            // Save / load
            (Test_SaveLoad_ForwardIdentical_AllHeads, "SaveLoad: forward all heads identical after round-trip"),
            (Test_SaveLoad_WeightsAndContextPreserved, "SaveLoad: output heads/context/global weights preserved"),
            (Test_SaveLoad_MemoryAndPruningPreserved, "SaveLoad: memory and pruning config preserved"),
            (Test_SaveLoad_ContinueTrainingFinite, "SaveLoad: can continue training after load"),
            (Test_SaveLoad_SequentialMemoryPredictionIdentical, "SaveLoad: rolling-memory prediction identical after reload"),

            // Numerical stability
            (Test_Stability_InputScalesNoNaN, "Stability: large/small/zero/negative inputs produce finite outputs"),
            (Test_Stability_ManyStoriesNoNaN, "Stability: many simultaneous stories produce finite outputs"),
            (Test_Stability_NoNaNAfterManyEpochs, "Stability: no NaN after many epochs"),
        };
        // ---------------------------------------------------------------------
        // High-confidence functional tests
        // ---------------------------------------------------------------------

        private void Test_HighConfidence_OutputHeadFiniteDifferenceUpdateDirection()
        {
            var (inputs, targets) = TinyGradientData();

            void Check(string label, Func<MmtacModel, float> getter, Action<MmtacModel, float> setter)
            {
                var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, useConf: false, priceContextEnabled: false);

                cfg.Output.CloseDirectionConsistencyWeight = 0f;
                cfg.Output.CloseDirectionConsistencyMargin = 0f;

                var model = new MmtacModel(cfg, new Random(42));

                AssertOneStepUpdateOpposesFiniteDifference(
                    label,
                    model,
                    inputs,
                    targets,
                    getter,
                    setter);
            }

            Check(
                "RegressionBias[Close]",
                m => m.RegressionBias[2],
                (m, v) => m.RegressionBias[2] = v);

            Check(
                "RangeBias[0]",
                m => m.RangeBias[0],
                (m, v) => m.RangeBias[0] = v);

            Check(
                "QualityBias[0]",
                m => m.QualityBias[0],
                (m, v) => m.QualityBias[0] = v);

            Check(
                "DirectionBias[0]",
                m => m.DirectionBias[0],
                (m, v) => m.DirectionBias[0] = v);

            Check(
                "MidDirectionBias[0]",
                m => m.MidDirectionBias[0],
                (m, v) => m.MidDirectionBias[0] = v);
        }

        private void Test_HighConfidence_TargetAlignment_TrainSimpleLearnsNextRowTargets()
        {
            var (inputs, targets) = NextRowAlignmentData(sampleCount: 14, seqLen: 7);

            var cfg = Cfg(
                embDim: 24,
                numHeads: 4,
                numLayers: 1,
                ffnDim: 48,
                priceSeqLen: 8,
                priceContextEnabled: false);

            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var model = new MmtacModel(cfg, new Random(42));

            new MmtacTrainer(model, TC(lr: 0.004f, bs: 7, epochs: 120))
                .Train(inputs, targets);

            float nextErr = 0f;
            float currentErr = 0f;
            int count = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                int sl = inputs[i].PriceSequence.GetLength(0);
                var priceInp = SliceRows(inputs[i].PriceSequence, 0, sl - 1);

                var wrapped = new MultimodalInput
                {
                    PredictionTimestamp = inputs[i].PredictionTimestamp,
                    PriceSequence = priceInp
                };

                var (reg, _, _, _, _, _) = model.Forward(wrapped);

                for (int t = 0; t < reg.GetLength(0); t++)
                {
                    float predicted = reg[t, 2];

                    // Correct trainer alignment is input row t -> target row t + 1.
                    float correct = targets[i][t + 1].Close;

                    // This is the common off-by-one bug: input row t -> target row t.
                    float wrongCurrent = targets[i][t].Close;

                    nextErr += MathF.Abs(predicted - correct);
                    currentErr += MathF.Abs(predicted - wrongCurrent);
                    count++;
                }
            }

            nextErr /= Math.Max(1, count);
            currentErr /= Math.Max(1, count);

            Assert(nextErr < 0.12f,
                $"Model did not learn next-row target alignment. nextErr={nextErr:F6}");

            Assert(nextErr < currentErr * 0.70f,
                $"Predictions look closer to current-row targets than next-row targets. nextErr={nextErr:F6}, currentErr={currentErr:F6}");
        }

        private void Test_HighConfidence_TargetAlignment_PriceContextLearnsNextRowTargets()
        {
            // seqLen=7 with MinHistory=3 and MinCurrent=3 makes TrainWithPriceContext choose split=3.
            // Current rows are [3,4,5], and the correct targets are [4,5,6].
            var (inputs, targets) = NextRowAlignmentData(sampleCount: 14, seqLen: 7);

            var cfg = Cfg(
                embDim: 24,
                numHeads: 4,
                numLayers: 1,
                ffnDim: 48,
                priceSeqLen: 8,
                priceContextEnabled: true);

            cfg.PriceContext.MinHistoryLength = 3;
            cfg.PriceContext.MinCurrentLength = 3;
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var model = new MmtacModel(cfg, new Random(42));

            new MmtacTrainer(model, TC(lr: 0.004f, bs: 7, epochs: 140))
                .Train(inputs, targets);

            int split = 3;

            float nextErr = 0f;
            float currentErr = 0f;
            int count = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                int sl = inputs[i].PriceSequence.GetLength(0);

                var hist = SliceRows(inputs[i].PriceSequence, 0, split);
                var current = SliceRows(inputs[i].PriceSequence, split, sl - 1);

                var priceCtxH = model.EncodePriceHistory(hist);
                var priceCtxT = Enumerable.Range(0, split)
                    .Select(t => -(float)(split - t))
                    .ToArray();

                var wrapped = new MultimodalInput
                {
                    PredictionTimestamp = inputs[i].PredictionTimestamp,
                    PriceSequence = current
                };

                var cache = new MmtacForwardCache(
                    cfg.Text.NumLayers,
                    cfg.Price.NumLayers);

                var (reg, _, _, _, _, _) = InvokeForwardWithPriceContextAndCache(
                    model,
                    wrapped,
                    priceCtxH,
                    priceCtxT,
                    cache,
                    isTraining: false);

                for (int t = 0; t < reg.GetLength(0); t++)
                {
                    float predicted = reg[t, 2];

                    // Correct price-context alignment:
                    // current local row t corresponds to original row split + t,
                    // target should be original row split + t + 1.
                    float correct = targets[i][split + t + 1].Close;

                    // Common off-by-one bug:
                    // target original row split + t.
                    float wrongCurrent = targets[i][split + t].Close;

                    nextErr += MathF.Abs(predicted - correct);
                    currentErr += MathF.Abs(predicted - wrongCurrent);
                    count++;
                }
            }

            nextErr /= Math.Max(1, count);
            currentErr /= Math.Max(1, count);

            Assert(nextErr < 0.14f,
                $"Price-context branch did not learn next-row target alignment. nextErr={nextErr:F6}");

            Assert(nextErr < currentErr * 0.70f,
                $"Price-context predictions look closer to current-row targets than next-row targets. nextErr={nextErr:F6}, currentErr={currentErr:F6}");
        }

        private void Test_HighConfidence_MultimodalHeldOutAblation()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) =
                CompositionalMultimodalData(trainN: 96, testN: 32, seqLen: 6);

            var cfg = Cfg(
                vocabSize: tok.VocabSize + 2,
                embDim: 32,
                numHeads: 4,
                numLayers: 1,
                ffnDim: 64,
                globalDim: 3);

            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var model = new MmtacModel(cfg, new Random(42));

            float before = new MmtacTrainer(model, TC(epochs: 1))
                .ValidateAligned(testInputs, testTargets);

            new MmtacTrainer(model, TC(lr: 0.003f, bs: 8, epochs: 140))
                .Train(trainInputs, trainTargets);

            var validator = new MmtacTrainer(model, TC(epochs: 1));

            float baseLoss = validator.ValidateAligned(testInputs, testTargets);

            var noPrice = testInputs
                .Select(x => CloneInput(x, keepPrice: false, keepNews: true, keepGlobals: true))
                .ToArray();

            var noNews = testInputs
                .Select(x => CloneInput(x, keepPrice: true, keepNews: false, keepGlobals: true))
                .ToArray();

            var noGlobal = testInputs
                .Select(x => CloneInput(x, keepPrice: true, keepNews: true, keepGlobals: false))
                .ToArray();

            float noPriceLoss = validator.ValidateAligned(noPrice, testTargets);
            float noNewsLoss = validator.ValidateAligned(noNews, testTargets);
            float noGlobalLoss = validator.ValidateAligned(noGlobal, testTargets);

            Assert(baseLoss < before * 0.65f,
                $"Multimodal held-out loss did not improve enough. before={before:F6}, after={baseLoss:F6}");

            AssertAblationHurts("price", baseLoss, noPriceLoss);
            AssertAblationHurts("news", baseLoss, noNewsLoss);
            AssertAblationHurts("global", baseLoss, noGlobalLoss);
        }

        private void Test_HighConfidence_SequentialMemoryCarryTask()
        {
            int seqLen = 6;

            var (trainInputs, trainTargets, trainTs) =
                SequentialCarryPairsData(pairCount: 40, seqLen: seqLen, offset: 0);

            var (testInputs, testTargets, testTs) =
                SequentialCarryPairsData(pairCount: 16, seqLen: seqLen, offset: 10_000);

            var cfg = Cfg(
                embDim: 32,
                numHeads: 4,
                numLayers: 1,
                ffnDim: 64,
                priceSeqLen: 8);

            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var model = new MmtacModel(cfg, new Random(42));

            new MmtacTrainer(model, TC(lr: 0.003f, bs: 1, epochs: 110))
                .TrainSequential(
                    trainInputs,
                    trainTargets,
                    trainTs,
                    timeUnitsPerPosition: 1.0,
                    maxNewsMemory: 0,
                    maxPriceMemory: seqLen);

            float withMemoryErr = AverageCarryCloseErrorWithMemory(
                model,
                testInputs,
                testTargets,
                testTs,
                maxPriceMemory: seqLen);

            float statelessErr = AverageCarryCloseErrorStateless(
                model,
                testInputs,
                testTargets);

            Assert(withMemoryErr < statelessErr * 0.75f,
                $"Rolling price memory should materially improve carried-regime prediction. withMemory={withMemoryErr:F6}, stateless={statelessErr:F6}");

            Assert(withMemoryErr < 0.22f,
                $"Rolling memory carry error is still high. withMemory={withMemoryErr:F6}");
        }
        // ---------------------------------------------------------------------
        // Construction / config
        // ---------------------------------------------------------------------

        private void Test_Config_ValidPresets()
        {
            Cfg().Validate();
            MmtacConfig.Small(vocabSize: 1000, priceFeatureDim: 5).Validate();
            MmtacConfig.Standard(vocabSize: 5000, priceFeatureDim: 5, globalDim: 8).Validate();
        }

        private void Test_Construction_CoreShapes()
        {
            var m = new MmtacModel(Cfg(vocabSize: 60, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceFeatures: 7), new Random(42));

            Assert(m.TextTokenEmbedding.GetLength(0) == 60, "TextTokenEmbedding rows");
            Assert(m.TextTokenEmbedding.GetLength(1) == 24, "TextTokenEmbedding cols");
            Assert(m.PriceInputProjection.GetLength(0) == 24, "PriceInputProjection rows");
            Assert(m.PriceInputProjection.GetLength(1) == 7, "PriceInputProjection cols");
            Assert(m.RegressionProjection.GetLength(0) == MmtacOutputConfig.RegressionOutputCount, "RegressionProjection rows");
            Assert(m.RegressionProjection.GetLength(1) == 24, "RegressionProjection cols");
            Assert(m.RangeProjection.GetLength(0) == 1 && m.RangeProjection.GetLength(1) == 24, "RangeProjection shape");
            Assert(m.QualityProjection.GetLength(0) == 1 && m.QualityProjection.GetLength(1) == 24, "QualityProjection shape");
            Assert(m.DirectionProjection.GetLength(0) == 1 && m.DirectionProjection.GetLength(1) == 24, "DirectionProjection shape");
            Assert(m.MidDirectionProjection.GetLength(0) == 1 && m.MidDirectionProjection.GetLength(1) == 24, "MidDirectionProjection shape");
            Assert(m.ContextTypeEmbedding.GetLength(0) == MmtacConfig.ContextTypeCount, "ContextTypeEmbedding rows");
            Assert(m.ContextTypeEmbedding.GetLength(1) == 24, "ContextTypeEmbedding cols");
            Assert(m.PriceBlocks.Length == 2, "PriceBlocks length");
            Assert(m.PriceBlocks.All(b => b.SelfAttention.WQ.GetLength(0) == 24 && b.SelfAttention.WQ.GetLength(1) == 24), "PriceBlock SelfAttention.WQ shape");
        }

        private void Test_Construction_OptionalHeadAndGlobalShapes()
        {
            var disabled = new MmtacModel(Cfg(useConf: false), new Random(42));
            Assert(disabled.ConfidenceProjection == null, "ConfidenceProjection should be null when disabled");
            Assert(disabled.ConfidenceBias == null, "ConfidenceBias should be null when disabled");

            var enabled = new MmtacModel(Cfg(embDim: 16, useConf: true, globalDim: 4), new Random(42));
            Assert(enabled.ConfidenceProjection != null, "ConfidenceProjection should exist when enabled");
            Assert(enabled.ConfidenceProjection.GetLength(0) == 1 && enabled.ConfidenceProjection.GetLength(1) == 16, "ConfidenceProjection shape");
            Assert(enabled.GlobalFeatureProjection != null, "GlobalFeatureProjection should exist when globalDim > 0");
            Assert(enabled.GlobalFeatureProjection.GetLength(0) == 16 && enabled.GlobalFeatureProjection.GetLength(1) == 4, "GlobalFeatureProjection shape");
        }

        private void Test_Construction_RuntimePruningConfigCopied()
        {
            var cfg = Cfg();
            cfg.Pruning.AttentionScoreAlpha = 0.37f;
            cfg.Pruning.MinQueryCountForPruning = 9;
            cfg.Pruning.NewEntryReserveFraction = 0.23f;
            cfg.Pruning.UseAttentionBasedPruning = false;

            var m = new MmtacModel(cfg, new Random(42));

            Assert(MathF.Abs(m.PruningConfig.AttentionScoreAlpha - cfg.Pruning.AttentionScoreAlpha) < 1e-6f, "AttentionScoreAlpha was not copied");
            Assert(m.PruningConfig.MinQueryCountForPruning == cfg.Pruning.MinQueryCountForPruning, "MinQueryCountForPruning was not copied");
            Assert(MathF.Abs(m.PruningConfig.NewEntryReserveFraction - cfg.Pruning.NewEntryReserveFraction) < 1e-6f, "NewEntryReserveFraction was not copied");
            Assert(m.PruningConfig.UseAttentionBasedPruning == cfg.Pruning.UseAttentionBasedPruning, "UseAttentionBasedPruning was not copied");
        }

        // ---------------------------------------------------------------------
        // Forward contracts
        // ---------------------------------------------------------------------

        private void Test_Forward_PredictNextFiniteAndConstrained_ConfidenceDisabled()
        {
            var m = new MmtacModel(Cfg(useConf: false), new Random(42));
            var p = m.PredictNext(MakeInput(seqLen: 8));
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
            Assert(p.Confidence == 1f, "Confidence should be 1 when confidence head is disabled");
        }

        private void Test_Forward_PredictNextFiniteAndConstrained_ConfidenceEnabled()
        {
            var m = new MmtacModel(Cfg(useConf: true), new Random(42));
            var p = m.PredictNext(MakeInput(seqLen: 8));
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: true);
        }

        private void Test_Forward_NullAndEmptyStoriesWork()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            AssertPredictionFiniteAndConstrained(m.PredictNext(MakeInput(stories: null)), expectConfidenceHead: false);
            AssertPredictionFiniteAndConstrained(m.PredictNext(MakeInput(stories: Array.Empty<NewsStory>())), expectConfidenceHead: false);
        }

        private void Test_Forward_SeqLenOneWorks()
        {
            var m = new MmtacModel(Cfg(priceSeqLen: 4), new Random(42));
            var p = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = RandomMatrix(1, 5, new Random(1), 0.1f) });
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
        }

        private void Test_Forward_AllHeadShapesMatchSequenceLength()
        {
            int seqLen = 9;
            var m = new MmtacModel(Cfg(useConf: true), new Random(42));
            var (reg, range, quality, dir, midDir, conf) = m.Forward(MakeInput(seqLen: seqLen));

            AssertMatrixShape(reg, seqLen, 3, nameof(reg));
            AssertMatrixShape(range, seqLen, 1, nameof(range));
            AssertMatrixShape(quality, seqLen, 1, nameof(quality));
            AssertMatrixShape(dir, seqLen, 1, nameof(dir));
            AssertMatrixShape(midDir, seqLen, 1, nameof(midDir));
            AssertMatrixShape(conf, seqLen, 1, nameof(conf));
            AssertAllFinite(reg, "regression");
            AssertAllFinite(range, "range");
            AssertAllFinite(quality, "quality");
            AssertAllFinite(dir, "direction");
            AssertAllFinite(midDir, "midDirection");
            AssertAllFinite(conf, "confidence");

            var pred = m.PredictNext(MakeInput(seqLen: seqLen));
            var (reg2, range2, quality2, dir2, mid2, conf2) = m.Forward(MakeInput(seqLen: seqLen));
            int last = seqLen - 1;
            Assert(MathF.Abs(pred.High - reg2[last, 0]) < 1e-6f, "PredictNext.High must equal last regression timestep");
            Assert(MathF.Abs(pred.Low - reg2[last, 1]) < 1e-6f, "PredictNext.Low must equal last regression timestep");
            Assert(MathF.Abs(pred.Close - reg2[last, 2]) < 1e-6f, "PredictNext.Close must equal last regression timestep");
            Assert(MathF.Abs(pred.Range - range2[last, 0]) < 1e-6f, "PredictNext.Range must equal last range timestep");
            Assert(MathF.Abs(pred.Quality - quality2[last, 0]) < 1e-6f, "PredictNext.Quality must equal last quality timestep");
            Assert(MathF.Abs(pred.DirectionProb - dir2[last, 0]) < 1e-6f, "PredictNext.Direction must equal last direction timestep");
            Assert(MathF.Abs(pred.MidWindowDirectionProb - mid2[last, 0]) < 1e-6f, "PredictNext.MidDirection must equal last mid-direction timestep");
            Assert(MathF.Abs(pred.Confidence - conf2[last, 0]) < 1e-6f, "PredictNext.Confidence must equal last confidence timestep");
        }

        private void Test_Forward_DeterministicAndDoesNotMutateMemory()
        {
            var (tok, inputs, _) = RandomData(n: 1, withNews: true, globalDim: 3);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, useConf: true, globalDim: 3), new Random(42));

            int newsBefore = m.NewsMemory.Count;
            int priceBefore = m.PriceMemory.Count;
            double lastBefore = m.LastPriceTimestamp;

            var p1 = m.PredictNext(inputs[0]);
            var p2 = m.PredictNext(inputs[0]);

            AssertPredictionsClose(p1, p2, 0f, "PredictNext must be exactly deterministic for repeated calls");
            Assert(m.NewsMemory.Count == newsBefore, "PredictNext mutated NewsMemory");
            Assert(m.PriceMemory.Count == priceBefore, "PredictNext mutated PriceMemory");
            Assert(m.LastPriceTimestamp == lastBefore, "PredictNext mutated LastPriceTimestamp");

            var f1 = m.Forward(inputs[0]);
            var f2 = m.Forward(inputs[0]);
            AssertMatricesClose(f1.regression, f2.regression, 0f, "Forward regression deterministic");
            AssertMatricesClose(f1.range, f2.range, 0f, "Forward range deterministic");
            AssertMatricesClose(f1.quality, f2.quality, 0f, "Forward quality deterministic");
            AssertMatricesClose(f1.direction, f2.direction, 0f, "Forward direction deterministic");
            AssertMatricesClose(f1.midDirection, f2.midDirection, 0f, "Forward midDirection deterministic");
            AssertMatricesClose(f1.confidence, f2.confidence, 0f, "Forward confidence deterministic");
        }

        private void Test_Forward_StoryAndGlobalChangePrediction()
        {
            var corpus = new[] { "bullish demand growth", "bearish credit stress" };
            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 64, minFrequency: 1);

            var cfg = Cfg(tok.VocabSize + 2, globalDim: 3);
            var m = new MmtacModel(cfg, new Random(42));
            var ps = ConstantPriceSequence(8, 0.1f, 0.2f, -0.1f, 0.05f, 1f);

            var basePred = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps });
            var storyPred = m.PredictNext(new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ps,
                NewsStories = new[] { new NewsStory(tok.Encode(corpus[0], addSpecialTokens: true), 0f) }
            });
            var globalPred = m.PredictNext(new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ps,
                GlobalFeatures = new[] { 1f, 0f, 0.5f }
            });

            Assert(PredictionsDiffer(basePred, storyPred, 1e-7f), "Adding a story should change at least one prediction field");
            Assert(PredictionsDiffer(basePred, globalPred, 1e-7f), "Adding global features should change at least one prediction field");
        }

        private void Test_Forward_WithCacheMatchesForward_AllHeads()
        {
            var corpus = new[] { "policy support growth demand" };
            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 32, minFrequency: 1);

            var cfg = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, useConf: true, globalDim: 3);
            var m = new MmtacModel(cfg, new Random(42));

            var input = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandomMatrix(7, 5, new Random(11), 0.3f),
                NewsStories = new[] { new NewsStory(tok.Encode(corpus[0], addSpecialTokens: true), -2f) },
                GlobalFeatures = new[] { 0.2f, 0.7f, 0.1f }
            };

            var expected = m.Forward(input);
            var cache = new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers);
            var actual = InvokeForwardWithCache(m, input, cache, isTraining: false);

            AssertMatricesClose(expected.regression, actual.regression, 1e-5f, "regression mismatch");
            AssertMatricesClose(expected.range, actual.range, 1e-5f, "range mismatch");
            AssertMatricesClose(expected.quality, actual.quality, 1e-5f, "quality mismatch");
            AssertMatricesClose(expected.direction, actual.direction, 1e-5f, "direction mismatch");
            AssertMatricesClose(expected.midDirection, actual.midDirection, 1e-5f, "midDirection mismatch");
            AssertMatricesClose(expected.confidence, actual.confidence, 1e-5f, "confidence mismatch");
        }

        private void Test_Forward_Causal_NoFutureLeakage_PriceOnly()
        {
            var m = new MmtacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 16, useConf: true), new Random(42));
            AssertNoFutureLeakage(m, stories: null, globals: null);
        }

        private void Test_Forward_Causal_NoFutureLeakage_WithNewsAndGlobal()
        {
            var stories = new[] { new NewsStory(new[] { 1, 2, 3, 4 }, 0f), new NewsStory(new[] { 5, 6, 7 }, -2f) };
            var globals = new[] { 0.25f, 0.75f, 0.5f };
            var m = new MmtacModel(Cfg(vocabSize: 64, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 16, useConf: true, globalDim: 3), new Random(42));
            AssertNoFutureLeakage(m, stories, globals);
        }

        private void Test_Forward_WrongPriceFeatureDimThrowsArgumentException()
        {
            var m = new MmtacModel(Cfg(priceFeatures: 5), new Random(42));
            var bad = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = new float[8, 6] };
            AssertThrows<ArgumentException>(() => m.PredictNext(bad), "wrong price feature dimension should throw ArgumentException");
        }

        private void Test_Forward_WrongGlobalFeatureDimThrowsArgumentException()
        {
            var m = new MmtacModel(Cfg(globalDim: 3), new Random(42));
            var bad = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                GlobalFeatures = new[] { 1f, 0f }
            };
            AssertThrows<ArgumentException>(() => m.PredictNext(bad), "short global feature vector should throw ArgumentException");
        }

        private void Test_Forward_GlobalFeatureExtraDimThrowsArgumentException()
        {
            var m = new MmtacModel(Cfg(globalDim: 3), new Random(42));
            var bad = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                GlobalFeatures = new[] { 1f, 0f, 0.5f, 99f }
            };
            AssertThrows<ArgumentException>(() => m.PredictNext(bad), "extra global features should throw ArgumentException, not be silently ignored");
        }

        // ---------------------------------------------------------------------
        // Decay / context wiring
        // ---------------------------------------------------------------------

        private void Test_Decay_TimeValueReachesDecoder()
        {
            var cfg = Cfg(
                vocabSize: 64,
                embDim: 16,
                numHeads: 2,
                numLayers: 1,
                ffnDim: 32,
                decayEnabled: true);

            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var m = new MmtacModel(cfg, new Random(42));

            var ps = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);

            var ctx = new float[2, cfg.Price.EmbeddingDim];

            for (int d = 0; d < cfg.Price.EmbeddingDim; d++)
            {
                ctx[0, d] = 0.10f * (d + 1);
                ctx[1, d] = -0.07f * (d + 1);
            }

            // Same two keys, different relative recency.
            var hFirstRecent = m.ForwardPriceDecoderWithCache(
                ps,
                0,
                ps.GetLength(0),
                ctx,
                new[] { 0f, -1000f },
                new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers),
                isTraining: false);

            var hSecondRecent = m.ForwardPriceDecoderWithCache(
                ps,
                0,
                ps.GetLength(0),
                ctx,
                new[] { -1000f, 0f },
                new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers),
                isTraining: false);

            Assert(
                MaxAbsDiff(hFirstRecent, hSecondRecent) > 1e-7f,
                "Changing relative context times should affect decoder hidden state when decay is enabled.");
        }

        private void Test_Decay_ContextContentReachesDecoder()
        {
            var cfg = Cfg(vocabSize: 64, embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32);
            var m = new MmtacModel(cfg, new Random(42));
            var ps = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);
            var ctxA = new float[1, cfg.Price.EmbeddingDim];
            var ctxB = new float[1, cfg.Price.EmbeddingDim];
            for (int d = 0; d < cfg.Price.EmbeddingDim; d++) { ctxA[0, d] = 0.1f * (d + 1); ctxB[0, d] = -ctxA[0, d]; }

            var hA = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctxA, new[] { -2f }, new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers), isTraining: false);
            var hB = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctxB, new[] { -2f }, new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers), isTraining: false);

            Assert(MaxAbsDiff(hA, hB) > 1e-7f, "Changing context content should affect decoder hidden state");
        }

        private void Test_Decay_DisabledIgnoresContextTimes()
        {
            var cfg = Cfg(vocabSize: 64, embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, decayEnabled: false);
            var m = new MmtacModel(cfg, new Random(42));
            var ps = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);
            var ctx = new float[1, cfg.Price.EmbeddingDim];
            for (int d = 0; d < cfg.Price.EmbeddingDim; d++) ctx[0, d] = 0.1f * (d + 1);

            var h1 = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctx, new[] { 0f }, new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers), isTraining: false);
            var h2 = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctx, new[] { -1000f }, new MmtacForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers), isTraining: false);

            Assert(MaxAbsDiff(h1, h2) < 1e-6f, "When Decay.Enabled=false, context times should not change the output");
        }

        // ---------------------------------------------------------------------
        // Tokenizer
        // ---------------------------------------------------------------------

        private void Test_Tokenizer_SetTokenizeAndOversizedThrows()
        {
            var tok = new BPETokenizer();
            tok.Train(new[] { "stock rose sharply", "market crashed today" }, vocabSize: 100, minFrequency: 1);

            var m = new MmtacModel(Cfg(vocabSize: tok.VocabSize + 5), new Random(42));
            m.SetTokenizer(tok);
            var stories = m.TokenizeStories(new[] { "stock rose sharply", "market crashed today" }, new[] { 0f, -1f });

            Assert(stories.Length == 2, "story count");
            Assert(stories.All(s => s.TokenIds != null && s.TokenIds.Length > 0), "empty tokenization");
            Assert(MathF.Abs(stories[1].ArrivalTime + 1f) < 1e-6f, "arrival time not preserved");
            AssertThrows<ArgumentException>(() => m.TokenizeStories(new[] { "one" }, new[] { 0f, 1f }), "texts/arrivalTimes length mismatch should throw");

            var small = new MmtacModel(Cfg(vocabSize: 2), new Random(42));
            AssertThrows<ArgumentException>(() => small.SetTokenizer(tok), "oversized tokenizer should throw");
            AssertThrows<ArgumentNullException>(() => small.SetTokenizer(null), "null tokenizer should throw");
        }

        private void Test_Tokenizer_SaveLoadRoundTrip()
        {
            var corpus = new[] { "central bank support growth demand", "credit stress recession slowdown", "routine market update commentary" };
            var m = new MmtacModel(Cfg(vocabSize: 80), new Random(42));
            m.TrainTokenizer(corpus, minFrequency: 1);

            var texts = new[] { corpus[0], corpus[1], corpus[2] };
            var times = new[] { 0f, -1f, -2f };
            var before = m.TokenizeStories(texts, times);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var loaded = MmtacModel.Load(dir);
                Assert(loaded.Tokenizer != null, "Tokenizer should load");
                var after = loaded.TokenizeStories(texts, times);
                Assert(before.Length == after.Length, "tokenized story count mismatch");

                for (int i = 0; i < before.Length; i++)
                {
                    Assert(before[i].ArrivalTime == after[i].ArrivalTime, $"arrival mismatch {i}");
                    Assert(before[i].TokenIds.SequenceEqual(after[i].TokenIds), $"token mismatch {i}");
                }
            }
            finally { DeleteDir(dir); }
        }

        // ---------------------------------------------------------------------
        // Validation / argument contracts
        // ---------------------------------------------------------------------

        private void Test_PublicMethods_LengthMismatchThrow()
        {
            var (tok, inputs, targets) = RandomData(n: 3, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));
            var ts = new[] { 10d, 20d, 30d };

            AssertThrows<ArgumentException>(() => tr.Train(inputs, targets.Take(2).ToArray()), "Train should throw for mismatched inputs/targets lengths");
            AssertThrows<ArgumentException>(() => tr.Validate(inputs, targets.Take(2).ToArray()), "Validate should throw for mismatched inputs/targets lengths");
            AssertThrows<ArgumentException>(() => tr.ValidateAligned(inputs, targets.Take(2).ToArray()), "ValidateAligned should throw for mismatched inputs/targets lengths");
            AssertThrows<ArgumentException>(() => tr.TrainSequential(inputs, targets.Take(2).ToArray(), ts), "TrainSequential should throw for mismatched inputs/targets lengths");
            AssertThrows<ArgumentException>(() => tr.TrainSequential(inputs, targets, ts.Take(2).ToArray()), "TrainSequential should throw for mismatched timestamps length");
            AssertThrows<ArgumentException>(() => tr.ValidateSequential(inputs, targets, ts.Take(2).ToArray()), "ValidateSequential should throw for mismatched timestamps length");
            AssertThrows<ArgumentException>(() => tr.ValidateSequentialAligned(inputs, targets, ts.Take(2).ToArray()), "ValidateSequentialAligned should throw for mismatched timestamps length");
        }

        private void Test_PublicMethods_ZeroTimeUnitsThrow()
        {
            var (tok, inputs, targets) = RandomData(n: 2, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));
            var ts = new[] { 10d, 20d };

            AssertThrows<ArgumentOutOfRangeException>(() => m.PredictWithMemory(inputs[0], 10.0, timeUnitsPerPosition: 0.0), "PredictWithMemory should reject zero timeUnitsPerPosition");
            AssertThrows<ArgumentOutOfRangeException>(() => tr.TrainSequential(inputs, targets, ts, timeUnitsPerPosition: 0.0), "TrainSequential should reject zero timeUnitsPerPosition");
            AssertThrows<ArgumentOutOfRangeException>(() => tr.ValidateSequential(inputs, targets, ts, timeUnitsPerPosition: 0.0), "ValidateSequential should reject zero timeUnitsPerPosition");
            AssertThrows<ArgumentOutOfRangeException>(() => tr.ValidateSequentialAligned(inputs, targets, ts, timeUnitsPerPosition: 0.0), "ValidateSequentialAligned should reject zero timeUnitsPerPosition");
        }

        private void Test_Validate_ReturnsFiniteNonNegative()
        {
            var (tok, inputs, targets) = RandomData(n: 5, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            float legacy = tr.Validate(inputs, targets);
            float aligned = tr.ValidateAligned(inputs, targets);
            float seq = tr.ValidateSequential(inputs, targets, Enumerable.Range(0, inputs.Length).Select(i => (double)i).ToArray());
            float seqAligned = tr.ValidateSequentialAligned(inputs, targets, Enumerable.Range(0, inputs.Length).Select(i => (double)i).ToArray());

            Assert(float.IsFinite(legacy) && legacy >= 0f, $"Validate invalid: {legacy}");
            Assert(float.IsFinite(aligned) && aligned >= 0f, $"ValidateAligned invalid: {aligned}");
            Assert(float.IsFinite(seq) && seq >= 0f, $"ValidateSequential invalid: {seq}");
            Assert(float.IsFinite(seqAligned) && seqAligned >= 0f, $"ValidateSequentialAligned invalid: {seqAligned}");
        }

        private void Test_Validate_DoesNotMutateWeights()
        {
            var (tok, inputs, targets) = RandomData(n: 4, seqLen: 6, withNews: true, globalDim: 3);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 3, useConf: true), new Random(42));
            var snap = SnapshotWeights(m);

            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(float.IsFinite(loss), $"non-finite validation loss {loss}");
            AssertWeightsUnchanged(snap, m, "Validate");
        }

        private void Test_ValidateSequential_DoesNotMutateWeights()
        {
            var (tok, inputs, targets) = RandomData(n: 4, seqLen: 6, withNews: true, globalDim: 3);
            var ts = new[] { 30d, 10d, 40d, 20d };
            var cfg = Cfg(tok.VocabSize + 2, globalDim: 3, useConf: true);
            cfg.Runtime.AccelerationType = AccelerationType.CPU;
            var m = new MmtacModel(cfg, new Random(42));
            var snap = SnapshotWeights(m);

            float loss = new MmtacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);
            Assert(float.IsFinite(loss), $"non-finite validation loss {loss}");
            AssertWeightsUnchanged(snap, m, "ValidateSequential");
        }

        private void Test_ValidateSequential_OrderInvariant()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: true, globalDim: 3);
            var ts = new[] { 500d, 100d, 600d, 200d, 700d, 300d };
            int[] perm = { 2, 5, 1, 4, 0, 3 };

            var cfg = Cfg(tok.VocabSize + 2, globalDim: 3);
            var m = new MmtacModel(cfg, new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            float v1 = tr.ValidateSequential(inputs, targets, ts);
            float v2 = tr.ValidateSequential(perm.Select(i => inputs[i]).ToArray(), perm.Select(i => targets[i]).ToArray(), perm.Select(i => ts[i]).ToArray());

            Assert(MathF.Abs(v1 - v2) < 1e-6f, $"ValidateSequential should sort by timestamp. v1={v1:F6}, v2={v2:F6}");
        }

        private void Test_ValidateSequential_MatchesReferenceLoop()
        {
            var (inputs, targets, ts) = SequentialReferenceData();
            var cfg = Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48);
            var m = new MmtacModel(cfg, new Random(42));
            var tr = new MmtacTrainer(m, TC(lr: 0.002f, bs: 1, epochs: 8));
            tr.TrainSequential(inputs, targets, ts);

            float actual = tr.ValidateSequential(inputs, targets, ts);
            float expected = ValidateSequentialReference(tr, m, inputs, targets, ts);
            Assert(MathF.Abs(actual - expected) < 1e-4f, $"ValidateSequential mismatch. actual={actual:F6}, expected={expected:F6}");
        }

        private void Test_ValidateSequential_ClearsMemoryOnExit()
        {
            var (tok, inputs, targets) = RandomData(n: 3, seqLen: 6, withNews: true);
            var ts = new[] { 100d, 200d, 300d };
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PredictWithMemory(inputs[0], 50.0);
            Assert(m.NewsMemory.Count > 0 || m.PriceMemory.Count > 0, "precondition failed: memory did not populate");

            float loss = new MmtacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);
            Assert(float.IsFinite(loss), "ValidateSequential returned non-finite loss");
            Assert(m.NewsMemory.Count == 0, "ValidateSequential should clear NewsMemory on exit");
            Assert(m.PriceMemory.Count == 0, "ValidateSequential should clear PriceMemory on exit");
        }

        // ---------------------------------------------------------------------
        // Training mechanics
        // ---------------------------------------------------------------------

        private void Test_Train_LossDecreases_OnDeterministicPriceData()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = PriceHeldOutData(trainN: 48, testN: 16, seqLen: 6);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            float before = tr.ValidateAligned(testInputs, testTargets);
            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 60)).Train(trainInputs, trainTargets);
            float after = new MmtacTrainer(m, TC(epochs: 1)).ValidateAligned(testInputs, testTargets);

            Assert(after < before, $"loss did not decrease. before={before:F6}, after={after:F6}");
        }

        private void Test_Train_SingleSampleOverfit_LossDecreases()
        {
            var (tok, inputs, targets) = RandomData(n: 1, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            float before = new MmtacTrainer(m, TC(epochs: 1)).ValidateAligned(inputs, targets);
            new MmtacTrainer(m, TC(lr: 0.004f, bs: 1, epochs: 120)).Train(inputs, targets);
            float after = new MmtacTrainer(m, TC(epochs: 1)).ValidateAligned(inputs, targets);
            Assert(after < before, $"single-sample loss did not decrease. before={before:F6}, after={after:F6}");
        }

        private void Test_Train_MixedNullNewsFinite()
        {
            var (tok, inputs, targets) = RandomData(n: 8, seqLen: 6, withNews: true);
            for (int i = 0; i < inputs.Length; i += 2)
                inputs[i] = CloneInput(inputs[i], keepNews: false);

            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.002f, bs: 4, epochs: 8)).Train(inputs, targets);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(float.IsFinite(loss) && loss >= 0f, $"invalid loss after mixed-null-news training: {loss}");
        }

        private void Test_Train_ParameterUpdateCoverage()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));

            var priceProj = Copy(m.PriceInputProjection);
            var selfWq = Copy(m.PriceBlocks[0].SelfAttention.WQ);
            var crossWk = Copy(m.PriceBlocks[0].CrossAttention.WK);
            var reg = Copy(m.RegressionProjection);
            var range = Copy(m.RangeProjection);
            var quality = Copy(m.QualityProjection);
            var dir = Copy(m.DirectionProjection);
            var mid = Copy(m.MidDirectionProjection);
            var context = Copy(m.ContextTypeEmbedding);

            new MmtacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);

            Assert(Changed(priceProj, m.PriceInputProjection), "PriceInputProjection did not update");
            Assert(Changed(selfWq, m.PriceBlocks[0].SelfAttention.WQ), "SelfAttention.WQ did not update");
            Assert(Changed(crossWk, m.PriceBlocks[0].CrossAttention.WK), "CrossAttention.WK did not update");
            Assert(Changed(reg, m.RegressionProjection), "RegressionProjection did not update");
            Assert(Changed(range, m.RangeProjection), "RangeProjection did not update");
            Assert(Changed(quality, m.QualityProjection), "QualityProjection did not update");
            Assert(Changed(dir, m.DirectionProjection), "DirectionProjection did not update");
            Assert(Changed(mid, m.MidDirectionProjection), "MidDirectionProjection did not update");
            Assert(Changed(context, m.ContextTypeEmbedding), "ContextTypeEmbedding did not update");
        }

        private void Test_Train_FrozenTextUnchanged_UnfrozenTextChanges()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: true);

            var frozen = new MmtacModel(Cfg(tok.VocabSize + 2, freezeText: true), new Random(42));
            var frozenBefore = Copy(frozen.TextTokenEmbedding);
            new MmtacTrainer(frozen, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);
            Assert(!Changed(frozenBefore, frozen.TextTokenEmbedding, 1e-8f), "frozen TextTokenEmbedding changed");

            var unfrozen = new MmtacModel(Cfg(tok.VocabSize + 2, freezeText: false), new Random(42));
            var unfrozenBefore = Copy(unfrozen.TextTokenEmbedding);
            new MmtacTrainer(unfrozen, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);
            Assert(Changed(unfrozenBefore, unfrozen.TextTokenEmbedding), "unfrozen TextTokenEmbedding did not change");
        }

        private void Test_Train_GlobalProjectionUpdated()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: true, globalDim: 4);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 4), new Random(42));
            var before = Copy(m.GlobalFeatureProjection);
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);
            Assert(Changed(before, m.GlobalFeatureProjection), "GlobalFeatureProjection did not update");
        }

        private void Test_Train_ConfidenceHeadUpdated()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, useConf: true), new Random(42));
            var before = Copy(m.ConfidenceProjection);
            new MmtacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);
            Assert(Changed(before, m.ConfidenceProjection), "ConfidenceProjection did not update");
        }

        private void Test_Train_LossWeightsControlDirectionHeadGradient()
        {
            var (inputs, targets) = DirectionSignalData(n: 12, seqLen: 6);

            var zeroCfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32);
            zeroCfg.Output.DirectionLossWeight = 0f;
            var zero = new MmtacModel(zeroCfg, new Random(42));
            var zeroBefore = Copy(zero.DirectionProjection);
            new MmtacTrainer(zero, TC(lr: 0.01f, bs: 6, epochs: 3)).Train(inputs, targets);
            Assert(!Changed(zeroBefore, zero.DirectionProjection, 1e-8f), "DirectionProjection changed even though DirectionLossWeight=0");

            var activeCfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32);
            activeCfg.Output.DirectionLossWeight = 2f;
            var active = new MmtacModel(activeCfg, new Random(42));
            var activeBefore = Copy(active.DirectionProjection);
            new MmtacTrainer(active, TC(lr: 0.01f, bs: 6, epochs: 3)).Train(inputs, targets);
            Assert(Changed(activeBefore, active.DirectionProjection), "DirectionProjection did not change when DirectionLossWeight>0");
        }

        private void Test_Train_ConfidenceLossWeightZeroDisablesConfidenceGradient()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: false);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, useConf: true), new Random(42));
            var before = Copy(m.ConfidenceProjection);

            var tc = TC(lr: 0.01f, bs: 3, epochs: 3);
            tc.ConfidenceLossWeight = 0f;
            new MmtacTrainer(m, tc).Train(inputs, targets);

            Assert(!Changed(before, m.ConfidenceProjection, 1e-8f), "ConfidenceProjection changed even though ConfidenceLossWeight=0");
        }

        private void Test_Train_GradientClippingBoundsUpdate()
        {
            var (tok, inputs, targets) = RandomData(n: 3, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var before = Copy(m.PriceInputProjection);

            new MmtacTrainer(m, new TrainingConfig
            {
                LearningRate = 10f,
                BatchSize = 3,
                Epochs = 1,
                UseGradientClipping = true,
                GradientClipThreshold = 0.01f,
                Verbose = false
            }).Train(inputs, targets);

            Assert(MaxAbsDiff(before, m.PriceInputProjection) < 1.0f, "update too large despite tight clipping");
        }

        private void Test_Train_HighLearningRateWithClippingStaysFinite()
        {
            var (tok, inputs, targets) = RandomData(n: 5, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.5f, bs: 5, epochs: 4, clip: true)).Train(inputs, targets);
            AssertPredictionFiniteAndConstrained(m.PredictNext(inputs[0]), expectConfidenceHead: false);
        }

        private void Test_Train_PriceContextPathRunsFinite()
        {
            var (tok, inputs, targets) = RandomData(n: 5, seqLen: 10, withNews: true);
            var cfg = Cfg(tok.VocabSize + 2, priceSeqLen: 12, priceContextEnabled: true);
            var m = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(m, TC(lr: 0.001f, bs: 5, epochs: 4)).Train(inputs, targets);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(float.IsFinite(loss) && loss >= 0f, $"invalid loss after price-context training: {loss}");
        }

        private void Test_Train_DecayNetworkUpdatedWithContext()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var before = Copy3D(m.PriceBlocks[0].DecayNetwork.QueryProjection);

            new MmtacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);

            Assert(Changed(before, m.PriceBlocks[0].DecayNetwork.QueryProjection), "DecayNetwork.QueryProjection did not update");
        }

        private void Test_Train_ZeroLearningRateDoesNotChangeWeights()
        {
            var (tok, inputs, targets) = RandomData(n: 4, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var snap = SnapshotWeights(m);
            new MmtacTrainer(m, TC(lr: 0f, bs: 2, epochs: 2)).Train(inputs, targets);
            AssertWeightsUnchanged(snap, m, "Train lr=0");
        }

        // ---------------------------------------------------------------------
        // Semantic smoke tests
        // ---------------------------------------------------------------------

        private void Test_Signal_PriceHeldOutLossImproves()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = PriceHeldOutData(trainN: 64, testN: 20, seqLen: 6);
            var cfg = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48);
            var m = new MmtacModel(cfg, new Random(42));

            float before = new MmtacTrainer(m, TC(epochs: 1)).ValidateAligned(testInputs, testTargets);
            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 80)).Train(trainInputs, trainTargets);
            float after = new MmtacTrainer(m, TC(epochs: 1)).ValidateAligned(testInputs, testTargets);

            Assert(after < before * 0.75f, $"held-out price loss did not improve enough. before={before:F6}, after={after:F6}");
        }

        private void Test_Signal_NewsSeparationAfterTraining()
        {
            var (tok, inputs, targets) = NewsSignalData(n: 48, seqLen: 6);
            var cfg = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48);
            var m = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 80)).Train(inputs, targets);

            var bull = tok.Encode("strong bullish outlook demand growth", addSpecialTokens: true);
            var bear = tok.Encode("weak bearish outlook contraction risk", addSpecialTokens: true);
            var ps = ConstantPriceSequence(6, 0f, 0f, 0f, 0f, 1f);

            var bullPred = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, NewsStories = new[] { new NewsStory(bull, 0f) } });
            var bearPred = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, NewsStories = new[] { new NewsStory(bear, 0f) } });

            Assert(bullPred.Close > bearPred.Close + 0.05f, $"bull close should exceed bear close. bull={bullPred.Close:F6}, bear={bearPred.Close:F6}");
            Assert(bullPred.DirectionProb > bearPred.DirectionProb + 0.05f, $"bull dir should exceed bear dir. bull={bullPred.DirectionProb:F6}, bear={bearPred.DirectionProb:F6}");
        }

        private void Test_Signal_GlobalSeparationAfterTraining()
        {
            var (tok, inputs, targets) = GlobalSignalData(n: 48, seqLen: 6);
            var cfg = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3);
            var m = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 80)).Train(inputs, targets);

            var ps = ConstantPriceSequence(6, 0f, 0f, 0f, 0f, 1f);
            var on = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, GlobalFeatures = new[] { 1f, 0f, 1f } });
            var off = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, GlobalFeatures = new[] { 0f, 1f, 0f } });

            Assert(on.Close > off.Close + 0.05f, $"risk-on close should exceed risk-off close. on={on.Close:F6}, off={off.Close:F6}");
            Assert(on.DirectionProb > off.DirectionProb + 0.05f, $"risk-on dir should exceed risk-off dir. on={on.DirectionProb:F6}, off={off.DirectionProb:F6}");
        }

        private void Test_Signal_DirectionAndMidDirectionCanLearnOppositeLabels()
        {
            var (inputs, targets) = OppositeDirectionData(n: 24, seqLen: 6);
            var m = new MmtacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.004f, bs: 8, epochs: 100)).Train(inputs, targets);

            int dirCorrect = 0, midCorrect = 0, total = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                bool dirUp = i < inputs.Length / 2;
                var (_, _, _, dir, mid, _) = m.Forward(inputs[i]);
                for (int t = 0; t < dir.GetLength(0); t++)
                {
                    if ((dir[t, 0] > 0.5f) == dirUp) dirCorrect++;
                    if ((mid[t, 0] > 0.5f) != dirUp) midCorrect++;
                    total++;
                }
            }

            float dirAcc = (float)dirCorrect / total;
            float midAcc = (float)midCorrect / total;
            Assert(dirAcc > 0.60f, $"Direction accuracy too low: {dirAcc:P0}");
            Assert(midAcc > 0.60f, $"MidDirection accuracy too low: {midAcc:P0}");
        }

        private void Test_Signal_RangeHeadLearnsRangeTarget()
        {
            var (inputs, targets) = RangeSignalData(n: 24, seqLen: 6);
            var m = new MmtacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            float before = MeanRangeError(m, inputs, targets);
            new MmtacTrainer(m, TC(lr: 0.004f, bs: 8, epochs: 100)).Train(inputs, targets);
            float after = MeanRangeError(m, inputs, targets);
            Assert(after < before * 0.65f, $"Range error did not improve. before={before:F6}, after={after:F6}");
        }

        // ---------------------------------------------------------------------
        // Memory
        // ---------------------------------------------------------------------

        private void Test_Memory_InitiallyEmptyAndClearMethods()
        {
            var (tok, inputs, _) = RandomData(n: 1, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            Assert(m.NewsMemory.Count == 0, "NewsMemory should start empty");
            Assert(m.PriceMemory.Count == 0, "PriceMemory should start empty");

            m.PredictWithMemory(inputs[0], 100.0);
            int priceCount = m.PriceMemory.Count;
            int newsCount = m.NewsMemory.Count;
            Assert(priceCount > 0 && newsCount > 0, "memory did not populate");

            m.ClearNewsMemory();
            Assert(m.NewsMemory.Count == 0, "ClearNewsMemory failed");
            Assert(m.PriceMemory.Count == priceCount, "ClearNewsMemory should not clear price memory");

            m.ClearPriceMemory();
            Assert(m.PriceMemory.Count == 0, "ClearPriceMemory failed");

            m.PredictWithMemory(inputs[0], 200.0);
            m.ClearAllMemory();
            Assert(m.NewsMemory.Count == 0 && m.PriceMemory.Count == 0, "ClearAllMemory failed");
        }

        private void Test_Memory_PredictWithMemoryExactGrowthAndTimestamps()
        {
            var m = new MmtacModel(Cfg(vocabSize: 64, priceSeqLen: 8), new Random(42));
            var input = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(5, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                NewsStories = new[] { new NewsStory(new[] { 1, 2, 3 }, 0f), new NewsStory(new[] { 4, 5, 6 }, -1f) }
            };

            m.PredictWithMemory(input, currentAbsoluteTimestamp: 100.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);

            Assert(m.NewsMemory.Count == 2, $"expected 2 news entries, got {m.NewsMemory.Count}");
            Assert(m.PriceMemory.Count == 5, $"expected 5 price entries, got {m.PriceMemory.Count}");
            Assert(Math.Abs(m.NewsMemory[0].AbsoluteTimestamp - 100.0) < 1e-9, "news timestamp 0");
            Assert(Math.Abs(m.NewsMemory[1].AbsoluteTimestamp - 99.0) < 1e-9, "news timestamp 1");
            for (int i = 0; i < 5; i++)
                Assert(Math.Abs(m.PriceMemory[i].AbsoluteTimestamp - (100.0 + i)) < 1e-9, $"price timestamp {i}");
            Assert(Math.Abs(m.LastPriceTimestamp - 104.0) < 1e-9, "LastPriceTimestamp");
        }

        private void Test_Memory_CustomTimeUnitsScaleTimestamps()
        {
            var m = new MmtacModel(Cfg(vocabSize: 64, priceSeqLen: 8), new Random(42));
            var input = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(4, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                NewsStories = new[] { new NewsStory(new[] { 1, 2, 3 }, -2f) }
            };

            m.PredictWithMemory(input, currentAbsoluteTimestamp: 50.0, timeUnitsPerPosition: 0.25, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);
            Assert(Math.Abs(m.NewsMemory[0].AbsoluteTimestamp - 49.5) < 1e-9, "scaled news timestamp");
            for (int i = 0; i < 4; i++)
                Assert(Math.Abs(m.PriceMemory[i].AbsoluteTimestamp - (50.0 + 0.25 * i)) < 1e-9, $"scaled price timestamp {i}");
            Assert(Math.Abs(m.LastPriceTimestamp - 50.75) < 1e-9, "scaled LastPriceTimestamp");
        }

        private void Test_Memory_ZeroBudgetsKeepBanksEmpty()
        {
            var (tok, inputs, _) = RandomData(n: 1, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var p = m.PredictWithMemory(inputs[0], 100.0, maxNewsMemorySize: 0, maxPriceMemorySize: 0);
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
            Assert(m.NewsMemory.Count == 0, "NewsMemory should stay empty with budget 0");
            Assert(m.PriceMemory.Count == 0, "PriceMemory should stay empty with budget 0");
        }

        private void Test_Memory_PrunesToLimits()
        {
            var (tok, inputs, _) = RandomData(n: 5, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PruningConfig.UseAttentionBasedPruning = false;
            for (int i = 0; i < inputs.Length; i++)
                m.PredictWithMemory(inputs[i], i * 100.0, maxNewsMemorySize: 4, maxPriceMemorySize: 7);

            Assert(m.NewsMemory.Count <= 4, $"news memory too large: {m.NewsMemory.Count}");
            Assert(m.PriceMemory.Count <= 7, $"price memory too large: {m.PriceMemory.Count}");
            Assert(m.PriceMemory.Max(e => e.AbsoluteTimestamp) >= 400.0, "recent price memory was not retained");
        }

        private void Test_Memory_AttentionPruningKeepsHighScore_NewsAndPrice()
        {
            var m = new MmtacModel(Cfg(vocabSize: 64), new Random(42));
            m.PruningConfig.UseAttentionBasedPruning = true;
            m.PruningConfig.NewEntryReserveFraction = 0f;
            m.PruningConfig.MinQueryCountForPruning = 1;

            m.NewsMemory = MakeNewsMemoryEntries(m.Config.Price.EmbeddingDim,
                (10.0, 0.10f, 5), (20.0, 0.90f, 5), (30.0, 0.20f, 5), (40.0, 0.80f, 5));
            m.PriceMemory = MakePriceMemoryEntries(m.Config.Price.EmbeddingDim,
                (10.0, 0.10f, 5), (20.0, 0.90f, 5), (30.0, 0.20f, 5), (40.0, 0.80f, 5));

            m.PruneNewsMemory(2);
            InvokePricePruneMemory(m, 2);

            var newsTs = m.NewsMemory.Select(e => e.AbsoluteTimestamp).OrderBy(x => x).ToArray();
            var priceTs = m.PriceMemory.Select(e => e.AbsoluteTimestamp).OrderBy(x => x).ToArray();
            Assert(newsTs.SequenceEqual(new[] { 20.0, 40.0 }), "news pruning did not keep high-score entries");
            Assert(priceTs.SequenceEqual(newsTs), "price pruning should mirror news pruning");
        }

        private void Test_Memory_AttentionScoresUpdated()
        {
            var (tok, inputs, _) = RandomData(n: 3, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));

            m.PredictWithMemory(inputs[0], 0.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);
            m.PredictWithMemory(inputs[1], 100.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);
            m.PredictWithMemory(inputs[2], 200.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);

            Assert(m.NewsMemory.Any(e => e.QueryCount > 0), "no news memory QueryCount was incremented");
            Assert(m.PriceMemory.Any(e => e.QueryCount > 0), "no price memory QueryCount was incremented");
        }

        private void Test_Memory_GlobalTokenNotPersistedAsMemory()
        {
            var cfg = Cfg(globalDim: 3);
            var m = new MmtacModel(cfg, new Random(42));
            var noGlobal = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ConstantPriceSequence(6, 0.2f, 0.1f, -0.1f, 0.05f, 1f) };
            var withGlobal = CloneInput(noGlobal);
            withGlobal.GlobalFeatures = new[] { 1f, 0f, 0.5f };

            m.ClearAllMemory();
            var p0 = m.PredictWithMemory(noGlobal, 100.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);
            int n0 = m.NewsMemory.Count, pMem0 = m.PriceMemory.Count;

            m.ClearAllMemory();
            var p1 = m.PredictWithMemory(withGlobal, 100.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);
            int n1 = m.NewsMemory.Count, pMem1 = m.PriceMemory.Count;

            Assert(n0 == 0 && n1 == 0, "global token should not create news memory entries");
            Assert(pMem0 == noGlobal.PriceSequence.GetLength(0) && pMem1 == noGlobal.PriceSequence.GetLength(0), "global token should not change price memory append count");
            Assert(PredictionsDiffer(p0, p1, 1e-7f), "global token should affect prediction");
        }

        private void Test_Memory_PriceOnlyWorksWithExistingPriceMemory()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            var input = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = RandomMatrix(6, 5, new Random(12), 0.2f) };
            m.PredictWithMemory(input, 0.0);
            Assert(m.PriceMemory.Count > 0, "price memory did not populate");
            var p = m.PredictWithMemory(input, 100.0);
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
        }

        private void Test_Memory_NullStoryEntryDoesNotCrashPredictNext()
        {
            var m = new MmtacModel(Cfg(vocabSize: 64), new Random(42));
            var input = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                NewsStories = new NewsStory[] { null, new NewsStory(new[] { 1, 2, 3 }, 0f) }
            };
            AssertPredictionFiniteAndConstrained(m.PredictNext(input), expectConfidenceHead: false);
        }

        private void Test_Memory_NullStoryEntryDoesNotCrashPredictWithMemory()
        {
            var m = new MmtacModel(Cfg(vocabSize: 64), new Random(42));
            var input = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                NewsStories = new NewsStory[] { null, new NewsStory(new[] { 1, 2, 3 }, 0f) }
            };
            AssertPredictionFiniteAndConstrained(m.PredictWithMemory(input, 100.0), expectConfidenceHead: false);
        }

        // ---------------------------------------------------------------------
        // Sequential
        // ---------------------------------------------------------------------

        private void Test_Sequential_CommitsFullObservedSampleAndScaledTimestamps()
        {
            int seqLen = 6;
            var input = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ConstantPriceSequence(seqLen, 0.2f, 0.1f, -0.1f, 0.05f, 1f) };
            var target = ConstantTargets(seqLen, close: 0.5f, range: 0.2f, quality: 0.9f, direction: 1, midDirection: 1);
            var m = new MmtacModel(Cfg(priceSeqLen: 8), new Random(42));
            new MmtacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(new[] { input }, new[] { target }, new[] { 50.0 }, timeUnitsPerPosition: 0.25);

            Assert(m.PriceMemory.Count == seqLen, $"expected full sample commit count {seqLen}, got {m.PriceMemory.Count}");
            for (int t = 0; t < seqLen; t++)
                Assert(Math.Abs(m.PriceMemory[t].AbsoluteTimestamp - (50.0 + 0.25 * t)) < 1e-9, $"timestamp {t}");
            Assert(Math.Abs(m.LastPriceTimestamp - (50.0 + 0.25 * (seqLen - 1))) < 1e-9, "LastPriceTimestamp");
        }

        private void Test_Sequential_SkippedInvalidSampleDoesNotCommit()
        {
            var valid = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ConstantPriceSequence(5, 0.2f, 0.1f, -0.1f, 0.05f, 1f) };
            var skipped = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ConstantPriceSequence(1, 0.9f, 0.9f, 0.9f, 0.9f, 1f), NewsStories = new[] { new NewsStory(new[] { 1, 2, 3 }, 0f) } };
            var validT = ConstantTargets(5, 0.5f, 0.2f, 0.9f, 1, 1);
            var skippedT = ConstantTargets(1, 0.9f, 0.1f, 0.9f, 1, 1);

            var m = new MmtacModel(Cfg(vocabSize: 64, priceSeqLen: 8), new Random(42));
            new MmtacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(new[] { valid, skipped }, new[] { validT, skippedT }, new[] { 100.0, 200.0 });

            Assert(m.NewsMemory.Count == 0, $"skipped sample's news should not be committed, count={m.NewsMemory.Count}");
            Assert(m.PriceMemory.Count == valid.PriceSequence.GetLength(0), "only valid sample price rows should be committed");
            Assert(Math.Abs(m.LastPriceTimestamp - 104.0) < 1e-9, "skipped sample should not change LastPriceTimestamp");
        }

        private void Test_Sequential_MemoryClearedBetweenEpochs()
        {
            var (tok, inputs, targets) = RandomData(n: 4, seqLen: 6, withNews: true, globalDim: 3);
            var ts = new[] { 100d, 200d, 300d, 400d };
            var cfg1 = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3);
            var cfg2 = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3);
            var m1 = new MmtacModel(cfg1, new Random(42));
            var m2 = new MmtacModel(cfg2, new Random(42));

            new MmtacTrainer(m1, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            new MmtacTrainer(m2, TC(lr: 0f, bs: 1, epochs: 2)).TrainSequential(inputs, targets, ts);

            AssertMemoryEquivalent(m1, m2, "one epoch vs two epochs with lr=0");
        }

        private void Test_Sequential_TimestampShuffleInvariantAtZeroLearningRate()
        {
            var (tok, inputs, targets) = RandomData(n: 6, seqLen: 6, withNews: true, globalDim: 3);
            var ts = new[] { 500d, 100d, 600d, 200d, 700d, 300d };
            int[] perm = { 2, 5, 1, 4, 0, 3 };
            var cfg = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3);

            var m1 = new MmtacModel(cfg, new Random(42));
            var m2 = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3), new Random(42));

            new MmtacTrainer(m1, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            new MmtacTrainer(m2, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(perm.Select(i => inputs[i]).ToArray(), perm.Select(i => targets[i]).ToArray(), perm.Select(i => ts[i]).ToArray());

            AssertMemoryEquivalent(m1, m2, "timestamp shuffle invariant");
        }

        private void Test_Sequential_ZeroLearningRateDoesNotChangeWeights()
        {
            var (tok, inputs, targets) = RandomData(n: 4, seqLen: 6, withNews: true, globalDim: 3);
            var ts = new[] { 100d, 200d, 300d, 400d };
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 3), new Random(42));
            var snap = SnapshotWeights(m);
            new MmtacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            AssertWeightsUnchanged(snap, m, "TrainSequential lr=0");
        }

        private void Test_Sequential_WithGlobalTokenRunsFinite()
        {
            var (tok, inputs, targets) = RandomData(n: 5, seqLen: 6, withNews: true, globalDim: 3);
            var ts = Enumerable.Range(0, inputs.Length).Select(i => i * 100.0).ToArray();
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, globalDim: 3), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.002f, bs: 1, epochs: 4)).TrainSequential(inputs, targets, ts);
            var p = m.PredictWithMemory(inputs[0], 1000.0);
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
        }

        private void Test_Sequential_MemoryAccumulates()
        {
            var (tok, inputs, targets) = RandomData(n: 3, seqLen: 6, withNews: true);
            var ts = new[] { 100d, 200d, 300d };
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            Assert(m.NewsMemory.Count > 0, "NewsMemory did not accumulate");
            Assert(m.PriceMemory.Count > 0, "PriceMemory did not accumulate");
            Assert(m.PriceMemory.All(e => e.HiddenState.Length == m.Config.Price.EmbeddingDim), "price memory hidden dim mismatch");
        }

        // ---------------------------------------------------------------------
        // Save / Load
        // ---------------------------------------------------------------------

        private void Test_SaveLoad_ForwardIdentical_AllHeads()
        {
            var (tok, inputs, targets) = RandomData(n: 4, seqLen: 6, withNews: true, globalDim: 3);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, useConf: true, globalDim: 3), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.002f, bs: 2, epochs: 4)).Train(inputs, targets);
            var before = m.Forward(inputs[0]);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var loaded = MmtacModel.Load(dir);
                var after = loaded.Forward(inputs[0]);

                AssertMatricesClose(before.regression, after.regression, 1e-5f, "regression mismatch");
                AssertMatricesClose(before.range, after.range, 1e-5f, "range mismatch");
                AssertMatricesClose(before.quality, after.quality, 1e-5f, "quality mismatch");
                AssertMatricesClose(before.direction, after.direction, 1e-5f, "direction mismatch");
                AssertMatricesClose(before.midDirection, after.midDirection, 1e-5f, "midDirection mismatch");
                AssertMatricesClose(before.confidence, after.confidence, 1e-5f, "confidence mismatch");
            }
            finally { DeleteDir(dir); }
        }

        private void Test_SaveLoad_WeightsAndContextPreserved()
        {
            var m = new MmtacModel(Cfg(useConf: true, globalDim: 4), new Random(42));
            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                Assert(!Changed(m.RegressionProjection, ld.RegressionProjection, 1e-6f), "RegressionProjection mismatch");
                Assert(!Changed(m.RangeProjection, ld.RangeProjection, 1e-6f), "RangeProjection mismatch");
                Assert(!Changed(m.QualityProjection, ld.QualityProjection, 1e-6f), "QualityProjection mismatch");
                Assert(!Changed(m.DirectionProjection, ld.DirectionProjection, 1e-6f), "DirectionProjection mismatch");
                Assert(!Changed(m.MidDirectionProjection, ld.MidDirectionProjection, 1e-6f), "MidDirectionProjection mismatch");
                Assert(!Changed(m.ConfidenceProjection, ld.ConfidenceProjection, 1e-6f), "ConfidenceProjection mismatch");
                Assert(!Changed(m.ContextTypeEmbedding, ld.ContextTypeEmbedding, 1e-6f), "ContextTypeEmbedding mismatch");
                Assert(!Changed(m.GlobalFeatureProjection, ld.GlobalFeatureProjection, 1e-6f), "GlobalFeatureProjection mismatch");
            }
            finally { DeleteDir(dir); }
        }

        private void Test_SaveLoad_MemoryAndPruningPreserved()
        {
            var (tok, inputs, _) = RandomData(n: 1, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            m.PruningConfig.AttentionScoreAlpha = 0.25f;
            m.PruningConfig.MinQueryCountForPruning = 7;
            m.PruningConfig.NewEntryReserveFraction = 0.33f;
            m.PruningConfig.UseAttentionBasedPruning = false;
            m.PredictWithMemory(inputs[0], 100.0);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                Assert(ld.NewsMemory.Count == m.NewsMemory.Count, "NewsMemory count mismatch");
                Assert(ld.PriceMemory.Count == m.PriceMemory.Count, "PriceMemory count mismatch");
                Assert(Math.Abs(ld.LastPriceTimestamp - m.LastPriceTimestamp) < 1e-9, "LastPriceTimestamp mismatch");
                Assert(MathF.Abs(ld.PruningConfig.AttentionScoreAlpha - 0.25f) < 1e-6f, "pruning alpha mismatch");
                Assert(ld.PruningConfig.MinQueryCountForPruning == 7, "pruning min query mismatch");
                Assert(MathF.Abs(ld.PruningConfig.NewEntryReserveFraction - 0.33f) < 1e-6f, "pruning reserve mismatch");
                Assert(!ld.PruningConfig.UseAttentionBasedPruning, "pruning mode mismatch");
            }
            finally { DeleteDir(dir); }
        }

        private void Test_SaveLoad_ContinueTrainingFinite()
        {
            var (tok, inputs, targets) = RandomData(n: 5, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.002f, bs: 5, epochs: 4)).Train(inputs, targets);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                new MmtacTrainer(ld, TC(lr: 0.002f, bs: 5, epochs: 4)).Train(inputs, targets);
                float loss = new MmtacTrainer(ld, TC(epochs: 1)).Validate(inputs, targets);
                Assert(float.IsFinite(loss) && loss >= 0f, $"invalid loss after continued training: {loss}");
            }
            finally { DeleteDir(dir); }
        }

        private void Test_SaveLoad_SequentialMemoryPredictionIdentical()
        {
            var (tok, inputs, targets) = RandomData(n: 4, seqLen: 6, withNews: true, globalDim: 3);
            var cfg = Cfg(tok.VocabSize + 2, globalDim: 3);
            var m = new MmtacModel(cfg, new Random(42));
            var ts = new[] { 100d, 200d, 300d, 400d };
            new MmtacTrainer(m, TC(lr: 0.002f, bs: 1, epochs: 4)).TrainSequential(inputs, targets, ts);
            m.PredictWithMemory(inputs[0], 1000.0);
            m.PredictWithMemory(inputs[1], 1200.0);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);
                var p1 = m.PredictWithMemory(inputs[2], 1500.0);
                var p2 = ld.PredictWithMemory(inputs[2], 1500.0);
                AssertPredictionsClose(p1, p2, 1e-5f, "rolling-memory prediction changed after reload");
            }
            finally { DeleteDir(dir); }
        }

        // ---------------------------------------------------------------------
        // Stability
        // ---------------------------------------------------------------------

        private void Test_Stability_InputScalesNoNaN()
        {
            var m = new MmtacModel(Cfg(), new Random(42));
            float[] scales = { 100f, 1e-6f, 0f, -5f };
            foreach (float scale in scales)
            {
                var ps = new float[8, 5];
                for (int i = 0; i < ps.GetLength(0); i++)
                    for (int j = 0; j < ps.GetLength(1); j++)
                        ps[i, j] = scale;
                AssertPredictionFiniteAndConstrained(m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps }), expectConfidenceHead: false);
            }
        }

        private void Test_Stability_ManyStoriesNoNaN()
        {
            var m = new MmtacModel(Cfg(vocabSize: 64), new Random(42));
            var stories = Enumerable.Range(0, 12).Select(i => new NewsStory(new[] { 1, 2, 3, 4 }, -i)).ToArray();
            var p = m.PredictNext(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = RandomMatrix(8, 5, new Random(42), 0.2f), NewsStories = stories });
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
        }

        private void Test_Stability_NoNaNAfterManyEpochs()
        {
            var (tok, inputs, targets) = RandomData(n: 5, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            new MmtacTrainer(m, TC(lr: 0.001f, bs: 5, epochs: 40)).Train(inputs, targets);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(float.IsFinite(loss) && loss >= 0f, $"invalid loss after many epochs: {loss}");
            AssertPredictionFiniteAndConstrained(m.PredictNext(inputs[0]), expectConfidenceHead: false);
        }

        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        private MmtacConfig Cfg(
            int vocabSize = 50,
            int embDim = 16,
            int numHeads = 2,
            int numLayers = 1,
            int ffnDim = 32,
            int priceFeatures = 5,
            int priceSeqLen = 12,
            bool useConf = false,
            bool freezeText = false,
            int globalDim = 0,
            bool decayEnabled = true,
            bool bypassDecay = true,
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
                Global = new MmtacGlobalConfig
                {
                    GlobalFeatureDim = globalDim,
                    BypassDecay = bypassDecay
                },
                Output = new MmtacOutputConfig
                {
                    UseConfidenceHead = useConf,
                    DirectionLossWeight = 1f,
                    MidDirectionLossWeight = 0.5f,
                    RangeLossWeight = 1f,
                    QualityLossWeight = 1f,
                    CloseDirectionConsistencyWeight = 0f,
                    CloseDirectionConsistencyMargin = 0f
                },
                Decay = new DecayNetworkConfig
                {
                    Enabled = decayEnabled,
                    ProjectionDim = Math.Min(8, embDim),
                    HiddenDim = Math.Max(8, embDim),
                    TimeEncodingBases = 8,
                    MemAttentionDropout = 0f,
                    MlpDropout = 0f,
                    WeightDecay = 0f
                },
                Reg = new RegularizationConfig
                {
                    L2RegulationLamda = 0f,
                    GradientClippingThreshold = 1f
                },
                Runtime = new RuntimeConfig
                {
                    FFNActivationType = ActivationType.Relu,
                    AccelerationType = AccelerationType.CPU,
                    AccelerationDeviceId = 0
                },
                PriceContext = new PriceContextConfig
                {
                    Enabled = priceContextEnabled,
                    MinHistoryLength = 3,
                    MinCurrentLength = 3
                },
                Pruning = new MemoryPruningConfig
                {
                    AttentionScoreAlpha = 0.1f,
                    MinQueryCountForPruning = 3,
                    NewEntryReserveFraction = 0.1f,
                    UseAttentionBasedPruning = true
                }
            };
            cfg.Validate();
            return cfg;
        }

        private TrainingConfig TC(float lr = 0.001f, int bs = 4, int epochs = 10, bool clip = true)
        {
            return new TrainingConfig
            {
                LearningRate = lr,
                BatchSize = bs,
                Epochs = epochs,
                UseGradientClipping = clip,
                GradientClipThreshold = 1f,
                Verbose = false
            };
        }

        private static float[,] RandomMatrix(int rows, int cols, Random rng, float scale)
        {
            var m = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = ((float)rng.NextDouble() * 2f - 1f) * scale;
            return m;
        }

        private static float[,] ConstantPriceSequence(int seqLen, params float[] features)
        {
            var ps = new float[seqLen, features.Length];
            for (int t = 0; t < seqLen; t++)
                for (int f = 0; f < features.Length; f++)
                    ps[t, f] = features[f];
            return ps;
        }

        private static ModelTarget[] ConstantTargets(int seqLen, float close, float range, float quality, int direction, int midDirection)
        {
            var out_ = new ModelTarget[seqLen];
            float high = close + range * 0.5f;
            float low = close - range * 0.5f;
            for (int t = 0; t < seqLen; t++)
            {
                out_[t] = new ModelTarget
                {
                    High = high,
                    Low = low,
                    Close = close,
                    Range = range,
                    Quality = quality,
                    Direction = direction,
                    MidWindowDirection = midDirection
                };
            }
            return out_;
        }

        private MultimodalInput MakeInput(int seqLen = 8, int priceFeatures = 5, int globalDim = 0, NewsStory[] stories = null)
        {
            var rng = new Random(42);
            return new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandomMatrix(seqLen, priceFeatures, rng, 0.3f),
                NewsStories = stories,
                GlobalFeatures = globalDim > 0 ? Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray() : null
            };
        }

        private (BPETokenizer tok, MultimodalInput[] inputs, ModelTarget[][] targets) RandomData(int n = 8, int seqLen = 6, int priceFeatures = 5, int seed = 42, bool withNews = true, int globalDim = 0)
        {
            var rng = new Random(seed);
            string[] corpus = { "stock rose sharply", "market crashed today", "bullish outlook strong", "bearish data weak" };
            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 200, minFrequency: 1);

            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int s = 0; s < n; s++)
            {
                var price = RandomMatrix(seqLen, priceFeatures, rng, 0.3f);
                NewsStory[] stories = null;
                if (withNews)
                {
                    int storyCount = 1 + rng.Next(2);
                    stories = new NewsStory[storyCount];
                    for (int i = 0; i < storyCount; i++)
                        stories[i] = new NewsStory(tok.Encode(corpus[rng.Next(corpus.Length)], addSpecialTokens: true), -i);
                }

                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(s),
                    PriceSequence = price,
                    NewsStories = stories,
                    GlobalFeatures = globalDim > 0 ? Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray() : null
                };

                targets[s] = new ModelTarget[seqLen];
                for (int t = 0; t < seqLen; t++)
                {
                    float close = 0.5f + 0.15f * price[t, 0];
                    float range = 0.1f + 0.03f * MathF.Abs(price[t, 1]);
                    targets[s][t] = new ModelTarget
                    {
                        High = close + range * 0.5f,
                        Low = close - range * 0.5f,
                        Close = close,
                        Range = range,
                        Quality = 0.75f,
                        Direction = close >= 0.5f ? 1f : 0f,
                        MidWindowDirection = price[t, 1] >= 0f ? 1f : 0f
                    };
                }
            }

            return (tok, inputs, targets);
        }

        private (BPETokenizer tok, MultimodalInput[] trainInputs, ModelTarget[][] trainTargets, MultimodalInput[] testInputs, ModelTarget[][] testTargets) PriceHeldOutData(int trainN = 64, int testN = 20, int seqLen = 6, int seed = 123)
        {
            var rng = new Random(seed);
            var tok = new BPETokenizer();
            tok.Train(new[] { "unused token" }, vocabSize: 32, minFrequency: 1);

            MultimodalInput MakeInput(float x, float y, int idx) => new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                PriceSequence = ConstantPriceSequence(seqLen, x, y, x * y, x * x, 1f)
            };

            ModelTarget[] MakeTarget(float x, float y)
            {
                float close = 0.5f + 0.22f * x - 0.1f * y;
                float range = 0.08f + 0.04f * MathF.Abs(x);
                float quality = Clamp01(0.4f + 0.5f * y);
                return ConstantTargets(seqLen, close, range, quality, close > 0.5f ? 1 : 0, x + y > 0f ? 1 : 0);
            }

            void Fill(int n, int offset, out MultimodalInput[] inputs, out ModelTarget[][] targets)
            {
                inputs = new MultimodalInput[n];
                targets = new ModelTarget[n][];
                for (int i = 0; i < n; i++)
                {
                    float x = (float)(rng.NextDouble() * 2.0 - 1.0);
                    float y = (float)(rng.NextDouble() * 2.0 - 1.0);
                    inputs[i] = MakeInput(x, y, offset + i);
                    targets[i] = MakeTarget(x, y);
                }
            }

            Fill(trainN, 0, out var trainInputs, out var trainTargets);
            Fill(testN, 1000, out var testInputs, out var testTargets);
            return (tok, trainInputs, trainTargets, testInputs, testTargets);
        }

        private (BPETokenizer tok, MultimodalInput[] inputs, ModelTarget[][] targets) NewsSignalData(int n = 48, int seqLen = 6)
        {
            string bullText = "strong bullish outlook demand growth";
            string bearText = "weak bearish outlook contraction risk";
            var tok = new BPETokenizer();
            tok.Train(new[] { bullText, bearText }, vocabSize: 80, minFrequency: 1);
            int[] bull = tok.Encode(bullText, addSpecialTokens: true);
            int[] bear = tok.Encode(bearText, addSpecialTokens: true);

            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            for (int i = 0; i < n; i++)
            {
                bool isBull = i % 2 == 0;
                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    NewsStories = new[] { new NewsStory(isBull ? bull : bear, 0f) }
                };
                targets[i] = ConstantTargets(seqLen, isBull ? 0.78f : 0.22f, 0.1f, 0.9f, isBull ? 1 : 0, isBull ? 1 : 0);
            }
            return (tok, inputs, targets);
        }

        private (BPETokenizer tok, MultimodalInput[] inputs, ModelTarget[][] targets) GlobalSignalData(int n = 48, int seqLen = 6)
        {
            var tok = new BPETokenizer();
            tok.Train(new[] { "unused token" }, vocabSize: 32, minFrequency: 1);
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            for (int i = 0; i < n; i++)
            {
                bool riskOn = i % 2 == 0;
                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    GlobalFeatures = riskOn ? new[] { 1f, 0f, 1f } : new[] { 0f, 1f, 0f }
                };
                targets[i] = ConstantTargets(seqLen, riskOn ? 0.78f : 0.22f, 0.1f, 0.9f, riskOn ? 1 : 0, riskOn ? 1 : 0);
            }
            return (tok, inputs, targets);
        }

        private static (MultimodalInput[] inputs, ModelTarget[][] targets) DirectionSignalData(int n = 12, int seqLen = 6)
        {
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            for (int i = 0; i < n; i++)
            {
                bool up = i < n / 2;
                inputs[i] = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow.AddMinutes(i), PriceSequence = ConstantPriceSequence(seqLen, up ? 0.8f : 0.2f, 0.5f, 0.5f, 0.5f, 1f) };
                targets[i] = ConstantTargets(seqLen, 0.5f, 0.1f, 0.7f, up ? 1 : 0, up ? 1 : 0);
            }
            return (inputs, targets);
        }

        private static (MultimodalInput[] inputs, ModelTarget[][] targets) OppositeDirectionData(int n = 24, int seqLen = 6)
        {
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            for (int i = 0; i < n; i++)
            {
                bool dirUp = i < n / 2;
                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, dirUp ? 0.8f : 0.2f, dirUp ? 0.2f : 0.8f, 0.5f, 0.5f, 1f)
                };
                targets[i] = ConstantTargets(seqLen, 0.5f, 0.1f, 0.7f, dirUp ? 1 : 0, dirUp ? 0 : 1);
            }
            return (inputs, targets);
        }

        private static (MultimodalInput[] inputs, ModelTarget[][] targets) RangeSignalData(int n = 24, int seqLen = 6)
        {
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            for (int i = 0; i < n; i++)
            {
                float x = i < n / 2 ? 0.2f : 0.8f;
                float range = i < n / 2 ? 0.08f : 0.30f;
                inputs[i] = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow.AddMinutes(i), PriceSequence = ConstantPriceSequence(seqLen, x, 1f - x, x * x, 0.5f, 1f) };
                targets[i] = ConstantTargets(seqLen, 0.5f, range, 0.8f, 1, 1);
            }
            return (inputs, targets);
        }

        private static (MultimodalInput[] inputs, ModelTarget[][] targets, double[] ts) SequentialReferenceData()
        {
            int n = 8, seqLen = 6;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            var ts = new double[n];
            for (int i = 0; i < n; i++)
            {
                float x = (i % 2 == 0) ? 0.4f : -0.4f;
                inputs[i] = new MultimodalInput { PredictionTimestamp = DateTime.UtcNow.AddMinutes(i), PriceSequence = ConstantPriceSequence(seqLen, x, x * x, MathF.Sign(x), 0.5f * x, 1f) };
                targets[i] = ConstantTargets(seqLen, 0.5f + 0.2f * x, 0.1f, 0.8f, x > 0 ? 1 : 0, x > 0 ? 1 : 0);
                ts[i] = i * 10.0;
            }
            return (inputs, targets, ts);
        }

        private static float Clamp01(float x) => MathF.Max(0f, MathF.Min(1f, x));

        private static float MeanRangeError(MmtacModel m, MultimodalInput[] inputs, ModelTarget[][] targets)
        {
            float total = 0f;
            int count = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var (_, range, _, _, _, _) = m.Forward(inputs[i]);
                for (int t = 0; t < range.GetLength(0); t++)
                {
                    total += MathF.Abs(range[t, 0] - targets[i][t].Range);
                    count++;
                }
            }
            return total / Math.Max(1, count);
        }

        private static MultimodalInput CloneInput(MultimodalInput src, bool keepPrice = true, bool keepNews = true, bool keepGlobals = true)
        {
            return new MultimodalInput
            {
                PredictionTimestamp = src.PredictionTimestamp,
                PriceSequence = keepPrice ? Copy(src.PriceSequence) : new float[src.PriceSequence.GetLength(0), src.PriceSequence.GetLength(1)],
                NewsStories = keepNews ? CopyStories(src.NewsStories) : null,
                GlobalFeatures = keepGlobals ? Copy(src.GlobalFeatures) : null
            };
        }

        private static NewsStory[] CopyStories(NewsStory[] stories)
        {
            if (stories == null) return null;
            var copy = new NewsStory[stories.Length];
            for (int i = 0; i < stories.Length; i++)
                copy[i] = stories[i] == null ? null : new NewsStory(Copy(stories[i].TokenIds), stories[i].ArrivalTime);
            return copy;
        }

        private static int[] Copy(int[] src)
        {
            if (src == null) return null;
            var dst = new int[src.Length];
            Array.Copy(src, dst, src.Length);
            return dst;
        }

        private static float[] Copy(float[] src)
        {
            if (src == null) return null;
            var dst = new float[src.Length];
            Array.Copy(src, dst, src.Length);
            return dst;
        }

        private static float[,] Copy(float[,] src)
        {
            if (src == null) return null;
            var dst = new float[src.GetLength(0), src.GetLength(1)];
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    dst[i, j] = src[i, j];
            return dst;
        }

        private static float[,,] Copy3D(float[,,] src)
        {
            if (src == null) return null;
            var dst = new float[src.GetLength(0), src.GetLength(1), src.GetLength(2)];
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    for (int k = 0; k < src.GetLength(2); k++)
                        dst[i, j, k] = src[i, j, k];
            return dst;
        }

        private static bool Changed(float[,] before, float[,] after, float tol = 1e-8f) => MaxAbsDiff(before, after) > tol;

        private static bool Changed(float[,,] before, float[,,] after, float tol = 1e-8f)
        {
            if (before == null || after == null) return before != after;
            if (before.GetLength(0) != after.GetLength(0) || before.GetLength(1) != after.GetLength(1) || before.GetLength(2) != after.GetLength(2)) return true;
            for (int i = 0; i < before.GetLength(0); i++)
                for (int j = 0; j < before.GetLength(1); j++)
                    for (int k = 0; k < before.GetLength(2); k++)
                        if (MathF.Abs(before[i, j, k] - after[i, j, k]) > tol) return true;
            return false;
        }

        private static float MaxAbsDiff(float[,] a, float[,] b)
        {
            if (a == null || b == null) return a == b ? 0f : float.PositiveInfinity;
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1)) return float.PositiveInfinity;
            float max = 0f;
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    max = MathF.Max(max, MathF.Abs(a[i, j] - b[i, j]));
            return max;
        }

        private static float MaxAbsDiff(float[] a, float[] b)
        {
            if (a == null || b == null) return a == b ? 0f : float.PositiveInfinity;
            if (a.Length != b.Length) return float.PositiveInfinity;
            float max = 0f;
            for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
            return max;
        }

        private void AssertMatricesClose(float[,] expected, float[,] actual, float tol, string message)
        {
            float diff = MaxAbsDiff(expected, actual);
            Assert(diff <= tol, $"{message}; maxAbsDiff={diff:E6}, tol={tol:E6}");
        }

        private void AssertPredictionFiniteAndConstrained(ModelPrediction p, bool expectConfidenceHead)
        {
            Assert(float.IsFinite(p.High), "High not finite");
            Assert(float.IsFinite(p.Low), "Low not finite");
            Assert(float.IsFinite(p.Close), "Close not finite");
            Assert(float.IsFinite(p.Range) && p.Range >= 0f, $"Range invalid: {p.Range}");
            Assert(float.IsFinite(p.Quality) && p.Quality >= 0f && p.Quality <= 1f, $"Quality invalid: {p.Quality}");
            Assert(float.IsFinite(p.DirectionProb) && p.DirectionProb >= 0f && p.DirectionProb <= 1f, $"DirectionProb invalid: {p.DirectionProb}");
            Assert(float.IsFinite(p.MidWindowDirectionProb) && p.MidWindowDirectionProb >= 0f && p.MidWindowDirectionProb <= 1f, $"MidWindowDirectionProb invalid: {p.MidWindowDirectionProb}");
            Assert(float.IsFinite(p.Confidence), "Confidence not finite");
            if (expectConfidenceHead)
                Assert(p.Confidence >= 0f && p.Confidence <= 1f, $"Confidence invalid: {p.Confidence}");
            else
                Assert(p.Confidence == 1f, $"Confidence should be 1 when disabled, got {p.Confidence}");
        }

        private static bool PredictionsDiffer(ModelPrediction a, ModelPrediction b, float tol)
        {
            return MathF.Abs(a.High - b.High) > tol
                || MathF.Abs(a.Low - b.Low) > tol
                || MathF.Abs(a.Close - b.Close) > tol
                || MathF.Abs(a.Range - b.Range) > tol
                || MathF.Abs(a.Quality - b.Quality) > tol
                || MathF.Abs(a.DirectionProb - b.DirectionProb) > tol
                || MathF.Abs(a.MidWindowDirectionProb - b.MidWindowDirectionProb) > tol
                || MathF.Abs(a.Confidence - b.Confidence) > tol;
        }

        private void AssertPredictionsClose(ModelPrediction a, ModelPrediction b, float tol, string message)
        {
            Assert(MathF.Abs(a.High - b.High) <= tol, message + " High");
            Assert(MathF.Abs(a.Low - b.Low) <= tol, message + " Low");
            Assert(MathF.Abs(a.Close - b.Close) <= tol, message + " Close");
            Assert(MathF.Abs(a.Range - b.Range) <= tol, message + " Range");
            Assert(MathF.Abs(a.Quality - b.Quality) <= tol, message + " Quality");
            Assert(MathF.Abs(a.DirectionProb - b.DirectionProb) <= tol, message + " DirectionProb");
            Assert(MathF.Abs(a.MidWindowDirectionProb - b.MidWindowDirectionProb) <= tol, message + " MidWindowDirectionProb");
            Assert(MathF.Abs(a.Confidence - b.Confidence) <= tol, message + " Confidence");
        }

        private void AssertAllFinite(float[,] m, string name)
        {
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    Assert(float.IsFinite(m[i, j]), $"{name}[{i},{j}] not finite: {m[i, j]}");
        }

        private void AssertMatrixShape(float[,] m, int rows, int cols, string name)
        {
            Assert(m != null, $"{name} is null");
            Assert(m.GetLength(0) == rows, $"{name} rows actual={m.GetLength(0)} expected={rows}");
            Assert(m.GetLength(1) == cols, $"{name} cols actual={m.GetLength(1)} expected={cols}");
        }

        private void AssertThrows<T>(Action action, string message) where T : Exception
        {
            try
            {
                action();
            }
            catch (TargetInvocationException ex) when (ex.InnerException is T)
            {
                return;
            }
            catch (T)
            {
                return;
            }
            catch (Exception ex)
            {
                Assert(false, $"{message}; expected {typeof(T).Name}, got {ex.GetType().Name}: {ex.Message}");
                return;
            }

            Assert(false, $"{message}; expected {typeof(T).Name}, got no exception");
        }

        private void AssertNoFutureLeakage(MmtacModel model, NewsStory[] stories, float[] globals)
        {
            int prefixLen = 5;
            int extraLen = 4;
            var full = RandomMatrix(prefixLen + extraLen, 5, new Random(42), 0.3f);
            var prefix = new float[prefixLen, 5];
            for (int t = 0; t < prefixLen; t++)
                for (int f = 0; f < 5; f++)
                    prefix[t, f] = full[t, f];
            for (int t = prefixLen; t < full.GetLength(0); t++)
                for (int f = 0; f < 5; f++)
                    full[t, f] = 100f + t * 7f + f;

            var a = model.Forward(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = prefix, NewsStories = stories, GlobalFeatures = globals });
            var b = model.Forward(new MultimodalInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = full, NewsStories = stories, GlobalFeatures = globals });

            const float tol = 1e-5f;
            for (int t = 0; t < prefixLen; t++)
            {
                for (int j = 0; j < 3; j++) Assert(MathF.Abs(a.regression[t, j] - b.regression[t, j]) < tol, $"regression future leakage t={t}, j={j}");
                Assert(MathF.Abs(a.range[t, 0] - b.range[t, 0]) < tol, $"range future leakage t={t}");
                Assert(MathF.Abs(a.quality[t, 0] - b.quality[t, 0]) < tol, $"quality future leakage t={t}");
                Assert(MathF.Abs(a.direction[t, 0] - b.direction[t, 0]) < tol, $"direction future leakage t={t}");
                Assert(MathF.Abs(a.midDirection[t, 0] - b.midDirection[t, 0]) < tol, $"midDirection future leakage t={t}");
                if (a.confidence != null || b.confidence != null)
                    Assert(MathF.Abs(a.confidence[t, 0] - b.confidence[t, 0]) < tol, $"confidence future leakage t={t}");
            }
        }

        private static (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence) InvokeForwardWithCache(MmtacModel model, MultimodalInput input, MmtacForwardCache cache, bool isTraining)
        {
            var method = typeof(MmtacModel).GetMethod(
                "ForwardWithCache",
                BindingFlags.Instance | BindingFlags.NonPublic,
                binder: null,
                types: new[] { typeof(MultimodalInput), typeof(MmtacForwardCache), typeof(bool), typeof(Random) },
                modifiers: null);

            if (method == null) throw new InvalidOperationException("Could not find ForwardWithCache overload");
            var result = method.Invoke(model, new object[] { input, cache, isTraining, null });
            return ((float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence))result;
        }

        private static float[,] SliceRows(float[,] m, int start, int endExclusive)
        {
            int len = endExclusive - start;
            int cols = m.GetLength(1);
            var out_ = new float[len, cols];
            for (int i = 0; i < len; i++)
                for (int j = 0; j < cols; j++)
                    out_[i, j] = m[start + i, j];
            return out_;
        }

        private static float ValidateSequentialReference(MmtacTrainer trainer, MmtacModel model, MultimodalInput[] inputs, ModelTarget[][] targets, double[] timestamps, double timeUnitsPerPosition = 1.0, int maxNewsMemory = 100, int maxPriceMemory = 200)
        {
            var predict = typeof(MmtacTrainer).GetMethod("PredictWithCurrentMemoryNoCommit", BindingFlags.Instance | BindingFlags.NonPublic);
            var commit = typeof(MmtacTrainer).GetMethod("CommitObservedSampleToMemory", BindingFlags.Instance | BindingFlags.NonPublic);
            if (predict == null || commit == null) throw new InvalidOperationException("Required private sequential validation helpers were not found");

            float total = 0f;
            int count = 0;
            int[] ordered = Enumerable.Range(0, inputs.Length).OrderBy(i => timestamps[i]).ToArray();
            model.ClearAllMemory();
            try
            {
                foreach (int idx in ordered)
                {
                    var input = inputs[idx];
                    var sampleTargets = targets[idx];
                    if (input?.PriceSequence == null || sampleTargets == null) continue;
                    int sl = input.PriceSequence.GetLength(0);
                    if (sl < 2 || sampleTargets.Length < sl) continue;

                    var wrapped = new MultimodalInput
                    {
                        PredictionTimestamp = input.PredictionTimestamp,
                        PriceSequence = SliceRows(input.PriceSequence, 0, sl - 1),
                        NewsStories = input.NewsStories,
                        GlobalFeatures = input.GlobalFeatures
                    };

                    var pred = (ModelPrediction)predict.Invoke(trainer, new object[] { wrapped, timestamps[idx], timeUnitsPerPosition });
                    total += SequentialLegacyLoss(model.Config, pred, sampleTargets[sl - 1]);
                    count++;
                    commit.Invoke(trainer, new object[] { input, timestamps[idx], timeUnitsPerPosition, maxNewsMemory, maxPriceMemory });
                }
                return count > 0 ? total / count : 0f;
            }
            finally { model.ClearAllMemory(); }
        }

        private static float SequentialLegacyLoss(MmtacConfig cfg, ModelPrediction pred, ModelTarget tgt)
        {
            float total = 0f;
            total += Square(pred.High - tgt.High);
            total += Square(pred.Low - tgt.Low);
            total += Square(pred.Close - tgt.Close);
            total += cfg.Output.RangeLossWeight * Square(pred.Range - tgt.Range);
            total += cfg.Output.QualityLossWeight * Square(pred.Quality - tgt.Quality);
            total += cfg.Output.DirectionLossWeight * Square(pred.DirectionProb - tgt.Direction);
            total += cfg.Output.MidDirectionLossWeight * Square(pred.MidWindowDirectionProb - tgt.MidWindowDirection);
            return total;
        }

        private static float Square(float x) => x * x;

        private static void InvokePricePruneMemory(MmtacModel model, int maxSize)
        {
            var method = typeof(MmtacModel).GetMethod("PricePruneMemory", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (method == null) throw new InvalidOperationException("Could not find PricePruneMemory");
            method.Invoke(model, new object[] { maxSize });
        }

        private static List<NewsMemoryEntry> MakeNewsMemoryEntries(int embDim, params (double ts, float score, int qc)[] specs)
        {
            return specs.Select((s, i) => new NewsMemoryEntry
            {
                AbsoluteTimestamp = s.ts,
                AttentionScore = s.score,
                QueryCount = s.qc,
                HiddenState = Enumerable.Range(0, embDim).Select(d => d == 0 ? (float)(i + 1) : 0f).ToArray()
            }).ToList();
        }

        private static List<PriceMemoryEntry> MakePriceMemoryEntries(int embDim, params (double ts, float score, int qc)[] specs)
        {
            return specs.Select((s, i) => new PriceMemoryEntry
            {
                AbsoluteTimestamp = s.ts,
                AttentionScore = s.score,
                QueryCount = s.qc,
                HiddenState = Enumerable.Range(0, embDim).Select(d => d == 0 ? (float)(i + 1) : 0f).ToArray()
            }).ToList();
        }

        private static string TmpDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "mmtac_test_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        private static void DeleteDir(string dir)
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }

        private sealed class WeightSnapshot
        {
            public float[,] TextTokenEmbedding { get; set; }
            public float[,] PriceInputProjection { get; set; }
            public float[,] RegressionProjection { get; set; }
            public float[,] RangeProjection { get; set; }
            public float[,] QualityProjection { get; set; }
            public float[,] DirectionProjection { get; set; }
            public float[,] MidDirectionProjection { get; set; }
            public float[,] ConfidenceProjection { get; set; }
            public float[,] GlobalFeatureProjection { get; set; }
            public float[,] ContextTypeEmbedding { get; set; }
            public float[,] PriceSelfWq { get; set; }
            public float[,] PriceCrossWk { get; set; }
        }

        private static WeightSnapshot SnapshotWeights(MmtacModel m)
        {
            return new WeightSnapshot
            {
                TextTokenEmbedding = Copy(m.TextTokenEmbedding),
                PriceInputProjection = Copy(m.PriceInputProjection),
                RegressionProjection = Copy(m.RegressionProjection),
                RangeProjection = Copy(m.RangeProjection),
                QualityProjection = Copy(m.QualityProjection),
                DirectionProjection = Copy(m.DirectionProjection),
                MidDirectionProjection = Copy(m.MidDirectionProjection),
                ConfidenceProjection = Copy(m.ConfidenceProjection),
                GlobalFeatureProjection = Copy(m.GlobalFeatureProjection),
                ContextTypeEmbedding = Copy(m.ContextTypeEmbedding),
                PriceSelfWq = Copy(m.PriceBlocks[0].SelfAttention.WQ),
                PriceCrossWk = Copy(m.PriceBlocks[0].CrossAttention.WK)
            };
        }

        private void AssertWeightsUnchanged(WeightSnapshot s, MmtacModel m, string label)
        {
            Assert(!Changed(s.TextTokenEmbedding, m.TextTokenEmbedding, 1e-8f), label + ": TextTokenEmbedding changed");
            Assert(!Changed(s.PriceInputProjection, m.PriceInputProjection, 1e-8f), label + ": PriceInputProjection changed");
            Assert(!Changed(s.RegressionProjection, m.RegressionProjection, 1e-8f), label + ": RegressionProjection changed");
            Assert(!Changed(s.RangeProjection, m.RangeProjection, 1e-8f), label + ": RangeProjection changed");
            Assert(!Changed(s.QualityProjection, m.QualityProjection, 1e-8f), label + ": QualityProjection changed");
            Assert(!Changed(s.DirectionProjection, m.DirectionProjection, 1e-8f), label + ": DirectionProjection changed");
            Assert(!Changed(s.MidDirectionProjection, m.MidDirectionProjection, 1e-8f), label + ": MidDirectionProjection changed");
            Assert(!Changed(s.ConfidenceProjection, m.ConfidenceProjection, 1e-8f), label + ": ConfidenceProjection changed");
            Assert(!Changed(s.GlobalFeatureProjection, m.GlobalFeatureProjection, 1e-8f), label + ": GlobalFeatureProjection changed");
            Assert(!Changed(s.ContextTypeEmbedding, m.ContextTypeEmbedding, 1e-8f), label + ": ContextTypeEmbedding changed");
            Assert(!Changed(s.PriceSelfWq, m.PriceBlocks[0].SelfAttention.WQ, 1e-8f), label + ": Price SelfAttention.WQ changed");
            Assert(!Changed(s.PriceCrossWk, m.PriceBlocks[0].CrossAttention.WK, 1e-8f), label + ": Price CrossAttention.WK changed");
        }

        private void AssertMemoryEquivalent(MmtacModel a, MmtacModel b, string label)
        {
            Assert(a.NewsMemory.Count == b.NewsMemory.Count, label + ": NewsMemory count mismatch");
            Assert(a.PriceMemory.Count == b.PriceMemory.Count, label + ": PriceMemory count mismatch");

            Assert(Math.Abs(a.LastPriceTimestamp - b.LastPriceTimestamp) < 1e-9,
                label + ": LastPriceTimestamp mismatch");

            for (int i = 0; i < a.NewsMemory.Count; i++)
            {
                Assert(Math.Abs(a.NewsMemory[i].AbsoluteTimestamp - b.NewsMemory[i].AbsoluteTimestamp) < 1e-9,
                    label + $": news timestamp mismatch {i}");

                Assert(a.NewsMemory[i].HiddenState != null,
                    label + $": left news hidden state null {i}");

                Assert(b.NewsMemory[i].HiddenState != null,
                    label + $": right news hidden state null {i}");

                Assert(a.NewsMemory[i].HiddenState.Length == b.NewsMemory[i].HiddenState.Length,
                    label + $": news hidden dim mismatch {i}");

                Assert(a.NewsMemory[i].HiddenState.Length == a.Config.Price.EmbeddingDim,
                    label + $": news hidden dim does not match embedding dim {i}");
            }

            for (int i = 0; i < a.PriceMemory.Count; i++)
            {
                Assert(Math.Abs(a.PriceMemory[i].AbsoluteTimestamp - b.PriceMemory[i].AbsoluteTimestamp) < 1e-9,
                    label + $": price timestamp mismatch {i}");

                Assert(a.PriceMemory[i].HiddenState != null,
                    label + $": left price hidden state null {i}");

                Assert(b.PriceMemory[i].HiddenState != null,
                    label + $": right price hidden state null {i}");

                Assert(a.PriceMemory[i].HiddenState.Length == b.PriceMemory[i].HiddenState.Length,
                    label + $": price hidden dim mismatch {i}");

                Assert(a.PriceMemory[i].HiddenState.Length == a.Config.Price.EmbeddingDim,
                    label + $": price hidden dim does not match embedding dim {i}");
            }
        }

        private (MultimodalInput[] inputs, ModelTarget[][] targets) TinyGradientData()
        {
            var inputs = new[]
            {
        new MultimodalInput
        {
            PredictionTimestamp = DateTime.UtcNow,
            PriceSequence = ConstantPriceSequence(
                5,
                0.30f,
                -0.20f,
                0.10f,
                0.05f,
                1f)
        }
    };

            var targets = new[]
            {
        ConstantTargets(
            seqLen: 5,
            close: 0.85f,
            range: 0.28f,
            quality: 0.90f,
            direction: 1,
            midDirection: 0)
    };

            return (inputs, targets);
        }

        private void AssertOneStepUpdateOpposesFiniteDifference(
            string label,
            MmtacModel model,
            MultimodalInput[] inputs,
            ModelTarget[][] targets,
            Func<MmtacModel, float> getter,
            Action<MmtacModel, float> setter)
        {
            const float eps = 1e-3f;
            const float lr = 1e-3f;

            float original = getter(model);

            setter(model, original + eps);
            float lossPlus = new MmtacTrainer(model, TC(epochs: 1))
                .ValidateAligned(inputs, targets);

            setter(model, original - eps);
            float lossMinus = new MmtacTrainer(model, TC(epochs: 1))
                .ValidateAligned(inputs, targets);

            setter(model, original);

            float finiteDiff = (lossPlus - lossMinus) / (2f * eps);

            Assert(float.IsFinite(finiteDiff),
                $"{label}: finite-difference gradient is not finite: {finiteDiff}");

            Assert(MathF.Abs(finiteDiff) > 1e-7f,
                $"{label}: finite-difference gradient is too close to zero: {finiteDiff:E6}");

            var tc = TC(lr: lr, bs: inputs.Length, epochs: 1, clip: false);

            new MmtacTrainer(model, tc)
                .Train(inputs, targets);

            float delta = getter(model) - original;

            Assert(MathF.Abs(delta) > 1e-10f,
                $"{label}: parameter did not move after one training step");

            Assert(delta * finiteDiff < 0f,
                $"{label}: update moved with the finite-difference gradient instead of against it. fd={finiteDiff:E6}, delta={delta:E6}");
        }

        private (MultimodalInput[] inputs, ModelTarget[][] targets) NextRowAlignmentData(
            int sampleCount,
            int seqLen)
        {
            var inputs = new MultimodalInput[sampleCount];
            var targets = new ModelTarget[sampleCount][];

            for (int s = 0; s < sampleCount; s++)
            {
                var ps = new float[seqLen, 5];
                var tg = new ModelTarget[seqLen];

                for (int t = 0; t < seqLen; t++)
                {
                    // Alternating target rows make current-row vs next-row alignment visibly different.
                    float close = ((s + t) % 2 == 0) ? 0.82f : 0.18f;

                    // Small sample-dependent perturbation prevents a pure two-value memorisation shortcut.
                    close += 0.02f * ((s % 3) - 1);
                    close = Clamp01(close);

                    float range = close > 0.5f ? 0.18f : 0.10f;

                    tg[t] = new ModelTarget
                    {
                        High = close + range * 0.5f,
                        Low = close - range * 0.5f,
                        Close = close,
                        Range = range,
                        Quality = close > 0.5f ? 0.90f : 0.65f,
                        Direction = close > 0.5f ? 1f : 0f,
                        MidWindowDirection = ((s + t) % 3 == 0) ? 1f : 0f
                    };
                }

                // Row t directly contains the value that should be predicted for target row t + 1.
                for (int t = 0; t < seqLen - 1; t++)
                {
                    float nextClose = tg[t + 1].Close;
                    ps[t, 0] = nextClose;
                    ps[t, 1] = 1f - nextClose;
                    ps[t, 2] = nextClose * nextClose;
                    ps[t, 3] = nextClose > 0.5f ? 1f : -1f;
                    ps[t, 4] = 1f;
                }

                // Final row is not used by TrainSimple/TrainWithPriceContext target alignment.
                ps[seqLen - 1, 0] = 0f;
                ps[seqLen - 1, 1] = 0f;
                ps[seqLen - 1, 2] = 0f;
                ps[seqLen - 1, 3] = 0f;
                ps[seqLen - 1, 4] = 1f;

                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(s),
                    PriceSequence = ps
                };

                targets[s] = tg;
            }

            return (inputs, targets);
        }

        private static (float[,] reg, float[,] range, float[,] quality, float[,] dir, float[,] midDir, float[,] conf)
            InvokeForwardWithPriceContextAndCache(
                MmtacModel model,
                MultimodalInput input,
                float[,] priceCtxHidden,
                float[] priceCtxTimes,
                MmtacForwardCache cache,
                bool isTraining)
        {
            var method = typeof(MmtacModel).GetMethod(
                "ForwardWithPriceContextAndCache",
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);

            if (method == null)
                throw new InvalidOperationException("Could not find MmtacModel.ForwardWithPriceContextAndCache via reflection.");

            var result = method.Invoke(
                model,
                new object[]
                {
            input,
            priceCtxHidden,
            priceCtxTimes,
            cache,
            isTraining,
            null
                });

            var tuple = ((float[,] reg,
                          float[,] range,
                          float[,] quality,
                          float[,] dir,
                          float[,] midDir,
                          float[,] conf))result;

            return tuple;
        }

        private (BPETokenizer tok,
                 MultimodalInput[] trainInputs,
                 ModelTarget[][] trainTargets,
                 MultimodalInput[] testInputs,
                 ModelTarget[][] testTargets)
            CompositionalMultimodalData(
                int trainN = 96,
                int testN = 32,
                int seqLen = 6,
                int seed = 12345)
        {
            string bullText = "central bank support growth demand";
            string bearText = "credit stress slowdown recession";

            var tok = new BPETokenizer();
            tok.Train(new[] { bullText, bearText }, vocabSize: 80, minFrequency: 1);

            int[] bull = tok.Encode(bullText, addSpecialTokens: true);
            int[] bear = tok.Encode(bearText, addSpecialTokens: true);

            var rng = new Random(seed);

            MultimodalInput MakeInput(float priceSignal, int newsSign, int globalSign, int idx)
            {
                return new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),

                    PriceSequence = ConstantPriceSequence(
                        seqLen,
                        priceSignal,
                        priceSignal * priceSignal,
                        MathF.Sign(priceSignal),
                        0.5f * priceSignal,
                        1f),

                    NewsStories = new[]
                    {
                new NewsStory(newsSign > 0 ? bull : bear, 0f)
            },

                    GlobalFeatures = new[]
                    {
                globalSign > 0 ? 1f : 0f,
                globalSign < 0 ? 1f : 0f,
                MathF.Abs(priceSignal)
            }
                };
            }

            ModelTarget[] MakeTarget(float priceSignal, int newsSign, int globalSign)
            {
                // All three modalities matter with similar strength.
                float score =
                    0.42f * priceSignal +
                    0.38f * newsSign +
                    0.34f * globalSign;

                float close = 0.50f + 0.24f * MathF.Tanh(score);
                float range = 0.07f + 0.04f * MathF.Abs(score);
                float quality = Clamp01(0.50f + 0.35f * MathF.Tanh(score));
                int direction = score > 0f ? 1 : 0;
                int midDirection = (0.55f * priceSignal + 0.45f * globalSign) > 0f ? 1 : 0;

                return ConstantTargets(seqLen, close, range, quality, direction, midDirection);
            }

            void Fill(int n, int offset, out MultimodalInput[] inputs, out ModelTarget[][] targets)
            {
                inputs = new MultimodalInput[n];
                targets = new ModelTarget[n][];

                for (int i = 0; i < n; i++)
                {
                    float priceSignal = (float)(rng.NextDouble() * 2.0 - 1.0);
                    int newsSign = rng.Next(2) == 0 ? -1 : 1;
                    int globalSign = rng.Next(2) == 0 ? -1 : 1;

                    inputs[i] = MakeInput(priceSignal, newsSign, globalSign, offset + i);
                    targets[i] = MakeTarget(priceSignal, newsSign, globalSign);
                }
            }

            Fill(trainN, 0, out var trainInputs, out var trainTargets);
            Fill(testN, 10_000, out var testInputs, out var testTargets);

            return (tok, trainInputs, trainTargets, testInputs, testTargets);
        }

        private void AssertAblationHurts(string modality, float baseLoss, float ablatedLoss)
        {
            bool relativeHurt = ablatedLoss > baseLoss * 1.15f;
            bool absoluteHurt = ablatedLoss > baseLoss + 0.02f;

            Assert(relativeHurt || absoluteHurt, $"{modality} ablation should hurt held-out loss. base={baseLoss:F6}, ablated={ablatedLoss:F6}");
        }

        private static (MultimodalInput[] inputs, ModelTarget[][] targets, double[] timestamps) SequentialCarryPairsData(int pairCount, int seqLen, int offset)
        {
            var inputs = new MultimodalInput[pairCount * 2];
            var targets = new ModelTarget[pairCount * 2][];
            var timestamps = new double[pairCount * 2];

            for (int pair = 0; pair < pairCount; pair++)
            {
                bool positive = pair % 2 == 0;

                float driverSignal = positive ? 1f : -1f;
                float targetClose = positive ? 0.80f : 0.20f;
                int direction = positive ? 1 : 0;

                int driverIdx = pair * 2;
                int carryIdx = driverIdx + 1;

                inputs[driverIdx] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(offset + driverIdx),
                    PriceSequence = ConstantPriceSequence(
                        seqLen,
                        driverSignal,
                        driverSignal * driverSignal,
                        MathF.Sign(driverSignal),
                        0.5f * driverSignal,
                        1f)
                };

                targets[driverIdx] = ConstantTargets(
                    seqLen,
                    close: targetClose,
                    range: 0.12f,
                    quality: 0.90f,
                    direction: direction,
                    midDirection: direction);

                // The carry sample is neutral. Its target depends on the immediately previous driver.
                inputs[carryIdx] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(offset + carryIdx),
                    PriceSequence = ConstantPriceSequence(
                        seqLen,
                        0f,
                        0f,
                        0f,
                        0f,
                        1f)
                };

                targets[carryIdx] = ConstantTargets(
                    seqLen,
                    close: targetClose,
                    range: 0.12f,
                    quality: 0.90f,
                    direction: direction,
                    midDirection: direction);

                timestamps[driverIdx] = (offset + driverIdx) * 10.0;
                timestamps[carryIdx] = (offset + carryIdx) * 10.0;
            }

            return (inputs, targets, timestamps);
        }

        private float AverageCarryCloseErrorWithMemory(MmtacModel model, MultimodalInput[] inputs, ModelTarget[][] targets, double[] timestamps, int maxPriceMemory)
        {
            float total = 0f;
            int count = 0;

            for (int i = 0; i < inputs.Length; i += 2)
            {
                model.ClearAllMemory();

                _ = model.PredictWithMemory(
                    inputs[i],
                    timestamps[i],
                    timeUnitsPerPosition: 1.0,
                    maxNewsMemorySize: 0,
                    maxPriceMemorySize: maxPriceMemory);

                var carryPred = model.PredictWithMemory(
                    inputs[i + 1],
                    timestamps[i + 1],
                    timeUnitsPerPosition: 1.0,
                    maxNewsMemorySize: 0,
                    maxPriceMemorySize: maxPriceMemory);

                float targetClose = targets[i + 1][targets[i + 1].Length - 1].Close;

                total += MathF.Abs(carryPred.Close - targetClose);
                count++;
            }

            model.ClearAllMemory();

            return total / Math.Max(1, count);
        }

        private float AverageCarryCloseErrorStateless(MmtacModel model, MultimodalInput[] inputs, ModelTarget[][] targets)
        {
            float total = 0f;
            int count = 0;

            for (int i = 1; i < inputs.Length; i += 2)
            {
                var pred = model.PredictNext(inputs[i]);
                float targetClose = targets[i][targets[i].Length - 1].Close;

                total += MathF.Abs(pred.Close - targetClose);
                count++;
            }

            return total / Math.Max(1, count);
        }
    }
}
