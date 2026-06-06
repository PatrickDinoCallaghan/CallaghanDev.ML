using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.PriceTAC;
using CallaghanDev.ML.Transformers.TACAMT;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    /// <summary>
    /// Correctness-focused tests for PriceTAC.
    ///
    /// PriceTAC is the price-only MMTAC variant: it keeps the price decoder,
    /// optional global feature token, price-memory context, learned content-aware
    /// time decay, output heads, price-context training, rolling memory, validation,
    /// and save/load paths, but has no news/text/tokenizer surface.
    /// </summary>
    internal sealed class PriceTacTests : TestBase
    {
        private readonly AccelerationType _accelerationType;
        private readonly int _deviceId;

        public PriceTacTests(AccelerationType accelerationType, int deviceId)
        {
            _accelerationType = accelerationType;
            _deviceId = deviceId;
        }

        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * PriceTAC (Price-only Market Transformer with TAC decay)");
        }

        private (Action test, string name)[] Tests() => new (Action, string)[]
        {
            // Core fixes / invariants
            (Test_Forward_OutputHeadCoherentOhlcRange_AllTimesteps, "Fix: output head guarantees coherent OHLC/range"),
            (Test_CrossAttention_ContextRowsArePermutationEquivariant, "Fix: cross-attention has no context-row RoPE ordering leak"),
            (Test_CrossAttention_DecayDisabledStillMasksFutureContext, "Fix: decay disabled still masks future context"),
            (Test_DecayNetwork_KeyRefinementMasksFutureKeys, "Fix: decay key-refinement masks future keys"),
            (Test_ForwardWithPriceContextSlice_MatchesExplicitSlice, "Fix: sliced price-context forward matches explicit slice"),
            (Test_Timestamp_WindowEndPredictionMatchesManualWindowStart, "Fix: window-end timestamp overload matches manual window-start conversion"),
            (Test_Train_InvalidSampleSameBatchDiscardsBatchWithoutDirtyUpdate, "Fix: invalid sample does not contaminate batch gradients"),
            (Test_ClearAllMemory_ResetsLastPriceTimestamp, "Fix: ClearAllMemory resets LastPriceTimestamp"),

            // Construction / configuration
            (Test_Config_ValidPresets, "Config: presets validate"),
            (Test_Construction_CoreShapes, "Construction: core tensor shapes are correct"),
            (Test_Construction_OptionalHeadAndGlobalShapes, "Construction: confidence/global optional shapes are correct"),
            (Test_Construction_RuntimePruningConfigCopied, "Construction: runtime pruning config is copied from config"),

            // Forward contracts
            (Test_Forward_PredictNextFiniteAndConstrained_ConfidenceDisabled, "Forward: PredictNext finite and constrained, confidence disabled"),
            (Test_Forward_PredictNextFiniteAndConstrained_ConfidenceEnabled, "Forward: PredictNext finite and constrained, confidence enabled"),
            (Test_Forward_NullGlobalFeaturesWork, "Forward: null global features work"),
            (Test_Forward_SeqLenOneWorks, "Forward: single timestep works"),
            (Test_Forward_AllHeadShapesMatchSequenceLength, "Forward: all head shapes match sequence length"),
            (Test_Forward_DeterministicAndDoesNotMutateMemory, "Forward: deterministic and does not mutate memory"),
            (Test_Forward_GlobalFeatureChangesPrediction, "Forward: global context changes prediction"),
            (Test_Forward_WithCacheMatchesForward_AllHeads, "ForwardWithCache: matches Forward for all heads"),
            (Test_Forward_Causal_NoFutureLeakage_PriceOnly, "Causal: future price rows do not affect previous outputs"),
            (Test_Forward_Causal_NoFutureLeakage_WithGlobal, "Causal: future price rows do not affect previous outputs with global context"),
            (Test_Forward_WrongPriceFeatureDimThrowsArgumentException, "Forward: wrong price feature dimension throws ArgumentException"),
            (Test_Forward_WrongGlobalFeatureDimThrowsArgumentException, "Forward: short global features throw ArgumentException"),
            (Test_Forward_GlobalFeatureExtraDimThrowsArgumentException, "Forward: extra global features throw ArgumentException"),

            // Decay/context wiring
            (Test_Decay_TimeValueReachesDecoder, "Decay: changing context time changes decoder hidden state"),
            (Test_Decay_ContextContentReachesDecoder, "Decay: changing context content changes decoder output"),
            (Test_Decay_DisabledIgnoresContextTimes, "Decay: disabled decay ignores context times"),

            // Public validation / argument contracts
            (Test_PublicMethods_LengthMismatchThrow, "Validation: public methods throw on mismatched array lengths"),
            (Test_PublicMethods_ZeroTimeUnitsThrow, "Validation: zero timeUnitsPerPosition throws"),
            (Test_Validate_ReturnsFiniteNonNegative, "Validate: returns finite non-negative values"),
            (Test_Validate_DoesNotMutateWeights, "Validate: does not mutate weights"),
            (Test_ValidateSequential_DoesNotMutateWeightsAndClearsMemory, "ValidateSequential: does not mutate weights and clears memory"),
            (Test_ValidateSequential_OrderInvariant, "ValidateSequential: timestamp order invariant"),

            // Training mechanics
            (Test_Train_LossDecreases_OnDeterministicPriceData, "Train: deterministic price loss decreases"),
            (Test_Train_SingleSampleOverfit_LossDecreases, "Train: single sample overfit decreases loss"),
            (Test_Train_ParameterUpdateCoverage_WithPriceContext, "Train: key price/context parameter groups update"),
            (Test_Train_GlobalProjectionUpdated, "Train: global projection updates"),
            (Test_Train_ConfidenceHeadUpdated, "Train: confidence head updates when enabled"),
            (Test_Train_DirectionLossWeightControlsDirectionHeadGradient, "Train: direction loss weight controls direction-head gradient"),
            (Test_Train_ConfidenceLossWeightZeroDisablesConfidenceGradient, "Train: confidence loss weight 0 disables confidence-head gradient"),
            (Test_Train_GradientClippingBoundsUpdate, "Train: gradient clipping bounds update magnitude"),
            (Test_Train_HighLearningRateWithClippingStaysFinite, "Train: high LR with clipping stays finite"),
            (Test_Train_PriceContextPathRunsFinite, "Train: price-context branch runs and stays finite"),
            (Test_Train_DecayNetworkUpdatedWithPriceContext, "Train: decay network receives gradients when price context exists"),
            (Test_Train_ZeroLearningRateDoesNotChangeWeights, "Train: zero learning rate does not change weights"),

            // Deterministic signal smoke tests
            (Test_Signal_PriceHeldOutLossImproves, "Signal: held-out deterministic price mapping improves"),
            (Test_Signal_GlobalSeparationAfterTraining, "Signal: trained global signal separates predictions"),
            (Test_Signal_DirectionAndMidDirectionCanLearnOppositeLabels, "Signal: Direction and MidDirection can learn opposite labels"),
            (Test_Signal_RangeHeadLearnsRangeTarget, "Signal: Range head learns range target"),

            // Rolling price memory
            (Test_Memory_InitiallyEmptyAndClearMethods, "Memory: starts empty and clear methods work"),
            (Test_Memory_PredictWithMemoryExactGrowthAndTimestamps, "Memory: PredictWithMemory exact growth and timestamps"),
            (Test_Memory_CustomTimeUnitsScaleTimestamps, "Memory: custom timeUnitsPerPosition scales timestamps"),
            (Test_Memory_ZeroBudgetKeepsPriceMemoryEmpty, "Memory: zero budget keeps price memory empty"),
            (Test_Memory_PrunesToLimit, "Memory: pruning respects max size"),
            (Test_Memory_AttentionPruningKeepsHighScore, "Memory: attention pruning keeps high-score entries"),
            (Test_Memory_AttentionScoresUpdated, "Memory: attention scores/query counts update"),
            (Test_Memory_GlobalTokenNotPersistedAsMemory, "Memory: global token affects output but is not persisted"),
            (Test_Memory_PriceOnlyWorksWithExistingPriceMemory, "Memory: existing rolling price memory works"),

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
            (Test_Stability_NoNaNAfterManyEpochs, "Stability: no NaN after many epochs"),
        };

        // ---------------------------------------------------------------------
        // Core fixes / invariants
        // ---------------------------------------------------------------------

        private void Test_Forward_OutputHeadCoherentOhlcRange_AllTimesteps()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, useConf: true, globalDim: 3);
            var model = new PriceTacModel(cfg, new Random(42));
            var input = MakeInput(seqLen: 10, globalDim: 3);

            var (reg, range, quality, dir, midDir, conf) = model.Forward(input);

            AssertAllFinite(reg, "regression");
            AssertAllFinite(range, "range");
            AssertAllFinite(quality, "quality");
            AssertAllFinite(dir, "direction");
            AssertAllFinite(midDir, "midDirection");
            AssertAllFinite(conf, "confidence");

            const float tol = 1e-5f;
            for (int t = 0; t < reg.GetLength(0); t++)
            {
                float high = reg[t, 0];
                float low = reg[t, 1];
                float close = reg[t, 2];
                float predictedRange = range[t, 0];

                Assert(high + tol >= close, $"High must be >= Close at t={t}. high={high}, close={close}");
                Assert(close + tol >= low, $"Close must be >= Low at t={t}. close={close}, low={low}");
                Assert(predictedRange >= 0f, $"Range must be non-negative at t={t}. range={predictedRange}");

                float ohlcRange = high - low;
                Assert(MathF.Abs(ohlcRange - predictedRange) <= 2e-5f,
                    $"Range head must equal High-Low at t={t}. high-low={ohlcRange}, range={predictedRange}");
            }
        }

        private void Test_CrossAttention_ContextRowsArePermutationEquivariant()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, decayEnabled: true);
            var model = new PriceTacModel(cfg, new Random(42));
            var price = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);

            var ctxA = new float[2, cfg.Price.EmbeddingDim];
            var ctxB = new float[2, cfg.Price.EmbeddingDim];

            for (int d = 0; d < cfg.Price.EmbeddingDim; d++)
            {
                ctxA[0, d] = 0.07f * (d + 1);
                ctxA[1, d] = -0.05f * (d + 1);
                ctxB[0, d] = ctxA[1, d];
                ctxB[1, d] = ctxA[0, d];
            }

            var outA = model.ForwardPriceDecoderWithCache(
                price,
                0,
                price.GetLength(0),
                ctxA,
                new[] { -2f, -1f },
                new PriceTacForwardCache(cfg.Price.NumLayers),
                isTraining: false);

            var outB = model.ForwardPriceDecoderWithCache(
                price,
                0,
                price.GetLength(0),
                ctxB,
                new[] { -1f, -2f },
                new PriceTacForwardCache(cfg.Price.NumLayers),
                isTraining: false);

            AssertMatricesClose(outA, outB, 1e-5f,
                "Permuting context rows and their times together should not change output. If this fails, cross-attention is still using row-order positional encoding.");
        }

        private void Test_CrossAttention_DecayDisabledStillMasksFutureContext()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, decayEnabled: false);
            var model = new PriceTacModel(cfg, new Random(42));
            var price = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);

            var ctxA = new float[2, cfg.Price.EmbeddingDim];
            var ctxB = new float[2, cfg.Price.EmbeddingDim];

            for (int d = 0; d < cfg.Price.EmbeddingDim; d++)
            {
                ctxA[0, d] = 0.10f * (d + 1);
                ctxB[0, d] = ctxA[0, d];
                ctxA[1, d] = 100f + d;
                ctxB[1, d] = -100f - d;
            }

            var times = new[] { -1f, 10f };
            var cacheA = new PriceTacForwardCache(cfg.Price.NumLayers);
            var outA = model.ForwardPriceDecoderWithCache(price, 0, price.GetLength(0), ctxA, times, cacheA, isTraining: false);
            var outB = model.ForwardPriceDecoderWithCache(price, 0, price.GetLength(0), ctxB, times, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);

            AssertMatricesClose(outA, outB, 1e-6f,
                "Changing a future context row must not affect output when decay is disabled.");

            Assert(cacheA.PriceBlockCaches[0].TimeDiffs != null, "TimeDiffs should still be cached when decay is disabled.");
            Assert(cacheA.PriceBlockCaches[0].DecayCache == null, "DecayCache should be null when Decay.Enabled=false.");

            var weights = cacheA.PriceBlockCaches[0].CrossAttentionWeights;
            Assert(weights != null, "Cross-attention weights should be cached.");

            for (int h = 0; h < weights.Length; h++)
            {
                for (int q = 0; q < weights[h].GetLength(0); q++)
                {
                    Assert(weights[h][q, 1] == 0f,
                        $"Future context row should be masked. head={h}, query={q}, weight={weights[h][q, 1]}");
                }
            }
        }

        private void Test_DecayNetwork_KeyRefinementMasksFutureKeys()
        {
            var rng = new Random(42);
            var accel = new AccelerationCPU();
            var net = new ContentAwareDecayNetwork(
                numHeads: 2,
                contentDim: 16,
                projectionDim: 8,
                hiddenDim: 16,
                random: rng,
                memAttnDropout: 0f,
                mlpDropout: 0f);

            var query = RandomMatrix(2, 16, new Random(201), 0.2f);
            var key = RandomMatrix(3, 16, new Random(202), 0.2f);
            var keyTimes = new[] { -2f, -1f, 1f };
            var timeDiffs = accel.ComputeTimeDiffMatrix(query.GetLength(0), keyTimes);

            var (_, cache) = accel.ContentAwareDecayForward(query, key, timeDiffs, keyTimes, net, isTraining: false);

            for (int h = 0; h < net.NumHeads; h++)
            {
                Assert(cache.MemAttnWeights[h, 0, 1] == 0f, "Key t=-2 must not refine itself using future key t=-1.");
                Assert(cache.MemAttnWeights[h, 0, 2] == 0f, "Key t=-2 must not refine itself using future key t=1.");
                Assert(cache.MemAttnWeights[h, 1, 2] == 0f, "Key t=-1 must not refine itself using future key t=1.");
                Assert(cache.MemAttnWeights[h, 0, 0] > 0f, "A key should still be allowed to attend to itself.");
                Assert(cache.MemAttnWeights[h, 2, 0] > 0f &&
                       cache.MemAttnWeights[h, 2, 1] > 0f &&
                       cache.MemAttnWeights[h, 2, 2] > 0f,
                    "The newest key should be able to refine from past and current keys.");
            }
        }

        private void Test_ForwardWithPriceContextSlice_MatchesExplicitSlice()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, priceSeqLen: 16);
            var model = new PriceTacModel(cfg, new Random(42));
            var fullPrice = RandomMatrix(12, 5, new Random(300), 0.2f);
            int rowStart = 5;
            int rowCount = 4;

            var priceCtx = model.EncodePriceHistory(fullPrice, 0, 3);
            var priceCtxTimes = new[] { -3f, -2f, -1f };

            var fullInput = new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = fullPrice
            };

            var explicitSliceInput = new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = SliceRows(fullPrice, rowStart, rowStart + rowCount)
            };

            var sliced = InvokeForwardWithPriceContextSliceAndCache(
                model,
                fullInput,
                rowStart,
                rowCount,
                priceCtx,
                priceCtxTimes,
                new PriceTacForwardCache(cfg.Price.NumLayers),
                isTraining: false);

            var explicit_ = InvokeForwardWithPriceContextAndCache(
                model,
                explicitSliceInput,
                priceCtx,
                priceCtxTimes,
                new PriceTacForwardCache(cfg.Price.NumLayers),
                isTraining: false);

            AssertMatricesClose(explicit_.regression, sliced.regression, 1e-5f, "rowStart regression mismatch");
            AssertMatricesClose(explicit_.range, sliced.range, 1e-5f, "rowStart range mismatch");
            AssertMatricesClose(explicit_.quality, sliced.quality, 1e-5f, "rowStart quality mismatch");
            AssertMatricesClose(explicit_.direction, sliced.direction, 1e-5f, "rowStart direction mismatch");
            AssertMatricesClose(explicit_.midDirection, sliced.midDirection, 1e-5f, "rowStart midDirection mismatch");
        }

        private void Test_Timestamp_WindowEndPredictionMatchesManualWindowStart()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32);
            var input = MakeInput(seqLen: 5);
            double timeUnits = 2.0;
            double windowStart = 100.0;
            double windowEnd = 108.0;

            double converted = PriceTacModel.WindowStartTimestampFromWindowEnd(windowEnd, input.PriceSequence.GetLength(0), timeUnits);
            Assert(Math.Abs(converted - windowStart) < 1e-9,
                $"Window-start conversion mismatch. expected={windowStart}, actual={converted}");

            var baseModel = new PriceTacModel(cfg, new Random(42));
            string dir = TmpDir();

            try
            {
                baseModel.Save(dir);
                var modelA = PriceTacModel.Load(dir, _accelerationType, _deviceId);
                var modelB = PriceTacModel.Load(dir, _accelerationType, _deviceId);

                var fromEnd = modelA.PredictWithMemoryAtWindowEnd(input, windowEnd, timeUnitsPerPosition: timeUnits, maxPriceMemorySize: 10);
                var fromStart = modelB.PredictWithMemory(input, windowStart, timeUnitsPerPosition: timeUnits, maxPriceMemorySize: 10);

                AssertPredictionsClose(fromStart, fromEnd, 1e-6f,
                    "PredictWithMemoryAtWindowEnd should match PredictWithMemory using manually converted window-start timestamp.");
                AssertMemoryEquivalent(modelB, modelA, "window-end prediction memory state");
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        private void Test_Train_InvalidSampleSameBatchDiscardsBatchWithoutDirtyUpdate()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, priceContextEnabled: false);
            var model = new PriceTacModel(cfg, new Random(42));
            var before = SnapshotWeights(model);

            var valid = MakeInput(seqLen: 6);
            var invalid = new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandomMatrix(6, cfg.Price.InputFeatureDim + 1, new Random(9), 0.2f)
            };

            var validTarget = ConstantTargets(6, close: 0.55f, range: 0.12f, quality: 0.9f, direction: 1, midDirection: 1);
            var invalidTarget = ConstantTargets(6, close: 0.20f, range: 0.08f, quality: 0.6f, direction: 0, midDirection: 0);

            new PriceTacTrainer(model, TC(lr: 0.01f, bs: 2, epochs: 1, clip: false))
                .Train(new[] { valid, invalid }, new[] { validTarget, invalidTarget });

            AssertWeightsUnchanged(before, model,
                "A valid sample followed by an invalid sample in the same batch must not leave dirty partial gradients or update weights.");
        }

        private void Test_ClearAllMemory_ResetsLastPriceTimestamp()
        {
            var model = new PriceTacModel(Cfg(), new Random(42));
            model.PredictWithMemory(MakeInput(seqLen: 5), currentAbsoluteTimestamp: 100.0, timeUnitsPerPosition: 2.0, maxPriceMemorySize: 10);

            Assert(model.PriceMemory.Count > 0, "Precondition failed: PriceMemory did not populate.");
            Assert(model.LastPriceTimestamp != 0.0, "Precondition failed: LastPriceTimestamp did not update.");

            model.ClearAllMemory();

            Assert(model.PriceMemory.Count == 0, "ClearAllMemory should clear PriceMemory.");
            Assert(model.LastPriceTimestamp == 0.0, "ClearAllMemory should reset LastPriceTimestamp.");
        }

        // ---------------------------------------------------------------------
        // Construction / config
        // ---------------------------------------------------------------------

        private void Test_Config_ValidPresets()
        {
            Cfg().Validate();
            PriceTacConfig.Small(priceFeatureDim: 5).Validate();
            PriceTacConfig.Standard(priceFeatureDim: 5, globalDim: 8).Validate();
        }

        private void Test_Construction_CoreShapes()
        {
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceFeatures: 7), new Random(42));

            Assert(m.PriceInputProjection.GetLength(0) == 24, "PriceInputProjection rows");
            Assert(m.PriceInputProjection.GetLength(1) == 7, "PriceInputProjection cols");
            Assert(m.PriceInputProjectionBias.Length == 24, "PriceInputProjectionBias length");
            Assert(m.RegressionProjection.GetLength(0) == PriceTacOutputConfig.RegressionOutputCount, "RegressionProjection rows");
            Assert(m.RegressionProjection.GetLength(1) == 24, "RegressionProjection cols");
            Assert(m.RangeProjection.GetLength(0) == 1 && m.RangeProjection.GetLength(1) == 24, "RangeProjection shape");
            Assert(m.QualityProjection.GetLength(0) == 1 && m.QualityProjection.GetLength(1) == 24, "QualityProjection shape");
            Assert(m.DirectionProjection.GetLength(0) == 1 && m.DirectionProjection.GetLength(1) == 24, "DirectionProjection shape");
            Assert(m.MidDirectionProjection.GetLength(0) == 1 && m.MidDirectionProjection.GetLength(1) == 24, "MidDirectionProjection shape");
            Assert(m.ContextTypeEmbedding.GetLength(0) == PriceTacConfig.ContextTypeCount, "ContextTypeEmbedding rows");
            Assert(m.ContextTypeEmbedding.GetLength(1) == 24, "ContextTypeEmbedding cols");
            Assert(m.PriceBlocks.Length == 2, "PriceBlocks length");
            Assert(m.PriceBlocks.All(b => b.SelfAttention.WQ.GetLength(0) == 24 && b.SelfAttention.WQ.GetLength(1) == 24), "PriceBlock SelfAttention.WQ shape");
        }

        private void Test_Construction_OptionalHeadAndGlobalShapes()
        {
            var disabled = new PriceTacModel(Cfg(useConf: false), new Random(42));
            Assert(disabled.ConfidenceProjection == null, "ConfidenceProjection should be null when disabled");
            Assert(disabled.ConfidenceBias == null, "ConfidenceBias should be null when disabled");
            Assert(disabled.GlobalFeatureProjection == null, "GlobalFeatureProjection should be null when GlobalFeatureDim=0");

            var enabled = new PriceTacModel(Cfg(embDim: 16, useConf: true, globalDim: 4), new Random(42));
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

            var m = new PriceTacModel(cfg, new Random(42));

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
            var m = new PriceTacModel(Cfg(useConf: false), new Random(42));
            var p = m.PredictNext(MakeInput(seqLen: 8));
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
            Assert(p.Confidence == 1f, "Confidence should be 1 when confidence head is disabled");
        }

        private void Test_Forward_PredictNextFiniteAndConstrained_ConfidenceEnabled()
        {
            var m = new PriceTacModel(Cfg(useConf: true), new Random(42));
            var p = m.PredictNext(MakeInput(seqLen: 8));
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: true);
        }

        private void Test_Forward_NullGlobalFeaturesWork()
        {
            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            AssertPredictionFiniteAndConstrained(m.PredictNext(MakeInput(globalDim: 0)), expectConfidenceHead: false);
        }

        private void Test_Forward_SeqLenOneWorks()
        {
            var m = new PriceTacModel(Cfg(priceSeqLen: 4), new Random(42));
            var p = m.PredictNext(new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandomMatrix(1, 5, new Random(1), 0.1f)
            });
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
        }

        private void Test_Forward_AllHeadShapesMatchSequenceLength()
        {
            int seqLen = 9;
            var m = new PriceTacModel(Cfg(useConf: true), new Random(42));
            var input = MakeInput(seqLen: seqLen);
            var (reg, range, quality, dir, midDir, conf) = m.Forward(input);

            AssertMatrixShape(reg, seqLen, 3, nameof(reg));
            AssertMatrixShape(range, seqLen, 1, nameof(range));
            AssertMatrixShape(quality, seqLen, 1, nameof(quality));
            AssertMatrixShape(dir, seqLen, 1, nameof(dir));
            AssertMatrixShape(midDir, seqLen, 1, nameof(midDir));
            AssertMatrixShape(conf, seqLen, 1, nameof(conf));

            var pred = m.PredictNext(input);
            int last = seqLen - 1;
            Assert(MathF.Abs(pred.High - reg[last, 0]) < 1e-6f, "PredictNext.High must equal last regression timestep");
            Assert(MathF.Abs(pred.Low - reg[last, 1]) < 1e-6f, "PredictNext.Low must equal last regression timestep");
            Assert(MathF.Abs(pred.Close - reg[last, 2]) < 1e-6f, "PredictNext.Close must equal last regression timestep");
            Assert(MathF.Abs(pred.Range - range[last, 0]) < 1e-6f, "PredictNext.Range must equal last range timestep");
            Assert(MathF.Abs(pred.Quality - quality[last, 0]) < 1e-6f, "PredictNext.Quality must equal last quality timestep");
            Assert(MathF.Abs(pred.DirectionProb - dir[last, 0]) < 1e-6f, "PredictNext.Direction must equal last direction timestep");
            Assert(MathF.Abs(pred.MidWindowDirectionProb - midDir[last, 0]) < 1e-6f, "PredictNext.MidDirection must equal last mid-direction timestep");
            Assert(MathF.Abs(pred.Confidence - conf[last, 0]) < 1e-6f, "PredictNext.Confidence must equal last confidence timestep");
        }

        private void Test_Forward_DeterministicAndDoesNotMutateMemory()
        {
            var input = RandomData(n: 1, withGlobal: true, globalDim: 3).inputs[0];
            var m = new PriceTacModel(Cfg(useConf: true, globalDim: 3), new Random(42));

            int memoryBefore = m.PriceMemory.Count;
            double lastBefore = m.LastPriceTimestamp;

            var p1 = m.PredictNext(input);
            var p2 = m.PredictNext(input);

            AssertPredictionsClose(p1, p2, 0f, "PredictNext must be exactly deterministic for repeated calls");
            Assert(m.PriceMemory.Count == memoryBefore, "PredictNext mutated PriceMemory");
            Assert(m.LastPriceTimestamp == lastBefore, "PredictNext mutated LastPriceTimestamp");

            var f1 = m.Forward(input);
            var f2 = m.Forward(input);
            AssertMatricesClose(f1.regression, f2.regression, 0f, "Forward regression deterministic");
            AssertMatricesClose(f1.range, f2.range, 0f, "Forward range deterministic");
            AssertMatricesClose(f1.quality, f2.quality, 0f, "Forward quality deterministic");
            AssertMatricesClose(f1.direction, f2.direction, 0f, "Forward direction deterministic");
            AssertMatricesClose(f1.midDirection, f2.midDirection, 0f, "Forward midDirection deterministic");
            AssertMatricesClose(f1.confidence, f2.confidence, 0f, "Forward confidence deterministic");
        }

        private void Test_Forward_GlobalFeatureChangesPrediction()
        {
            var cfg = Cfg(globalDim: 3);
            var m = new PriceTacModel(cfg, new Random(42));
            var ps = ConstantPriceSequence(8, 0.1f, 0.2f, -0.1f, 0.05f, 1f);

            var basePred = m.PredictNext(new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps });
            var globalPred = m.PredictNext(new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ps,
                GlobalFeatures = new[] { 1f, 0f, 0.5f }
            });

            Assert(PredictionsDiffer(basePred, globalPred, 1e-7f), "Adding global features should change at least one prediction field");
        }

        private void Test_Forward_WithCacheMatchesForward_AllHeads()
        {
            var cfg = Cfg(embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, useConf: true, globalDim: 3);
            var m = new PriceTacModel(cfg, new Random(42));

            var input = new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandomMatrix(7, 5, new Random(11), 0.3f),
                GlobalFeatures = new[] { 0.2f, 0.7f, 0.1f }
            };

            var expected = m.Forward(input);
            var actual = InvokeForwardWithCache(m, input, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);

            AssertMatricesClose(expected.regression, actual.regression, 1e-5f, "regression mismatch");
            AssertMatricesClose(expected.range, actual.range, 1e-5f, "range mismatch");
            AssertMatricesClose(expected.quality, actual.quality, 1e-5f, "quality mismatch");
            AssertMatricesClose(expected.direction, actual.direction, 1e-5f, "direction mismatch");
            AssertMatricesClose(expected.midDirection, actual.midDirection, 1e-5f, "midDirection mismatch");
            AssertMatricesClose(expected.confidence, actual.confidence, 1e-5f, "confidence mismatch");
        }

        private void Test_Forward_Causal_NoFutureLeakage_PriceOnly()
        {
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 16, useConf: true), new Random(42));
            AssertNoFutureLeakage(m, globals: null);
        }

        private void Test_Forward_Causal_NoFutureLeakage_WithGlobal()
        {
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 16, useConf: true, globalDim: 3), new Random(42));
            AssertNoFutureLeakage(m, globals: new[] { 0.25f, 0.75f, 0.5f });
        }

        private void Test_Forward_WrongPriceFeatureDimThrowsArgumentException()
        {
            var m = new PriceTacModel(Cfg(priceFeatures: 5), new Random(42));
            var bad = new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = new float[8, 6] };
            AssertThrows<ArgumentException>(() => m.PredictNext(bad), "wrong price feature dimension should throw ArgumentException");
        }

        private void Test_Forward_WrongGlobalFeatureDimThrowsArgumentException()
        {
            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            var bad = new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                GlobalFeatures = new[] { 1f, 0f }
            };
            AssertThrows<ArgumentException>(() => m.PredictNext(bad), "short global feature vector should throw ArgumentException");
        }

        private void Test_Forward_GlobalFeatureExtraDimThrowsArgumentException()
        {
            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            var bad = new PriceTacInput
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
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, decayEnabled: true);
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var m = new PriceTacModel(cfg, new Random(42));
            var ps = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);
            var ctx = new float[2, cfg.Price.EmbeddingDim];

            for (int d = 0; d < cfg.Price.EmbeddingDim; d++)
            {
                ctx[0, d] = 0.10f * (d + 1);
                ctx[1, d] = -0.07f * (d + 1);
            }

            var hFirstRecent = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctx, new[] { 0f, -1000f }, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);
            var hSecondRecent = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctx, new[] { -1000f, 0f }, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);

            Assert(MaxAbsDiff(hFirstRecent, hSecondRecent) > 1e-7f,
                "Changing relative context times should affect decoder hidden state when decay is enabled.");
        }

        private void Test_Decay_ContextContentReachesDecoder()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32);
            var m = new PriceTacModel(cfg, new Random(42));
            var ps = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);
            var ctxA = new float[1, cfg.Price.EmbeddingDim];
            var ctxB = new float[1, cfg.Price.EmbeddingDim];
            for (int d = 0; d < cfg.Price.EmbeddingDim; d++)
            {
                ctxA[0, d] = 0.1f * (d + 1);
                ctxB[0, d] = -ctxA[0, d];
            }

            var hA = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctxA, new[] { -2f }, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);
            var hB = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctxB, new[] { -2f }, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);

            Assert(MaxAbsDiff(hA, hB) > 1e-7f, "Changing context content should affect decoder hidden state");
        }

        private void Test_Decay_DisabledIgnoresContextTimes()
        {
            var cfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, decayEnabled: false);
            var m = new PriceTacModel(cfg, new Random(42));
            var ps = ConstantPriceSequence(6, 0.1f, 0.2f, -0.1f, 0.05f, 1f);
            var ctx = new float[1, cfg.Price.EmbeddingDim];
            for (int d = 0; d < cfg.Price.EmbeddingDim; d++) ctx[0, d] = 0.1f * (d + 1);

            var h1 = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctx, new[] { 0f }, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);
            var h2 = m.ForwardPriceDecoderWithCache(ps, 0, ps.GetLength(0), ctx, new[] { -1000f }, new PriceTacForwardCache(cfg.Price.NumLayers), isTraining: false);

            Assert(MaxAbsDiff(h1, h2) < 1e-6f, "When Decay.Enabled=false, context times should not change the output");
        }

        // ---------------------------------------------------------------------
        // Validation / argument contracts
        // ---------------------------------------------------------------------

        private void Test_PublicMethods_LengthMismatchThrow()
        {
            var (inputs, targets) = RandomData(n: 3, seqLen: 6);
            var m = new PriceTacModel(Cfg(), new Random(42));
            var tr = new PriceTacTrainer(m, TC(epochs: 1));
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
            var (inputs, targets) = RandomData(n: 2, seqLen: 6);
            var m = new PriceTacModel(Cfg(), new Random(42));
            var tr = new PriceTacTrainer(m, TC(epochs: 1));
            var ts = new[] { 10d, 20d };

            AssertThrows<ArgumentOutOfRangeException>(() => m.PredictWithMemory(inputs[0], 10.0, timeUnitsPerPosition: 0.0), "PredictWithMemory should reject zero timeUnitsPerPosition");
            AssertThrows<ArgumentOutOfRangeException>(() => tr.TrainSequential(inputs, targets, ts, timeUnitsPerPosition: 0.0), "TrainSequential should reject zero timeUnitsPerPosition");
            AssertThrows<ArgumentOutOfRangeException>(() => tr.ValidateSequential(inputs, targets, ts, timeUnitsPerPosition: 0.0), "ValidateSequential should reject zero timeUnitsPerPosition");
            AssertThrows<ArgumentOutOfRangeException>(() => tr.ValidateSequentialAligned(inputs, targets, ts, timeUnitsPerPosition: 0.0), "ValidateSequentialAligned should reject zero timeUnitsPerPosition");
        }

        private void Test_Validate_ReturnsFiniteNonNegative()
        {
            var (inputs, targets) = RandomData(n: 5, seqLen: 6, withGlobal: true, globalDim: 3);
            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            var tr = new PriceTacTrainer(m, TC(epochs: 1));
            var ts = Enumerable.Range(0, inputs.Length).Select(i => (double)i).ToArray();

            float legacy = tr.Validate(inputs, targets);
            float aligned = tr.ValidateAligned(inputs, targets);
            float seq = tr.ValidateSequential(inputs, targets, ts);
            float seqAligned = tr.ValidateSequentialAligned(inputs, targets, ts);

            Assert(float.IsFinite(legacy) && legacy >= 0f, $"Validate invalid: {legacy}");
            Assert(float.IsFinite(aligned) && aligned >= 0f, $"ValidateAligned invalid: {aligned}");
            Assert(float.IsFinite(seq) && seq >= 0f, $"ValidateSequential invalid: {seq}");
            Assert(float.IsFinite(seqAligned) && seqAligned >= 0f, $"ValidateSequentialAligned invalid: {seqAligned}");
        }

        private void Test_Validate_DoesNotMutateWeights()
        {
            var (inputs, targets) = RandomData(n: 4, seqLen: 6, withGlobal: true, globalDim: 3);
            var m = new PriceTacModel(Cfg(globalDim: 3, useConf: true), new Random(42));
            var snap = SnapshotWeights(m);

            float loss = new PriceTacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(float.IsFinite(loss), $"non-finite validation loss {loss}");
            AssertWeightsUnchanged(snap, m, "Validate");
        }

        private void Test_ValidateSequential_DoesNotMutateWeightsAndClearsMemory()
        {
            var (inputs, targets) = RandomData(n: 4, seqLen: 6, withGlobal: true, globalDim: 3);
            var ts = new[] { 30d, 10d, 40d, 20d };
            var cfg = Cfg(globalDim: 3, useConf: true);
            cfg.Runtime.AccelerationType = AccelerationType.CPU;
            var m = new PriceTacModel(cfg, new Random(42));
            m.PredictWithMemory(inputs[0], 5.0);
            Assert(m.PriceMemory.Count > 0, "precondition failed: memory did not populate");

            var snap = SnapshotWeights(m);
            float loss = new PriceTacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);
            Assert(float.IsFinite(loss), $"non-finite validation loss {loss}");
            AssertWeightsUnchanged(snap, m, "ValidateSequential");
            Assert(m.PriceMemory.Count == 0, "ValidateSequential should clear PriceMemory on exit");
        }

        private void Test_ValidateSequential_OrderInvariant()
        {
            var (inputs, targets) = RandomData(n: 6, seqLen: 6, withGlobal: true, globalDim: 3);
            var ts = new[] { 500d, 100d, 600d, 200d, 700d, 300d };
            int[] perm = { 2, 5, 1, 4, 0, 3 };

            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            var tr = new PriceTacTrainer(m, TC(epochs: 1));

            float v1 = tr.ValidateSequential(inputs, targets, ts);
            float v2 = tr.ValidateSequential(perm.Select(i => inputs[i]).ToArray(), perm.Select(i => targets[i]).ToArray(), perm.Select(i => ts[i]).ToArray());

            Assert(MathF.Abs(v1 - v2) < 1e-6f, $"ValidateSequential should sort by timestamp. v1={v1:F6}, v2={v2:F6}");
        }

        // ---------------------------------------------------------------------
        // Training mechanics
        // ---------------------------------------------------------------------

        private void Test_Train_LossDecreases_OnDeterministicPriceData()
        {
            var (trainInputs, trainTargets, testInputs, testTargets) = PriceHeldOutData(trainN: 48, testN: 16, seqLen: 6);
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));

            float before = new PriceTacTrainer(m, TC(epochs: 1)).ValidateAligned(testInputs, testTargets);
            new PriceTacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 60)).Train(trainInputs, trainTargets);
            float after = new PriceTacTrainer(m, TC(epochs: 1)).ValidateAligned(testInputs, testTargets);

            Assert(after < before, $"loss did not decrease. before={before:F6}, after={after:F6}");
        }

        private void Test_Train_SingleSampleOverfit_LossDecreases()
        {
            var (inputs, targets) = RandomData(n: 1, seqLen: 6);
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            float before = new PriceTacTrainer(m, TC(epochs: 1)).ValidateAligned(inputs, targets);
            new PriceTacTrainer(m, TC(lr: 0.004f, bs: 1, epochs: 120)).Train(inputs, targets);
            float after = new PriceTacTrainer(m, TC(epochs: 1)).ValidateAligned(inputs, targets);
            Assert(after < before, $"single-sample loss did not decrease. before={before:F6}, after={after:F6}");
        }

        private void Test_Train_ParameterUpdateCoverage_WithPriceContext()
        {
            var (inputs, targets) = RandomData(n: 6, seqLen: 10);
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, priceSeqLen: 12, priceContextEnabled: true), new Random(42));

            var priceProj = Copy(m.PriceInputProjection);
            var selfWq = Copy(m.PriceBlocks[0].SelfAttention.WQ);
            var crossWk = Copy(m.PriceBlocks[0].CrossAttention.WK);
            var reg = Copy(m.RegressionProjection);
            var range = Copy(m.RangeProjection);
            var quality = Copy(m.QualityProjection);
            var dir = Copy(m.DirectionProjection);
            var mid = Copy(m.MidDirectionProjection);
            var context = Copy(m.ContextTypeEmbedding);

            new PriceTacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);

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

        private void Test_Train_GlobalProjectionUpdated()
        {
            var (inputs, targets) = RandomData(n: 6, seqLen: 6, withGlobal: true, globalDim: 4);
            var m = new PriceTacModel(Cfg(globalDim: 4), new Random(42));
            var before = Copy(m.GlobalFeatureProjection);
            new PriceTacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);
            Assert(Changed(before, m.GlobalFeatureProjection), "GlobalFeatureProjection did not update");
        }

        private void Test_Train_ConfidenceHeadUpdated()
        {
            var (inputs, targets) = RandomData(n: 6, seqLen: 6);
            var confTargets = ConfidenceTargets(inputs, value: 0.8f);
            var m = new PriceTacModel(Cfg(useConf: true), new Random(42));
            var before = Copy(m.ConfidenceProjection);
            new PriceTacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets, confTargets);
            Assert(Changed(before, m.ConfidenceProjection), "ConfidenceProjection did not update");
        }

        private void Test_Train_DirectionLossWeightControlsDirectionHeadGradient()
        {
            var (inputs, targets) = DirectionSignalData(n: 12, seqLen: 6);

            var zeroCfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32);
            zeroCfg.Output.DirectionLossWeight = 0f;
            var zero = new PriceTacModel(zeroCfg, new Random(42));
            var zeroBefore = Copy(zero.DirectionProjection);
            new PriceTacTrainer(zero, TC(lr: 0.01f, bs: 6, epochs: 3)).Train(inputs, targets);
            Assert(!Changed(zeroBefore, zero.DirectionProjection, 1e-8f), "DirectionProjection changed even though DirectionLossWeight=0");

            var activeCfg = Cfg(embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32);
            activeCfg.Output.DirectionLossWeight = 2f;
            var active = new PriceTacModel(activeCfg, new Random(42));
            var activeBefore = Copy(active.DirectionProjection);
            new PriceTacTrainer(active, TC(lr: 0.01f, bs: 6, epochs: 3)).Train(inputs, targets);
            Assert(Changed(activeBefore, active.DirectionProjection), "DirectionProjection did not change when DirectionLossWeight>0");
        }

        private void Test_Train_ConfidenceLossWeightZeroDisablesConfidenceGradient()
        {
            var (inputs, targets) = RandomData(n: 6, seqLen: 6);
            var confTargets = ConfidenceTargets(inputs, value: 0.2f);
            var m = new PriceTacModel(Cfg(useConf: true), new Random(42));
            var before = Copy(m.ConfidenceProjection);

            var tc = TC(lr: 0.01f, bs: 3, epochs: 3);
            tc.ConfidenceLossWeight = 0f;
            new PriceTacTrainer(m, tc).Train(inputs, targets, confTargets);

            Assert(!Changed(before, m.ConfidenceProjection, 1e-8f), "ConfidenceProjection changed even though ConfidenceLossWeight=0");
        }

        private void Test_Train_GradientClippingBoundsUpdate()
        {
            var (inputs, targets) = RandomData(n: 3, seqLen: 6);
            var m = new PriceTacModel(Cfg(), new Random(42));
            var before = Copy(m.PriceInputProjection);

            new PriceTacTrainer(m, new TrainingConfig
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
            var (inputs, targets) = RandomData(n: 5, seqLen: 6);
            var m = new PriceTacModel(Cfg(), new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.5f, bs: 5, epochs: 4, clip: true)).Train(inputs, targets);
            AssertPredictionFiniteAndConstrained(m.PredictNext(inputs[0]), expectConfidenceHead: false);
        }

        private void Test_Train_PriceContextPathRunsFinite()
        {
            var (inputs, targets) = RandomData(n: 5, seqLen: 10);
            var cfg = Cfg(priceSeqLen: 12, priceContextEnabled: true);
            var m = new PriceTacModel(cfg, new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.001f, bs: 5, epochs: 4)).Train(inputs, targets);
            float loss = new PriceTacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(float.IsFinite(loss) && loss >= 0f, $"invalid loss after price-context training: {loss}");
        }

        private void Test_Train_DecayNetworkUpdatedWithPriceContext()
        {
            var (inputs, targets) = RandomData(n: 6, seqLen: 10);
            var m = new PriceTacModel(Cfg(priceSeqLen: 12, priceContextEnabled: true), new Random(42));
            var before = Copy3D(m.PriceBlocks[0].DecayNetwork.QueryProjection);

            new PriceTacTrainer(m, TC(lr: 0.005f, bs: 3, epochs: 8)).Train(inputs, targets);

            Assert(Changed(before, m.PriceBlocks[0].DecayNetwork.QueryProjection), "DecayNetwork.QueryProjection did not update");
        }

        private void Test_Train_ZeroLearningRateDoesNotChangeWeights()
        {
            var (inputs, targets) = RandomData(n: 4, seqLen: 6);
            var m = new PriceTacModel(Cfg(), new Random(42));
            var snap = SnapshotWeights(m);
            new PriceTacTrainer(m, TC(lr: 0f, bs: 2, epochs: 2)).Train(inputs, targets);
            AssertWeightsUnchanged(snap, m, "Train lr=0");
        }

        // ---------------------------------------------------------------------
        // Semantic smoke tests
        // ---------------------------------------------------------------------

        private void Test_Signal_PriceHeldOutLossImproves()
        {
            var (trainInputs, trainTargets, testInputs, testTargets) = PriceHeldOutData(trainN: 64, testN: 20, seqLen: 6);
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));

            float before = new PriceTacTrainer(m, TC(epochs: 1)).ValidateAligned(testInputs, testTargets);
            new PriceTacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 80)).Train(trainInputs, trainTargets);
            float after = new PriceTacTrainer(m, TC(epochs: 1)).ValidateAligned(testInputs, testTargets);

            Assert(after < before * 0.80f, $"held-out price loss did not improve enough. before={before:F6}, after={after:F6}");
        }

        private void Test_Signal_GlobalSeparationAfterTraining()
        {
            var (inputs, targets) = GlobalSignalData(n: 48, seqLen: 6);
            var cfg = Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3);
            var m = new PriceTacModel(cfg, new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 80)).Train(inputs, targets);

            var ps = ConstantPriceSequence(6, 0f, 0f, 0f, 0f, 1f);
            var on = m.PredictNext(new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, GlobalFeatures = new[] { 1f, 0f, 1f } });
            var off = m.PredictNext(new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps, GlobalFeatures = new[] { 0f, 1f, 0f } });

            Assert(on.Close > off.Close + 0.05f, $"risk-on close should exceed risk-off close. on={on.Close:F6}, off={off.Close:F6}");
            Assert(on.DirectionProb > off.DirectionProb + 0.05f, $"risk-on dir should exceed risk-off dir. on={on.DirectionProb:F6}, off={off.DirectionProb:F6}");
        }

        private void Test_Signal_DirectionAndMidDirectionCanLearnOppositeLabels()
        {
            var (inputs, targets) = OppositeDirectionData(n: 24, seqLen: 6);
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 90)).Train(inputs, targets);

            var up = m.PredictNext(inputs[0]);
            var down = m.PredictNext(inputs[inputs.Length - 1]);

            Assert(up.DirectionProb > down.DirectionProb + 0.05f,
                $"direction head did not separate. up={up.DirectionProb:F6}, down={down.DirectionProb:F6}");
            Assert(down.MidWindowDirectionProb > up.MidWindowDirectionProb + 0.05f,
                $"mid-direction head did not learn opposite label. up={up.MidWindowDirectionProb:F6}, down={down.MidWindowDirectionProb:F6}");
        }

        private void Test_Signal_RangeHeadLearnsRangeTarget()
        {
            var (inputs, targets) = RangeSignalData(n: 24, seqLen: 6);
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            float before = MeanRangeError(m, inputs, targets);
            new PriceTacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 90)).Train(inputs, targets);
            float after = MeanRangeError(m, inputs, targets);
            Assert(after < before * 0.70f, $"range error did not improve enough. before={before:F6}, after={after:F6}");
        }

        // ---------------------------------------------------------------------
        // Memory
        // ---------------------------------------------------------------------

        private void Test_Memory_InitiallyEmptyAndClearMethods()
        {
            var m = new PriceTacModel(Cfg(), new Random(42));
            Assert(m.PriceMemory.Count == 0, "PriceMemory should start empty");
            Assert(m.LastPriceTimestamp == 0.0, "LastPriceTimestamp should start at 0");

            m.PriceMemory.AddRange(MakePriceMemoryEntries(m.Config.Price.EmbeddingDim, (1.0, 0.2f, 1), (2.0, 0.3f, 2)));
            m.LastPriceTimestamp = 2.0;
            m.ClearPriceMemory();
            Assert(m.PriceMemory.Count == 0, "ClearPriceMemory should clear PriceMemory");
            Assert(m.LastPriceTimestamp == 2.0, "ClearPriceMemory should not reset LastPriceTimestamp");

            m.PriceMemory.AddRange(MakePriceMemoryEntries(m.Config.Price.EmbeddingDim, (3.0, 0.2f, 1)));
            m.ClearAllMemory();
            Assert(m.PriceMemory.Count == 0, "ClearAllMemory should clear PriceMemory");
            Assert(m.LastPriceTimestamp == 0.0, "ClearAllMemory should reset LastPriceTimestamp");
        }

        private void Test_Memory_PredictWithMemoryExactGrowthAndTimestamps()
        {
            var m = new PriceTacModel(Cfg(), new Random(42));
            var input = MakeInput(seqLen: 5);
            m.PredictWithMemory(input, currentAbsoluteTimestamp: 100.0, timeUnitsPerPosition: 2.0, maxPriceMemorySize: 20);

            Assert(m.PriceMemory.Count == 5, $"PriceMemory count mismatch: {m.PriceMemory.Count}");
            for (int i = 0; i < 5; i++)
            {
                Assert(Math.Abs(m.PriceMemory[i].AbsoluteTimestamp - (100.0 + i * 2.0)) < 1e-9,
                    $"timestamp mismatch at {i}: {m.PriceMemory[i].AbsoluteTimestamp}");
                Assert(m.PriceMemory[i].HiddenState.Length == m.Config.Price.EmbeddingDim, "price memory hidden dim mismatch");
            }
            Assert(Math.Abs(m.LastPriceTimestamp - 108.0) < 1e-9, "LastPriceTimestamp mismatch");
        }

        private void Test_Memory_CustomTimeUnitsScaleTimestamps()
        {
            var m = new PriceTacModel(Cfg(), new Random(42));
            var input = MakeInput(seqLen: 4);
            m.PredictWithMemory(input, currentAbsoluteTimestamp: 50.0, timeUnitsPerPosition: 0.25, maxPriceMemorySize: 10);

            Assert(m.PriceMemory.Count == 4, "PriceMemory count mismatch");
            Assert(Math.Abs(m.PriceMemory[1].AbsoluteTimestamp - 50.25) < 1e-9, "timeUnitsPerPosition was not applied at row 1");
            Assert(Math.Abs(m.PriceMemory[3].AbsoluteTimestamp - 50.75) < 1e-9, "timeUnitsPerPosition was not applied at row 3");
            Assert(Math.Abs(m.LastPriceTimestamp - 50.75) < 1e-9, "LastPriceTimestamp mismatch");
        }

        private void Test_Memory_ZeroBudgetKeepsPriceMemoryEmpty()
        {
            var m = new PriceTacModel(Cfg(), new Random(42));
            var pred = m.PredictWithMemory(MakeInput(seqLen: 5), 100.0, maxPriceMemorySize: 0);
            AssertPredictionFiniteAndConstrained(pred, expectConfidenceHead: false);
            Assert(m.PriceMemory.Count == 0, "zero maxPriceMemorySize should keep PriceMemory empty");
        }

        private void Test_Memory_PrunesToLimit()
        {
            var m = new PriceTacModel(Cfg(), new Random(42));
            m.PredictWithMemory(MakeInput(seqLen: 5), 100.0, maxPriceMemorySize: 6);
            m.PredictWithMemory(MakeInput(seqLen: 5), 200.0, maxPriceMemorySize: 6);
            Assert(m.PriceMemory.Count == 6, $"PriceMemory should be pruned to 6, got {m.PriceMemory.Count}");
        }

        private void Test_Memory_AttentionPruningKeepsHighScore()
        {
            var m = new PriceTacModel(Cfg(), new Random(42));
            m.PruningConfig.UseAttentionBasedPruning = true;
            m.PruningConfig.NewEntryReserveFraction = 0f;
            m.PruningConfig.MinQueryCountForPruning = 1;
            m.PriceMemory = MakePriceMemoryEntries(
                m.Config.Price.EmbeddingDim,
                (1.0, 0.10f, 5),
                (2.0, 0.90f, 5),
                (3.0, 0.30f, 5),
                (4.0, 0.70f, 5));

            InvokePricePruneMemory(m, 2);

            Assert(m.PriceMemory.Count == 2, "PricePruneMemory should keep exactly 2 entries");
            Assert(m.PriceMemory.Any(e => MathF.Abs(e.AttentionScore - 0.90f) < 1e-6f), "highest-score entry was not kept");
            Assert(m.PriceMemory.Any(e => MathF.Abs(e.AttentionScore - 0.70f) < 1e-6f), "second-highest-score entry was not kept");
        }

        private void Test_Memory_AttentionScoresUpdated()
        {
            var m = new PriceTacModel(Cfg(), new Random(42));
            m.PruningConfig.AttentionScoreAlpha = 0.5f;
            m.PriceMemory = MakePriceMemoryEntries(m.Config.Price.EmbeddingDim, (80.0, 0.0f, 0));
            m.PredictWithMemory(MakeInput(seqLen: 5), currentAbsoluteTimestamp: 100.0, maxPriceMemorySize: 20);

            Assert(m.PriceMemory[0].QueryCount == 1, $"existing price memory QueryCount should increment, got {m.PriceMemory[0].QueryCount}");
            Assert(m.PriceMemory[0].AttentionScore > 0f, "existing price memory AttentionScore should update");
        }

        private void Test_Memory_GlobalTokenNotPersistedAsMemory()
        {
            var baseModel = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            var m = CloneModel(baseModel);
            var noGlobalModel = CloneModel(baseModel);

            var input = MakeInput(seqLen: 5, globalDim: 3);
            var withGlobal = m.PredictWithMemory(input, 100.0, maxPriceMemorySize: 20);
            var withoutGlobal = noGlobalModel.PredictNext(new PriceTacInput
            {
                PredictionTimestamp = input.PredictionTimestamp,
                PriceSequence = input.PriceSequence
            });

            Assert(PredictionsDiffer(withGlobal, withoutGlobal, 1e-7f), "global token should affect output");
            Assert(m.PriceMemory.Count == input.PriceSequence.GetLength(0), "global token should not be persisted as a memory row");
        }

        private void Test_Memory_PriceOnlyWorksWithExistingPriceMemory()
        {
            var cfg = Cfg();
            var baseModel = new PriceTacModel(cfg, new Random(42));
            var withMemory = CloneModel(baseModel);
            var withoutMemory = CloneModel(baseModel);

            int embDim = cfg.Price.EmbeddingDim;
            withMemory.PriceMemory.Add(new PriceMemoryEntry
            {
                AbsoluteTimestamp = 80.0,
                HiddenState = Enumerable.Range(0, embDim).Select(d => d % 2 == 0 ? 3.0f + d * 0.01f : -2.0f - d * 0.01f).ToArray()
            });

            var input = MakeInput(seqLen: 6);
            var memPred = withMemory.PredictWithMemory(input, 100.0, maxPriceMemorySize: 20);
            var noMemPred = withoutMemory.PredictWithMemory(input, 100.0, maxPriceMemorySize: 20);

            Assert(PredictionsDiffer(memPred, noMemPred, 1e-7f), "existing price memory should affect prediction");
        }

        // ---------------------------------------------------------------------
        // Sequential training / rolling memory
        // ---------------------------------------------------------------------

        private void Test_Sequential_CommitsFullObservedSampleAndScaledTimestamps()
        {
            var valid = MakeInput(seqLen: 5);
            var validT = ConstantTargets(5, 0.55f, 0.12f, 0.8f, 1, 1);
            var m = new PriceTacModel(Cfg(), new Random(42));

            new PriceTacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1))
                .TrainSequential(new[] { valid }, new[] { validT }, new[] { 100.0 }, timeUnitsPerPosition: 3.0, maxPriceMemory: 20);

            Assert(m.PriceMemory.Count == 5, "full observed sample should be committed");
            Assert(Math.Abs(m.PriceMemory[4].AbsoluteTimestamp - 112.0) < 1e-9, "scaled timestamp mismatch");
            Assert(Math.Abs(m.LastPriceTimestamp - 112.0) < 1e-9, "LastPriceTimestamp mismatch");
        }

        private void Test_Sequential_SkippedInvalidSampleDoesNotCommit()
        {
            var valid = MakeInput(seqLen: 5);
            var skipped = new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = new float[5, 6] };
            var validT = ConstantTargets(5, 0.55f, 0.12f, 0.8f, 1, 1);
            var skippedT = ConstantTargets(5, 0.20f, 0.08f, 0.7f, 0, 0);
            var m = new PriceTacModel(Cfg(priceSeqLen: 8), new Random(42));

            new PriceTacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1))
                .TrainSequential(new[] { valid, skipped }, new[] { validT, skippedT }, new[] { 100.0, 200.0 });

            Assert(m.PriceMemory.Count == valid.PriceSequence.GetLength(0), "only valid sample price rows should be committed");
            Assert(Math.Abs(m.LastPriceTimestamp - 104.0) < 1e-9, "skipped sample should not change LastPriceTimestamp");
        }

        private void Test_Sequential_MemoryClearedBetweenEpochs()
        {
            var (inputs, targets) = RandomData(n: 4, seqLen: 6, withGlobal: true, globalDim: 3);
            var ts = new[] { 100d, 200d, 300d, 400d };
            var baseModel = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3), new Random(42));
            var m1 = CloneModel(baseModel);
            var m2 = CloneModel(baseModel);

            new PriceTacTrainer(m1, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            new PriceTacTrainer(m2, TC(lr: 0f, bs: 1, epochs: 2)).TrainSequential(inputs, targets, ts);

            AssertMemoryEquivalent(m1, m2, "one epoch vs two epochs with lr=0");
        }

        private void Test_Sequential_TimestampShuffleInvariantAtZeroLearningRate()
        {
            var (inputs, targets) = RandomData(n: 6, seqLen: 6, withGlobal: true, globalDim: 3);
            var ts = new[] { 500d, 100d, 600d, 200d, 700d, 300d };
            int[] perm = { 2, 5, 1, 4, 0, 3 };

            var baseModel = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48, globalDim: 3), new Random(42));
            var m1 = CloneModel(baseModel);
            var m2 = CloneModel(baseModel);

            new PriceTacTrainer(m1, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            new PriceTacTrainer(m2, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(
                perm.Select(i => inputs[i]).ToArray(),
                perm.Select(i => targets[i]).ToArray(),
                perm.Select(i => ts[i]).ToArray());

            AssertMemoryEquivalent(m1, m2, "timestamp shuffle invariant");
        }

        private void Test_Sequential_ZeroLearningRateDoesNotChangeWeights()
        {
            var (inputs, targets) = RandomData(n: 4, seqLen: 6, withGlobal: true, globalDim: 3);
            var ts = new[] { 100d, 200d, 300d, 400d };
            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            var snap = SnapshotWeights(m);
            new PriceTacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            AssertWeightsUnchanged(snap, m, "TrainSequential lr=0");
        }

        private void Test_Sequential_WithGlobalTokenRunsFinite()
        {
            var (inputs, targets) = RandomData(n: 5, seqLen: 6, withGlobal: true, globalDim: 3);
            var ts = Enumerable.Range(0, inputs.Length).Select(i => i * 100.0).ToArray();
            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.002f, bs: 1, epochs: 4)).TrainSequential(inputs, targets, ts);
            var p = m.PredictWithMemory(inputs[0], 1000.0);
            AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
        }

        private void Test_Sequential_MemoryAccumulates()
        {
            var (inputs, targets) = RandomData(n: 3, seqLen: 6);
            var ts = new[] { 100d, 200d, 300d };
            var m = new PriceTacModel(Cfg(), new Random(42));
            new PriceTacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts, maxPriceMemory: 100);
            Assert(m.PriceMemory.Count == inputs.Sum(x => x.PriceSequence.GetLength(0)), "PriceMemory did not accumulate all samples");
            Assert(m.PriceMemory.All(e => e.HiddenState.Length == m.Config.Price.EmbeddingDim), "price memory hidden dim mismatch");
        }

        // ---------------------------------------------------------------------
        // Save / Load
        // ---------------------------------------------------------------------

        private void Test_SaveLoad_ForwardIdentical_AllHeads()
        {
            var (inputs, targets) = RandomData(n: 4, seqLen: 6, withGlobal: true, globalDim: 3);
            var m = new PriceTacModel(Cfg(useConf: true, globalDim: 3), new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.002f, bs: 2, epochs: 4)).Train(inputs, targets, ConfidenceTargets(inputs, 0.8f));
            var before = m.Forward(inputs[0]);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var loaded = PriceTacModel.Load(dir, _accelerationType, _deviceId);
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
            var m = new PriceTacModel(Cfg(useConf: true, globalDim: 4), new Random(42));
            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = PriceTacModel.Load(dir, _accelerationType, _deviceId);
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
            var input = MakeInput(seqLen: 6);
            var m = new PriceTacModel(Cfg(), new Random(42));
            m.PruningConfig.AttentionScoreAlpha = 0.25f;
            m.PruningConfig.MinQueryCountForPruning = 7;
            m.PruningConfig.NewEntryReserveFraction = 0.33f;
            m.PruningConfig.UseAttentionBasedPruning = false;
            m.PredictWithMemory(input, 100.0);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = PriceTacModel.Load(dir, _accelerationType, _deviceId);
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
            var (inputs, targets) = RandomData(n: 5, seqLen: 6);
            var m = new PriceTacModel(Cfg(), new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.002f, bs: 5, epochs: 4)).Train(inputs, targets);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = PriceTacModel.Load(dir, _accelerationType, _deviceId);
                new PriceTacTrainer(ld, TC(lr: 0.002f, bs: 5, epochs: 4)).Train(inputs, targets);
                float loss = new PriceTacTrainer(ld, TC(epochs: 1)).Validate(inputs, targets);
                Assert(float.IsFinite(loss) && loss >= 0f, $"invalid loss after continued training: {loss}");
            }
            finally { DeleteDir(dir); }
        }

        private void Test_SaveLoad_SequentialMemoryPredictionIdentical()
        {
            var (inputs, targets) = RandomData(n: 4, seqLen: 6, withGlobal: true, globalDim: 3);
            var m = new PriceTacModel(Cfg(globalDim: 3), new Random(42));
            var ts = new[] { 100d, 200d, 300d, 400d };
            new PriceTacTrainer(m, TC(lr: 0.002f, bs: 1, epochs: 4)).TrainSequential(inputs, targets, ts);
            m.PredictWithMemory(inputs[0], 1000.0);
            m.PredictWithMemory(inputs[1], 1200.0);

            string dir = TmpDir();
            try
            {
                m.Save(dir);
                var ld = PriceTacModel.Load(dir, _accelerationType, _deviceId);
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
            var m = new PriceTacModel(Cfg(), new Random(42));
            float[] scales = { 100f, 1e-6f, 0f, -5f };
            foreach (float scale in scales)
            {
                var ps = new float[8, 5];
                for (int i = 0; i < ps.GetLength(0); i++)
                    for (int j = 0; j < ps.GetLength(1); j++)
                        ps[i, j] = scale;

                var p = m.PredictNext(new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = ps });
                AssertPredictionFiniteAndConstrained(p, expectConfidenceHead: false);
            }
        }

        private void Test_Stability_NoNaNAfterManyEpochs()
        {
            var (inputs, targets) = RandomData(n: 12, seqLen: 6);
            var m = new PriceTacModel(Cfg(embDim: 24, numHeads: 4, numLayers: 1, ffnDim: 48), new Random(42));
            new PriceTacTrainer(m, TC(lr: 0.003f, bs: 4, epochs: 50)).Train(inputs, targets);
            float loss = new PriceTacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(float.IsFinite(loss) && loss >= 0f, $"invalid loss after many epochs: {loss}");
            AssertPredictionFiniteAndConstrained(m.PredictNext(inputs[0]), expectConfidenceHead: false);
        }

        // ---------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------

        private PriceTacConfig Cfg(
            int embDim = 16,
            int numHeads = 2,
            int numLayers = 1,
            int ffnDim = 32,
            int priceFeatures = 5,
            int priceSeqLen = 12,
            bool useConf = false,
            int globalDim = 0,
            bool decayEnabled = true,
            bool bypassDecay = true,
            bool priceContextEnabled = false)
        {
            var cfg = new PriceTacConfig
            {
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
                Global = new PriceTacGlobalConfig
                {
                    GlobalFeatureDim = globalDim,
                    BypassDecay = bypassDecay
                },
                Output = new PriceTacOutputConfig
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
                    AccelerationType = _accelerationType,
                    AccelerationDeviceId = _deviceId
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

        private static TrainingConfig TC(float lr = 0.001f, int bs = 4, int epochs = 10, bool clip = true)
        {
            return new TrainingConfig
            {
                LearningRate = lr,
                BatchSize = bs,
                Epochs = epochs,
                UseGradientClipping = clip,
                GradientClipThreshold = 1f,
                ConfidenceLossWeight = 1f,
                Verbose = false
            };
        }

        private PriceTacInput MakeInput(int seqLen = 8, int priceFeatures = 5, int globalDim = 0)
        {
            var rng = new Random(42);
            return new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandomMatrix(seqLen, priceFeatures, rng, 0.3f),
                GlobalFeatures = globalDim > 0 ? Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray() : null
            };
        }

        private static (PriceTacInput[] inputs, PriceTacTarget[][] targets) RandomData(int n = 8, int seqLen = 6, int priceFeatures = 5, int seed = 42, bool withGlobal = false, int globalDim = 0)
        {
            var rng = new Random(seed);
            var inputs = new PriceTacInput[n];
            var targets = new PriceTacTarget[n][];

            for (int s = 0; s < n; s++)
            {
                var price = RandomMatrix(seqLen, priceFeatures, rng, 0.3f);
                float[] globals = null;
                float globalSignal = 0f;
                if (withGlobal && globalDim > 0)
                {
                    globals = Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray();
                    globalSignal = globals[0] - (globalDim > 1 ? globals[1] : 0f);
                }

                inputs[s] = new PriceTacInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(s),
                    PriceSequence = price,
                    GlobalFeatures = globals
                };

                targets[s] = new PriceTacTarget[seqLen];
                for (int t = 0; t < seqLen; t++)
                {
                    float close = Clamp01(0.5f + 0.15f * price[t, 0] - 0.05f * price[t, Math.Min(2, priceFeatures - 1)] + 0.08f * globalSignal);
                    float range = 0.1f + 0.03f * MathF.Abs(price[t, Math.Min(1, priceFeatures - 1)]);
                    targets[s][t] = new PriceTacTarget
                    {
                        High = close + range * 0.5f,
                        Low = close - range * 0.5f,
                        Close = close,
                        Range = range,
                        Quality = Clamp01(0.75f + 0.1f * price[t, 0]),
                        Direction = close >= 0.5f ? 1f : 0f,
                        MidWindowDirection = price[t, Math.Min(1, priceFeatures - 1)] >= 0f ? 1f : 0f
                    };
                }
            }

            return (inputs, targets);
        }

        private static (PriceTacInput[] trainInputs, PriceTacTarget[][] trainTargets, PriceTacInput[] testInputs, PriceTacTarget[][] testTargets) PriceHeldOutData(int trainN = 64, int testN = 20, int seqLen = 6, int seed = 123)
        {
            var rng = new Random(seed);

            PriceTacInput MakeIn(float x, float y, int idx) => new PriceTacInput
            {
                PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                PriceSequence = ConstantPriceSequence(seqLen, x, y, x * y, x * x, 1f)
            };

            PriceTacTarget[] MakeT(float x, float y)
            {
                float close = Clamp01(0.5f + 0.22f * x - 0.1f * y);
                float range = 0.08f + 0.04f * MathF.Abs(x);
                float quality = Clamp01(0.4f + 0.5f * y);
                return ConstantTargets(seqLen, close, range, quality, close > 0.5f ? 1 : 0, x + y > 0f ? 1 : 0);
            }

            void Fill(int n, int offset, out PriceTacInput[] inputs, out PriceTacTarget[][] targets)
            {
                inputs = new PriceTacInput[n];
                targets = new PriceTacTarget[n][];
                for (int i = 0; i < n; i++)
                {
                    float x = (float)(rng.NextDouble() * 2.0 - 1.0);
                    float y = (float)(rng.NextDouble() * 2.0 - 1.0);
                    inputs[i] = MakeIn(x, y, offset + i);
                    targets[i] = MakeT(x, y);
                }
            }

            Fill(trainN, 0, out var trainInputs, out var trainTargets);
            Fill(testN, 10_000, out var testInputs, out var testTargets);
            return (trainInputs, trainTargets, testInputs, testTargets);
        }

        private static (PriceTacInput[] inputs, PriceTacTarget[][] targets) GlobalSignalData(int n = 48, int seqLen = 6)
        {
            var inputs = new PriceTacInput[n];
            var targets = new PriceTacTarget[n][];
            for (int i = 0; i < n; i++)
            {
                bool riskOn = i % 2 == 0;
                inputs[i] = new PriceTacInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    GlobalFeatures = riskOn ? new[] { 1f, 0f, 1f } : new[] { 0f, 1f, 0f }
                };
                targets[i] = ConstantTargets(seqLen, riskOn ? 0.78f : 0.22f, 0.1f, 0.9f, riskOn ? 1 : 0, riskOn ? 1 : 0);
            }
            return (inputs, targets);
        }

        private static (PriceTacInput[] inputs, PriceTacTarget[][] targets) DirectionSignalData(int n = 12, int seqLen = 6)
        {
            var inputs = new PriceTacInput[n];
            var targets = new PriceTacTarget[n][];
            for (int i = 0; i < n; i++)
            {
                bool up = i < n / 2;
                inputs[i] = new PriceTacInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, up ? 0.8f : 0.2f, 0.5f, 0.5f, 0.5f, 1f)
                };
                targets[i] = ConstantTargets(seqLen, 0.5f, 0.1f, 0.7f, up ? 1 : 0, up ? 1 : 0);
            }
            return (inputs, targets);
        }

        private static (PriceTacInput[] inputs, PriceTacTarget[][] targets) OppositeDirectionData(int n = 24, int seqLen = 6)
        {
            var inputs = new PriceTacInput[n];
            var targets = new PriceTacTarget[n][];
            for (int i = 0; i < n; i++)
            {
                bool dirUp = i < n / 2;
                inputs[i] = new PriceTacInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, dirUp ? 0.8f : 0.2f, dirUp ? 0.2f : 0.8f, 0.5f, 0.5f, 1f)
                };
                targets[i] = ConstantTargets(seqLen, 0.5f, 0.1f, 0.7f, dirUp ? 1 : 0, dirUp ? 0 : 1);
            }
            return (inputs, targets);
        }

        private static (PriceTacInput[] inputs, PriceTacTarget[][] targets) RangeSignalData(int n = 24, int seqLen = 6)
        {
            var inputs = new PriceTacInput[n];
            var targets = new PriceTacTarget[n][];
            for (int i = 0; i < n; i++)
            {
                float x = i < n / 2 ? 0.2f : 0.8f;
                float range = i < n / 2 ? 0.08f : 0.30f;
                inputs[i] = new PriceTacInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, x, 1f - x, x * x, 0.5f, 1f)
                };
                targets[i] = ConstantTargets(seqLen, 0.5f, range, 0.8f, 1, 1);
            }
            return (inputs, targets);
        }

        private static float[][] ConfidenceTargets(PriceTacInput[] inputs, float value)
        {
            var c = new float[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                int sl = inputs[i].PriceSequence.GetLength(0);
                c[i] = Enumerable.Repeat(value, sl).ToArray();
            }
            return c;
        }

        private static PriceTacTarget[] ConstantTargets(int seqLen, float close, float range, float quality, int direction, int midDirection)
        {
            var out_ = new PriceTacTarget[seqLen];
            float high = close + range * 0.5f;
            float low = close - range * 0.5f;
            for (int t = 0; t < seqLen; t++)
            {
                out_[t] = new PriceTacTarget
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

        private static float[,] ConstantPriceSequence(int seqLen, params float[] features)
        {
            var ps = new float[seqLen, features.Length];
            for (int t = 0; t < seqLen; t++)
                for (int f = 0; f < features.Length; f++)
                    ps[t, f] = features[f];
            return ps;
        }

        private static float[,] RandomMatrix(int rows, int cols, Random rng, float scale)
        {
            var m = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = ((float)rng.NextDouble() * 2f - 1f) * scale;
            return m;
        }

        private static float Clamp01(float x) => MathF.Max(0f, MathF.Min(1f, x));

        private static float MeanRangeError(PriceTacModel m, PriceTacInput[] inputs, PriceTacTarget[][] targets)
        {
            float total = 0f;
            int count = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var p = m.PredictNext(inputs[i]);
                total += MathF.Abs(p.Range - targets[i][targets[i].Length - 1].Range);
                count++;
            }
            return count > 0 ? total / count : 0f;
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

        private static (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence)
        InvokeForwardWithCache(PriceTacModel model, PriceTacInput input, PriceTacForwardCache cache, bool isTraining)
        {
            var method = typeof(PriceTacModel).GetMethod(
                "ForwardWithCache",
                BindingFlags.Instance | BindingFlags.NonPublic,
                binder: null,
                types: new[]
                {
                    typeof(PriceTacInput),
                    typeof(PriceTacForwardCache),
                    typeof(bool),
                    typeof(Random)
                },
                modifiers: null);

            if (method == null)
                throw new InvalidOperationException("Could not find ForwardWithCache overload.");

            var result = method.Invoke(model, new object[] { input, cache, isTraining, null });
            return ((float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence))result;
        }

        private static (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence)
        InvokeForwardWithPriceContextAndCache(PriceTacModel model, PriceTacInput input, float[,] priceCtxHidden, float[] priceCtxTimes, PriceTacForwardCache cache, bool isTraining)
        {
            var method = typeof(PriceTacModel).GetMethod(
                "ForwardWithPriceContextAndCache",
                BindingFlags.Instance | BindingFlags.NonPublic,
                binder: null,
                types: new[]
                {
                    typeof(PriceTacInput),
                    typeof(float[,]),
                    typeof(float[]),
                    typeof(PriceTacForwardCache),
                    typeof(bool),
                    typeof(Random)
                },
                modifiers: null);

            if (method == null)
                throw new InvalidOperationException("Could not find ForwardWithPriceContextAndCache overload.");

            var result = method.Invoke(model, new object[] { input, priceCtxHidden, priceCtxTimes, cache, isTraining, null });
            return ((float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence))result;
        }

        private static (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence)
        InvokeForwardWithPriceContextSliceAndCache(PriceTacModel model, PriceTacInput input, int rowStart, int rowCount, float[,] priceCtxHidden, float[] priceCtxTimes, PriceTacForwardCache cache, bool isTraining)
        {
            var method = typeof(PriceTacModel).GetMethod(
                "ForwardWithPriceContextSliceAndCache",
                BindingFlags.Instance | BindingFlags.NonPublic,
                binder: null,
                types: new[]
                {
                    typeof(PriceTacInput),
                    typeof(int),
                    typeof(int),
                    typeof(float[,]),
                    typeof(float[]),
                    typeof(PriceTacForwardCache),
                    typeof(bool),
                    typeof(Random)
                },
                modifiers: null);

            if (method == null)
                throw new InvalidOperationException("Could not find ForwardWithPriceContextSliceAndCache overload.");

            var result = method.Invoke(model, new object[] { input, rowStart, rowCount, priceCtxHidden, priceCtxTimes, cache, isTraining, null });
            return ((float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence))result;
        }

        private static void InvokePricePruneMemory(PriceTacModel model, int maxSize)
        {
            var method = typeof(PriceTacModel).GetMethod("PricePruneMemory", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (method == null)
                throw new InvalidOperationException("Could not find PricePruneMemory");
            method.Invoke(model, new object[] { maxSize });
        }

        private void AssertNoFutureLeakage(PriceTacModel m, float[] globals)
        {
            int seqLen = 8;
            int cut = 4;
            var a = RandomMatrix(seqLen, m.Config.Price.InputFeatureDim, new Random(700), 0.2f);
            var b = Copy(a);
            for (int t = cut; t < seqLen; t++)
                for (int f = 0; f < b.GetLength(1); f++)
                    b[t, f] = 50f + 7f * t + f;

            var inputA = new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = a, GlobalFeatures = globals };
            var inputB = new PriceTacInput { PredictionTimestamp = DateTime.UtcNow, PriceSequence = b, GlobalFeatures = globals };

            var fa = m.Forward(inputA);
            var fb = m.Forward(inputB);

            AssertPrefixClose(fa.regression, fb.regression, cut, 1e-5f, "regression future leakage");
            AssertPrefixClose(fa.range, fb.range, cut, 1e-5f, "range future leakage");
            AssertPrefixClose(fa.quality, fb.quality, cut, 1e-5f, "quality future leakage");
            AssertPrefixClose(fa.direction, fb.direction, cut, 1e-5f, "direction future leakage");
            AssertPrefixClose(fa.midDirection, fb.midDirection, cut, 1e-5f, "midDirection future leakage");
            AssertPrefixClose(fa.confidence, fb.confidence, cut, 1e-5f, "confidence future leakage");
        }

        private void AssertPredictionFiniteAndConstrained(PriceTacPrediction p, bool expectConfidenceHead)
        {
            Assert(float.IsFinite(p.High), "High not finite");
            Assert(float.IsFinite(p.Low), "Low not finite");
            Assert(float.IsFinite(p.Close), "Close not finite");
            Assert(float.IsFinite(p.Range) && p.Range >= 0f, $"Range invalid: {p.Range}");
            Assert(float.IsFinite(p.Quality) && p.Quality >= 0f && p.Quality <= 1f, $"Quality invalid: {p.Quality}");
            Assert(float.IsFinite(p.DirectionProb) && p.DirectionProb >= 0f && p.DirectionProb <= 1f, $"DirectionProb invalid: {p.DirectionProb}");
            Assert(float.IsFinite(p.MidWindowDirectionProb) && p.MidWindowDirectionProb >= 0f && p.MidWindowDirectionProb <= 1f, $"MidWindowDirectionProb invalid: {p.MidWindowDirectionProb}");
            Assert(float.IsFinite(p.Confidence), "Confidence not finite");

            const float tol = 1e-5f;
            Assert(p.High + tol >= p.Close, $"High must be >= Close. high={p.High}, close={p.Close}");
            Assert(p.Close + tol >= p.Low, $"Close must be >= Low. close={p.Close}, low={p.Low}");
            Assert(MathF.Abs((p.High - p.Low) - p.Range) <= 2e-5f,
                $"Range must equal High-Low. high-low={p.High - p.Low}, range={p.Range}");

            if (expectConfidenceHead)
                Assert(p.Confidence >= 0f && p.Confidence <= 1f, $"Confidence invalid: {p.Confidence}");
            else
                Assert(p.Confidence == 1f, $"Confidence should be 1 when disabled, got {p.Confidence}");
        }

        private static bool PredictionsDiffer(PriceTacPrediction a, PriceTacPrediction b, float tol)
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

        private void AssertPredictionsClose(PriceTacPrediction a, PriceTacPrediction b, float tol, string message)
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
            Assert(m != null, name + " is null");
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

        private void AssertMatricesClose(float[,] expected, float[,] actual, float tol, string message)
        {
            float diff = MaxAbsDiff(expected, actual);
            Assert(diff <= tol, $"{message}; maxAbsDiff={diff:E6}, tol={tol:E6}");
        }

        private void AssertPrefixClose(float[,] a, float[,] b, int rows, float tol, string message)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    float diff = MathF.Abs(a[i, j] - b[i, j]);
                    Assert(diff <= tol, $"{message} at [{i},{j}]: diff={diff:E6}, tol={tol:E6}");
                }
            }
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
                throw new Exception($"{message}: expected {typeof(T).Name}, got {ex.GetType().Name}: {ex.Message}");
            }

            throw new Exception($"{message}: expected {typeof(T).Name}, no exception was thrown");
        }

        private PriceTacModel CloneModel(PriceTacModel source)
        {
            string dir = TmpDir();
            try
            {
                source.Save(dir);
                return PriceTacModel.Load(dir, _accelerationType, _deviceId);
            }
            finally
            {
                DeleteDir(dir);
            }
        }

        private void AssertMemoryEquivalent(PriceTacModel a, PriceTacModel b, string label)
        {
            Assert(a.PriceMemory.Count == b.PriceMemory.Count, label + ": PriceMemory count mismatch");
            Assert(Math.Abs(a.LastPriceTimestamp - b.LastPriceTimestamp) < 1e-9, label + ": LastPriceTimestamp mismatch");

            for (int i = 0; i < a.PriceMemory.Count; i++)
            {
                Assert(Math.Abs(a.PriceMemory[i].AbsoluteTimestamp - b.PriceMemory[i].AbsoluteTimestamp) < 1e-9,
                    label + $": price timestamp mismatch {i}");
                Assert(a.PriceMemory[i].HiddenState != null, label + $": left price hidden state null {i}");
                Assert(b.PriceMemory[i].HiddenState != null, label + $": right price hidden state null {i}");
                AssertVectorClose(a.PriceMemory[i].HiddenState, b.PriceMemory[i].HiddenState, 1e-5f, label + $": price hidden state mismatch {i}");
            }
        }

        private sealed class WeightSnapshot
        {
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

        private static WeightSnapshot SnapshotWeights(PriceTacModel m)
        {
            return new WeightSnapshot
            {
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

        private void AssertWeightsUnchanged(WeightSnapshot s, PriceTacModel m, string label)
        {
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

        private static float[,] Copy(float[,] m)
        {
            if (m == null) return null;
            var r = new float[m.GetLength(0), m.GetLength(1)];
            Array.Copy(m, r, m.Length);
            return r;
        }

        private static float[] Copy(float[] v)
        {
            if (v == null) return null;
            var r = new float[v.Length];
            Array.Copy(v, r, v.Length);
            return r;
        }

        private static float[,,] Copy3D(float[,,] a)
        {
            if (a == null) return null;
            var b = new float[a.GetLength(0), a.GetLength(1), a.GetLength(2)];
            Array.Copy(a, b, a.Length);
            return b;
        }

        private static bool Changed(float[,] a, float[,] b, float tol = 1e-10f)
        {
            if (a == null || b == null) return a != b;
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1)) return true;
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    if (MathF.Abs(a[i, j] - b[i, j]) > tol) return true;
            return false;
        }

        private static bool Changed(float[,,] a, float[,,] b, float tol = 1e-10f)
        {
            if (a == null || b == null) return a != b;
            if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1) || a.GetLength(2) != b.GetLength(2)) return true;
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    for (int k = 0; k < a.GetLength(2); k++)
                        if (MathF.Abs(a[i, j, k] - b[i, j, k]) > tol) return true;
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
            for (int i = 0; i < a.Length; i++)
                max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
            return max;
        }

        private void AssertVectorClose(float[] expected, float[] actual, float tol, string message)
        {
            float diff = MaxAbsDiff(expected, actual);
            Assert(diff <= tol, $"{message}; maxAbsDiff={diff:E6}, tol={tol:E6}");
        }

        private static string TmpDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "pricetac_test_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        private static void DeleteDir(string dir)
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
