using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.MMTAC;
using CallaghanDev.ML.Transformers.TACAMT;
using System.Reflection;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class MmtacTests : TestBase
    {
        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * MMTAC (Multimodal Market Transformer with Additional Context)");
        }
        private (Action, string)[] Tests() => new (Action, string)[]
        {
            // contract / validation gaps
(Test_Train_LengthMismatch_Throws, "Train: mismatched inputs/targets lengths throw"),
(Test_Validate_LengthMismatch_Throws, "Validate: mismatched inputs/targets lengths throw"),
(Test_PredictNext_WrongPriceFeatureDim_Throws, "PredictNext: wrong price feature dim throws"),
(Test_PredictNext_GlobalFeatureDimMismatch_Throws, "PredictNext: wrong global feature dim throws"),

// validation must not mutate weights
(Test_Validate_DoesNotMutateWeights, "Validate: does not mutate model weights"),
(Test_ValidateSequential_DoesNotMutateWeights, "ValidateSequential: does not mutate model weights"),

// boundary / time-scale gaps
(Test_PredictWithMemory_ZeroMemoryBudget_KeepsBanksEmpty, "Memory: zero memory budget keeps both banks empty"),
(Test_PredictWithMemory_CustomTimeUnits_ScalesTimestamps, "Memory: custom timeUnitsPerPosition scales stored timestamps"),
(Test_Sequential_CustomTimeUnits_CommitsScaledTimestamps, "Sequential: custom timeUnitsPerPosition scales committed timestamps"),

// config branch gaps
(Test_Causal_NoFutureLeakage_WithNewsAndGlobalToken, "Causal: no future leakage still holds with news + global token"),
(Test_Decay_Disabled_ForwardAndTrain_Work, "Decay: disabled path works for forward and training"),
            //Failing tests
            (Test_Causal_NoFutureLeakage, "Causal: appending future rows does not change earlier predictions"),
            (Test_Decay_TimeMonotonic_ForIdenticalContent, "Decay: newer identical news has stronger effect than older news"),
            (Test_Decay_ContentAware_SameAgeDifferentContent_DifferentImpact, "Decay: same-age news impact depends on content"),
            (Test_Sequential_Training_CommitsFullObservedSample_NotPrefix, "Sequential: training commits full observed sample to memory"),
            (Test_Sequential_SkippedSample_DoesNotCommitMemory, "Sequential: skipped invalid sample does not commit memory"),
            (Test_Sequential_MemoryHelpsOnAutoregressive, "Semantic: Sequential memory improves autoregressive prediction"),
            (Test_Sequential_LiveNews_Affects_Current_Step_Immediately, "Sequential: live news affects current step immediately"),
            (Test_Train_Generalises_PriceMapping_HeldOut, "Train: held-out price-only deterministic mapping generalises"),
            (Test_Train_Generalises_MultimodalCombinedSignal_HeldOut, "Train: held-out combined price/news/global signal generalises"),
            (Test_LiveRollingMemory_Uses_Previous_Sample_Causally, "Sequential: rolling price memory influences the next sample causally"),
            (Test_ValidateSequential_Matches_Reference_NoCommitThenCommit, "Sequential: validator matches no-commit predict then single commit reference"),


            (Test_Robustness_NeutralDistractorStories_DoNot_Overwhelm_Signal, "Robustness: neutral distractor stories do not overwhelm news signal"),
            (Test_Sequential_NewsMemory_Helps_On_RegimeCarryTask, "Sequential: news memory helps on carried-regime synthetic task"),
            (Test_Train_UnfrozenText_Updated,         "Train: unfrozen text encoder updated"),
            (Test_LossWeights_AffectWhichHeadLearns,    "Train: loss weights affect which head trains"),

            (Test_TrainSequential_ZeroLearningRate_DoesNotChangeWeights, "Sequential: zero learning rate does not change weights"),

            (Test_ValidateSequential_OrderOnly_Diagnostic,"Sequential: ValidateSequentialOrderOnly_Diagnostic") ,
            (Test_Sequential_TimestampShuffle_Invariant, "Sequential: timestamp shuffle invariant"),
            (Test_Sequential_PriceMemory_Is_Contextual_Not_ContextFree, "Sequential: price memory is contextual, not context-free"),
            (Test_Memory_ExactGrowthCounts_BeforePrune, "Memory: exact growth counts before prune"),
            (Test_SaveLoad_SequentialMemoryPredictionIdentical, "SaveLoad: sequential memory prediction identical after reload"),
            (Test_GlobalToken_NotPersisted_AsMemory, "GlobalToken: affects output but is not persisted as memory"),
            (Test_ValidateSequential_ClearsMemoryOnExit, "ValidateSequential: clears memory on exit"),
            (Test_PricePruning_Mirrors_NewsPruning, "Memory: price pruning mirrors news pruning"),
            (Test_Tokenizer_SaveLoad_RoundTrip, "Tokenizer: save/load round-trip preserves tokenization"),
            (Test_TrainSequential_ZeroTimeUnits_Throws, "Sequential: zero timeUnitsPerPosition throws"),
            (Test_ValidateSequential_LengthMismatch_Throws, "ValidateSequential: mismatched lengths throw"),
            (Test_Ablation_Multimodal_Each_Modality_Matters, "Ablation: each modality matters on combined synthetic task"),
            (Test_Robustness_MixedMissingModalities_Still_Trains, "Robustness: mixed missing modalities still train and generalise"),
            (Test_SaveLoad_HeldOutValidationIdentical, "Save/Load: held-out validation is preserved after reload"),

            (Test_Training_MultipleSeeds_AllRemainFinite_And_Learn, "Training: multiple seeds remain finite and learn on held-out data"),
            (Test_Train_Generalises_NewsSignal_HeldOut,  "Train: held-out news sentiment signal generalises"),
            (Test_Train_Generalises_GlobalFeatureSignal_HeldOut, "Train: held-out global feature signal generalises"),

            //  construction & dims 
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

            //  forward: PredictNext 
            (Test_PredictNext_NoError,                      "PredictNext: runs without error"),
            (Test_PredictNext_NoNaN,                        "PredictNext: all output fields are finite"),
            (Test_PredictNext_NullStories,                  "PredictNext: null stories (price-only) works"),
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

            //  training 
            (Test_Train_LossDecreases,                "Train: loss decreases over training"),
            (Test_Train_LossFinite,                   "Train: training loss stays finite"),
            (Test_Train_AllPriceParamsUpdated,        "Train: price-decoder params receive gradients"),
            (Test_Train_RegressionHeadUpdated,        "Train: RegressionProjection updated"),
            (Test_Train_RangeHeadUpdated,             "Train: RangeProjection updated"),
            (Test_Train_DirectionHeadUpdated,         "Train: DirectionProjection updated"),
            (Test_Train_ContextTypeEmbeddingUpdated,  "Train: ContextTypeEmbedding updated"),
            (Test_Train_FrozenText_NotUpdated,        "Train: frozen text encoder unchanged"),
            (Test_Train_ConfidenceHead_Updated,       "Train: ConfidenceProjection updated when enabled"),
            (Test_Train_GradientClipping_NoNaN,       "Train: high LR + clipping prevents NaN"),
            (Test_Train_SingleSampleOverfit,          "Train: single sample overfits (loss decreases)"),
            (Test_Train_PriceOnly_LossDecreases,      "Train: price-only (no news) loss decreases"),
            (Test_Train_MixedBatch_SomeNullNews,      "Train: mixed batch with some null news works"),
            (Test_Train_DecayNetwork_Weights_Updated, "Train: DecayNetwork weights updated during backprop"),
            (Test_Train_PriceContext_NoError,         "Train: PriceContext enabled path runs without error"),
            (Test_Train_PriceContext_LossFinite,      "Train: PriceContext enabled loss stays finite"),
            (Test_Validate_Loss_ConsistentWithTraining, "Train: validation loss consistent with training loss"),
            (Test_GradClip_BoundsUpdateMagnitude,        "Train: gradient clipping bounds actual update magnitude"),

            //  global token 
            (Test_GlobalToken_ForwardNoError,            "GlobalToken: forward works when globalDim > 0"),
            (Test_GlobalToken_ChangesOutput,             "GlobalToken: different global features produce different output"),
            (Test_GlobalToken_ProjectionUpdated,         "GlobalToken: GlobalFeatureProjection updated in training"),
            (Test_GlobalToken_PredictWithMemory_NoError, "GlobalToken: PredictWithMemory works with globalDim > 0"),

            //  multi-output head correctness 
            (Test_Outputs_RangeVsHighLow,     "Outputs: Range head is consistent with High-Low constraint"),
            (Test_Outputs_MultipleTimesteps,  "Outputs: all outputs shaped [seqLen] correctly"),

            //  memory 
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

            //  sequential training 
            (Test_Sequential_MemoryAccumulates,              "Sequential: memory grows across samples"),
            (Test_Sequential_LossDecreases,                  "Sequential: loss decreases over sequential epochs"),
            (Test_Sequential_PriceMemoryPopulated,           "Sequential: price memory has correct embedding dim"),
            (Test_Sequential_MemoryClearedBetweenEpochs,     "Sequential: memory cleared at start of each epoch"),
            (Test_Sequential_WithGlobalToken,                "Sequential: global token path in sequential training works"),
            (Test_Sequential_Produces_Different_Weights_Than_Batch, "Sequential: produces different weights than batch training"),

            //  tokenizer 
            (Test_Tokenizer_SetAndTokenize,           "Tokenizer: SetTokenizer + TokenizeStories works"),
            (Test_Tokenizer_VocabSizeMismatch_Throws, "Tokenizer: oversized tokenizer throws on SetTokenizer"),

            //  validation 
            (Test_Validate_ReturnsFiniteValue,        "Validate: returns finite non-negative value"),

            //  numerical stability 
            (Test_Stability_LargeInputs,         "Stability: large inputs (100x) produce no NaN"),
            (Test_Stability_SmallInputs,         "Stability: tiny inputs (1e-6) produce no NaN"),
            (Test_Stability_ZeroInputs,          "Stability: all-zero inputs produce no NaN"),
            (Test_Stability_NegativeInputs,      "Stability: all-negative inputs produce no NaN"),
            (Test_Stability_ManyStories,         "Stability: 12 simultaneous stories produce no NaN"),
            (Test_Stability_NoNaN_AfterManyEpochs, "Stability: no NaN after 60 training epochs"),

            //  save / load 
            (Test_SaveLoad_ForwardIdentical,    "SaveLoad: forward output identical after round-trip"),
            (Test_SaveLoad_AllHeadsPreserved,   "SaveLoad: all output head weights preserved"),
            (Test_SaveLoad_ContextTypeEmbedding,"SaveLoad: ContextTypeEmbedding preserved"),
            (Test_SaveLoad_GlobalProjection,    "SaveLoad: GlobalFeatureProjection preserved"),
            (Test_SaveLoad_Memory,              "SaveLoad: news + price memory preserved"),
            (Test_SaveLoad_PruningConfig,       "SaveLoad: PruningConfig preserved"),
            (Test_SaveLoad_ContinueTraining,    "SaveLoad: can continue training after load"),

            //  config validation 
            (Test_Config_Validate_Good,    "Config: Validate accepts valid config"),
            (Test_Config_SmallPreset,      "Config: Small() preset validates"),
            (Test_Config_StandardPreset,   "Config: Standard() preset validates"),

            //  gradient checks 
            (Test_GradCheck_RegressionProjection,   "GradCheck: RegressionProjection finite-difference non-zero"),
            (Test_GradCheck_RangeProjection,        "GradCheck: RangeProjection finite-difference non-zero"),
            (Test_GradCheck_DirectionProjection,    "GradCheck: DirectionProjection finite-difference non-zero"),
            (Test_GradCheck_QualityProjection,      "GradCheck: QualityProjection finite-difference non-zero"),
            (Test_GradCheck_MidDirectionProjection, "GradCheck: MidDirectionProjection finite-difference non-zero"),
            (Test_GradCheck_TextEmbedding,          "GradCheck: TextEmbedding affects prediction through cross-attn"),

            //  end-to-end signal learning 
            (Test_E2E_BullBear, "E2E: bull vs bear news drives different predictions"),

            //  semantic correctness 
            (Test_DirectionHead_LearnsBinarySignal,       "Semantic: Direction head learns up/down classification"),
            (Test_RangeHead_ConsistentWithHighLow,        "Semantic: Range head consistent with High-Low after training"),
            (Test_ConfidenceHead_CorrelatesWithAccuracy,  "Semantic: Confidence higher on predictable vs noisy data"),
            (Test_QualityHead_LearnsPredictableData,      "Semantic: Quality head higher on predictable inputs"),
            (Test_GlobalToken_MacroSignalDrivesOutput,    "Semantic: Global macro signal drives prediction direction"),
            (Test_MultiHead_AllOutputsIndependent,        "Semantic: Separate heads can learn conflicting targets"),
            (Test_MidDir_And_Direction_Learn_Different_Signals, "Semantic: MidWindowDirection independent from Direction"),
        };
        void Test_ValidateSequential_Matches_Reference_NoCommitThenCommit()
        {
            var (inputs, targets, ts) = BuildSomeSequentialDataset();

            var cfg = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));
            var trainer = new MmtacTrainer(m, TC(lr: 0.003f, epochs: 40));

            trainer.TrainSequential(inputs, targets, ts);

            float actual = trainer.ValidateSequential(inputs, targets, ts);
            float reference = ValidateSequentialReference(trainer, m, inputs, targets, ts);

            Assert(Math.Abs(actual - reference) < 1e-4f,
                $"ValidateSequential should match reference no-commit/commit-once loop. actual={actual:F6}, ref={reference:F6}");
        }
        void Test_Causal_NoFutureLeakage()
        {
            // Appending arbitrary future rows must not change outputs for earlier timesteps.
            // This directly checks decoder-only causality.
            var cfg = Cfg(embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 16, useConf: true);
            var m = new MmtacModel(cfg, new Random(42));

            var rng = new Random(42);
            int prefixLen = 6;
            int extraLen = 4;

            var full = RandMatrix(prefixLen + extraLen, 5, rng, 0.5f);
            var prefix = new float[prefixLen, 5];

            for (int t = 0; t < prefixLen; t++)
                for (int f = 0; f < 5; f++)
                    prefix[t, f] = full[t, f];

            // Make future rows obviously different so any leakage would show up clearly.
            for (int t = prefixLen; t < prefixLen + extraLen; t++)
                for (int f = 0; f < 5; f++)
                    full[t, f] = 10f + t * 3f + f;

            var inpPrefix = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = prefix
            };

            var inpFull = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = full
            };

            var (regP, rangeP, qualityP, dirP, midP, confP) = m.Forward(inpPrefix);
            var (regF, rangeF, qualityF, dirF, midF, confF) = m.Forward(inpFull);

            const float tol = 1e-5f;

            for (int t = 0; t < prefixLen; t++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert(MathF.Abs(regP[t, j] - regF[t, j]) < tol,
                        $"Future leakage in regression at t={t}, j={j}. prefix={regP[t, j]:F6}, full={regF[t, j]:F6}");
                }

                Assert(MathF.Abs(rangeP[t, 0] - rangeF[t, 0]) < tol,
                    $"Future leakage in range at t={t}. prefix={rangeP[t, 0]:F6}, full={rangeF[t, 0]:F6}");

                Assert(MathF.Abs(qualityP[t, 0] - qualityF[t, 0]) < tol,
                    $"Future leakage in quality at t={t}. prefix={qualityP[t, 0]:F6}, full={qualityF[t, 0]:F6}");

                Assert(MathF.Abs(dirP[t, 0] - dirF[t, 0]) < tol,
                    $"Future leakage in direction at t={t}. prefix={dirP[t, 0]:F6}, full={dirF[t, 0]:F6}");

                Assert(MathF.Abs(midP[t, 0] - midF[t, 0]) < tol,
                    $"Future leakage in mid-direction at t={t}. prefix={midP[t, 0]:F6}, full={midF[t, 0]:F6}");

                Assert(confP != null && confF != null, "Confidence head should be enabled for this test");
                Assert(MathF.Abs(confP[t, 0] - confF[t, 0]) < tol,
                    $"Future leakage in confidence at t={t}. prefix={confP[t, 0]:F6}, full={confF[t, 0]:F6}");
            }
        }
        void Test_Train_LengthMismatch_Throws()
        {
            var (tok, inputs, targets) = Data(n: 3, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            bool threw = false;
            try
            {
                tr.Train(inputs, targets.Take(2).ToArray());
            }
            catch (ArgumentException)
            {
                threw = true;
            }

            Assert(threw, "Train should throw on mismatched inputs/targets lengths");
        }

        void Test_Validate_LengthMismatch_Throws()
        {
            var (tok, inputs, targets) = Data(n: 3, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            bool threw = false;
            try
            {
                tr.Validate(inputs, targets.Take(2).ToArray());
            }
            catch (ArgumentException)
            {
                threw = true;
            }

            Assert(threw, "Validate should throw on mismatched inputs/targets lengths");
        }

        void Test_PredictNext_WrongPriceFeatureDim_Throws()
        {
            var m = new MmtacModel(Cfg(priceFeatures: 5), new Random(42));

            var bad = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = new float[8, 6] // wrong feature count
            };

            bool threw = false;
            try
            {
                _ = m.PredictNext(bad);
            }
            catch (ArgumentException)
            {
                threw = true;
            }

            Assert(threw, "PredictNext should throw when PriceSequence feature dimension does not match config");
        }

        void Test_PredictNext_GlobalFeatureDimMismatch_Throws()
        {
            var m = new MmtacModel(Cfg(globalDim: 3), new Random(42));

            var bad = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                GlobalFeatures = new[] { 1f, 0f } // wrong length
            };

            bool threw = false;
            try
            {
                _ = m.PredictNext(bad);
            }
            catch (ArgumentException)
            {
                threw = true;
            }

            Assert(threw, "PredictNext should throw when GlobalFeatures length does not match config.GlobalFeatureDim");
        }

        void Test_Validate_DoesNotMutateWeights()
        {
            var (tok, inputs, targets) = Data(n: 5, withNews: true, globalDim: 3);

            var m = new MmtacModel(
                Cfg(tok.VocabSize + 2, useConf: true, globalDim: 3),
                new Random(42));

            var priceBefore = CopyMatrix(m.PriceInputProjection);
            var regBefore = CopyMatrix(m.RegressionProjection);
            var rangeBefore = CopyMatrix(m.RangeProjection);
            var dirBefore = CopyMatrix(m.DirectionProjection);
            var ctxBefore = CopyMatrix(m.ContextTypeEmbedding);

            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);

            Assert(float.IsFinite(loss), $"Validate returned non-finite loss: {loss}");

            Assert(!MatrixChanged(priceBefore, m.PriceInputProjection, 1e-8f),
                "Validate should not mutate PriceInputProjection");

            Assert(!MatrixChanged(regBefore, m.RegressionProjection, 1e-8f),
                "Validate should not mutate RegressionProjection");

            Assert(!MatrixChanged(rangeBefore, m.RangeProjection, 1e-8f),
                "Validate should not mutate RangeProjection");

            Assert(!MatrixChanged(dirBefore, m.DirectionProjection, 1e-8f),
                "Validate should not mutate DirectionProjection");

            Assert(!MatrixChanged(ctxBefore, m.ContextTypeEmbedding, 1e-8f),
                "Validate should not mutate ContextTypeEmbedding");
        }

        void Test_ValidateSequential_DoesNotMutateWeights()
        {
            var (tok, inputs, targets) = Data(n: 5, seqLen: 8, withNews: true, globalDim: 3);
            var ts = new[] { 500d, 100d, 600d, 200d, 700d };

            var cfg = Cfg(tok.VocabSize + 2, useConf: true, globalDim: 3, priceSeqLen: 10);
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var m = new MmtacModel(cfg, new Random(42));

            var priceBefore = CopyMatrix(m.PriceInputProjection);
            var regBefore = CopyMatrix(m.RegressionProjection);
            var rangeBefore = CopyMatrix(m.RangeProjection);
            var dirBefore = CopyMatrix(m.DirectionProjection);
            var ctxBefore = CopyMatrix(m.ContextTypeEmbedding);

            float loss = new MmtacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);

            Assert(float.IsFinite(loss), $"ValidateSequential returned non-finite loss: {loss}");

            Assert(!MatrixChanged(priceBefore, m.PriceInputProjection, 1e-8f),
                "ValidateSequential should not mutate PriceInputProjection");

            Assert(!MatrixChanged(regBefore, m.RegressionProjection, 1e-8f),
                "ValidateSequential should not mutate RegressionProjection");

            Assert(!MatrixChanged(rangeBefore, m.RangeProjection, 1e-8f),
                "ValidateSequential should not mutate RangeProjection");

            Assert(!MatrixChanged(dirBefore, m.DirectionProjection, 1e-8f),
                "ValidateSequential should not mutate DirectionProjection");

            Assert(!MatrixChanged(ctxBefore, m.ContextTypeEmbedding, 1e-8f),
                "ValidateSequential should not mutate ContextTypeEmbedding");
        }

        void Test_PredictWithMemory_ZeroMemoryBudget_KeepsBanksEmpty()
        {
            var (tok, inputs, _) = Data(n: 1, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));

            var pred = m.PredictWithMemory(
                inputs[0],
                currentAbsoluteTimestamp: 100.0,
                timeUnitsPerPosition: 1.0,
                maxNewsMemorySize: 0,
                maxPriceMemorySize: 0);

            Assert(!HasNaNPrediction(pred), "PredictWithMemory should still produce a finite prediction with zero memory budgets");
            Assert(m.NewsMemory.Count == 0, $"NewsMemory should remain empty when maxNewsMemorySize=0, actual={m.NewsMemory.Count}");
            Assert(m.PriceMemory.Count == 0, $"PriceMemory should remain empty when maxPriceMemorySize=0, actual={m.PriceMemory.Count}");
        }

        void Test_PredictWithMemory_CustomTimeUnits_ScalesTimestamps()
        {
            var (tok, inputs, _) = Data(n: 1, seqLen: 4, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 8), new Random(42));

            double t0 = 100.0;
            double timeUnits = 0.5;

            _ = m.PredictWithMemory(inputs[0], t0, timeUnits, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);

            Assert(m.PriceMemory.Count == 4, $"Expected 4 price entries, got {m.PriceMemory.Count}");

            for (int i = 0; i < 4; i++)
            {
                double expectedTs = t0 + i * timeUnits;
                Assert(Math.Abs(m.PriceMemory[i].AbsoluteTimestamp - expectedTs) < 1e-9,
                    $"Scaled price timestamp mismatch at {i}. actual={m.PriceMemory[i].AbsoluteTimestamp}, expected={expectedTs}");
            }

            double expectedLast = t0 + 3 * timeUnits;
            Assert(Math.Abs(m.LastPriceTimestamp - expectedLast) < 1e-9,
                $"LastPriceTimestamp mismatch. actual={m.LastPriceTimestamp}, expected={expectedLast}");
        }

        void Test_Sequential_CustomTimeUnits_CommitsScaledTimestamps()
        {
            int seqLen = 4;
            double t0 = 50.0;
            double timeUnits = 0.25;

            var input = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0.2f, 0.1f, -0.1f, 0.05f, 1f),
                NewsStories = null
            };

            var target = ConstantTargets(
                seqLen,
                close: 0.45f,
                range: 0.20f,
                quality: 0.90f,
                direction: 1,
                midDirection: 1);

            var m = new MmtacModel(Cfg(vocabSize: 64, priceSeqLen: 8), new Random(42));
            var tr = new MmtacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1));

            tr.TrainSequential(
                new[] { input },
                new[] { target },
                new[] { t0 },
                timeUnitsPerPosition: timeUnits);

            Assert(m.PriceMemory.Count == seqLen,
                $"Sequential commit should store the full observed sample. expected={seqLen}, actual={m.PriceMemory.Count}");

            for (int i = 0; i < seqLen; i++)
            {
                double expectedTs = t0 + i * timeUnits;
                Assert(Math.Abs(m.PriceMemory[i].AbsoluteTimestamp - expectedTs) < 1e-9,
                    $"Scaled sequential timestamp mismatch at {i}. actual={m.PriceMemory[i].AbsoluteTimestamp}, expected={expectedTs}");
            }

            double expectedLast = t0 + (seqLen - 1) * timeUnits;
            Assert(Math.Abs(m.LastPriceTimestamp - expectedLast) < 1e-9,
                $"Scaled LastPriceTimestamp mismatch. actual={m.LastPriceTimestamp}, expected={expectedLast}");
        }

        void Test_Causal_NoFutureLeakage_WithNewsAndGlobalToken()
        {
            var cfg = Cfg(
                vocabSize: 64,
                embDim: 24,
                numHeads: 4,
                numLayers: 2,
                ffnDim: 48,
                priceSeqLen: 16,
                useConf: true,
                globalDim: 3);

            var m = new MmtacModel(cfg, new Random(42));

            int prefixLen = 6;
            int extraLen = 4;
            var rng = new Random(42);

            var full = RandMatrix(prefixLen + extraLen, 5, rng, 0.5f);
            var prefix = new float[prefixLen, 5];

            for (int t = 0; t < prefixLen; t++)
                for (int f = 0; f < 5; f++)
                    prefix[t, f] = full[t, f];

            for (int t = prefixLen; t < prefixLen + extraLen; t++)
                for (int f = 0; f < 5; f++)
                    full[t, f] = 100f + t * 7f + f;

            var stories = new[]
            {
        new NewsStory(new[] { 1, 2, 3, 4 }, 0f),
        new NewsStory(new[] { 5, 6, 7 }, -2f)
    };

            var globals = new[] { 1f, 0f, 0.5f };

            var inpPrefix = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = prefix,
                NewsStories = stories,
                GlobalFeatures = globals
            };

            var inpFull = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = full,
                NewsStories = stories,
                GlobalFeatures = globals
            };

            var (regP, rangeP, qualityP, dirP, midP, confP) = m.Forward(inpPrefix);
            var (regF, rangeF, qualityF, dirF, midF, confF) = m.Forward(inpFull);

            const float tol = 1e-5f;

            for (int t = 0; t < prefixLen; t++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert(MathF.Abs(regP[t, j] - regF[t, j]) < tol,
                        $"Future leakage with news/global in regression at t={t}, j={j}. prefix={regP[t, j]:F6}, full={regF[t, j]:F6}");
                }

                Assert(MathF.Abs(rangeP[t, 0] - rangeF[t, 0]) < tol,
                    $"Future leakage with news/global in range at t={t}. prefix={rangeP[t, 0]:F6}, full={rangeF[t, 0]:F6}");

                Assert(MathF.Abs(qualityP[t, 0] - qualityF[t, 0]) < tol,
                    $"Future leakage with news/global in quality at t={t}. prefix={qualityP[t, 0]:F6}, full={qualityF[t, 0]:F6}");

                Assert(MathF.Abs(dirP[t, 0] - dirF[t, 0]) < tol,
                    $"Future leakage with news/global in direction at t={t}. prefix={dirP[t, 0]:F6}, full={dirF[t, 0]:F6}");

                Assert(MathF.Abs(midP[t, 0] - midF[t, 0]) < tol,
                    $"Future leakage with news/global in mid-direction at t={t}. prefix={midP[t, 0]:F6}, full={midF[t, 0]:F6}");

                Assert(confP != null && confF != null, "Confidence head should be enabled for this test");
                Assert(MathF.Abs(confP[t, 0] - confF[t, 0]) < tol,
                    $"Future leakage with news/global in confidence at t={t}. prefix={confP[t, 0]:F6}, full={confF[t, 0]:F6}");
            }
        }

        void Test_Decay_Disabled_ForwardAndTrain_Work()
        {
            var (tok, inputs, targets) = Data(n: 6, seqLen: 8, withNews: true, globalDim: 3);

            var cfg = Cfg(
                vocabSize: tok.VocabSize + 2,
                embDim: 24,
                numHeads: 4,
                numLayers: 2,
                ffnDim: 48,
                useConf: true,
                globalDim: 3,
                decayEnabled: false);

            var m = new MmtacModel(cfg, new Random(42));

            var pBefore = m.PredictNext(inputs[0]);
            Assert(!HasNaNPrediction(pBefore), "Decay-disabled PredictNext produced NaN before training");

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 3, epochs: 10)).Train(inputs, targets);

            var pAfter = m.PredictNext(inputs[0]);
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);

            Assert(!HasNaNPrediction(pAfter), "Decay-disabled PredictNext produced NaN after training");
            Assert(float.IsFinite(loss) && loss >= 0f, $"Decay-disabled Validate returned invalid loss: {loss}");
        }
        void Test_Decay_TimeMonotonic_ForIdenticalContent()
        {
            // Same story content, same price input, only arrival time differs.
            // The learned effect should be stronger for newer stories than older stories.
            string storyText = "policy support growth demand";
            var tok = new BPETokenizer();
            tok.Train(new[] { storyText }, vocabSize: 32, minFrequency: 1);
            int[] story = tok.Encode(storyText, addSpecialTokens: true);

            int seqLen = 8;
            int nPerBucket = 24;
            float[] arrivals = { 0f, -3f, -6f };
            float[] closes = { 0.85f, 0.55f, 0.25f };

            int n = nPerBucket * 3;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int i = 0; i < n; i++)
            {
                int bucket = i % 3;
                float arrival = arrivals[bucket];
                float close = closes[bucket];

                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    NewsStories = new[]
                    {
                new NewsStory(story, arrival)
            }
                };

                targets[i] = ConstantTargets(
                    seqLen,
                    close: close,
                    range: 0.20f,
                    quality: 0.95f,
                    direction: close > 0.5f ? 1 : 0,
                    midDirection: close > 0.5f ? 1 : 0);
            }

            var cfg = Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10);
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(m, TC(lr: 0.003f, bs: 12, epochs: 140)).Train(inputs, targets);

            var recentProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[] { new NewsStory(story, 0f) }
            };

            var midProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[] { new NewsStory(story, -3f) }
            };

            var oldProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[] { new NewsStory(story, -6f) }
            };

            var (regRecent, _, _, dirRecent, _, _) = m.Forward(recentProbe);
            var (regMid, _, _, dirMid, _, _) = m.Forward(midProbe);
            var (regOld, _, _, dirOld, _, _) = m.Forward(oldProbe);

            float closeRecent = regRecent[0, 2];
            float closeMid = regMid[0, 2];
            float closeOld = regOld[0, 2];

            Assert(closeRecent > closeMid + 0.08f,
                $"Newer identical story should influence Close more than mid-aged story. recent={closeRecent:F6}, mid={closeMid:F6}");

            Assert(closeMid > closeOld + 0.08f,
                $"Mid-aged identical story should influence Close more than old story. mid={closeMid:F6}, old={closeOld:F6}");

            Assert(dirRecent[0, 0] > dirOld[0, 0] + 0.15f,
                $"Newer identical story should influence Direction more than old story. recent={dirRecent[0, 0]:F6}, old={dirOld[0, 0]:F6}");
        }

        void Test_Decay_ContentAware_SameAgeDifferentContent_DifferentImpact()
        {
            // Same price input, same age, different story content.
            // This checks that impact is not only a function of time.
            string bullText = "central bank support growth demand";
            string bearText = "credit stress slowdown recession";

            var tok = new BPETokenizer();
            tok.Train(new[] { bullText, bearText }, vocabSize: 64, minFrequency: 1);

            int[] bull = tok.Encode(bullText, addSpecialTokens: true);
            int[] bear = tok.Encode(bearText, addSpecialTokens: true);

            int n = 60;
            int seqLen = 8;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];

            for (int i = 0; i < n; i++)
            {
                bool isBull = (i % 2 == 0);

                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    NewsStories = new[]
                    {
                new NewsStory(isBull ? bull : bear, -4f) // same age for both
            }
                };

                targets[i] = ConstantTargets(
                    seqLen,
                    close: isBull ? 0.82f : 0.18f,
                    range: 0.20f,
                    quality: 0.95f,
                    direction: isBull ? 1 : 0,
                    midDirection: isBull ? 1 : 0);
            }

            var cfg = Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10);
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(m, TC(lr: 0.003f, bs: 10, epochs: 140)).Train(inputs, targets);

            var bullProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[] { new NewsStory(bull, -4f) }
            };

            var bearProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[] { new NewsStory(bear, -4f) }
            };

            var (regBull, _, _, dirBull, _, _) = m.Forward(bullProbe);
            var (regBear, _, _, dirBear, _, _) = m.Forward(bearProbe);

            Assert(regBull[0, 2] > regBear[0, 2] + 0.20f,
                $"At the same age, content should still matter. bullClose={regBull[0, 2]:F6}, bearClose={regBear[0, 2]:F6}");

            Assert(dirBull[0, 0] > dirBear[0, 0] + 0.25f,
                $"At the same age, content should still matter for Direction. bullDir={dirBull[0, 0]:F6}, bearDir={dirBear[0, 0]:F6}");
        }

        void Test_Sequential_Training_CommitsFullObservedSample_NotPrefix()
        {
            // TrainSequential predicts on prefix rows but should commit the FULL observed sample to memory.
            int seqLen = 8;

            var input = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0.25f, 0.10f, -0.15f, 0.05f, 1f),
                NewsStories = null
            };

            var target = ConstantTargets(
                seqLen,
                close: 0.50f,
                range: 0.20f,
                quality: 0.90f,
                direction: 1,
                midDirection: 1);

            var cfg = Cfg(vocabSize: 64, priceSeqLen: 12);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));
            var tr = new MmtacTrainer(m, TC(lr: 0f, bs: 1, epochs: 1)); // lr=0 isolates memory-commit behavior

            tr.TrainSequential(
                new[] { input },
                new[] { target },
                new[] { 100.0 });

            Assert(m.NewsMemory.Count == 0,
                $"No news stories were supplied, so NewsMemory should remain empty. count={m.NewsMemory.Count}");

            Assert(m.PriceMemory.Count == seqLen,
                $"Sequential training should commit the FULL observed sample, not the prefix. expected={seqLen}, actual={m.PriceMemory.Count}");

            Assert(Math.Abs(m.LastPriceTimestamp - (100.0 + seqLen - 1)) < 1e-9,
                $"LastPriceTimestamp should reflect the full observed sample. actual={m.LastPriceTimestamp}, expected={100.0 + seqLen - 1}");

            for (int t = 0; t < seqLen; t++)
            {
                double expectedTs = 100.0 + t;
                Assert(Math.Abs(m.PriceMemory[t].AbsoluteTimestamp - expectedTs) < 1e-9,
                    $"PriceMemory timestamp mismatch at row {t}. actual={m.PriceMemory[t].AbsoluteTimestamp}, expected={expectedTs}");
            }
        }

        void Test_Sequential_SkippedSample_DoesNotCommitMemory()
        {
            // A skipped sample (seqLen < 2) must not alter rolling memory.
            var cfgA = Cfg(vocabSize: 64, priceSeqLen: 12);
            var cfgB = Cfg(vocabSize: 64, priceSeqLen: 12);

            cfgA.Output.CloseDirectionConsistencyWeight = 0f;
            cfgA.Output.CloseDirectionConsistencyMargin = 0f;

            cfgB.Output.CloseDirectionConsistencyWeight = 0f;
            cfgB.Output.CloseDirectionConsistencyMargin = 0f;

            var validInput = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(6, 0.2f, 0.1f, -0.1f, 0.05f, 1f),
                NewsStories = null
            };

            var validTarget = ConstantTargets(
                6,
                close: 0.50f,
                range: 0.20f,
                quality: 0.85f,
                direction: 1,
                midDirection: 1);

            // This sample should be skipped by TrainSequential because seqLen < 2.
            var skippedInput = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(1, 0.9f, 0.9f, 0.9f, 0.9f, 1f),
                NewsStories = new[]
                {
            new NewsStory(new[] { 1, 2, 3 }, 0f)
        }
            };

            var skippedTarget = ConstantTargets(
                1,
                close: 0.95f,
                range: 0.10f,
                quality: 0.95f,
                direction: 1,
                midDirection: 1);

            var mOnlyValid = new MmtacModel(cfgA, new Random(42));
            var mWithSkipped = new MmtacModel(cfgB, new Random(42));

            var trOnlyValid = new MmtacTrainer(mOnlyValid, TC(lr: 0f, bs: 1, epochs: 1));
            var trWithSkipped = new MmtacTrainer(mWithSkipped, TC(lr: 0f, bs: 1, epochs: 1));

            trOnlyValid.TrainSequential(
                new[] { validInput },
                new[] { validTarget },
                new[] { 100.0 });

            trWithSkipped.TrainSequential(
                new[] { validInput, skippedInput },
                new[] { validTarget, skippedTarget },
                new[] { 100.0, 200.0 });

            Assert(mWithSkipped.NewsMemory.Count == mOnlyValid.NewsMemory.Count,
                $"Skipped sample should not commit news memory. validOnly={mOnlyValid.NewsMemory.Count}, withSkipped={mWithSkipped.NewsMemory.Count}");

            Assert(mWithSkipped.PriceMemory.Count == mOnlyValid.PriceMemory.Count,
                $"Skipped sample should not commit price memory. validOnly={mOnlyValid.PriceMemory.Count}, withSkipped={mWithSkipped.PriceMemory.Count}");

            Assert(Math.Abs(mWithSkipped.LastPriceTimestamp - mOnlyValid.LastPriceTimestamp) < 1e-9,
                $"Skipped sample should not change LastPriceTimestamp. validOnly={mOnlyValid.LastPriceTimestamp}, withSkipped={mWithSkipped.LastPriceTimestamp}");

            Assert(mWithSkipped.NewsMemory.Count == 0,
                $"The only sample with news was skipped, so NewsMemory should remain empty. count={mWithSkipped.NewsMemory.Count}");
        }
        static (MultimodalInput[] inputs, ModelTarget[][] targets, double[] timestamps) BuildSomeSequentialDataset()
        {
            var rng = new Random(123);
            int n = 12;
            int seqLen = 8;

            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            var timestamps = new double[n];

            float val = 0.25f;

            for (int s = 0; s < n; s++)
            {
                var ps = new float[seqLen, 5];
                var tg = new ModelTarget[seqLen];

                for (int t = 0; t < seqLen; t++)
                {
                    ps[t, 0] = val;
                    ps[t, 1] = val * val;
                    ps[t, 2] = val > 0f ? 1f : -1f;
                    ps[t, 3] = val * 0.5f;
                    ps[t, 4] = 1f;

                    tg[t] = new ModelTarget
                    {
                        High = val + 0.05f,
                        Low = val - 0.05f,
                        Close = val,
                        Range = 0.10f,
                        Quality = 0.90f,
                        Direction = val > 0f ? 1f : 0f,
                        MidWindowDirection = val > 0f ? 1f : 0f
                    };

                    val = 0.85f * val + 0.15f * ((float)rng.NextDouble() - 0.5f);
                }

                inputs[s] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = ps
                };

                targets[s] = tg;
                timestamps[s] = s * 10.0;
            }

            return (inputs, targets, timestamps);
        }

        static float ComputeSequentialValLossReference(MmtacConfig cfg, ModelPrediction pred, ModelTarget tgt)
        {
            float totalLoss = 0f;

            float dh = pred.High - tgt.High;
            totalLoss += dh * dh;

            float dl = pred.Low - tgt.Low;
            totalLoss += dl * dl;

            float dc = pred.Close - tgt.Close;
            totalLoss += dc * dc;

            float dr = pred.Range - tgt.Range;
            totalLoss += cfg.Output.RangeLossWeight * dr * dr;

            float dq = pred.Quality - tgt.Quality;
            totalLoss += cfg.Output.QualityLossWeight * dq * dq;

            float dd = pred.DirectionProb - tgt.Direction;
            totalLoss += cfg.Output.DirectionLossWeight * dd * dd;

            float dmd = pred.MidWindowDirectionProb - tgt.MidWindowDirection;
            totalLoss += cfg.Output.MidDirectionLossWeight * dmd * dmd;

            return totalLoss;
        }

        static float[,] SliceRowsForTest(float[,] m, int start, int endExclusive)
        {
            int len = endExclusive - start;
            int cols = m.GetLength(1);

            var result = new float[len, cols];

            for (int i = 0; i < len; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = m[start + i, j];
                }
            }

            return result;
        }

        static float ValidateSequentialReference(MmtacTrainer trainer, MmtacModel model, MultimodalInput[] inputs, ModelTarget[][] targets, double[] timestamps, double timeUnitsPerPosition = 1.0, int maxNewsMemory = 100, int maxPriceMemory = 200)
        {
            var predictMethod = typeof(MmtacTrainer).GetMethod(
                "PredictWithCurrentMemoryNoCommit",
                BindingFlags.Instance | BindingFlags.NonPublic);

            var commitMethod = typeof(MmtacTrainer).GetMethod(
                "CommitObservedSampleToMemory",
                BindingFlags.Instance | BindingFlags.NonPublic);

            if (predictMethod == null)
                throw new InvalidOperationException(
                    "Could not find MmtacTrainer.PredictWithCurrentMemoryNoCommit. Add that helper first.");

            if (commitMethod == null)
                throw new InvalidOperationException(
                    "Could not find MmtacTrainer.CommitObservedSampleToMemory. Add that helper first.");

            float total = 0f;
            int count = 0;

            int[] ordered = Enumerable.Range(0, inputs.Length)
                                      .OrderBy(i => timestamps[i])
                                      .ToArray();

            model.ClearAllMemory();

            try
            {
                foreach (int idx in ordered)
                {
                    var input = inputs[idx];
                    var sampleTargets = targets[idx];

                    if (input == null || input.PriceSequence == null || sampleTargets == null)
                        continue;

                    int sl = input.PriceSequence.GetLength(0);

                    if (sl < 2 || sampleTargets.Length < sl)
                        continue;

                    var priceInp = SliceRowsForTest(input.PriceSequence, 0, sl - 1);

                    var wrappedInput = new MultimodalInput
                    {
                        PredictionTimestamp = input.PredictionTimestamp,
                        PriceSequence = priceInp,
                        GlobalFeatures = input.GlobalFeatures,
                        NewsStories = input.NewsStories
                    };

                    var pred = (ModelPrediction)predictMethod.Invoke(trainer, new object[] { wrappedInput, timestamps[idx], timeUnitsPerPosition });

                    var tgt = sampleTargets[sl - 1];

                    total += ComputeSequentialValLossReference(model.Config, pred, tgt);
                    count++;

                    commitMethod.Invoke(
                        trainer,
                        new object[]
                        {
                            input,
                            timestamps[idx],
                            timeUnitsPerPosition,
                            maxNewsMemory,
                            maxPriceMemory
                        });
                }

                return count > 0 ? total / count : 0f;
            }
            finally
            {
                model.ClearAllMemory();
            }
        }
        void Test_LiveRollingMemory_Uses_Previous_Sample_Causally()
        {
            int n = 80;
            int seqLen = 6;

            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            var ts = new double[n];

            // Sample i target depends on sample i-1 regime.
            // Current sample itself is mostly neutral, so memory has to matter.
            float[] regime = new float[n];
            regime[0] = 0f;

            for (int i = 1; i < n; i++)
            {
                regime[i] = (i % 2 == 0) ? 0.8f : -0.8f;
            }

            for (int i = 0; i < n; i++)
            {
                var ps = new float[seqLen, 5];
                var tg = new ModelTarget[seqLen];

                // Current sample price input is deliberately weak/neutral.
                for (int t = 0; t < seqLen; t++)
                {
                    ps[t, 0] = 0f;
                    ps[t, 1] = 0f;
                    ps[t, 2] = 0f;
                    ps[t, 3] = 0f;
                    ps[t, 4] = 1f;
                }

                // Target depends on PREVIOUS sample regime only.
                float prev = i == 0 ? 0f : regime[i - 1];

                for (int t = 0; t < seqLen; t++)
                {
                    tg[t] = new ModelTarget
                    {
                        High = prev + 0.10f,
                        Low = prev - 0.10f,
                        Close = prev,
                        Range = 0.20f,
                        Quality = 0.95f,
                        Direction = prev > 0f ? 1f : 0f,
                        MidWindowDirection = prev > 0f ? 1f : 0f
                    };
                }

                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = ps
                };

                targets[i] = tg;
                ts[i] = i * 10.0;
            }

            var cfg = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: seqLen + 2);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(m, TC(lr: 0.003f, epochs: 100)).TrainSequential(inputs, targets, ts);

            var positiveDriver = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0.8f, 0f, 0f, 0f, 1f)
            };

            var negativeDriver = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, -0.8f, 0f, 0f, 0f, 1f)
            };

            var neutralProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f)
            };

            // Stateless baseline
            m.ClearAllMemory();
            var freshPred = m.PredictNext(neutralProbe);

            // Positive memory path
            m.ClearAllMemory();
            _ = m.PredictWithMemory(positiveDriver, currentAbsoluteTimestamp: 0.0);
            var posPred = m.PredictWithMemory(neutralProbe, currentAbsoluteTimestamp: 10.0);

            // Negative memory path
            m.ClearAllMemory();
            _ = m.PredictWithMemory(negativeDriver, currentAbsoluteTimestamp: 0.0);
            var negPred = m.PredictWithMemory(neutralProbe, currentAbsoluteTimestamp: 10.0);

            Assert(posPred.Close > negPred.Close + 0.20f,
                $"Rolling price memory should shift Close prediction. pos={posPred.Close:F6}, neg={negPred.Close:F6}");

            Assert(posPred.DirectionProb > negPred.DirectionProb + 0.20f,
                $"Rolling price memory should shift DirectionProb. pos={posPred.DirectionProb:F6}, neg={negPred.DirectionProb:F6}");

            Assert(Math.Abs(posPred.Close) > Math.Abs(freshPred.Close) + 0.05f ||
                   Math.Abs(negPred.Close) > Math.Abs(freshPred.Close) + 0.05f,
                $"Memory-backed probe should differ materially from fresh probe. fresh={freshPred.Close:F6}, pos={posPred.Close:F6}, neg={negPred.Close:F6}");
        }

        void Test_Ablation_Multimodal_Each_Modality_Matters()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = MultimodalHeldOutData();
            var m = new MmtacModel(
                Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, globalDim: 3),
                new Random(42));

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 160)).Train(trainInputs, trainTargets);

            float baseLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

            var noNews = testInputs.Select(x => CloneInput(x, keepPrice: true, keepNews: false, keepGlobals: true)).ToArray();
            var noGlobals = testInputs.Select(x => CloneInput(x, keepPrice: true, keepNews: true, keepGlobals: false)).ToArray();
            var noPrice = testInputs.Select(x => CloneInput(x, keepPrice: false, keepNews: true, keepGlobals: true)).ToArray();

            float noNewsLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(noNews, testTargets);
            float noGlobalsLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(noGlobals, testTargets);
            float noPriceLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(noPrice, testTargets);

            Assert(noNewsLoss > baseLoss * 1.10f,
                $"Removing news should hurt. base={baseLoss:F6}, noNews={noNewsLoss:F6}");

            Assert(noGlobalsLoss > baseLoss * 1.08f,
                $"Removing globals should hurt. base={baseLoss:F6}, noGlobals={noGlobalsLoss:F6}");

            Assert(noPriceLoss > baseLoss * 1.10f,
                $"Removing price should hurt. base={baseLoss:F6}, noPrice={noPriceLoss:F6}");
        }

        void Test_Sequential_NewsMemory_Helps_On_RegimeCarryTask()
        {
            var (tok, trainInputs, trainTargets, trainTs, testInputs, testTargets, testTs) = RegimeCarryData();

            var cfg = Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10);

            var model = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(model, TC(lr: 0.003f, bs: 4, epochs: 80)).TrainSequential(trainInputs, trainTargets, trainTs);

            float withNewsMemoryErr = AverageCarryCloseError_NewsMemoryOnly(model, testInputs, testTargets, testTs);

            float withoutMemoryErr = 0f;
            int count = 0;

            for (int i = 0; i < testInputs.Length; i++)
            {
                // Only evaluate carry positions, where memory is supposed to matter.
                if ((i % 4) == 0)
                    continue;

                var pred = model.PredictNext(testInputs[i]);
                float tgt = testTargets[i][testTargets[i].Length - 1].Close;
                withoutMemoryErr += MathF.Abs(pred.Close - tgt);
                count++;
            }

            withoutMemoryErr /= Math.Max(1, count);

            Assert(withNewsMemoryErr < withoutMemoryErr * 0.90f,
                $"News memory should help on carried-regime task. withNewsMemory={withNewsMemoryErr:F6}, withoutMemory={withoutMemoryErr:F6}");

            Assert(withNewsMemoryErr < 0.30f,
                $"News-memory carry error still too high. withNewsMemory={withNewsMemoryErr:F6}");
        }
        private float AverageCarryCloseError_NewsMemoryOnly(MmtacModel model, MultimodalInput[] inputs, ModelTarget[][] targets, double[] timestamps, int blockSize = 4, double timeUnitsPerPosition = 1.0)
        {
            model.ClearAllMemory();

            float err = 0f;
            int count = 0;

            for (int i = 0; i < inputs.Length; i++)
            {
                // Keep only news memory active for this synthetic task.
                model.ClearPriceMemory();

                var pred = model.PredictWithMemory(inputs[i], timestamps[i], timeUnitsPerPosition);

                // Only evaluate the carry positions, not the first sample where the story first appears.
                if ((i % blockSize) != 0)
                {
                    float tgt = targets[i][targets[i].Length - 1].Close;
                    err += MathF.Abs(pred.Close - tgt);
                    count++;
                }
            }

            model.ClearAllMemory();
            return err / Math.Max(1, count);
        }
        void Test_Robustness_MixedMissingModalities_Still_Trains()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = MixedMissingModalitiesData();

            var m = new MmtacModel(
                Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, globalDim: 3),
                new Random(42));

            float beforeLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 150)).Train(trainInputs, trainTargets);

            float afterLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

            Assert(float.IsFinite(afterLoss), $"afterLoss not finite: {afterLoss}");
            Assert(afterLoss < beforeLoss * 0.55f,
                $"Mixed-modality held-out loss did not improve enough. before={beforeLoss:F6}, after={afterLoss:F6}");

            AssertAllPredictionsFinite(m, testInputs, "mixed-missing-modalities");
        }

        void Test_Robustness_NeutralDistractorStories_DoNot_Overwhelm_Signal()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = NewsHeldOutData();

            var m = new MmtacModel(
                Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64),
                new Random(42));

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 120)).Train(trainInputs, trainTargets);

            int[] bull = tok.Encode("strong bullish outlook demand growth", addSpecialTokens: true);
            int[] bear = tok.Encode("weak bearish outlook contraction risk", addSpecialTokens: true);
            int[] neutral = tok.Encode("routine market update commentary", addSpecialTokens: true);

            var bullProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[]
                {
            new NewsStory(bull, 0f),
            new NewsStory(neutral, -1f),
            new NewsStory(neutral, -2f),
            new NewsStory(neutral, -3f),
            new NewsStory(neutral, -4f),
            new NewsStory(neutral, -5f),
        }
            };

            var bearProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[]
                {
            new NewsStory(bear, 0f),
            new NewsStory(neutral, -1f),
            new NewsStory(neutral, -2f),
            new NewsStory(neutral, -3f),
            new NewsStory(neutral, -4f),
            new NewsStory(neutral, -5f),
        }
            };

            var bullPred = m.PredictNext(bullProbe);
            var bearPred = m.PredictNext(bearProbe);

            Assert(bullPred.DirectionProb > bearPred.DirectionProb + 0.20f,
                $"Neutral distractors should not overwhelm news signal. BullDir={bullPred.DirectionProb:F6}, BearDir={bearPred.DirectionProb:F6}");

            Assert(bullPred.Close > bearPred.Close + 0.15f,
                $"Neutral distractors should not collapse Close separation. BullClose={bullPred.Close:F6}, BearClose={bearPred.Close:F6}");
        }

        void Test_SaveLoad_HeldOutValidationIdentical()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = MultimodalHeldOutData();

            var m = new MmtacModel(
                Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, globalDim: 3),
                new Random(42));

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 120)).Train(trainInputs, trainTargets);

            float lossBefore = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);
            var predBefore = m.PredictNext(testInputs[0]);

            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var loaded = MmtacModel.Load(dir);

                float lossAfter = new MmtacTrainer(loaded, TC(epochs: 1)).Validate(testInputs, testTargets);
                var predAfter = loaded.PredictNext(testInputs[0]);

                Assert(MathF.Abs(lossBefore - lossAfter) < 1e-5f,
                    $"Held-out validation loss changed after reload. before={lossBefore:F6}, after={lossAfter:F6}");

                Assert(MathF.Abs(predBefore.Close - predAfter.Close) < 1e-5f, "Close mismatch after reload");
                Assert(MathF.Abs(predBefore.DirectionProb - predAfter.DirectionProb) < 1e-5f, "Direction mismatch after reload");
                Assert(MathF.Abs(predBefore.Quality - predAfter.Quality) < 1e-5f, "Quality mismatch after reload");
            }
            finally
            {
                Cleanup(dir);
            }
        }

        void Test_Training_MultipleSeeds_AllRemainFinite_And_Learn()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = PriceHeldOutData();

            for (int seed = 1; seed <= 3; seed++)
            {
                var m = new MmtacModel(
                    Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64),
                    new Random(seed));

                float beforeLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

                new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 120)).Train(trainInputs, trainTargets);

                float afterLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

                Assert(float.IsFinite(afterLoss), $"seed {seed}: afterLoss not finite ({afterLoss})");
                Assert(afterLoss < beforeLoss * 0.60f,
                    $"seed {seed}: held-out loss did not improve enough. before={beforeLoss:F6}, after={afterLoss:F6}");

                AssertAllPredictionsFinite(m, testInputs, $"multiple-seeds seed={seed}");
            }
        }
        void Test_Train_Generalises_PriceMapping_HeldOut()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = PriceHeldOutData();

            var cfg = Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64);
            cfg.PriceContext.Enabled = false;
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));

            float beforeLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);
            float beforeCloseErr = MeanCloseAbsError(m, testInputs, testTargets);

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 120)).Train(trainInputs, trainTargets);

            float afterLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);
            float afterCloseErr = MeanCloseAbsError(m, testInputs, testTargets);

            Assert(afterLoss < beforeLoss * 0.45f,
                $"Held-out loss did not improve enough. Before={beforeLoss:F6}, After={afterLoss:F6}");

            Assert(afterCloseErr < beforeCloseErr * 0.45f,
                $"Held-out close error did not improve enough. Before={beforeCloseErr:F6}, After={afterCloseErr:F6}");

            Assert(afterCloseErr < 0.08f,
                $"Held-out close error still too high for deterministic price mapping. Err={afterCloseErr:F6}");
        }

        void Test_Train_Generalises_NewsSignal_HeldOut()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = NewsHeldOutData();
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64), new Random(42));

            float beforeLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 120)).Train(trainInputs, trainTargets);

            float afterLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

            Assert(afterLoss < beforeLoss * 0.35f, $"Held-out news loss did not improve enough. Before={beforeLoss:F6}, After={afterLoss:F6}");

            var bullProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[]
                {
                    new NewsStory(tok.Encode("strong bullish outlook demand growth", addSpecialTokens: true), 0f),
                    new NewsStory(tok.Encode("routine market update commentary", addSpecialTokens: true), -3f)
                }
            };

            var bearProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[]
                {
                    new NewsStory(tok.Encode("weak bearish outlook contraction risk", addSpecialTokens: true), 0f),
                    new NewsStory(tok.Encode("routine market update commentary", addSpecialTokens: true), -3f)
                }
            };

            var bullPred = m.PredictNext(bullProbe);
            var bearPred = m.PredictNext(bearProbe);

            Assert(bullPred.DirectionProb > bearPred.DirectionProb + 0.30f, $"Bull news should raise DirectionProb. Bull={bullPred.DirectionProb:F6}, Bear={bearPred.DirectionProb:F6}");

            Assert(bullPred.Close > bearPred.Close + 0.20f, $"Bull news should raise Close prediction. Bull={bullPred.Close:F6}, Bear={bearPred.Close:F6}");
        }

        void Test_Train_Generalises_GlobalFeatureSignal_HeldOut()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = GlobalHeldOutData();
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, globalDim: 3), new Random(42));

            float beforeLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 140)).Train(trainInputs, trainTargets);

            float afterLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);

            Assert(afterLoss < beforeLoss * 0.45f, $"Held-out global-feature loss did not improve enough. Before={beforeLoss:F6}, After={afterLoss:F6}");

            var riskOn = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                GlobalFeatures = new[] { 1f, 0f, 1f }
            };

            var riskOff = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                GlobalFeatures = new[] { 0f, 1f, 0f }
            };

            var onPred = m.PredictNext(riskOn);
            var offPred = m.PredictNext(riskOff);

            Assert(onPred.DirectionProb > offPred.DirectionProb + 0.25f,
                $"Risk-on globals should raise DirectionProb. On={onPred.DirectionProb:F6}, Off={offPred.DirectionProb:F6}");

            Assert(onPred.Close > offPred.Close + 0.15f,
                $"Risk-on globals should raise Close prediction. On={onPred.Close:F6}, Off={offPred.Close:F6}");
        }


        void Test_Train_Generalises_MultimodalCombinedSignal_HeldOut()
        {
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = MultimodalHeldOutData();

            var cfg = Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, globalDim: 3);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));

            float beforeLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);
            float beforeCloseErr = MeanCloseAbsError(m, testInputs, testTargets);

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 160)).Train(trainInputs, trainTargets);

            float afterLoss = new MmtacTrainer(m, TC(epochs: 1)).Validate(testInputs, testTargets);
            float afterCloseErr = MeanCloseAbsError(m, testInputs, testTargets);

            Assert(afterLoss < beforeLoss * 0.50f,
                $"Held-out multimodal loss did not improve enough. Before={beforeLoss:F6}, After={afterLoss:F6}");

            Assert(afterCloseErr < beforeCloseErr * 0.50f,
                $"Held-out multimodal close error did not improve enough. Before={beforeCloseErr:F6}, After={afterCloseErr:F6}");

            var positiveProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.8f, 0.64f, 1f, 0.4f, 1f),
                NewsStories = new[]
                {
            new NewsStory(tok.Encode("central bank support growth demand", addSpecialTokens: true), 0f)
        },
                GlobalFeatures = new[] { 1f, 0f, 0.8f }
            };

            var negativeProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, -0.8f, 0.64f, -1f, -0.4f, 1f),
                NewsStories = new[]
                {
            new NewsStory(tok.Encode("credit stress slowdown recession", addSpecialTokens: true), 0f)
        },
                GlobalFeatures = new[] { 0f, 1f, 0.8f }
            };

            var posPred = m.PredictNext(positiveProbe);
            var negPred = m.PredictNext(negativeProbe);

            Assert(posPred.DirectionProb > negPred.DirectionProb + 0.30f,
                $"Positive multimodal regime should raise DirectionProb. Pos={posPred.DirectionProb:F6}, Neg={negPred.DirectionProb:F6}");

            Assert(posPred.Close > negPred.Close + 0.20f,
                $"Positive multimodal regime should raise Close prediction. Pos={posPred.Close:F6}, Neg={negPred.Close:F6}");

            Assert(posPred.Quality > negPred.Quality,
                $"Quality should be higher on the stronger positive synthetic regime. Pos={posPred.Quality:F6}, Neg={negPred.Quality:F6}");
        }
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
        void Test_ValidateSequential_OrderOnly_Diagnostic()
        {
            var (tok, inputs, targets) = Data(n: 6, seqLen: 8, withNews: true, globalDim: 3);
            var ts = new[] { 500d, 100d, 600d, 200d, 700d, 300d };

            int[] perm = { 2, 5, 1, 4, 0, 3 };
            var shufInputs = perm.Select(i => inputs[i]).ToArray();
            var shufTargets = perm.Select(i => targets[i]).ToArray();
            var shufTs = perm.Select(i => ts[i]).ToArray();

            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            float v1 = tr.ValidateSequential(inputs, targets, ts);
            float v2 = tr.ValidateSequential(shufInputs, shufTargets, shufTs);

            Assert(MathF.Abs(v1 - v2) < 1e-6f, $"ValidateSequential is not order-invariant. v1={v1:F6}, v2={v2:F6}");
        }
        void Test_Sequential_TimestampShuffle_Invariant()
        {
            var (tok, inputs, targets) = Data(n: 6, seqLen: 8, withNews: true, globalDim: 3);

            var ts = new[] { 500d, 100d, 600d, 200d, 700d, 300d };

            int[] perm = { 2, 5, 1, 4, 0, 3 };
            var shufInputs = perm.Select(i => inputs[i]).ToArray();
            var shufTargets = perm.Select(i => targets[i]).ToArray();
            var shufTs = perm.Select(i => ts[i]).ToArray();

            var cfg1 = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3);
            cfg1.Decay.MemAttentionDropout = 0f;
            cfg1.Decay.MlpDropout = 0f;
            cfg1.Runtime.AccelerationType = AccelerationType.CPU;

            var cfg2 = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3);
            cfg2.Decay.MemAttentionDropout = 0f;
            cfg2.Decay.MlpDropout = 0f;
            cfg2.Runtime.AccelerationType = AccelerationType.CPU;

            var m1 = new MmtacModel(cfg1, new Random(42));
            var m2 = new MmtacModel(cfg2, new Random(42));

            // Zero LR removes actual learning; this test is about chronological processing.
            var tc = TC(lr: 0f, epochs: 1);

            new MmtacTrainer(m1, tc).TrainSequential(inputs, targets, ts);
            new MmtacTrainer(m2, tc).TrainSequential(shufInputs, shufTargets, shufTs);

            Assert(m1.NewsMemory.Count == m2.NewsMemory.Count,
                $"NewsMemory count mismatch after timestamp-sorted sequential pass. m1={m1.NewsMemory.Count}, m2={m2.NewsMemory.Count}");

            Assert(m1.PriceMemory.Count == m2.PriceMemory.Count,
                $"PriceMemory count mismatch after timestamp-sorted sequential pass. m1={m1.PriceMemory.Count}, m2={m2.PriceMemory.Count}");

            Assert(Math.Abs(m1.LastPriceTimestamp - m2.LastPriceTimestamp) < 1e-9,
                $"LastPriceTimestamp mismatch. m1={m1.LastPriceTimestamp}, m2={m2.LastPriceTimestamp}");

            var newsTs1 = m1.NewsMemory.Select(e => e.AbsoluteTimestamp).ToArray();
            var newsTs2 = m2.NewsMemory.Select(e => e.AbsoluteTimestamp).ToArray();
            Assert(newsTs1.SequenceEqual(newsTs2),
                "News memory timestamps differ after timestamp-sorted sequential pass");

            var priceTs1 = m1.PriceMemory.Select(e => e.AbsoluteTimestamp).ToArray();
            var priceTs2 = m2.PriceMemory.Select(e => e.AbsoluteTimestamp).ToArray();
            Assert(priceTs1.SequenceEqual(priceTs2),
                "Price memory timestamps differ after timestamp-sorted sequential pass");
        }
        void Test_TrainSequential_ZeroLearningRate_DoesNotChangeWeights()
        {
            var (tok, inputs, targets) = Data(n: 4, seqLen: 8, withNews: true, globalDim: 3);
            var ts = new[] { 100d, 200d, 300d, 400d };

            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3), new Random(42));

            var priceProjBefore = (float[,])m.PriceInputProjection.Clone();
            var regBefore = (float[,])m.RegressionProjection.Clone();
            var ctxBefore = (float[,])m.ContextTypeEmbedding.Clone();

            new MmtacTrainer(m, TC(lr: 0f, epochs: 1)).TrainSequential(inputs, targets, ts);

            Assert(!MatrixChanged(priceProjBefore, m.PriceInputProjection, 1e-8f),
                "PriceInputProjection changed despite zero learning rate");

            Assert(!MatrixChanged(regBefore, m.RegressionProjection, 1e-8f),
                "RegressionProjection changed despite zero learning rate");

            Assert(!MatrixChanged(ctxBefore, m.ContextTypeEmbedding, 1e-8f),
                "ContextTypeEmbedding changed despite zero learning rate");
        }
        void Test_Sequential_LiveNews_Affects_Current_Step_Immediately()
        {
            string[] corpus =
            {
                "strong bullish outlook demand growth",
                "weak bearish outlook contraction risk",
                "routine market update commentary"
            };

            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 64, minFrequency: 1);

            int[] bull = tok.Encode("strong bullish outlook demand growth", addSpecialTokens: true);
            int[] bear = tok.Encode("weak bearish outlook contraction risk", addSpecialTokens: true);

            int n = 40;
            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            var ts = new double[n];

            for (int i = 0; i < n; i++)
            {
                bool isBull = (i % 2 == 0);

                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow,
                    PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                    NewsStories = new[]
                    {
                        new NewsStory(isBull ? bull : bear, 0f)
                    }
                };

                targets[i] = Enumerable.Range(0, 8).Select(_ => new ModelTarget
                {
                    High = isBull ? 0.90f : 0.30f,
                    Low = isBull ? 0.70f : 0.10f,
                    Close = isBull ? 0.80f : 0.20f,
                    Range = 0.20f,
                    Quality = 0.95f,
                    Direction = isBull ? 1f : 0f,
                    MidWindowDirection = isBull ? 1f : 0f
                }).ToArray();

                ts[i] = i * 10.0;
            }

            var cfg = Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10);
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(m, TC(lr: 0.003f, epochs: 80)).TrainSequential(inputs, targets, ts);

            m.ClearAllMemory();

            var bullProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[] { new NewsStory(bull, 0f) }
            };

            var bearProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0f, 0f, 0f, 0f, 1f),
                NewsStories = new[] { new NewsStory(bear, 0f) }
            };

            var (bullReg, _, _, bullDir, _, _) = m.Forward(bullProbe);
            var (bearReg, _, _, bearDir, _, _) = m.Forward(bearProbe);

            // Immediate effect = timestep 0
            Assert(bullDir[0, 0] > bearDir[0, 0] + 0.20f, $"Current live news should affect current-step direction. bull={bullDir[0, 0]:F6}, bear={bearDir[0, 0]:F6}");

            Assert(bullReg[0, 2] > bearReg[0, 2] + 0.15f, $"Current live news should affect current-step close. bull={bullReg[0, 2]:F6}, bear={bearReg[0, 2]:F6}");
        }

        void Test_Memory_ExactGrowthCounts_BeforePrune()
        {
            var m = new MmtacModel(Cfg(vocabSize: 64, priceSeqLen: 10), new Random(42));

            var inp1 = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.1f, 0.2f, -0.1f, 0.05f, 1f),
                NewsStories = new[]
                {
            new NewsStory(new[] { 1, 2, 3 }, 0f),
            new NewsStory(new[] { 4, 5, 6 }, -1f),
            new NewsStory(new[] { 7, 8, 9 }, -2f),
        }
            };

            int n0 = m.NewsMemory.Count;
            int p0 = m.PriceMemory.Count;

            m.PredictWithMemory(inp1, 100.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);

            int newsDelta1 = m.NewsMemory.Count - n0;
            int priceDelta1 = m.PriceMemory.Count - p0;

            Assert(newsDelta1 == 3, $"Expected news delta 3, got {newsDelta1}");
            Assert(priceDelta1 == inp1.PriceSequence.GetLength(0),
                $"Expected price delta {inp1.PriceSequence.GetLength(0)}, got {priceDelta1}");

            var inp2 = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(5, -0.2f, 0.1f, 0.2f, -0.05f, 1f),
                NewsStories = new[]
                {
            new NewsStory(new[] { 10, 11, 12 }, 0f),
            new NewsStory(new[] { 13, 14, 15 }, -1f),
        }
            };

            int n1 = m.NewsMemory.Count;
            int p1 = m.PriceMemory.Count;

            m.PredictWithMemory(inp2, 200.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);

            int newsDelta2 = m.NewsMemory.Count - n1;
            int priceDelta2 = m.PriceMemory.Count - p1;

            Assert(newsDelta2 == 2, $"Expected news delta 2, got {newsDelta2}");
            Assert(priceDelta2 == inp2.PriceSequence.GetLength(0),
                $"Expected price delta {inp2.PriceSequence.GetLength(0)}, got {priceDelta2}");
        }

        void Test_SaveLoad_SequentialMemoryPredictionIdentical()
        {
            var (tok, inputs, targets) = Data(n: 4, seqLen: 8, withNews: true, globalDim: 3);

            var cfg = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3);
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var m = new MmtacModel(cfg, new Random(42));
            var trainer = new MmtacTrainer(m, TC(lr: 0.003f, epochs: 5));
            var ts = new[] { 100d, 200d, 300d, 400d };

            trainer.TrainSequential(inputs, targets, ts);

            // Build non-empty rolling memory state.
            m.PredictWithMemory(inputs[0], 1000.0);
            m.PredictWithMemory(inputs[1], 1200.0);

            int newsCountBefore = m.NewsMemory.Count;
            int priceCountBefore = m.PriceMemory.Count;
            double lastTsBefore = m.LastPriceTimestamp;

            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);

                Assert(ld.NewsMemory.Count == newsCountBefore, "news memory count mismatch after load");
                Assert(ld.PriceMemory.Count == priceCountBefore, "price memory count mismatch after load");
                Assert(Math.Abs(ld.LastPriceTimestamp - lastTsBefore) < 1e-9, "LastPriceTimestamp mismatch after load");

                var nextInput = inputs[2];
                double nextTs = 1500.0;

                var p1 = m.PredictWithMemory(nextInput, nextTs);
                var p2 = ld.PredictWithMemory(nextInput, nextTs);

                Assert(PredictionsClose(p1, p2, 1e-5f),
                    $"Sequential-memory prediction changed after reload. p1.Close={p1.Close:F6}, p2.Close={p2.Close:F6}");
            }
            finally
            {
                Cleanup(dir);
            }
        }

        void Test_GlobalToken_NotPersisted_AsMemory()
        {
            var cfg = Cfg(vocabSize: 64, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3);
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var m = new MmtacModel(cfg, new Random(42));

            var noGlobal = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.25f, -0.1f, 0.15f, 0.05f, 1f)
            };

            var withGlobal = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(8, 0.25f, -0.1f, 0.15f, 0.05f, 1f),
                GlobalFeatures = new[] { 1f, 0f, 0.5f }
            };

            m.ClearAllMemory();
            var p0 = m.PredictWithMemory(noGlobal, 100.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);
            int news0 = m.NewsMemory.Count;
            int price0 = m.PriceMemory.Count;

            m.ClearAllMemory();
            var p1 = m.PredictWithMemory(withGlobal, 100.0, maxNewsMemorySize: 1000, maxPriceMemorySize: 1000);
            int news1 = m.NewsMemory.Count;
            int price1 = m.PriceMemory.Count;

            Assert(news0 == 0 && news1 == 0,
                $"Global token should not create news memory entries. noGlobal={news0}, withGlobal={news1}");

            int expectedPrice = noGlobal.PriceSequence.GetLength(0);
            Assert(price0 == expectedPrice && price1 == expectedPrice,
                $"Global token should not change price-memory append count. noGlobal={price0}, withGlobal={price1}, expected={expectedPrice}");

            bool changed =
                MathF.Abs(p0.Close - p1.Close) > 1e-6f ||
                MathF.Abs(p0.DirectionProb - p1.DirectionProb) > 1e-6f ||
                MathF.Abs(p0.Quality - p1.Quality) > 1e-6f;

            Assert(changed, "Global token should affect prediction, not memory count only.");
        }

        void Test_ValidateSequential_ClearsMemoryOnExit()
        {
            var (tok, inputs, targets) = Data(n: 3, seqLen: 8, withNews: true);
            var ts = new[] { 100d, 200d, 300d };

            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));

            // Pre-populate memory so the test proves ValidateSequential clears existing state too.
            m.PredictWithMemory(inputs[0], 500.0);
            Assert(m.NewsMemory.Count > 0 || m.PriceMemory.Count > 0, "precondition failed: memory not populated");

            var tr = new MmtacTrainer(m, TC(epochs: 1));
            float loss = tr.ValidateSequential(inputs, targets, ts);

            Assert(float.IsFinite(loss), $"ValidateSequential returned non-finite loss: {loss}");
            Assert(m.NewsMemory.Count == 0, "ValidateSequential should clear news memory on exit");
            Assert(m.PriceMemory.Count == 0, "ValidateSequential should clear price memory on exit");
        }

        void Test_PricePruning_Mirrors_NewsPruning()
        {
            // Case 1: all entries eligible by query count.
            {
                var m = new MmtacModel(Cfg(vocabSize: 64), new Random(42));
                m.PruningConfig.UseAttentionBasedPruning = true;
                m.PruningConfig.NewEntryReserveFraction = 0f;
                m.PruningConfig.MinQueryCountForPruning = 1;

                m.NewsMemory = MakeNewsMemoryEntries(
                    (10.0, 0.10f, 1),
                    (20.0, 0.90f, 1),
                    (30.0, 0.20f, 1),
                    (40.0, 0.80f, 1));

                m.PriceMemory = MakePriceMemoryEntries(
             m.Config.Price.EmbeddingDim,
             (10.0, 0.10f, 1),
             (20.0, 0.90f, 1),
             (30.0, 0.20f, 1),
             (40.0, 0.80f, 1));

                m.PruneNewsMemory(2);
                InvokePricePruneMemory(m, 2);

                var newsTs = m.NewsMemory.Select(e => e.AbsoluteTimestamp).OrderBy(x => x).ToArray();
                var priceTs = m.PriceMemory.Select(e => e.AbsoluteTimestamp).OrderBy(x => x).ToArray();

                Assert(newsTs.SequenceEqual(priceTs),
                    "Price pruning should mirror news pruning when all entries are eligible.");
            }

            // Case 2: cold-start / low-query-count branch.
            {
                var m = new MmtacModel(Cfg(vocabSize: 64), new Random(42));
                m.PruningConfig.UseAttentionBasedPruning = true;
                m.PruningConfig.NewEntryReserveFraction = 0f;
                m.PruningConfig.MinQueryCountForPruning = 3;

                m.NewsMemory = MakeNewsMemoryEntries(
                    (10.0, 0.20f, 5),
                    (20.0, 0.10f, 5),
                    (30.0, 0.95f, 0),
                    (40.0, 0.85f, 0));

                m.PriceMemory = MakePriceMemoryEntries(
              m.Config.Price.EmbeddingDim,
              (10.0, 0.20f, 5),
              (20.0, 0.10f, 5),
              (30.0, 0.95f, 0),
              (40.0, 0.85f, 0));

                m.PruneNewsMemory(2);
                InvokePricePruneMemory(m, 2);

                var newsTs = m.NewsMemory.Select(e => e.AbsoluteTimestamp).OrderBy(x => x).ToArray();
                var priceTs = m.PriceMemory.Select(e => e.AbsoluteTimestamp).OrderBy(x => x).ToArray();

                Assert(newsTs.SequenceEqual(priceTs),
                    "Price pruning should mirror news pruning in the low-query-count branch.");
            }
        }

        void Test_Tokenizer_SaveLoad_RoundTrip()
        {
            var corpus = new[]
            {
        "central bank support growth demand",
        "credit stress recession slowdown",
        "routine market update commentary",
        "carry trade stable outlook"
    };

            var m = new MmtacModel(Cfg(vocabSize: 64), new Random(42));
            m.TrainTokenizer(corpus, minFrequency: 1);

            var texts = new[]
            {
        "central bank support growth demand",
        "credit stress recession slowdown",
        "routine market update commentary"
    };
            var arrivalTimes = new[] { 0f, -1f, -2f };

            var before = m.TokenizeStories(texts, arrivalTimes);

            var dir = TmpDir2();
            try
            {
                m.Save(dir);
                var ld = MmtacModel.Load(dir);

                Assert(ld.Tokenizer != null, "Tokenizer should be restored after load");

                var after = ld.TokenizeStories(texts, arrivalTimes);

                Assert(before.Length == after.Length, "Tokenized story count mismatch");

                for (int i = 0; i < before.Length; i++)
                {
                    Assert(MathF.Abs(before[i].ArrivalTime - after[i].ArrivalTime) < 1e-9f,
                        $"ArrivalTime mismatch at story {i}");

                    Assert(before[i].TokenIds.Length == after[i].TokenIds.Length,
                        $"Token count mismatch at story {i}");

                    for (int t = 0; t < before[i].TokenIds.Length; t++)
                    {
                        Assert(before[i].TokenIds[t] == after[i].TokenIds[t],
                            $"Token mismatch at story {i}, pos {t}");
                    }
                }
            }
            finally
            {
                Cleanup(dir);
            }
        }

        void Test_TrainSequential_ZeroTimeUnits_Throws()
        {
            var (tok, inputs, targets) = Data(n: 2, seqLen: 8, withNews: true);
            var ts = new[] { 100d, 200d };

            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            bool threw = false;
            try
            {
                tr.TrainSequential(inputs, targets, ts, timeUnitsPerPosition: 0.0);
            }
            catch (ArgumentOutOfRangeException)
            {
                threw = true;
            }

            Assert(threw, "TrainSequential should throw when timeUnitsPerPosition == 0");
        }

        void Test_ValidateSequential_LengthMismatch_Throws()
        {
            var (tok, inputs, targets) = Data(n: 2, seqLen: 8, withNews: true);
            var ts = new[] { 100d }; // intentionally wrong length

            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));
            var tr = new MmtacTrainer(m, TC(epochs: 1));

            bool threw = false;
            try
            {
                tr.ValidateSequential(inputs, targets, ts);
            }
            catch (ArgumentException)
            {
                threw = true;
            }

            Assert(threw, "ValidateSequential should throw on mismatched inputs/targets/timestamps lengths");
        }

        // ===========================
        // Helpers for new tests
        // ===========================

        private static bool PredictionsClose(ModelPrediction a, ModelPrediction b, float tol)
        {
            return MathF.Abs(a.High - b.High) <= tol
                && MathF.Abs(a.Low - b.Low) <= tol
                && MathF.Abs(a.Close - b.Close) <= tol
                && MathF.Abs(a.Range - b.Range) <= tol
                && MathF.Abs(a.Quality - b.Quality) <= tol
                && MathF.Abs(a.DirectionProb - b.DirectionProb) <= tol
                && MathF.Abs(a.MidWindowDirectionProb - b.MidWindowDirectionProb) <= tol
                && MathF.Abs(a.Confidence - b.Confidence) <= tol;
        }

        private static float MaxAbsDiff(float[] a, float[] b)
        {
            int n = Math.Min(a.Length, b.Length);
            float max = 0f;
            for (int i = 0; i < n; i++)
            {
                float d = MathF.Abs(a[i] - b[i]);
                if (d > max) max = d;
            }
            return max;
        }

        private List<NewsMemoryEntry> MakeNewsMemoryEntries(params (double ts, float score, int qc)[] specs)
        {
            int ed = 8;
            var list = new List<NewsMemoryEntry>();

            for (int i = 0; i < specs.Length; i++)
            {
                var hs = new float[ed];
                hs[0] = i + 1;

                list.Add(new NewsMemoryEntry
                {
                    AbsoluteTimestamp = specs[i].ts,
                    AttentionScore = specs[i].score,
                    QueryCount = specs[i].qc,
                    HiddenState = hs
                });
            }

            return list;
        }

        private List<PriceMemoryEntry> MakePriceMemoryEntries(int embDim, params (double ts, float score, int qc)[] specs)
        {
            var list = new List<PriceMemoryEntry>();

            for (int i = 0; i < specs.Length; i++)
            {
                var hs = new float[embDim];
                hs[0] = i + 1;

                list.Add(new PriceMemoryEntry
                {
                    AbsoluteTimestamp = specs[i].ts,
                    AttentionScore = specs[i].score,
                    QueryCount = specs[i].qc,
                    HiddenState = hs
                });
            }

            return list;
        }

        private void InvokePricePruneMemory(MmtacModel model, int maxSize)
        {
            var mi = typeof(MmtacModel).GetMethod(
                "PricePruneMemory",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

            Assert(mi != null, "Could not find internal PricePruneMemory via reflection");
            mi.Invoke(model, new object[] { maxSize });
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
            var (tok, inputs, _) = Data(n: 1, withNews: true, globalDim: 3);
            var m = new MmtacModel(
                Cfg(vocabSize: tok.VocabSize + 2, useConf: true, globalDim: 3),
                new Random(42));

            var input = inputs[0];

            int newsCountBefore = m.NewsMemory.Count;
            int priceCountBefore = m.PriceMemory.Count;
            double lastPriceTsBefore = m.LastPriceTimestamp;

            var p1 = m.PredictNext(input);
            var p2 = m.PredictNext(input);

            Assert(PredictionsClose(p1, p2, 0f),
                "PredictNext should be exactly deterministic for identical input on the same model instance");

            var (reg1, range1, quality1, dir1, mid1, conf1) = m.Forward(input);
            var (reg2, range2, quality2, dir2, mid2, conf2) = m.Forward(input);

            Assert(MatricesEqualExact(reg1, reg2), "Forward regression outputs changed across identical calls");
            Assert(MatricesEqualExact(range1, range2), "Forward range outputs changed across identical calls");
            Assert(MatricesEqualExact(quality1, quality2), "Forward quality outputs changed across identical calls");
            Assert(MatricesEqualExact(dir1, dir2), "Forward direction outputs changed across identical calls");
            Assert(MatricesEqualExact(mid1, mid2), "Forward mid-direction outputs changed across identical calls");

            Assert(conf1 != null && conf2 != null, "Confidence head should be enabled in this test");
            Assert(MatricesEqualExact(conf1, conf2), "Forward confidence outputs changed across identical calls");

            Assert(m.NewsMemory.Count == newsCountBefore,
                $"PredictNext should not mutate NewsMemory. before={newsCountBefore}, after={m.NewsMemory.Count}");

            Assert(m.PriceMemory.Count == priceCountBefore,
                $"PredictNext should not mutate PriceMemory. before={priceCountBefore}, after={m.PriceMemory.Count}");

            Assert(m.LastPriceTimestamp == lastPriceTsBefore,
                $"PredictNext should not mutate LastPriceTimestamp. before={lastPriceTsBefore}, after={m.LastPriceTimestamp}");
        }

        private static bool MatricesEqualExact(float[,] a, float[,] b)
        {
            if (ReferenceEquals(a, b))
                return true;

            if (a == null || b == null)
                return false;

            int rows = a.GetLength(0);
            int cols = a.GetLength(1);

            if (b.GetLength(0) != rows || b.GetLength(1) != cols)
                return false;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (a[i, j] != b[i, j])
                        return false;
                }
            }

            return true;
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
            // Exercises the raw Forward() return - all six head arrays, not just PredictNext.
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

        #region Training

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
            // This verifies those paths are actually wired - the projection weights must move.
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
                "DecayNetwork.QueryProjection unchanged after training - decay backprop not firing");
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

        #endregion

        #region Global Token

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

        #endregion

        #region Multi-output head correctness

        void Test_Outputs_RangeVsHighLow()
        {
            // Train on data where Range is always exactly High - Low, then verify the model
            // keeps those outputs approximately consistent on held-out samples.
            var (tok, trainInputs, trainTargets, testInputs, testTargets) = PriceHeldOutData();

            var cfg = Cfg(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;

            var m = new MmtacModel(cfg, new Random(42));

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 8, epochs: 140)).Train(trainInputs, trainTargets);

            float avgErr = 0f;
            int count = 0;

            foreach (var inp in testInputs)
            {
                var pred = m.PredictNext(inp);
                float implied = pred.High - pred.Low;
                avgErr += MathF.Abs(pred.Range - implied);
                count++;
            }

            avgErr /= Math.Max(1, count);

            Assert(avgErr < 0.15f,
                $"Predicted Range should be approximately consistent with predicted High-Low. avgErr={avgErr:F6}");
        }
        void Test_Outputs_MultipleTimesteps()
        {
            int seqLen = 10;

            var m = new MmtacModel(Cfg(priceSeqLen: 14, useConf: true), new Random(42));
            var inp = MakeInput(seqLen: seqLen);

            var (reg, range, quality, dir, midDir, conf) = m.Forward(inp);
            var pred = m.PredictNext(inp);

            Assert(reg.GetLength(0) == seqLen, $"Regression rows should equal seqLen. actual={reg.GetLength(0)}, expected={seqLen}");
            Assert(reg.GetLength(1) == 3, $"Regression cols should be 3. actual={reg.GetLength(1)}");

            Assert(range.GetLength(0) == seqLen && range.GetLength(1) == 1,
                $"Range head shape should be [{seqLen},1], actual=[{range.GetLength(0)},{range.GetLength(1)}]");

            Assert(quality.GetLength(0) == seqLen && quality.GetLength(1) == 1,
                $"Quality head shape should be [{seqLen},1], actual=[{quality.GetLength(0)},{quality.GetLength(1)}]");

            Assert(dir.GetLength(0) == seqLen && dir.GetLength(1) == 1,
                $"Direction head shape should be [{seqLen},1], actual=[{dir.GetLength(0)},{dir.GetLength(1)}]");

            Assert(midDir.GetLength(0) == seqLen && midDir.GetLength(1) == 1,
                $"MidDirection head shape should be [{seqLen},1], actual=[{midDir.GetLength(0)},{midDir.GetLength(1)}]");

            Assert(conf != null, "Confidence head should be enabled in this test");
            Assert(conf.GetLength(0) == seqLen && conf.GetLength(1) == 1,
                $"Confidence head shape should be [{seqLen},1], actual=[{conf.GetLength(0)},{conf.GetLength(1)}]");

            const float tol = 1e-6f;

            Assert(MathF.Abs(pred.High - reg[seqLen - 1, 0]) < tol, "PredictNext.High should equal final regression timestep High");
            Assert(MathF.Abs(pred.Low - reg[seqLen - 1, 1]) < tol, "PredictNext.Low should equal final regression timestep Low");
            Assert(MathF.Abs(pred.Close - reg[seqLen - 1, 2]) < tol, "PredictNext.Close should equal final regression timestep Close");
            Assert(MathF.Abs(pred.Range - range[seqLen - 1, 0]) < tol, "PredictNext.Range should equal final range timestep");
            Assert(MathF.Abs(pred.Quality - quality[seqLen - 1, 0]) < tol, "PredictNext.Quality should equal final quality timestep");
            Assert(MathF.Abs(pred.DirectionProb - dir[seqLen - 1, 0]) < tol, "PredictNext.DirectionProb should equal final direction timestep");
            Assert(MathF.Abs(pred.MidWindowDirectionProb - midDir[seqLen - 1, 0]) < tol, "PredictNext.MidWindowDirectionProb should equal final mid-direction timestep");
            Assert(MathF.Abs(pred.Confidence - conf[seqLen - 1, 0]) < tol, "PredictNext.Confidence should equal final confidence timestep");
        }

        #endregion

        #region Memory

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
            var (tok, inputs, _) = Data(n: 3, seqLen: 6, withNews: true);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, priceSeqLen: 10), new Random(42));

            m.PredictWithMemory(inputs[0], 100.0);
            m.PredictWithMemory(inputs[1], 200.0);
            m.PredictWithMemory(inputs[2], 300.0);

            var newsTs = m.NewsMemory.Select(e => e.AbsoluteTimestamp).ToArray();
            var priceTs = m.PriceMemory.Select(e => e.AbsoluteTimestamp).ToArray();

            for (int i = 1; i < newsTs.Length; i++)
            {
                Assert(newsTs[i] >= newsTs[i - 1],
                    $"NewsMemory timestamps are not nondecreasing at {i}. prev={newsTs[i - 1]}, cur={newsTs[i]}");
            }

            for (int i = 1; i < priceTs.Length; i++)
            {
                Assert(priceTs[i] >= priceTs[i - 1],
                    $"PriceMemory timestamps are not nondecreasing at {i}. prev={priceTs[i - 1]}, cur={priceTs[i]}");
            }

            Assert(newsTs.Length == 0 || newsTs[^1] >= 300.0,
                $"Latest news timestamp should reflect the latest call. latest={newsTs[^1]}");

            Assert(priceTs.Length == 0 || priceTs[^1] >= 300.0,
                $"Latest price timestamp should reflect the latest call. latest={priceTs[^1]}");
        }

        void Test_Memory_TimeDeltaReachesDecoder()
        {
            // After the SafeRecencyScale fix, relTime is passed raw into ctxT which
            // becomes the time-diff matrix fed to the DecayNetwork. Two identical
            // setups differing only in currentAbsoluteTimestamp must produce different
            // price decoder hidden states - this is structurally guaranteed regardless
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

            if (m.NewsMemory.Count < 2) return; // not enough entries - skip

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

        #endregion

        #region Sequential training 

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
            var (inputs, targets, ts) = BuildSomeSequentialDataset();

            var cfg = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 12);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var m = new MmtacModel(cfg, new Random(42));

            float before = new MmtacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 1, epochs: 60)).TrainSequential(inputs, targets, ts);

            float after = new MmtacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);

            Assert(after < before,
                $"Sequential validation loss did not decrease. before={before:F6}, after={after:F6}");
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
            // With lr=0 and identical initialisation, the final memory state after 2 epochs
            // should exactly match the final memory state after 1 epoch if each epoch starts clean.
            var (tok, inputs, targets) = Data(n: 4, seqLen: 8, withNews: true, globalDim: 3);
            var ts = new[] { 100d, 200d, 300d, 400d };

            var cfg1 = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3);
            var cfg2 = Cfg(tok.VocabSize + 2, embDim: 24, numHeads: 4, numLayers: 2, ffnDim: 48, priceSeqLen: 10, globalDim: 3);

            cfg1.Decay.MemAttentionDropout = 0f;
            cfg1.Decay.MlpDropout = 0f;
            cfg1.Runtime.AccelerationType = AccelerationType.CPU;

            cfg2.Decay.MemAttentionDropout = 0f;
            cfg2.Decay.MlpDropout = 0f;
            cfg2.Runtime.AccelerationType = AccelerationType.CPU;

            var m1 = new MmtacModel(cfg1, new Random(42));
            var m2 = new MmtacModel(cfg2, new Random(42));

            new MmtacTrainer(m1, TC(lr: 0f, bs: 1, epochs: 1)).TrainSequential(inputs, targets, ts);
            new MmtacTrainer(m2, TC(lr: 0f, bs: 1, epochs: 2)).TrainSequential(inputs, targets, ts);

            Assert(m1.NewsMemory.Count == m2.NewsMemory.Count,
                $"NewsMemory count mismatch. oneEpoch={m1.NewsMemory.Count}, twoEpochs={m2.NewsMemory.Count}");

            Assert(m1.PriceMemory.Count == m2.PriceMemory.Count,
                $"PriceMemory count mismatch. oneEpoch={m1.PriceMemory.Count}, twoEpochs={m2.PriceMemory.Count}");

            Assert(Math.Abs(m1.LastPriceTimestamp - m2.LastPriceTimestamp) < 1e-9,
                $"LastPriceTimestamp mismatch. oneEpoch={m1.LastPriceTimestamp}, twoEpochs={m2.LastPriceTimestamp}");

            var newsTs1 = m1.NewsMemory.Select(x => x.AbsoluteTimestamp).ToArray();
            var newsTs2 = m2.NewsMemory.Select(x => x.AbsoluteTimestamp).ToArray();
            Assert(newsTs1.SequenceEqual(newsTs2), "NewsMemory timestamps differ between one epoch and two epochs");

            var priceTs1 = m1.PriceMemory.Select(x => x.AbsoluteTimestamp).ToArray();
            var priceTs2 = m2.PriceMemory.Select(x => x.AbsoluteTimestamp).ToArray();
            Assert(priceTs1.SequenceEqual(priceTs2), "PriceMemory timestamps differ between one epoch and two epochs");

            for (int i = 0; i < m1.NewsMemory.Count; i++)
            {
                Assert(m1.NewsMemory[i].QueryCount == m2.NewsMemory[i].QueryCount,
                    $"NewsMemory QueryCount mismatch at {i}. oneEpoch={m1.NewsMemory[i].QueryCount}, twoEpochs={m2.NewsMemory[i].QueryCount}");

                Assert(MathF.Abs(m1.NewsMemory[i].AttentionScore - m2.NewsMemory[i].AttentionScore) < 1e-6f,
                    $"NewsMemory AttentionScore mismatch at {i}. oneEpoch={m1.NewsMemory[i].AttentionScore:F6}, twoEpochs={m2.NewsMemory[i].AttentionScore:F6}");

                Assert(MaxAbsDiff(m1.NewsMemory[i].HiddenState, m2.NewsMemory[i].HiddenState) < 1e-6f,
                    $"NewsMemory hidden state mismatch at {i}");
            }

            for (int i = 0; i < m1.PriceMemory.Count; i++)
            {
                Assert(m1.PriceMemory[i].QueryCount == m2.PriceMemory[i].QueryCount,
                    $"PriceMemory QueryCount mismatch at {i}. oneEpoch={m1.PriceMemory[i].QueryCount}, twoEpochs={m2.PriceMemory[i].QueryCount}");

                Assert(MathF.Abs(m1.PriceMemory[i].AttentionScore - m2.PriceMemory[i].AttentionScore) < 1e-6f,
                    $"PriceMemory AttentionScore mismatch at {i}. oneEpoch={m1.PriceMemory[i].AttentionScore:F6}, twoEpochs={m2.PriceMemory[i].AttentionScore:F6}");

                Assert(MaxAbsDiff(m1.PriceMemory[i].HiddenState, m2.PriceMemory[i].HiddenState) < 1e-6f,
                    $"PriceMemory hidden state mismatch at {i}");
            }
        }

        void Test_Sequential_WithGlobalToken()
        {
            int n = 48;
            int seqLen = 6;

            var inputs = new MultimodalInput[n];
            var targets = new ModelTarget[n][];
            var ts = new double[n];

            for (int i = 0; i < n; i++)
            {
                bool riskOn = (i % 2 == 0);

                inputs[i] = new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(i),
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    GlobalFeatures = riskOn ? new[] { 1f, 0f, 1f } : new[] { 0f, 1f, 0f },
                    NewsStories = null
                };

                targets[i] = ConstantTargets(
                    seqLen,
                    close: riskOn ? 0.80f : 0.20f,
                    range: 0.20f,
                    quality: 0.95f,
                    direction: riskOn ? 1 : 0,
                    midDirection: riskOn ? 1 : 0);

                ts[i] = i * 10.0;
            }

            var cfg = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10, globalDim: 3);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var m = new MmtacModel(cfg, new Random(42));

            float before = new MmtacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);

            new MmtacTrainer(m, TC(lr: 0.003f, bs: 1, epochs: 80)).TrainSequential(inputs, targets, ts);

            float after = new MmtacTrainer(m, TC(epochs: 1)).ValidateSequential(inputs, targets, ts);

            Assert(float.IsFinite(after), $"Sequential validation loss with global token is not finite: {after}");
            Assert(after < before * 0.55f,
                $"Sequential training with global token did not improve enough. before={before:F6}, after={after:F6}");

            var riskOnProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                GlobalFeatures = new[] { 1f, 0f, 1f },
                NewsStories = null
            };

            var riskOffProbe = new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                GlobalFeatures = new[] { 0f, 1f, 0f },
                NewsStories = null
            };

            m.ClearAllMemory();
            var onPred = m.PredictWithMemory(riskOnProbe, 1000.0);

            m.ClearAllMemory();
            var offPred = m.PredictWithMemory(riskOffProbe, 1000.0);

            Assert(onPred.Close > offPred.Close + 0.15f,
                $"Global token should materially affect sequential prediction. onClose={onPred.Close:F6}, offClose={offPred.Close:F6}");

            Assert(onPred.DirectionProb > offPred.DirectionProb + 0.20f,
                $"Global token should materially affect sequential direction. onDir={onPred.DirectionProb:F6}, offDir={offPred.DirectionProb:F6}");
        }
        #endregion

        #region Tokenizer

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

        #endregion

        #region Validation
        void Test_Validate_ReturnsFiniteValue()
        {
            var (tok, inputs, targets) = Data(n: 5);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2), new Random(42));
            float loss = new MmtacTrainer(m, TC(epochs: 1)).Validate(inputs, targets);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss) && loss >= 0, $"invalid loss {loss}");
        }

        #endregion

        #region Numerical stability 


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

        #endregion

        #region  Save / Load 

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
        #endregion

        #region Config validation 

        void Test_Config_Validate_Good() { Cfg().Validate(); }
        void Test_Config_SmallPreset() { MmtacConfig.Small(vocabSize: 1000, priceFeatureDim: 5).Validate(); }
        void Test_Config_StandardPreset() { MmtacConfig.Standard(vocabSize: 5000, priceFeatureDim: 5, globalDim: 8).Validate(); }

        #endregion

        #region Gradient checks (finite-difference) 

        void Test_GradCheck_RegressionProjection()
        {
            var (tok, inputs, targets) = Data(n: 1, seqLen: 6);
            var m = new MmtacModel(Cfg(tok.VocabSize + 2, embDim: 8, numHeads: 2), new Random(42));
            float eps = 1e-3f, orig = m.RegressionProjection[2, 0];
            m.RegressionProjection[2, 0] = orig + eps; float lp = MmtacMSELoss(m, inputs[0], targets[0]);
            m.RegressionProjection[2, 0] = orig - eps; float lm = MmtacMSELoss(m, inputs[0], targets[0]);
            m.RegressionProjection[2, 0] = orig;
            float fd = (lp - lm) / (2 * eps);
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} - RegressionProjection has no effect");
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
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} - RangeProjection has no effect");
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
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} - DirectionProjection has no effect");
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
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} - QualityProjection has no effect");
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
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} - MidDirectionProjection has no effect");
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
            Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, $"fd={fd:E5} - TextEmbedding has no effect");
        }

        #endregion

        #region End to end signal learning

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
                $"Range head vs implied High-Low avg error={avgErr:F4} - heads not learning consistently");
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
            Assert(avgDirH < avgDirL, $"Direction: high input ({avgDirH:F4}) should be LOWER than low ({avgDirL:F4}) - inverted target");
        }

        void Test_Sequential_MemoryHelpsOnAutoregressive()
        {
            // The carry sample is neutral on its own; its target depends on the immediately
            // previous driver sample. This directly checks that rolling sequential memory helps.
            static (MultimodalInput[] inputs, ModelTarget[][] targets, double[] ts) BuildCarryDataset(int pairCount, int seqLen, int offset)
            {
                var inputs = new MultimodalInput[pairCount * 2];
                var targets = new ModelTarget[pairCount * 2][];
                var ts = new double[pairCount * 2];

                for (int k = 0; k < pairCount; k++)
                {
                    float regime = (k % 2 == 0) ? 0.8f : -0.8f;

                    int driverIdx = 2 * k;
                    int carryIdx = 2 * k + 1;

                    inputs[driverIdx] = new MultimodalInput
                    {
                        PredictionTimestamp = DateTime.UtcNow.AddMinutes(offset + driverIdx),
                        PriceSequence = ConstantPriceSequence(seqLen, regime, regime * regime, MathF.Sign(regime), 0.5f * regime, 1f),
                        NewsStories = null,
                        GlobalFeatures = null
                    };

                    targets[driverIdx] = ConstantTargets(
                        seqLen,
                        close: regime,
                        range: 0.20f,
                        quality: 0.95f,
                        direction: regime > 0f ? 1 : 0,
                        midDirection: regime > 0f ? 1 : 0);

                    inputs[carryIdx] = new MultimodalInput
                    {
                        PredictionTimestamp = DateTime.UtcNow.AddMinutes(offset + carryIdx),
                        PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                        NewsStories = null,
                        GlobalFeatures = null
                    };

                    targets[carryIdx] = ConstantTargets(
                        seqLen,
                        close: regime,
                        range: 0.20f,
                        quality: 0.95f,
                        direction: regime > 0f ? 1 : 0,
                        midDirection: regime > 0f ? 1 : 0);

                    ts[driverIdx] = (offset + driverIdx) * 10.0;
                    ts[carryIdx] = (offset + carryIdx) * 10.0;
                }

                return (inputs, targets, ts);
            }

            var (trainInputs, trainTargets, trainTs) = BuildCarryDataset(pairCount: 40, seqLen: 6, offset: 0);
            var (testInputs, testTargets, testTs) = BuildCarryDataset(pairCount: 16, seqLen: 6, offset: 10_000);

            var cfg = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10);
            cfg.Output.CloseDirectionConsistencyWeight = 0f;
            cfg.Output.CloseDirectionConsistencyMargin = 0f;
            cfg.Decay.MemAttentionDropout = 0f;
            cfg.Decay.MlpDropout = 0f;
            cfg.Runtime.AccelerationType = AccelerationType.CPU;

            var model = new MmtacModel(cfg, new Random(42));
            new MmtacTrainer(model, TC(lr: 0.003f, bs: 1, epochs: 80)).TrainSequential(trainInputs, trainTargets, trainTs);

            float withMemoryErr = 0f;
            int withMemoryCount = 0;

            model.ClearAllMemory();
            for (int i = 0; i < testInputs.Length; i++)
            {
                var pred = model.PredictWithMemory(testInputs[i], testTs[i]);

                // Evaluate only the carry positions; those are the ones that require memory.
                if ((i % 2) == 1)
                {
                    float tgt = testTargets[i][testTargets[i].Length - 1].Close;
                    withMemoryErr += MathF.Abs(pred.Close - tgt);
                    withMemoryCount++;
                }
            }
            model.ClearAllMemory();
            withMemoryErr /= Math.Max(1, withMemoryCount);

            float withoutMemoryErr = 0f;
            int withoutMemoryCount = 0;

            for (int i = 0; i < testInputs.Length; i++)
            {
                if ((i % 2) == 0)
                    continue;

                var pred = model.PredictNext(testInputs[i]);
                float tgt = testTargets[i][testTargets[i].Length - 1].Close;
                withoutMemoryErr += MathF.Abs(pred.Close - tgt);
                withoutMemoryCount++;
            }
            withoutMemoryErr /= Math.Max(1, withoutMemoryCount);

            Assert(withMemoryErr < withoutMemoryErr * 0.65f,
                $"Sequential memory should materially help on carry/autoregressive data. withMemory={withMemoryErr:F6}, withoutMemory={withoutMemoryErr:F6}");

            Assert(withMemoryErr < 0.20f,
                $"Sequential memory error is still too high on carry/autoregressive data. withMemory={withMemoryErr:F6}");
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

            Assert(anyDiff, "All timestep predictions are identical - causal mask or positional encoding may be broken");
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

            Assert(valLoss < 0.1f, $"Validation loss ({valLoss:F6}) on training data should be low after overfitting - train and validate loss functions may differ");
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

                    for (int f = 1; f < 5; f++)
                    {
                        ps[t, f] = 0.5f;
                    }
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

            // Model A: direction weight = 0 - direction head gets no gradient
            var cfgA = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64);
            cfgA.Output.DirectionLossWeight = 0f;
            var mA = new MmtacModel(cfgA, new Random(42));
            new MmtacTrainer(mA, TC(lr: 0.005f, bs: 10, epochs: 200)).Train(inputs, targets);

            // Model B: direction weight = 5 - direction head gets strong gradient
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
                    if ((dirA[t, 0] > 0.5f) == up)
                    {
                        corrA++;
                    }
                    if ((dirB[t, 0] > 0.5f) == up)
                    {
                        corrB++;
                    }
                    total++;
                }
            }
            float accA = (float)corrA / total;
            float accB = (float)corrB / total;

            Assert(accB > accA, $"Higher direction weight (acc={accB:P0}) should outperform zero weight (acc={accA:P0}) - loss weights may not be controlling gradient flow");
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
            {
                for (int j = 0; j < projBefore.GetLength(1); j++)
                {
                    maxChange = MathF.Max(maxChange, MathF.Abs(m.PriceInputProjection[i, j] - projBefore[i, j]));
                }
            }

            Assert(maxChange < 1.0f, $"Max weight change {maxChange:F4} is too large with tight gradient clipping - clip threshold may not be controlling update magnitude");
        }

        void Test_Sequential_Produces_Different_Weights_Than_Batch()
        {
            // Same optimizer schedule on both sides. The only intended difference is that
            // sequential mode can use rolling cross-sample memory and batch mode cannot.
            static (MultimodalInput[] inputs, ModelTarget[][] targets, double[] ts) BuildCarryDataset(int pairCount, int seqLen, int offset)
            {
                var inputs = new MultimodalInput[pairCount * 2];
                var targets = new ModelTarget[pairCount * 2][];
                var ts = new double[pairCount * 2];

                for (int k = 0; k < pairCount; k++)
                {
                    float regime = (k % 2 == 0) ? 0.8f : -0.8f;

                    int driverIdx = 2 * k;
                    int carryIdx = 2 * k + 1;

                    inputs[driverIdx] = new MultimodalInput
                    {
                        PredictionTimestamp = DateTime.UtcNow.AddMinutes(offset + driverIdx),
                        PriceSequence = ConstantPriceSequence(seqLen, regime, regime * regime, MathF.Sign(regime), 0.5f * regime, 1f),
                        NewsStories = null,
                        GlobalFeatures = null
                    };

                    targets[driverIdx] = ConstantTargets(
                        seqLen,
                        close: regime,
                        range: 0.20f,
                        quality: 0.95f,
                        direction: regime > 0f ? 1 : 0,
                        midDirection: regime > 0f ? 1 : 0);

                    inputs[carryIdx] = new MultimodalInput
                    {
                        PredictionTimestamp = DateTime.UtcNow.AddMinutes(offset + carryIdx),
                        PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                        NewsStories = null,
                        GlobalFeatures = null
                    };

                    targets[carryIdx] = ConstantTargets(
                        seqLen,
                        close: regime,
                        range: 0.20f,
                        quality: 0.95f,
                        direction: regime > 0f ? 1 : 0,
                        midDirection: regime > 0f ? 1 : 0);

                    ts[driverIdx] = (offset + driverIdx) * 10.0;
                    ts[carryIdx] = (offset + carryIdx) * 10.0;
                }

                return (inputs, targets, ts);
            }

            var (trainInputs, trainTargets, trainTs) = BuildCarryDataset(pairCount: 36, seqLen: 6, offset: 0);
            var (testInputs, testTargets, testTs) = BuildCarryDataset(pairCount: 12, seqLen: 6, offset: 10_000);

            var cfgSeq = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10);
            var cfgBatch = Cfg(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 10);

            cfgSeq.Output.CloseDirectionConsistencyWeight = 0f;
            cfgSeq.Output.CloseDirectionConsistencyMargin = 0f;
            cfgSeq.Decay.MemAttentionDropout = 0f;
            cfgSeq.Decay.MlpDropout = 0f;
            cfgSeq.Runtime.AccelerationType = AccelerationType.CPU;

            cfgBatch.Output.CloseDirectionConsistencyWeight = 0f;
            cfgBatch.Output.CloseDirectionConsistencyMargin = 0f;
            cfgBatch.Decay.MemAttentionDropout = 0f;
            cfgBatch.Decay.MlpDropout = 0f;
            cfgBatch.Runtime.AccelerationType = AccelerationType.CPU;

            var mSeq = new MmtacModel(cfgSeq, new Random(42));
            var mBatch = new MmtacModel(cfgBatch, new Random(42));

            var tc = TC(lr: 0.003f, bs: 1, epochs: 80);

            new MmtacTrainer(mSeq, tc).TrainSequential(trainInputs, trainTargets, trainTs);
            new MmtacTrainer(mBatch, tc).Train(trainInputs, trainTargets);

            float seqCarryErr = 0f;
            int seqCount = 0;

            mSeq.ClearAllMemory();
            for (int i = 0; i < testInputs.Length; i++)
            {
                var pred = mSeq.PredictWithMemory(testInputs[i], testTs[i]);
                if ((i % 2) == 1)
                {
                    float tgt = testTargets[i][testTargets[i].Length - 1].Close;
                    seqCarryErr += MathF.Abs(pred.Close - tgt);
                    seqCount++;
                }
            }
            mSeq.ClearAllMemory();
            seqCarryErr /= Math.Max(1, seqCount);

            float batchCarryErr = 0f;
            int batchCount = 0;

            for (int i = 0; i < testInputs.Length; i++)
            {
                if ((i % 2) == 0)
                    continue;

                var pred = mBatch.PredictNext(testInputs[i]);
                float tgt = testTargets[i][testTargets[i].Length - 1].Close;
                batchCarryErr += MathF.Abs(pred.Close - tgt);
                batchCount++;
            }
            batchCarryErr /= Math.Max(1, batchCount);

            Assert(seqCarryErr < batchCarryErr * 0.70f,
                $"Sequential mode should outperform batch mode on cross-sample carry data. seq={seqCarryErr:F6}, batch={batchCarryErr:F6}");

            bool differ =
                MatrixChanged(mSeq.PriceInputProjection, mBatch.PriceInputProjection, 1e-6f) ||
                MatrixChanged(mSeq.RegressionProjection, mBatch.RegressionProjection, 1e-6f) ||
                MatrixChanged(mSeq.DirectionProjection, mBatch.DirectionProjection, 1e-6f);

            Assert(differ,
                "Sequential and batch training ended with identical key weights despite learning different behaviour.");
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

                    for (int f = 2; f < 5; f++)
                    {
                        ps[t, f] = 0.5f;

                    }
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
                    if ((dir[t, 0] > 0.5f) == dirUp)
                    {
                        corrDir++;
                    }
                    if ((midDir[t, 0] > 0.5f) != dirUp)
                    {
                        corrMid++;
                    }
                    total++;
                }
            }
            float accDir = (float)corrDir / total;
            float accMid = (float)corrMid / total;

            Assert(accDir > 0.60f, $"Direction accuracy {accDir:P0} too low - head may not be learning its signal");
            Assert(accMid > 0.60f, $"MidWindowDirection accuracy {accMid:P0} too low - head may be copying Direction instead of learning its own signal");
        }

        #endregion

        #region Helpers
        private MmtacConfig Cfg(int vocabSize = 50, int embDim = 16, int numHeads = 2, int numLayers = 1, int ffnDim = 32, int priceFeatures = 5, int priceSeqLen = 12, bool useConf = false, bool freezeText = false, int globalDim = 0, bool decayEnabled = true, bool bypassDecay = true, bool priceContextEnabled = false)
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

        private (BPETokenizer tok, MultimodalInput[] inputs, ModelTarget[][] targets) Data(int n = 10, int seqLen = 10, int priceFeatures = 5, int seed = 42, int globalDim = 0, bool withNews = true)
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
                    {
                        stories[i] = new NewsStory(tok.Encode(corpus[rng.Next(corpus.Length)], addSpecialTokens: true), (float)(i * 2));
                    }
                }

                float[] globalFeatures = globalDim > 0 ? Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray() : null;

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
        {
            return float.IsNaN(p.High) || float.IsNaN(p.Low) || float.IsNaN(p.Close) || float.IsNaN(p.Range) || float.IsNaN(p.Quality) || float.IsNaN(p.DirectionProb) || float.IsNaN(p.MidWindowDirectionProb);
        }

        private string TmpDir2()
        {
            var d = Path.Combine(Path.GetTempPath(), $"mmtac_test_{Guid.NewGuid():N}");
            Directory.CreateDirectory(d); return d;
        }

        private float MmtacMSELoss(MmtacModel m, MultimodalInput inp, ModelTarget[] tgt)
        {
            float diff = m.PredictNext(inp).Close - tgt[tgt.Length - 1].Close;
            return diff * diff;
        }

        private MultimodalInput MakeInput(int seqLen = 10, int priceFeatures = 5, int globalDim = 0, NewsStory[] stories = null)
        {
            var rng = new Random(42);
            return new MultimodalInput
            {
                PredictionTimestamp = DateTime.UtcNow,
                PriceSequence = RandMatrix(seqLen, priceFeatures, rng, 0.5f),
                NewsStories = stories,
                GlobalFeatures = globalDim > 0  ? Enumerable.Range(0, globalDim).Select(_ => (float)rng.NextDouble()).ToArray() : null,
            };
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
        private static float[,] ConstantPriceSequence(int seqLen, params float[] features)
        {
            var ps = new float[seqLen, features.Length];
            for (int t = 0; t < seqLen; t++)
            {
                for (int j = 0; j < features.Length; j++)
                {
                    ps[t, j] = features[j];
                }
            }
            return ps;
        }

        private static ModelTarget[] ConstantTargets(int seqLen, float close, float range, float quality, int direction, int midDirection)
        {
            float high = close + range * 0.5f;
            float low = close - range * 0.5f;

            var out_ = new ModelTarget[seqLen];
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

        private static float Clamp01(float x)
        {
            return MathF.Max(0f, MathF.Min(1f, x));
        }

        private static float Tanh01(float x)
        {
            return 0.5f + 0.5f * MathF.Tanh(x);
        }

        private static float MeanCloseAbsError(MmtacModel model, MultimodalInput[] inputs, ModelTarget[][] targets)
        {
            float err = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                var pred = model.PredictNext(inputs[i]);
                float tgt = targets[i][targets[i].Length - 1].Close;
                err += MathF.Abs(pred.Close - tgt);
            }
            return err / Math.Max(1, inputs.Length);
        }

        private (BPETokenizer tok, MultimodalInput[] trainInputs, ModelTarget[][] trainTargets, MultimodalInput[] testInputs, ModelTarget[][] testTargets) PriceHeldOutData(int trainN = 80, int testN = 24, int seqLen = 8, int priceFeatures = 5, int seed = 123)
        {
            var rng = new Random(seed);
            var tok = new BPETokenizer();
            tok.Train(new[] { "unused token" }, vocabSize: 32, minFrequency: 1);

            MultimodalInput MakeInput(float x, float y, int idx)
            {
                return new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                    PriceSequence = ConstantPriceSequence(
                        seqLen,
                        x,
                        y,
                        x * y,
                        x * x,
                        1f),
                    NewsStories = null,
                    GlobalFeatures = null
                };
            }

            ModelTarget[] MakeTarget(float x, float y)
            {
                float close = 0.55f + 0.22f * x - 0.10f * y;
                float range = 0.06f + 0.05f * MathF.Abs(x);
                float quality = Clamp01(0.20f + 0.70f * y);
                int dir = close > 0.55f ? 1 : 0;
                int mid = x + y > 0.50f ? 1 : 0;
                return ConstantTargets(seqLen, close, range, quality, dir, mid);
            }

            var trainInputs = new MultimodalInput[trainN];
            var trainTargets = new ModelTarget[trainN][];
            for (int i = 0; i < trainN; i++)
            {
                float x = (float)(rng.NextDouble() * 2.0 - 1.0);
                float y = (float)rng.NextDouble();
                trainInputs[i] = MakeInput(x, y, i);
                trainTargets[i] = MakeTarget(x, y);
            }

            var testInputs = new MultimodalInput[testN];
            var testTargets = new ModelTarget[testN][];
            for (int i = 0; i < testN; i++)
            {
                float x = (float)(rng.NextDouble() * 2.0 - 1.0);
                float y = (float)rng.NextDouble();
                testInputs[i] = MakeInput(x, y, 1000 + i);
                testTargets[i] = MakeTarget(x, y);
            }

            return (tok, trainInputs, trainTargets, testInputs, testTargets);
        }

        private (BPETokenizer tok, MultimodalInput[] trainInputs, ModelTarget[][] trainTargets, MultimodalInput[] testInputs, ModelTarget[][] testTargets) NewsHeldOutData(int trainN = 80, int testN = 24, int seqLen = 8, int priceFeatures = 5)
        {
            string bullText = "strong bullish outlook demand growth";
            string bearText = "weak bearish outlook contraction risk";
            string neutralText = "routine market update commentary";

            var tok = new BPETokenizer();
            tok.Train(new[] { bullText, bearText, neutralText }, vocabSize: 80, minFrequency: 1);

            int[] bull = tok.Encode(bullText, addSpecialTokens: true);
            int[] bear = tok.Encode(bearText, addSpecialTokens: true);
            int[] neutral = tok.Encode(neutralText, addSpecialTokens: true);

            MultimodalInput MakeInput(bool isBull, int idx)
            {
                return new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    NewsStories = new[]
                    {
                new NewsStory(isBull ? bull : bear, 0f),
                new NewsStory(neutral, -3f)
            }
                };
            }

            ModelTarget[] MakeTarget(bool isBull)
            {
                float close = isBull ? 0.78f : 0.22f;
                float range = isBull ? 0.11f : 0.11f;
                float quality = 0.90f;
                int dir = isBull ? 1 : 0;
                int mid = isBull ? 1 : 0;
                return ConstantTargets(seqLen, close, range, quality, dir, mid);
            }

            var trainInputs = new MultimodalInput[trainN];
            var trainTargets = new ModelTarget[trainN][];
            for (int i = 0; i < trainN; i++)
            {
                bool isBull = (i % 2 == 0);
                trainInputs[i] = MakeInput(isBull, i);
                trainTargets[i] = MakeTarget(isBull);
            }

            var testInputs = new MultimodalInput[testN];
            var testTargets = new ModelTarget[testN][];
            for (int i = 0; i < testN; i++)
            {
                bool isBull = (i % 2 == 0);
                testInputs[i] = MakeInput(isBull, 1000 + i);
                testTargets[i] = MakeTarget(isBull);
            }

            return (tok, trainInputs, trainTargets, testInputs, testTargets);
        }

        private (BPETokenizer tok, MultimodalInput[] trainInputs, ModelTarget[][] trainTargets, MultimodalInput[] testInputs, ModelTarget[][] testTargets) GlobalHeldOutData(int trainN = 80, int testN = 24, int seqLen = 8, int priceFeatures = 5, int globalDim = 3, int seed = 321)
        {
            var rng = new Random(seed);
            var tok = new BPETokenizer();
            tok.Train(new[] { "unused token" }, vocabSize: 32, minFrequency: 1);

            MultimodalInput MakeInput(float g0, float g1, float g2, int idx)
            {
                return new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    NewsStories = null,
                    GlobalFeatures = new[] { g0, g1, g2 }
                };
            }

            ModelTarget[] MakeTarget(float g0, float g1, float g2)
            {
                float score = 1.2f * g0 - 1.0f * g1 + 0.6f * g2;
                float close = 0.50f + 0.22f * MathF.Tanh(score);
                float range = 0.05f + 0.03f * MathF.Abs(score);
                float quality = Tanh01(0.8f * score);
                int dir = score > 0f ? 1 : 0;
                int mid = g2 > 0.5f ? 1 : 0;
                return ConstantTargets(seqLen, close, range, quality, dir, mid);
            }

            var trainInputs = new MultimodalInput[trainN];
            var trainTargets = new ModelTarget[trainN][];
            for (int i = 0; i < trainN; i++)
            {
                float g0 = (float)rng.NextDouble();
                float g1 = (float)rng.NextDouble();
                float g2 = (float)rng.NextDouble();
                trainInputs[i] = MakeInput(g0, g1, g2, i);
                trainTargets[i] = MakeTarget(g0, g1, g2);
            }

            var testInputs = new MultimodalInput[testN];
            var testTargets = new ModelTarget[testN][];
            for (int i = 0; i < testN; i++)
            {
                float g0 = (float)rng.NextDouble();
                float g1 = (float)rng.NextDouble();
                float g2 = (float)rng.NextDouble();
                testInputs[i] = MakeInput(g0, g1, g2, 1000 + i);
                testTargets[i] = MakeTarget(g0, g1, g2);
            }

            return (tok, trainInputs, trainTargets, testInputs, testTargets);
        }

        private (BPETokenizer tok, MultimodalInput[] trainInputs, ModelTarget[][] trainTargets, MultimodalInput[] testInputs, ModelTarget[][] testTargets) MultimodalHeldOutData(int trainN = 120, int testN = 32, int seqLen = 8, int priceFeatures = 5, int globalDim = 3, int seed = 999)
        {
            var rng = new Random(seed);

            string bullText = "central bank support growth demand";
            string bearText = "credit stress slowdown recession";
            var tok = new BPETokenizer();
            tok.Train(new[] { bullText, bearText }, vocabSize: 80, minFrequency: 1);

            int[] bull = tok.Encode(bullText, addSpecialTokens: true);
            int[] bear = tok.Encode(bearText, addSpecialTokens: true);

            MultimodalInput MakeInput(float p, int newsSign, int globalSign, int idx)
            {
                return new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                    PriceSequence = ConstantPriceSequence(
                        seqLen,
                        p,
                        p * p,
                        MathF.Sign(p),
                        0.5f * p,
                        1f),
                    NewsStories = new[]
                    {
                new NewsStory(newsSign > 0 ? bull : bear, 0f)
            },
                    GlobalFeatures = new[]
                    {
                globalSign > 0 ? 1f : 0f,
                globalSign < 0 ? 1f : 0f,
                MathF.Abs(p)
            }
                };
            }

            ModelTarget[] MakeTarget(float p, int newsSign, int globalSign)
            {
                float score = 0.55f * p + 0.35f * newsSign + 0.25f * globalSign;
                float close = 0.50f + 0.23f * MathF.Tanh(score);
                float range = 0.05f + 0.04f * MathF.Abs(score);
                float quality = Tanh01(score);
                int dir = score > 0f ? 1 : 0;
                int mid = (0.45f * p + 0.55f * globalSign) > 0f ? 1 : 0;
                return ConstantTargets(seqLen, close, range, quality, dir, mid);
            }

            var trainInputs = new MultimodalInput[trainN];
            var trainTargets = new ModelTarget[trainN][];
            for (int i = 0; i < trainN; i++)
            {
                float p = (float)(rng.NextDouble() * 2.0 - 1.0);
                int newsSign = rng.Next(2) == 0 ? -1 : 1;
                int globalSign = rng.Next(2) == 0 ? -1 : 1;
                trainInputs[i] = MakeInput(p, newsSign, globalSign, i);
                trainTargets[i] = MakeTarget(p, newsSign, globalSign);
            }

            var testInputs = new MultimodalInput[testN];
            var testTargets = new ModelTarget[testN][];
            for (int i = 0; i < testN; i++)
            {
                float p = (float)(rng.NextDouble() * 2.0 - 1.0);
                int newsSign = rng.Next(2) == 0 ? -1 : 1;
                int globalSign = rng.Next(2) == 0 ? -1 : 1;
                testInputs[i] = MakeInput(p, newsSign, globalSign, 1000 + i);
                testTargets[i] = MakeTarget(p, newsSign, globalSign);
            }

            return (tok, trainInputs, trainTargets, testInputs, testTargets);
        }
        private static float[,] CopyMatrix(float[,] src)
        {
            int r = src.GetLength(0), c = src.GetLength(1);
            var dst = new float[r, c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    dst[i, j] = src[i, j];
            return dst;
        }

        private static float[,] ZeroLike(float[,] src)
        {
            return new float[src.GetLength(0), src.GetLength(1)];
        }

        private static float[] CopyVector(float[] src)
        {
            if (src == null) return null;
            var dst = new float[src.Length];
            Array.Copy(src, dst, src.Length);
            return dst;
        }

        private static NewsStory[] CopyStories(NewsStory[] src)
        {
            if (src == null) return null;
            var dst = new NewsStory[src.Length];
            for (int i = 0; i < src.Length; i++)
                dst[i] = new NewsStory(src[i].TokenIds, src[i].ArrivalTime);
            return dst;
        }

        private static MultimodalInput CloneInput(
            MultimodalInput src,
            bool keepPrice = true,
            bool keepNews = true,
            bool keepGlobals = true)
        {
            return new MultimodalInput
            {
                PredictionTimestamp = src.PredictionTimestamp,
                PriceSequence = keepPrice ? CopyMatrix(src.PriceSequence) : ZeroLike(src.PriceSequence),
                NewsStories = keepNews ? CopyStories(src.NewsStories) : null,
                GlobalFeatures = keepGlobals ? CopyVector(src.GlobalFeatures) : null
            };
        }

        private void AssertAllPredictionsFinite(MmtacModel model, MultimodalInput[] inputs, string label)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var p = model.PredictNext(inputs[i]);
                Assert(float.IsFinite(p.High), $"{label}: non-finite High at {i}");
                Assert(float.IsFinite(p.Low), $"{label}: non-finite Low at {i}");
                Assert(float.IsFinite(p.Close), $"{label}: non-finite Close at {i}");
                Assert(float.IsFinite(p.Range), $"{label}: non-finite Range at {i}");
                Assert(float.IsFinite(p.Quality), $"{label}: non-finite Quality at {i}");
                Assert(float.IsFinite(p.DirectionProb), $"{label}: non-finite DirectionProb at {i}");
                Assert(float.IsFinite(p.MidWindowDirectionProb), $"{label}: non-finite MidWindowDirectionProb at {i}");
                Assert(float.IsFinite(p.Confidence), $"{label}: non-finite Confidence at {i}");
            }
        }

        private float AverageSequentialCloseError(
            MmtacModel model,
            MultimodalInput[] inputs,
            ModelTarget[][] targets,
            double[] timestamps,
            double timeUnitsPerPosition = 1.0)
        {
            model.ClearAllMemory();
            float err = 0f;

            for (int i = 0; i < inputs.Length; i++)
            {
                var pred = model.PredictWithMemory(inputs[i], timestamps[i], timeUnitsPerPosition);
                float tgt = targets[i][targets[i].Length - 1].Close;
                err += MathF.Abs(pred.Close - tgt);
            }

            model.ClearAllMemory();
            return err / Math.Max(1, inputs.Length);
        }

        private (BPETokenizer tok, MultimodalInput[] trainInputs, ModelTarget[][] trainTargets, double[] trainTs,
                 MultimodalInput[] testInputs, ModelTarget[][] testTargets, double[] testTs)
            RegimeCarryData(int trainN = 40, int testN = 16, int seqLen = 8)
        {
            string bullText = "regime bullish macro support";
            string bearText = "regime bearish macro stress";

            var tok = new BPETokenizer();
            tok.Train(new[] { bullText, bearText }, vocabSize: 64, minFrequency: 1);

            int[] bull = tok.Encode(bullText, addSpecialTokens: true);
            int[] bear = tok.Encode(bearText, addSpecialTokens: true);

            MultimodalInput MakeInput(int regimeSign, bool emitRegimeStory, int idx)
            {
                return new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                    PriceSequence = ConstantPriceSequence(seqLen, 0f, 0f, 0f, 0f, 1f),
                    NewsStories = emitRegimeStory
                        ? new[] { new NewsStory(regimeSign > 0 ? bull : bear, 0f) }
                        : null,
                    GlobalFeatures = null
                };
            }

            ModelTarget[] MakeTarget(int regimeSign)
            {
                float close = regimeSign > 0 ? 0.82f : 0.18f;
                float range = 0.10f;
                float quality = 0.90f;
                int dir = regimeSign > 0 ? 1 : 0;
                int mid = dir;
                return ConstantTargets(seqLen, close, range, quality, dir, mid);
            }

            void Fill(int n, int offset, out MultimodalInput[] inputs, out ModelTarget[][] targets, out double[] ts)
            {
                inputs = new MultimodalInput[n];
                targets = new ModelTarget[n][];
                ts = new double[n];

                int block = 4;
                for (int i = 0; i < n; i++)
                {
                    int blockIdx = i / block;
                    int regimeSign = (blockIdx % 2 == 0) ? 1 : -1;
                    bool emitStory = (i % block == 0);

                    inputs[i] = MakeInput(regimeSign, emitStory, offset + i);
                    targets[i] = MakeTarget(regimeSign);
                    ts[i] = (offset + i) * 100.0;
                }
            }

            Fill(trainN, 0, out var trainInputs, out var trainTargets, out var trainTs);
            Fill(testN, 1000, out var testInputs, out var testTargets, out var testTs);

            return (tok, trainInputs, trainTargets, trainTs, testInputs, testTargets, testTs);
        }

        private (BPETokenizer tok, MultimodalInput[] trainInputs, ModelTarget[][] trainTargets,
                 MultimodalInput[] testInputs, ModelTarget[][] testTargets)
            MixedMissingModalitiesData(int trainN = 120, int testN = 32, int seqLen = 8, int seed = 1234)
        {
            string bullText = "policy easing growth support";
            string bearText = "credit stress slowdown risk";
            var tok = new BPETokenizer();
            tok.Train(new[] { bullText, bearText }, vocabSize: 80, minFrequency: 1);

            int[] bull = tok.Encode(bullText, addSpecialTokens: true);
            int[] bear = tok.Encode(bearText, addSpecialTokens: true);

            var rng = new Random(seed);

            MultimodalInput MakeInput(float p, int newsSign, int globalSign, bool includeNews, bool includeGlobal, int idx)
            {
                return new MultimodalInput
                {
                    PredictionTimestamp = DateTime.UtcNow.AddMinutes(idx),
                    PriceSequence = ConstantPriceSequence(seqLen, p, p * p, MathF.Sign(p), 0.25f * p, 1f),
                    NewsStories = includeNews
                        ? new[] { new NewsStory(newsSign > 0 ? bull : bear, 0f) }
                        : null,
                    GlobalFeatures = includeGlobal
                        ? new[] { globalSign > 0 ? 1f : 0f, globalSign < 0 ? 1f : 0f, MathF.Abs(p) }
                        : null
                };
            }

            ModelTarget[] MakeTarget(float p, int newsSign, int globalSign, bool includeNews, bool includeGlobal)
            {
                float score = 0.55f * p;
                if (includeNews) score += 0.25f * newsSign;
                if (includeGlobal) score += 0.20f * globalSign;

                float close = 0.50f + 0.22f * MathF.Tanh(score);
                float range = 0.05f + 0.03f * MathF.Abs(score);
                float quality = Tanh01(score);
                int dir = score > 0f ? 1 : 0;
                int mid = (0.6f * p + (includeGlobal ? 0.4f * globalSign : 0f)) > 0f ? 1 : 0;

                return ConstantTargets(seqLen, close, range, quality, dir, mid);
            }

            void Fill(int n, int offset, out MultimodalInput[] inputs, out ModelTarget[][] targets)
            {
                inputs = new MultimodalInput[n];
                targets = new ModelTarget[n][];

                for (int i = 0; i < n; i++)
                {
                    float p = (float)(rng.NextDouble() * 2.0 - 1.0);
                    int newsSign = rng.Next(2) == 0 ? -1 : 1;
                    int globalSign = rng.Next(2) == 0 ? -1 : 1;

                    bool includeNews = (i % 4 != 1);
                    bool includeGlobal = (i % 4 != 2);

                    inputs[i] = MakeInput(p, newsSign, globalSign, includeNews, includeGlobal, offset + i);
                    targets[i] = MakeTarget(p, newsSign, globalSign, includeNews, includeGlobal);
                }
            }

            Fill(trainN, 0, out var trainInputs, out var trainTargets);
            Fill(testN, 1000, out var testInputs, out var testTargets);

            return (tok, trainInputs, trainTargets, testInputs, testTargets);
        }

        #endregion
    }
}
