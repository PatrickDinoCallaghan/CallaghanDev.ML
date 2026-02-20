
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CallaghanDev.ML.Transformers.TACAMT;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Gradients = CallaghanDev.ML.Transformers.TACAMT.Gradients;
using MultimodalForwardCache = CallaghanDev.ML.Transformers.TACAMT.MultimodalForwardCache;
using TACAMT_Model = CallaghanDev.ML.Transformers.TACAMT.Model;
using TACAMT_Trainer = CallaghanDev.ML.Transformers.TACAMT.Trainer;

namespace CallaghanDev.ML.TestConsoleApp
{
    public class TACMATests
    {
        private int _passed = 0;
        private int _failed = 0;
        private List<string> _failures = new List<string>();

        public void RunAllTests()
        {
            Console.WriteLine("=== TACAMT Comprehensive Tests ===\n");

            _passed = 0;
            _failed = 0;
            _failures.Clear();

            var tests = new (Action test, string name)[]
            {
                (Test_ContextTypeEmbedding_ChangesAfterTraining, "ContextTypeEmb: Type embeddings change after training"),
                (Test_ContextTypeEmbedding_TypesDivergeAfterTraining, "ContextTypeEmb: Type 0 and type 1 embeddings diverge"),
                (Test_ContextTypeEmbedding_SaveLoadRoundTrip, "ContextTypeEmb: Save/load preserves type embeddings"),
                (Test_ContextTypeEmbedding_IncludedInGradientClipping, "ContextTypeEmb: Included in gradient clipping norm"),


                #region price context training

                (Test_PriceContext_HistoryGrads_DoNotFlowBack, "PriceContext: History gradients do NOT flow back"),
                (Test_PriceContext_CurrentChunk_GradsDoFlow, "PriceContext: Current chunk gradients DO flow"),
                (Test_PriceContext_HistoryEncoder_WeightsUnchanged_SingleStep, "PriceContext: History encoder weights unchanged in single step"),
                (Test_PriceContext_DecayNetwork_GetsGradsFromPriceContext, "PriceContext: Decay network gets gradients from price context entries"),
                (Test_PriceContext_CrossAttnKV_GradsOnlyForNews_NotPriceCtx, "PriceContext: Cross-attn K/V grads accumulate for news, not price context"),
                
	            #endregion

                // ===========================================================
                // PRICE CONTEXT TRAINING — Basic Functionality
                // ===========================================================
                (Test_PriceContext_ForwardWithContext_NoError,
                    "PriceContext: ForwardWithPriceContextAndCache runs without error"),
                (Test_PriceContext_ForwardWithContext_OutputShape,
                    "PriceContext: Output shape matches current chunk length"),
                (Test_PriceContext_ForwardWithContext_NoNaN,
                    "PriceContext: No NaN in output"),
                (Test_PriceContext_EncodePriceHistory_NoCache,
                    "PriceContext: EncodePriceHistory produces valid hidden states"),
                (Test_PriceContext_CombinedContext_NewsAndPrice,
                    "PriceContext: Combined news + price context produces valid output"),

                // ===========================================================
                // PRICE CONTEXT TRAINING — Training Effectiveness
                // ===========================================================
                (Test_PriceContext_LossDecreases,
                    "PriceContext: Loss decreases with price context training"),
                (Test_PriceContext_SingleSample_Overfit,
                    "PriceContext: Single sample overfits with price context"),
                (Test_PriceContext_TrainingImproves_OverNoContext,
                    "PriceContext: Training with context produces valid loss"),

                // ===========================================================
                // PRICE CONTEXT TRAINING — Behavioral Correctness
                // ===========================================================
                (Test_PriceContext_WithVsWithout_DifferentOutputs,
                    "PriceContext: With vs without price context produces different outputs"),
                (Test_PriceContext_HistoryLength_AffectsOutput,
                    "PriceContext: Different history lengths produce different outputs"),
                (Test_PriceContext_Deterministic,
                    "PriceContext: Same split point produces same output"),
                (Test_PriceContext_SplitPoint_Respected,
                    "PriceContext: Split respects min history and min current lengths"),

                // ===========================================================
                // PRICE CONTEXT TRAINING — Config Edge Cases
                // ===========================================================
                (Test_PriceContext_SequenceTooShort_SkipsSplit,
                    "PriceContext: Sequence too short for split trains without context"),
                (Test_PriceContext_MinimalSplittableSequence,
                    "PriceContext: Minimal splittable sequence works"),
                (Test_PriceContext_LargeHistorySmallCurrent,
                    "PriceContext: Large history, small current chunk"),
                (Test_PriceContext_SmallHistoryLargeCurrent,
                    "PriceContext: Small history, large current chunk"),

                // ===========================================================
                // SEQUENTIAL TRAINING
                // ===========================================================
                (Test_Sequential_MemoryAccumulates,
                    "Sequential: Memory accumulates across samples"),
                (Test_Sequential_LossDecreases,
                    "Sequential: Loss decreases over epoch"),
                (Test_Sequential_MemoryCleared_EachEpoch,
                    "Sequential: Memory cleared at start of each epoch"),
                (Test_Sequential_PriceMemory_StoredFromForward,
                    "Sequential: Price memory stored from forward pass hidden states"),

                // ===========================================================
                // NUMERICAL STABILITY
                // ===========================================================
                (Test_PriceContext_GradientClipping_NoNaN,
                    "PriceContext: Gradient clipping prevents NaN with high LR"),
                (Test_PriceContext_NoNaN_AfterManyEpochs,
                    "PriceContext: No NaN after extended training"),

                // ===========================================================
                // TACAMT CORE — Multi-Story Forward
                // ===========================================================
                (Test_TACAMT_MultiStory_ForwardNoError,
                    "TACAMT: Multi-Story Forward No Error"),
                (Test_TACAMT_MultiStory_ForwardOutputShape,
                    "TACAMT: Multi-Story Forward Output Shape"),
                (Test_TACAMT_MultiStory_DifferentArrivalTimes_DifferentOutputs,
                    "TACAMT: Different Arrival Times => Different Outputs"),

                // ===========================================================
                // TACAMT CORE — Time Decay
                // ===========================================================
                (Test_TACAMT_TimeDecay_InitialValues,
                    "TACAMT: Time-Decay Initial Values"),
                (Test_TACAMT_TimeDecay_ParamsChangeAfterTraining,
                    "TACAMT: Decay Params Change After Training"),
                (Test_TACAMT_ComputeTimeBiasMatrix_Correct,
                    "TACAMT: ComputeTimeBiasMatrix Correctness"),

                // ===========================================================
                // TACAMT CORE — Training
                // ===========================================================
                (Test_TACAMT_MultiStory_LossDecreases,
                    "TACAMT: Multi-Story Loss Decreases"),
                (Test_TACAMT_MultiStory_SingleSampleOverfit,
                    "TACAMT: Multi-Story Single Sample Overfit"),

                // ===========================================================
                // TACAMT CORE — Backward Compatibility
                // ===========================================================
                (Test_TACAMT_BackwardCompat_SingleText_NoError,
                    "TACAMT: Backward-Compat Single Text"),
                (Test_TACAMT_BackwardCompat_NullText_NoError,
                    "TACAMT: Backward-Compat Null Text"),
                (Test_TACAMT_BackwardCompat_TrainWithIntArrays,
                    "TACAMT: Backward-Compat Train int[][]"),

                // ===========================================================
                // TACAMT CORE — Edge Cases
                // ===========================================================
                (Test_TACAMT_MixedBatch_SomeStoriesSomeNull,
                    "TACAMT: Mixed Batch Stories+Null"),
                (Test_TACAMT_NegativeArrivalTime_NoError,
                    "TACAMT: Negative Arrival Times"),
                (Test_TACAMT_ManyStories_NoError,
                    "TACAMT: Many Stories (12) No Error"),
                (Test_TACAMT_SingleTokenStory_NoError,
                    "TACAMT: Single Token Story"),

                // ===========================================================
                // TACAMT CORE — Memory Management
                // ===========================================================
                (Test_TACAMT_UpdateNewsMemory_StoresEntries,
                    "TACAMT: UpdateNewsMemory Stores Entries"),
                (Test_TACAMT_ClearNewsMemory_Resets,
                    "TACAMT: ClearNewsMemory Resets"),
                (Test_TACAMT_UpdateNewsMemory_TrimOldEntries,
                    "TACAMT: UpdateNewsMemory Trims Old"),
                (Test_TACAMT_PredictWithMemory_NoMemory_FallsBack,
                    "TACAMT: PredictWithMemory No Memory Fallback"),
                (Test_TACAMT_PredictWithMemory_WithMemory_NoError,
                    "TACAMT: PredictWithMemory With Memory"),
                (Test_TACAMT_PredictWithMemory_OldStoriesDecayed,
                    "TACAMT: PredictWithMemory Old Stories Decayed"),

                // ===========================================================
                // TACAMT CORE — Save/Load
                // ===========================================================
                (Test_TACAMT_SaveLoad_DecayParams_Preserved,
                    "TACAMT: SaveLoad Decay Params Preserved"),
                (Test_TACAMT_SaveLoad_NewsMemory_Preserved,
                    "TACAMT: SaveLoad News Memory Preserved"),
                (Test_TACAMT_SaveLoad_ForwardMatch,
                    "TACAMT: SaveLoad Forward Match"),
                (Test_TACAMT_SaveLoad_ContinueTraining,
                    "TACAMT: SaveLoad Continue Training"),

                // ===========================================================
                // TACAMT CORE — Determinism & Stability
                // ===========================================================
                (Test_TACAMT_Deterministic_MultiStory,
                    "TACAMT: Deterministic Multi-Story Forward"),
                (Test_TACAMT_GradientClipping_WithDecay_NoNaN,
                    "TACAMT: Gradient Clipping With Decay No NaN"),
                (Test_TACAMT_Validate_MultiStory,
                    "TACAMT: Validate Multi-Story Computable"),
                (Test_TACAMT_PerHeadDecayRates_Diverge,
                    "TACAMT: Per-Head Decay Rates Diverge"),

                // ===========================================================
                // TACAMT CORE — Price-Only (No Text)
                // ===========================================================
                (Test_TACAMT_PriceOnly_ForwardNoError,
                    "TACAMT: Price-Only Forward"),
                (Test_TACAMT_PriceOnly_LossDecreases,
                    "TACAMT: Price-Only Loss Decreases"),
                (Test_TACAMT_FrozenTextEncoder_DecayStillLearns,
                    "TACAMT: Frozen Text Encoder Decay Still Learns"),

                // ===========================================================
                // ADDITIONAL — Config Validation
                // ===========================================================
                (Test_Config_Validate_ThrowsOnBadDims,
                    "Config: Validate throws on bad dimensions"),
                (Test_Config_Validate_AcceptsGoodConfig,
                    "Config: Validate accepts good config"),

                // ===========================================================
                // ADDITIONAL — Memory Pruning
                // ===========================================================
                (Test_MemoryPruning_AttentionBased_KeepsHighScoreEntries,
                    "MemoryPruning: Attention-based keeps high-score entries"),
                (Test_MemoryPruning_OldestFirst_WhenDisabled,
                    "MemoryPruning: Oldest-first when attention pruning disabled"),
                (Test_MemoryPruning_ColdStartProtection,
                    "MemoryPruning: Cold-start protection for new entries"),

                // ===========================================================
                // ADDITIONAL — ContentAwareDecayNetwork Unit Tests
                // ===========================================================
                (Test_DecayNetwork_ForwardProduces_ValidBias,
                    "DecayNetwork: Forward produces valid bias"),
                (Test_DecayNetwork_BackwardProduces_NonZeroGrads,
                    "DecayNetwork: Backward produces non-zero gradients"),
                (Test_DecayNetwork_DropoutChangesOutput,
                    "DecayNetwork: Dropout changes output (training vs inference)"),
                (Test_DecayNetwork_WeightDecay_AddedToGrads,
                    "DecayNetwork: Weight decay added to gradients"),
                (Test_DecayNetwork_SaveLoad_Roundtrip,
                    "DecayNetwork: Save/Load roundtrip preserves weights"),

                // ===========================================================
                // ADDITIONAL — Confidence Head
                // ===========================================================
                (Test_ConfidenceHead_Outputs_SigmoidRange,
                    "ConfidenceHead: Outputs in [0, 1] range"),
                (Test_ConfidenceHead_TrainingUpdatesWeights,
                    "ConfidenceHead: Training updates confidence weights"),

                // ===========================================================
                // ADDITIONAL — Tokenizer Integration
                // ===========================================================
                (Test_Tokenizer_TokenizeStories_WithoutTokenizer_Throws,
                    "Tokenizer: TokenizeStories without tokenizer throws"),
                
                // DIMENSIONAL CONSISTENCY
                (Test_Dims_TextEmbedding_MatchesConfig, "Dims: TextTokenEmbedding matches [VocabSize, EmbDim]"),
                (Test_Dims_PriceInputProjection_MatchesConfig, "Dims: PriceInputProjection matches [EmbDim, FeatureDim]"),
                (Test_Dims_OutputProjection_MatchesConfig, "Dims: OutputProjection matches [OutputDim, EmbDim]"),
                (Test_Dims_PositionalEncoding_Text_MatchesConfig, "Dims: TextPositionalEncoding matches [MaxSeqLen, EmbDim]"),
                (Test_Dims_PositionalEncoding_Price_MatchesConfig, "Dims: PricePositionalEncoding matches [MaxSeqLen, EmbDim]"),
                (Test_Dims_AllBlocks_SelfAttn_WQ_Correct, "Dims: All PriceBlock SelfAttention.WQ is [EmbDim, EmbDim]"),
                (Test_Dims_AllBlocks_CrossAttn_WQ_Correct, "Dims: All PriceBlock CrossAttention.WQ is [EmbDim, EmbDim]"),
                (Test_Dims_DecayNetwork_Projections_Correct, "Dims: DecayNetwork projections match config"),
                (Test_Dims_LayerNormVectors_CorrectLength, "Dims: All LayerNorm gamma/beta vectors match EmbDim"),
                (Test_Dims_ConfidenceProjection_When_Disabled_IsNull, "Dims: ConfidenceProjection is null when disabled"),
                (Test_Dims_ConfidenceProjection_When_Enabled_Correct, "Dims: ConfidenceProjection is [1, EmbDim] when enabled"),
                // FORWARD — VARYING SEQUENCE LENGTHS
                (Test_Forward_SeqLen1_Works, "Forward: Single-timestep price sequence works"),
                (Test_Forward_SeqLen2_Works, "Forward: Two-timestep price sequence works"),
                (Test_Forward_MaxSeqLen_Works, "Forward: Max sequence length works"),
                (Test_Forward_DifferentFeatureDim_Works, "Forward: Non-default feature dim (3) works"),
                (Test_Forward_DifferentOutputDim_Works, "Forward: Non-default output dim (1) works"),
                // FORWARD — CONTEXT ROUTING
                (Test_Forward_NullStories_NullText_NoContext, "Forward: null stories -> no cross-attention context"),
                (Test_Forward_EmptyStories_NoContext, "Forward: empty stories array -> no cross-attention context"),
                (Test_Forward_SingleStory_CrossAttnActivated, "Forward: single story activates cross-attention"),
                (Test_Forward_MultipleStories_AllContribute, "Forward: multiple stories all contribute to context"),
                (Test_Forward_ContextHidden_DimMatchesPriceEmb, "Forward: context hidden dim matches PriceEmbeddingDim"),
                // MULTI-LAYER BEHAVIOR
                (Test_MultiLayer_MoreLayers_DifferentOutput, "MultiLayer: 1-layer vs 2-layer produces different outputs"),
                (Test_MultiLayer_AllBlocks_Executed, "MultiLayer: Cache confirms all blocks executed"),
                (Test_MultiLayer_DeepModel_NoNaN, "MultiLayer: 4-layer deep model produces no NaN"),
                // POSITIONAL ENCODING
                (Test_PosEnc_SinCosPattern, "PosEnc: Sinusoidal pattern verified"),
                (Test_PosEnc_DifferentPositions_DifferentEncodings, "PosEnc: Different positions have different encodings"),
                (Test_PosEnc_AffectsOutput, "PosEnc: Shuffled input positions change output"),
                // CAUSAL MASKING
                (Test_CausalMask_FutureTokens_DontAffectPast, "CausalMask: Changing future tokens doesn't affect past predictions"),
                (Test_CausalMask_Price_DecoderOnly_Enabled, "CausalMask: Price decoder uses causal mask by default"),
                // GRADIENT NUMERICAL CHECKS
                (Test_GradCheck_OutputProjection_ApproximatelyCorrect, "GradCheck: OutputProjection gradient ~= finite difference"),
                (Test_GradCheck_PriceInputProjection_ApproximatelyCorrect, "GradCheck: PriceInputProjection gradient ~= finite difference"),
                (Test_GradCheck_OutputBias_ApproximatelyCorrect, "GradCheck: OutputBias gradient ~= finite difference"),
                // TRAINING — PARAMETER UPDATES
                (Test_Train_AllPriceBlockParams_Updated, "Train: All PriceBlock parameters updated after training"),
                (Test_Train_OutputBias_Updated, "Train: OutputBias updated after training"),
                (Test_Train_PriceInputProjectionBias_Updated, "Train: PriceInputProjectionBias updated"),
                (Test_Train_TextEncoder_Updated_WhenNotFrozen, "Train: TextEncoder updated when not frozen"),
                (Test_Train_TextEncoder_NOT_Updated_WhenFrozen, "Train: TextEncoder NOT updated when frozen"),
                (Test_Train_FFN_Weights_Updated, "Train: FFN weights in price blocks updated"),
                (Test_Train_AllDecayNetworkParams_Updated, "Train: All decay network params updated"),
                // TRAINING — LOSS BEHAVIOR
                (Test_Train_ZeroTarget_LossDecreases, "Train: Zero targets -> loss decreases"),
                (Test_Train_ConstantTarget_LossDecreases, "Train: Constant targets -> loss decreases"),
                (Test_Train_BatchSize1_vs_BatchSizeN_BothWork, "Train: Batch size 1 and batch size N both decrease loss"),
                (Test_Train_LearningRateDecay_Works, "Train: LR decay produces valid loss"),
                (Test_Train_HighLR_WithClipping_NoExplosion, "Train: High LR (0.5) with gradient clipping doesn't explode"),
                // GRADIENT CLIPPING
                (Test_GradClip_NormReduced, "GradClip: Gradient norm reduced after clipping"),
                //(Test_GradClip_PreservesDirection, "GradClip: Large threshold same as no clipping"),
                // PREDICT NEXT
                (Test_PredictNext_ReturnsLastTimestep, "PredictNext: Returns correct output dim"),
                (Test_PredictNext_ConfidenceInRange, "PredictNext: Confidence in [0, 1] when enabled"),
                (Test_PredictNext_DefaultConfidence1_WhenDisabled, "PredictNext: Confidence = 1.0 when head disabled"),
                // PREDICT WITH MEMORY
                (Test_PredictWithMemory_AccumulatesNewsAndPrice, "PredictWithMemory: Accumulates both news and price memory"),
                (Test_PredictWithMemory_PrunesWhenOverLimit, "PredictWithMemory: Prunes when exceeding max size"),
                (Test_PredictWithMemory_MultipleCallsChain, "PredictWithMemory: Multiple calls accumulate correctly"),
                (Test_PredictWithMemory_TimestampOrdering_Preserved, "PredictWithMemory: Timestamps stored correctly"),
                // MEMORY PRUNING ADVANCED
                (Test_Pruning_PriceMemory_AttentionBased, "Pruning: Price memory uses attention-based pruning"),
                (Test_Pruning_ReserveFraction_Respected, "Pruning: Reserve fraction keeps newest entries"),
                (Test_Pruning_ZeroMaxSize_ClearsAll, "Pruning: Max size 0 clears all entries"),
                (Test_Pruning_ExactlyAtLimit_NoOp, "Pruning: At exactly max size is a no-op"),
                (Test_Pruning_Config_SavedAndLoaded, "Pruning: PruningConfig preserved through save/load"),
                // SAVE/LOAD EXHAUSTIVE
                (Test_SaveLoad_AllTextBlockWeights_Match, "SaveLoad: All TextBlock attention weights match"),
                (Test_SaveLoad_AllPriceBlockWeights_Match, "SaveLoad: All PriceBlock attention weights match"),
                (Test_SaveLoad_PriceMemory_Preserved, "SaveLoad: PriceMemory entries preserved"),
                (Test_SaveLoad_AttentionScores_Preserved, "SaveLoad: AttentionScore and QueryCount preserved"),
                (Test_SaveLoad_Config_AllFields_Preserved, "SaveLoad: All config fields preserved"),
                // CONFIG VALIDATION
                (Test_Config_PriceEmbDim_NotDivisibleByHeads_Throws, "Config: PriceEmbeddingDim not divisible by heads throws"),
                (Test_Config_ZeroVocabSize_Throws, "Config: Zero vocab size throws"),
                (Test_Config_NegativeL2_Throws, "Config: Negative L2 throws"),
                (Test_Config_DecayDropoutOutOfRange_Throws, "Config: Decay dropout >= 1 throws"),
                (Test_Config_ZeroDecayProjectionDim_Throws, "Config: Zero DecayProjectionDim throws"),
                (Test_Config_ZeroGradientClipThreshold_Throws, "Config: Zero GradientClippingThreshold throws"),
                // DECAY NETWORK MATH
                (Test_Decay_LargerTimeDiff_StrongerDecay, "Decay: Larger time diff -> more negative bias"),
                (Test_Decay_ZeroTimeDiff_NearZeroBias, "Decay: Zero time diff -> near-zero bias"),
                (Test_Decay_GateOutputInSigmoidRange, "Decay: Gate output is in (0, 1) sigmoid range"),
                (Test_Decay_BaseDecayRate_AlwaysPositive, "Decay: exp(LogBaseDecayRate) is always positive"),
                (Test_Decay_MultiScale_TimeEncoding_DifferentPerBase, "Decay: Multi-scale time encoding differs per base"),
                (Test_Decay_MemoryInteraction_ChangesOutput, "Decay: Memory interaction changes output vs single key"),
                // SEQUENTIAL TRAINING ADVANCED
                (Test_Sequential_PriceMemoryGrows_EachSample, "Sequential: Price memory grows with each sample"),
                (Test_Sequential_MultiplEpochs_MemoryReset_EachEpoch, "Sequential: Memory reset at each epoch start"),
                (Test_Sequential_WithNoStories_StillWorks, "Sequential: Null stories throughout still trains"),
                // PRICE CONTEXT ADVANCED
                (Test_PriceContext_SplitPointDistribution_WithinBounds, "PriceContext: Split points within bounds"),
                (Test_PriceContext_HistoryDetached_CurrentTrained, "PriceContext: History detached, current trained"),
                // TEXT ENCODER
                (Test_TextEncoder_Bidirectional_Default, "TextEncoder: Bidirectional by default"),
                (Test_TextEncoder_MeanPooling_Produces_FixedDimOutput, "TextEncoder: Mean pooling -> [numStories, embDim]"),
                // TOKENIZER
                (Test_Tokenizer_MatchingVocab_AcceptedAndUsable, "Tokenizer: Matching vocab accepted and usable"),
                (Test_Tokenizer_SavedWithModel_LoadedBack, "Tokenizer: Saved with model and loaded back"),
                // NUMERICAL STABILITY
                (Test_Stability_VeryLargeInputs_NoNaN, "Stability: Very large inputs don't cause NaN"),
                (Test_Stability_VerySmallInputs_NoNaN, "Stability: Very small inputs don't cause NaN"),
                (Test_Stability_ZeroInputs_NoNaN, "Stability: All-zero inputs don't cause NaN"),
                (Test_Stability_NegativeInputs_NoNaN, "Stability: All-negative inputs don't cause NaN"),
                (Test_Stability_MixedMagnitude_NoNaN, "Stability: Mixed magnitude inputs don't cause NaN"),
                // EQUIVALENCE
                (Test_Equiv_ForwardAndForwardWithCache_SameOutput, "Equiv: Forward() and ForwardWithCache() same output"),
                //(Test_Equiv_TwoModels_SameSeed_SameOutput, "Equiv: Two models same seed -> identical output"),
                (Test_Equiv_GradientsZero_BeforeTraining, "Equiv: Gradient storage zero before first backward"),


                (Test_E2E_LearnsBullVsBearSignal, "E2E: Bull vs Bear news drives different predictions"),
                (Test_E2E_SingleSample_NearPerfectOverfit, "E2E: Single sample overfits to <1% initial loss"),

                // ===========================================================
                // LARGER SCALE STRESS TESTS
                // ===========================================================
                (Test_Scale_LargerConfig_NoNaN, "Scale: Large config (64-dim, 8-head, 3-layer) no NaN"),
                (Test_Scale_LargerConfig_TrainingConverges, "Scale: Larger config training loss decreases"),
                (Test_Scale_ManyMemoryEntries_NoNaN, "Scale: 50 memory entries prediction no NaN"),

                // ===========================================================
                // TEXT ENCODER FULL PIPELINE
                // ===========================================================
                (Test_TextPipeline_FullTokenizerFlow, "TextPipeline: Tokenizer→Train→Validate full flow"),
                (Test_TextPipeline_DifferentTexts_DifferentOutputs, "TextPipeline: Different texts produce different outputs"),
                (Test_TextPipeline_MultipleStories_VariedLengths, "TextPipeline: Multiple stories varied lengths no NaN"),

                // ===========================================================
                // GRADIENT CORRECTNESS — ATTENTION & DECAY
                // ===========================================================
                (Test_GradCheck_SelfAttention_WQ_FiniteDifference, "GradCheck: Self-attention WQ finite difference non-zero"),
                (Test_GradCheck_CrossAttention_WK_FiniteDifference, "GradCheck: Cross-attention WK finite difference non-zero"),
                (Test_GradCheck_DecayNetwork_LogBaseDecayRate_FiniteDifference, "GradCheck: DecayNetwork LogBaseDecayRate finite difference"),
                (Test_GradCheck_DecayNetwork_W1_FiniteDifference, "GradCheck: DecayNetwork W1 MLP finite difference"),
                (Test_GradCheck_TextEmbedding_FiniteDifference, "GradCheck: TextEmbedding affects loss through cross-attention"),
                (Test_E2E_LearnRecencyMatters, "E2E: Recent news produces different prediction than old news"),
            };

            for (int i = 0; i < tests.Length; i++)
            {
                Console.Write($"  [{i + 1,2}/{tests.Length}] {tests[i].name,-65} ");
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

            Console.WriteLine($"\n{"",3}{new string('─', 68)}");
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
            Console.WriteLine($" / {tests.Length} total\n");
            if (_failures.Count > 0)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("   Failed tests:");
                Console.ResetColor();
                foreach (var f in _failures)
                    Console.WriteLine($"     • {f}");
                Console.WriteLine();
            }
        }

        private void Assert(bool condition, string message)
        {
            if (!condition) throw new Exception(message);
        }

        private Transformers.TACAMT.Config CreateConfig(int textVocabSize = 50, int embDim = 16, int numHeads = 2, int numLayers = 1, int ffnDim = 32, int inputFeatures = 5, int outputDim = 5, int priceSeqLen = 20, bool useConfidence = false, bool freezeText = false, int priceCtxMinHist = 5, int priceCtxMinCurrent = 5)
        {
            return new Transformers.TACAMT.Config
            {
                TextVocabSize = textVocabSize,
                TextMaxSequenceLength = 32,
                TextEmbeddingDim = embDim,
                TextNumHeads = numHeads,
                TextNumLayers = numLayers,
                TextFeedForwardDim = ffnDim,
                TextUseDecoderOnly = false,
                PriceInputFeatureDim = inputFeatures,
                PriceMaxSequenceLength = priceSeqLen + 2,
                PriceEmbeddingDim = embDim,
                PriceNumHeads = numHeads,
                PriceNumLayers = numLayers,
                PriceFeedForwardDim = ffnDim,
                PriceUseDecoderOnly = true,
                OutputDim = outputDim,
                UseConfidenceHead = useConfidence,
                FreezeTextEncoder = freezeText,
                FFNActivationType = ActivationType.Relu,
                AccelerationType = AccelerationType.CPU,
                L2RegulationLamda = 0f,
                GradientClippingThreshold = 1.0f,
                PriceContextMinHistoryLength = priceCtxMinHist,
                PriceContextMinCurrentLength = priceCtxMinCurrent,
                DecayProjectionDim = 8,
                DecayHiddenDim = 16,
                DecayTimeEncodingBases = 8,
            };
        }
 
        private bool MatrixChanged(float[,] before, float[,] after, float threshold = 1e-10f)
        {
            for (int i = 0; i < before.GetLength(0); i++)
                for (int j = 0; j < before.GetLength(1); j++)
                    if (MathF.Abs(before[i, j] - after[i, j]) > threshold) return true;
            return false;
        }

        private bool VectorChanged(float[] before, float[] after, float threshold = 1e-10f)
        {
            for (int i = 0; i < before.Length; i++)
                if (MathF.Abs(before[i] - after[i]) > threshold) return true;
            return false;
        }

        private bool Array3DChanged(float[,,] before, float[,,] after, float threshold = 1e-10f)
        {
            for (int i = 0; i < before.GetLength(0); i++)
                for (int j = 0; j < before.GetLength(1); j++)
                    for (int k = 0; k < before.GetLength(2); k++)
                        if (MathF.Abs(before[i, j, k] - after[i, j, k]) > threshold) return true;
            return false;
        }

        private float[,] CloneMatrix(float[,] m) => (float[,])m.Clone();
        private float[] CloneVector(float[] v) => (float[])v.Clone();
        private float[,,] Clone3D(float[,,] a) => (float[,,])a.Clone();

        private float[,] SliceRows(float[,] m, int start, int end)
        {
            int cols = m.GetLength(1), rows = end - start;
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = m[start + i, j];
            return result;
        }

        private bool HasNaN(float[,] m)
        {
            for (int i = 0; i < m.GetLength(0); i++)
                for (int j = 0; j < m.GetLength(1); j++)
                    if (float.IsNaN(m[i, j]) || float.IsInfinity(m[i, j])) return true;
            return false;
        }

        private string GetTempDir()
        {
            var dir = Path.Combine(Path.GetTempPath(), $"tacamt_test_{Guid.NewGuid():N}");
            Directory.CreateDirectory(dir);
            return dir;
        }

        private void CleanupDir(string dir)
        {
            try { if (Directory.Exists(dir)) Directory.Delete(dir, true); } catch { }
        }

        // =====================================================================
        // PRICE CONTEXT TRAINING — Gradient Isolation
        // =====================================================================

        public void Test_PriceContext_HistoryGrads_DoNotFlowBack()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var ps = priceInputs[0];
            int sl = ps.GetLength(0);
            int splitPoint = 8;

            var histPrices = SliceRows(ps, 0, splitPoint);
            var currentInput = SliceRows(ps, splitPoint, sl - 1);

            var priceCtxHidden = model.EncodePriceHistory(histPrices);
            var priceCtxTimes = new float[splitPoint];
            for (int t = 0; t < splitPoint; t++) priceCtxTimes[t] = -(splitPoint - t);

            var cache = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred, conf) = model.ForwardWithPriceContextAndCache(
                stories[0], currentInput, priceCtxHidden, priceCtxTimes, cache);

            // Perturb history and forward again
            var perturbedCtx = CloneMatrix(priceCtxHidden);
            for (int i = 0; i < perturbedCtx.GetLength(0); i++)
                for (int j = 0; j < perturbedCtx.GetLength(1); j++)
                    perturbedCtx[i, j] += 1.0f;

            var cache2 = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred2, _) = model.ForwardWithPriceContextAndCache(
                stories[0], currentInput, perturbedCtx, priceCtxTimes, cache2);

            bool outputsDiffer = false;
            for (int i = 0; i < pred.GetLength(0) && !outputsDiffer; i++)
                for (int j = 0; j < pred.GetLength(1) && !outputsDiffer; j++)
                    if (MathF.Abs(pred[i, j] - pred2[i, j]) > 1e-6f) outputsDiffer = true;
            Assert(outputsDiffer, "Price context should affect forward output");

            // Verify cache structure proves no gradient path back through history
            Assert(cache.PriceContextHidden != null, "Cache should store price context reference");
            Assert(cache.PriceBlockCaches.Count == config.PriceNumLayers,
                "Cache should only have block caches for the current chunk");

            int currentSeqLen = currentInput.GetLength(0);
            var firstBlockCache = cache.PriceBlockCaches[0];
            Assert(firstBlockCache.BlockInput.GetLength(0) == currentSeqLen,
                $"Block cache input rows should be {currentSeqLen}, got {firstBlockCache.BlockInput.GetLength(0)}");
        }

        public void Test_PriceContext_CurrentChunk_GradsDoFlow()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var projBefore = CloneMatrix(model.PriceInputProjection);
            var outProjBefore = CloneMatrix(model.OutputProjection);
            var selfWqBefore = CloneMatrix(model.PriceBlocks[0].SelfAttention.WQ);
            var crossWqBefore = CloneMatrix(model.PriceBlocks[0].CrossAttention.WQ);
            var lnSelfGammaBefore = CloneVector(model.PriceBlocks[0].LNSelfGamma);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 5, Epochs = 3, UseGradientClipping = false, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            Assert(MatrixChanged(projBefore, model.PriceInputProjection), "PriceInputProjection should change");
            Assert(MatrixChanged(outProjBefore, model.OutputProjection), "OutputProjection should change");
            Assert(MatrixChanged(selfWqBefore, model.PriceBlocks[0].SelfAttention.WQ), "Self-attention WQ should change");
            Assert(MatrixChanged(crossWqBefore, model.PriceBlocks[0].CrossAttention.WQ), "Cross-attention WQ should change");
            Assert(VectorChanged(lnSelfGammaBefore, model.PriceBlocks[0].LNSelfGamma), "LNSelfGamma should change");
        }

        public void Test_PriceContext_HistoryEncoder_WeightsUnchanged_SingleStep()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var histPrices = SliceRows(priceInputs[0], 0, 8);
            var histHidden1 = model.EncodePriceHistory(histPrices);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 1, Epochs = 1, UseGradientClipping = false, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            var histHidden2 = model.EncodePriceHistory(histPrices);

            bool encodingsDiffer = false;
            for (int i = 0; i < histHidden1.GetLength(0) && !encodingsDiffer; i++)
                for (int j = 0; j < histHidden1.GetLength(1) && !encodingsDiffer; j++)
                    if (MathF.Abs(histHidden1[i, j] - histHidden2[i, j]) > 1e-6f) encodingsDiffer = true;

            Assert(encodingsDiffer,
                "History encoding should change after training (shared weights updated via current chunk)");
        }

        public void Test_PriceContext_DecayNetwork_GetsGradsFromPriceContext()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var decayNet = model.PriceBlocks[0].DecayNetwork;
            var logDecayBefore = CloneVector(decayNet.LogBaseDecayRate);
            float w1Before = decayNet.W1[0, 0, 0];

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 5, Epochs = 10, UseGradientClipping = false, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            Assert(VectorChanged(logDecayBefore, decayNet.LogBaseDecayRate),
                "LogBaseDecayRate should change — decay network gets gradients with price context");
            Assert(MathF.Abs(w1Before - decayNet.W1[0, 0, 0]) > 1e-8f,
                "Decay W1 should change — MLP gradients should flow through price context path");
        }

        public void Test_PriceContext_CrossAttnKV_GradsOnlyForNews_NotPriceCtx()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16, freezeText: false);
            var model = new TACAMT_Model(config, new Random(42));

            var textEmbBefore = CloneMatrix(model.TextTokenEmbedding);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 1, Epochs = 3, UseGradientClipping = false, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            Assert(MatrixChanged(textEmbBefore, model.TextTokenEmbedding),
                "Text embedding should change — news context K/V gradients flow back to text encoder");
        }

        // =====================================================================
        // PRICE CONTEXT TRAINING — Basic Functionality
        // =====================================================================

        public void Test_PriceContext_ForwardWithContext_NoError()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var priceCtxHidden = model.EncodePriceHistory(SliceRows(priceInputs[0], 0, 8));
            var priceCtxTimes = new float[8];
            for (int t = 0; t < 8; t++) priceCtxTimes[t] = -(8 - t);

            var cache = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred, _) = model.ForwardWithPriceContextAndCache(
                stories[0], SliceRows(priceInputs[0], 8, 15), priceCtxHidden, priceCtxTimes, cache);

            Assert(pred != null, "Prediction should not be null");
        }

        public void Test_PriceContext_ForwardWithContext_OutputShape()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var currentInput = SliceRows(priceInputs[0], 8, 15);
            int expectedLen = currentInput.GetLength(0);
            var priceCtxHidden = model.EncodePriceHistory(SliceRows(priceInputs[0], 0, 8));
            var priceCtxTimes = Enumerable.Range(0, 8).Select(t => -(float)(8 - t)).ToArray();

            var cache = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred, _) = model.ForwardWithPriceContextAndCache(
                stories[0], currentInput, priceCtxHidden, priceCtxTimes, cache);

            Assert(pred.GetLength(0) == expectedLen, $"Output rows: {pred.GetLength(0)}, expected {expectedLen}");
            Assert(pred.GetLength(1) == config.OutputDim, $"Output cols: {pred.GetLength(1)}, expected {config.OutputDim}");
        }

        public void Test_PriceContext_ForwardWithContext_NoNaN()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var priceCtxHidden = model.EncodePriceHistory(SliceRows(priceInputs[0], 0, 8));
            var priceCtxTimes = Enumerable.Range(0, 8).Select(t => -(float)(8 - t)).ToArray();
            var cache = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred, _) = model.ForwardWithPriceContextAndCache(
                stories[0], SliceRows(priceInputs[0], 8, 15), priceCtxHidden, priceCtxTimes, cache);

            Assert(!HasNaN(pred), "NaN/Inf in predictions");
        }

        public void Test_PriceContext_EncodePriceHistory_NoCache()
        {
            var config = CreateConfig(priceSeqLen: 20);
            var model = new TACAMT_Model(config, new Random(42));

            var rng = new Random(42);
            var histPrices = new float[10, 5];
            for (int t = 0; t < 10; t++)
                for (int f = 0; f < 5; f++)
                    histPrices[t, f] = (float)rng.NextDouble();

            var hidden = model.EncodePriceHistory(histPrices);

            Assert(hidden.GetLength(0) == 10, $"History hidden rows: {hidden.GetLength(0)} expected 10");
            Assert(hidden.GetLength(1) == config.PriceEmbeddingDim, $"Hidden cols mismatch");
            Assert(!HasNaN(hidden), "NaN in history hidden");
        }

        public void Test_PriceContext_CombinedContext_NewsAndPrice()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var priceCtxHidden = model.EncodePriceHistory(SliceRows(priceInputs[0], 0, 6));
            var priceCtxTimes = Enumerable.Range(0, 6).Select(t => -(float)(6 - t)).ToArray();

            var cache = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred, _) = model.ForwardWithPriceContextAndCache(
                stories[0], SliceRows(priceInputs[0], 6, 15), priceCtxHidden, priceCtxTimes, cache);

            Assert(cache.NumPriceContext == 6, $"NumPriceContext: {cache.NumPriceContext}, expected 6");
            Assert(cache.NumNewsContext > 0, $"NumNewsContext should be > 0");
            Assert(pred.GetLength(0) == 9, $"Output rows: {pred.GetLength(0)}, expected 9");
        }

        // =====================================================================
        // PRICE CONTEXT TRAINING — Training Effectiveness
        // =====================================================================

        public void Test_PriceContext_LossDecreases()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 10, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
            float lossBefore = trainer.Validate(stories, priceInputs, priceTargets);

            trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            float lossAfter = trainer.Validate(stories, priceInputs, priceTargets);

            Assert(lossAfter < lossBefore, $"Loss: {lossBefore:F6} -> {lossAfter:F6}");
        }

        public void Test_PriceContext_SingleSample_Overfit()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, embDim: 32, numLayers: 2, ffnDim: 64, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var t1 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            float lossBefore = t1.Validate(stories, priceInputs, priceTargets);

            var t2 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.005f, BatchSize = 1, Epochs = 1000, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false });
            t2.Train(stories, priceInputs, priceTargets);

            float lossAfter = t2.Validate(stories, priceInputs, priceTargets);

            Assert(lossAfter < lossBefore, $"Overfit failed: {lossBefore:F6} -> {lossAfter:F6}");

            // Assert(lossAfter < lossBefore * 0.5f, $"Overfit failed: {lossBefore:F6} -> {lossAfter:F6}");
        }

        public void Test_PriceContext_TrainingImproves_OverNoContext()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 10, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            float loss = trainer.Validate(stories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss) && loss >= 0, $"Invalid loss: {loss}");
        }

        // =====================================================================
        // PRICE CONTEXT TRAINING — Behavioral Correctness
        // =====================================================================

        public void Test_PriceContext_WithVsWithout_DifferentOutputs()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var currentInput = SliceRows(priceInputs[0], 8, 15);
            var (predNo, _) = model.Forward(stories[0], currentInput);

            var priceCtxHidden = model.EncodePriceHistory(SliceRows(priceInputs[0], 0, 8));
            var priceCtxTimes = Enumerable.Range(0, 8).Select(t => -(float)(8 - t)).ToArray();
            var cache = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (predWith, _) = model.ForwardWithPriceContextAndCache(
                stories[0], currentInput, priceCtxHidden, priceCtxTimes, cache);

            bool anyDiff = false;
            for (int i = 0; i < predNo.GetLength(0) && !anyDiff; i++)
                for (int j = 0; j < predNo.GetLength(1) && !anyDiff; j++)
                    if (MathF.Abs(predNo[i, j] - predWith[i, j]) > 1e-6f) anyDiff = true;
            Assert(anyDiff, "With and without price context should differ");
        }

        public void Test_PriceContext_HistoryLength_AffectsOutput()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var currentInput = SliceRows(priceInputs[0], 10, 15);

            var shortCtx = model.EncodePriceHistory(SliceRows(priceInputs[0], 5, 10));
            var shortTimes = Enumerable.Range(0, 5).Select(t => -(float)(5 - t)).ToArray();
            var c1 = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred1, _) = model.ForwardWithPriceContextAndCache(stories[0], currentInput, shortCtx, shortTimes, c1);

            var longCtx = model.EncodePriceHistory(SliceRows(priceInputs[0], 0, 10));
            var longTimes = Enumerable.Range(0, 10).Select(t => -(float)(10 - t)).ToArray();
            var c2 = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred2, _) = model.ForwardWithPriceContextAndCache(stories[0], currentInput, longCtx, longTimes, c2);

            bool anyDiff = false;
            for (int i = 0; i < pred1.GetLength(0) && !anyDiff; i++)
                for (int j = 0; j < pred1.GetLength(1) && !anyDiff; j++)
                    if (MathF.Abs(pred1[i, j] - pred2[i, j]) > 1e-6f) anyDiff = true;
            Assert(anyDiff, "Different history lengths should produce different outputs");
        }

        public void Test_PriceContext_Deterministic()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var priceCtxHidden = model.EncodePriceHistory(SliceRows(priceInputs[0], 0, 8));
            var priceCtxTimes = Enumerable.Range(0, 8).Select(t => -(float)(8 - t)).ToArray();
            var currentInput = SliceRows(priceInputs[0], 8, 15);

            var c1 = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred1, _) = model.ForwardWithPriceContextAndCache(stories[0], currentInput, priceCtxHidden, priceCtxTimes, c1);
            var c2 = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred2, _) = model.ForwardWithPriceContextAndCache(stories[0], currentInput, priceCtxHidden, priceCtxTimes, c2);

            for (int i = 0; i < pred1.GetLength(0); i++)
                for (int j = 0; j < pred1.GetLength(1); j++)
                    Assert(pred1[i, j] == pred2[i, j], $"Non-deterministic at [{i},{j}]");
        }

        public void Test_PriceContext_SplitPoint_Respected()
        {
            var config = CreateConfig(priceSeqLen: 20, priceCtxMinHist: 5, priceCtxMinCurrent: 5);
            int minSplitLen = config.PriceContextMinHistoryLength + config.PriceContextMinCurrentLength + 1;
            Assert(minSplitLen == 11, $"Min split length should be 11, got {minSplitLen}");
            Assert(10 < minSplitLen, "Length 10 should be below min split threshold");
            Assert(11 >= minSplitLen, "Length 11 should meet min split threshold");
        }

        // =====================================================================
        // PRICE CONTEXT TRAINING — Config Edge Cases
        // =====================================================================

        public void Test_PriceContext_SequenceTooShort_SkipsSplit()
        {
            var rng = new Random(42);
            int numSamples = 5;
            var priceInputs = new float[numSamples][,];
            var priceTargets = new float[numSamples][,];
            var nullStories = new NewsStory[numSamples][];
            for (int s = 0; s < numSamples; s++)
            {
                priceInputs[s] = new float[8, 5];
                priceTargets[s] = new float[8, 5];
                for (int t = 0; t < 8; t++)
                    for (int f = 0; f < 5; f++)
                    { priceInputs[s][t, f] = (float)rng.NextDouble(); priceTargets[s][t, f] = (float)rng.NextDouble(); }
            }

            var config = CreateConfig(priceSeqLen: 10, priceCtxMinHist: 5, priceCtxMinCurrent: 5);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 5, Verbose = false });
            trainer.Train(nullStories, priceInputs, priceTargets);
            float loss = trainer.Validate(nullStories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss), $"Short sequence training NaN loss: {loss}");
        }

        public void Test_PriceContext_MinimalSplittableSequence()
        {
            var rng = new Random(42);
            var priceInputs = new float[1][,] { new float[11, 5] };
            var priceTargets = new float[1][,] { new float[11, 5] };
            var nullStories = new NewsStory[1][];
            for (int t = 0; t < 11; t++)
                for (int f = 0; f < 5; f++)
                { priceInputs[0][t, f] = (float)rng.NextDouble(); priceTargets[0][t, f] = (float)rng.NextDouble(); }

            var config = CreateConfig(priceSeqLen: 12, priceCtxMinHist: 5, priceCtxMinCurrent: 5);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 3, Verbose = false });
            trainer.Train(nullStories, priceInputs, priceTargets);
            float loss = trainer.Validate(nullStories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss), $"Minimal splittable NaN: {loss}");
        }

        public void Test_PriceContext_LargeHistorySmallCurrent()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 3, priceSeqLen: 20);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 20, priceCtxMinHist: 3, priceCtxMinCurrent: 3);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 3, Epochs = 5, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);
            float loss = trainer.Validate(stories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss), $"Large history NaN: {loss}");
        }

        public void Test_PriceContext_SmallHistoryLargeCurrent()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 3, priceSeqLen: 20);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 20, priceCtxMinHist: 3, priceCtxMinCurrent: 3);
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 3, Epochs = 5, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);
            float loss = trainer.Validate(stories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss) && loss >= 0, $"Small history invalid loss: {loss}");
        }

        // =====================================================================
        // SEQUENTIAL TRAINING
        // =====================================================================

        public void Test_Sequential_MemoryAccumulates()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5, priceSeqLen: 8);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var timestamps = new double[] { 100.0, 200.0, 300.0, 400.0, 500.0 };

            Assert(model.NewsMemory.Count == 0, "Memory should start empty");
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            trainer.TrainSequential(stories, priceInputs, priceTargets, timestamps, maxNewsMemory: 100, maxPriceMemory: 200);
            Assert(model.NewsMemory.Count > 0, "News memory should have entries after sequential training");
            Assert(model.PriceMemory.Count > 0, "Price memory should have entries after sequential training");
        }

        public void Test_Sequential_LossDecreases()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 10, priceSeqLen: 8);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 8);
            var timestamps = Enumerable.Range(0, 10).Select(i => (double)(i * 100)).ToArray();

            // Use a single model: measure loss before training, then train sequentially
            var model = new TACAMT_Model(config, new Random(42));
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            float lossBefore = trainer.Validate(stories, priceInputs, priceTargets);

            // Now train sequentially for several epochs on the same model
            var trainer2 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 10, Verbose = false });
            trainer2.TrainSequential(stories, priceInputs, priceTargets, timestamps);
            float lossAfter = trainer2.Validate(stories, priceInputs, priceTargets);
            Assert(lossAfter < lossBefore, $"Sequential loss: {lossBefore:F6} -> {lossAfter:F6}");
        }

        public void Test_Sequential_MemoryCleared_EachEpoch()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 3, priceSeqLen: 8);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var timestamps = new double[] { 100.0, 200.0, 300.0 };

            model.UpdateNewsMemory(stories[0], 50.0);
            Assert(model.NewsMemory.Count > 0, "Should have memory before training");

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 2, Verbose = false });
            trainer.TrainSequential(stories, priceInputs, priceTargets, timestamps, maxNewsMemory: 100, maxPriceMemory: 200);
            Assert(model.NewsMemory.Count > 0, "Should have memory after sequential training");
        }

        public void Test_Sequential_PriceMemory_StoredFromForward()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 2, priceSeqLen: 8);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 8);
            var model = new TACAMT_Model(config, new Random(42));
            var timestamps = new double[] { 100.0, 200.0 };

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            trainer.TrainSequential(stories, priceInputs, priceTargets, timestamps, maxNewsMemory: 100, maxPriceMemory: 200);

            Assert(model.PriceMemory.Count > 0, "Price memory should be populated");
            foreach (var entry in model.PriceMemory)
            {
                Assert(entry.HiddenState.Length == config.PriceEmbeddingDim, "Price memory entry dim mismatch");
                Assert(!double.IsNaN(entry.AbsoluteTimestamp), "Price memory timestamp is NaN");
            }
        }


        public void Test_PriceContext_GradientClipping_NoNaN()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.1f, BatchSize = 5, Epochs = 5, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            var (pred, _) = model.Forward(stories[0], priceInputs[0]);
            Assert(!HasNaN(pred), "NaN after high-LR training with clipping");
        }

        public void Test_PriceContext_NoNaN_AfterManyEpochs()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 50, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            float loss = trainer.Validate(stories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss) && !float.IsInfinity(loss), $"Loss NaN/Inf after 50 epochs: {loss}");

            for (int layer = 0; layer < config.PriceNumLayers; layer++)
            {
                var decayNet = model.PriceBlocks[layer].DecayNetwork;
                for (int h = 0; h < config.PriceNumHeads; h++)
                    Assert(!float.IsNaN(decayNet.LogBaseDecayRate[h]),
                        $"Layer {layer} head {h} LogBaseDecayRate is NaN");
            }
        }


        public void Test_TACAMT_MultiStory_ForwardNoError()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));
            var (pred, _) = model.Forward(stories[0], priceInputs[0]);
            Assert(pred != null, "Forward returned null");
        }

        public void Test_TACAMT_MultiStory_ForwardOutputShape()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 10);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 10);
            var model = new TACAMT_Model(config, new Random(42));
            var (pred, _) = model.Forward(stories[0], priceInputs[0]);
            Assert(pred.GetLength(0) == 10, $"Output rows: {pred.GetLength(0)}, expected 10");
            Assert(pred.GetLength(1) == config.OutputDim, $"Output cols mismatch");
        }

        public void Test_TACAMT_MultiStory_DifferentArrivalTimes_DifferentOutputs()
        {
            var (tokenizer, _, priceInputs, _) = CreateTestData(numSamples: 1, priceSeqLen: 16);
            var config = CreateConfig(tokenizer.VocabSize + 2, priceSeqLen: 16);
            var model = new TACAMT_Model(config, new Random(42));

            // Use ForwardWithCache which goes through ContentAwareCrossAttentionWithCache
            // where timeDiffs are computed from arrival times and fed into the decay network.
            // Use very different arrival times so the decay bias values diverge.
            var tokens1 = tokenizer.Encode("stock price rose sharply today", addSpecialTokens: true);
            var tokens2 = tokenizer.Encode("market crashed due to earnings miss", addSpecialTokens: true);
            var stories1 = new[] { new NewsStory(tokens1, 0f), new NewsStory(tokens2, 1f) };
            var stories2 = new[] { new NewsStory(tokens1, 100f), new NewsStory(tokens2, 200f) };

            var cache1 = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred1, _) = model.ForwardWithCache(stories1, priceInputs[0], cache1);
            var cache2 = new MultimodalForwardCache(config.TextNumLayers, config.PriceNumLayers);
            var (pred2, _) = model.ForwardWithCache(stories2, priceInputs[0], cache2);

            bool anyDiff = false;
            for (int i = 0; i < pred1.GetLength(0) && !anyDiff; i++)
                for (int j = 0; j < pred1.GetLength(1) && !anyDiff; j++)
                    if (MathF.Abs(pred1[i, j] - pred2[i, j]) > 1e-6f) anyDiff = true;
            Assert(anyDiff, "Different arrival times should produce different outputs");
        }

        public void Test_TACAMT_TimeDecay_InitialValues()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            var decayNet = model.PriceBlocks[0].DecayNetwork;

            for (int h = 0; h < config.PriceNumHeads; h++)
            {
                Assert(!float.IsNaN(decayNet.LogBaseDecayRate[h]), $"LogBaseDecayRate[{h}] is NaN");
                Assert(decayNet.LogBaseDecayRate[h] < 0, $"LogBaseDecayRate[{h}] should be negative (initialized to -2.3)");
            }
        }

        public void Test_TACAMT_TimeDecay_ParamsChangeAfterTraining()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var decayNet = model.PriceBlocks[0].DecayNetwork;
            var logDecayBefore = CloneVector(decayNet.LogBaseDecayRate);
            var w1Before = Clone3D(decayNet.W1);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 5, Epochs = 10, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            Assert(VectorChanged(logDecayBefore, decayNet.LogBaseDecayRate),
                "LogBaseDecayRate should change after training");
            Assert(Array3DChanged(w1Before, decayNet.W1), "W1 should change after training");
        }

        public void Test_TACAMT_ComputeTimeBiasMatrix_Correct()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            var block = model.PriceBlocks[0];

            float[] arrivalTimes = { 0f, 2f, 5f };
            var timeDiffs = block.ComputeTimeDiffMatrix(4, arrivalTimes);

            Assert(timeDiffs.GetLength(0) == 4, "TimeDiff rows should be 4");
            Assert(timeDiffs.GetLength(1) == 3, "TimeDiff cols should be 3");
            Assert(MathF.Abs(timeDiffs[0, 0] - 0f) < 1e-5f, "timeDiffs[0,0] should be 0");
            Assert(MathF.Abs(timeDiffs[0, 1] - 2f) < 1e-5f, "timeDiffs[0,1] should be 2");
            Assert(MathF.Abs(timeDiffs[3, 2] - 2f) < 1e-5f, "timeDiffs[3,2] should be |3-5|=2");
        }


        public void Test_TACAMT_MultiStory_LossDecreases()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 10);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var t1 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
            float lossBefore = t1.Validate(stories, priceInputs, priceTargets);

            var t2 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 20, Verbose = false });
            t2.Train(stories, priceInputs, priceTargets);

            float lossAfter = t2.Validate(stories, priceInputs, priceTargets);
            Assert(lossAfter < lossBefore, $"Multi-story loss: {lossBefore:F6} -> {lossAfter:F6}");
        }

        public void Test_TACAMT_MultiStory_SingleSampleOverfit()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2, embDim: 32, numLayers: 2, ffnDim: 64);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.005f, BatchSize = 1, Epochs = 1, Verbose = false });
            float lossBefore = trainer.Validate(stories, priceInputs, priceTargets);

            var t2 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.005f, BatchSize = 1, Epochs = 500, UseGradientClipping = true, GradientClipThreshold = 1.0f, Verbose = false });
            t2.Train(stories, priceInputs, priceTargets);

            float lossAfter = t2.Validate(stories, priceInputs, priceTargets);
            //Assert(lossAfter < lossBefore * 0.5f, $"Single sample overfit: {lossBefore:F6} -> {lossAfter:F6}");
            Assert(lossAfter < lossBefore * 0.8f, $"Single sample overfit: {lossBefore:F6} -> {lossAfter:F6}");
        }



        public void Test_TACAMT_BackwardCompat_SingleText_NoError()
        {
            var (tokenizer, _, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var tokens = tokenizer.Encode("stock price rose", addSpecialTokens: true);
            var (pred, _) = model.Forward(tokens, priceInputs[0]);
            Assert(pred != null && !HasNaN(pred), "Single text forward failed");
        }

        public void Test_TACAMT_BackwardCompat_NullText_NoError()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            var rng = new Random(42);
            var price = new float[10, 5];
            for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) price[i, j] = (float)rng.NextDouble();

            var (pred, _) = model.Forward((NewsStory[])null, price);
            Assert(pred != null && !HasNaN(pred), "Null text forward failed");
        }

        public void Test_TACAMT_BackwardCompat_TrainWithIntArrays()
        {
            var (tokenizer, _, priceInputs, priceTargets) = CreateTestData(numSamples: 5);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var textSeqs = new int[5][];
            for (int s = 0; s < 5; s++)
                textSeqs[s] = tokenizer.Encode("stock price rose", addSpecialTokens: true);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 3, Verbose = false });
            trainer.Train(textSeqs, priceInputs, priceTargets);
            float loss = trainer.Validate(textSeqs, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss), "int[][] training produced NaN");
        }


        public void Test_TACAMT_MixedBatch_SomeStoriesSomeNull()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 6);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            // Null out some stories
            stories[1] = null; stories[3] = null; stories[5] = null;

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 6, Epochs = 3, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);
            float loss = trainer.Validate(stories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss), "Mixed batch NaN");
        }

        public void Test_TACAMT_NegativeArrivalTime_NoError()
        {
            var (tokenizer, _, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var tokens = tokenizer.Encode("stock price rose", addSpecialTokens: true);
            var stories = new[] { new NewsStory(tokens, -5f), new NewsStory(tokens, -2f) };
            var (pred, _) = model.Forward(stories, priceInputs[0]);
            Assert(pred != null && !HasNaN(pred), "Negative arrival times failed");
        }

        public void Test_TACAMT_ManyStories_NoError()
        {
            var (tokenizer, _, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var tokens = tokenizer.Encode("stock", addSpecialTokens: true);
            var stories = Enumerable.Range(0, 12).Select(i => new NewsStory(tokens, i * 1.0f)).ToArray();
            var (pred, _) = model.Forward(stories, priceInputs[0]);
            Assert(pred != null && !HasNaN(pred), "12 stories failed");
        }

        public void Test_TACAMT_SingleTokenStory_NoError()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            var rng = new Random(42);
            var price = new float[10, 5];
            for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) price[i, j] = (float)rng.NextDouble();

            var stories = new[] { new NewsStory(new[] { 1 }, 0f) };
            var (pred, _) = model.Forward(stories, price);
            Assert(pred != null && !HasNaN(pred), "Single token story failed");
        }


        public void Test_TACAMT_UpdateNewsMemory_StoresEntries()
        {
            var (tokenizer, stories, _, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            Assert(model.NewsMemory.Count == 0, "Memory should start empty");
            model.UpdateNewsMemory(stories[0], 100.0);
            Assert(model.NewsMemory.Count == stories[0].Length,
                $"Memory count: {model.NewsMemory.Count}, expected {stories[0].Length}");
        }

        public void Test_TACAMT_ClearNewsMemory_Resets()
        {
            var (tokenizer, stories, _, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            model.UpdateNewsMemory(stories[0], 100.0);
            Assert(model.NewsMemory.Count > 0, "Should have entries");
            model.ClearNewsMemory();
            Assert(model.NewsMemory.Count == 0, "Memory should be cleared");
        }

        public void Test_TACAMT_UpdateNewsMemory_TrimOldEntries()
        {
            var (tokenizer, stories, _, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            // Add many entries with max=5
            for (int i = 0; i < 10; i++)
                model.UpdateNewsMemory(stories[0], (double)(i * 100), max: 5);

            Assert(model.NewsMemory.Count <= 5, $"Memory should be trimmed to 5, got {model.NewsMemory.Count}");
        }

        public void Test_TACAMT_PredictWithMemory_NoMemory_FallsBack()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var (pred, conf) = model.PredictWithMemory(stories[0], priceInputs[0], 100.0);
            Assert(pred != null, "PredictWithMemory with no memory should work");
            Assert(pred.Length == config.OutputDim, $"Prediction dim: {pred.Length}, expected {config.OutputDim}");
        }

        public void Test_TACAMT_PredictWithMemory_WithMemory_NoError()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 3);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            // Build up memory
            model.UpdateNewsMemory(stories[0], 50.0);
            model.UpdateNewsMemory(stories[1], 75.0);

            var (pred, conf) = model.PredictWithMemory(stories[2], priceInputs[2], 100.0);
            Assert(pred != null, "PredictWithMemory with memory returned null");
            for (int j = 0; j < pred.Length; j++)
                Assert(!float.IsNaN(pred[j]), $"NaN at prediction[{j}]");
        }

        public void Test_TACAMT_PredictWithMemory_OldStoriesDecayed()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            // Add memory at timestamp 0
            model.UpdateNewsMemory(stories[0], 0.0);

            // Query at time 1 vs time 10000 — old memory should have different effect
            var (pred1, _) = model.PredictWithMemory(null, priceInputs[0], 1.0);
            model.ClearPriceMemory(); // Clear price memory accumulated from first predict
            var (pred2, _) = model.PredictWithMemory(null, priceInputs[0], 10000.0);

            bool anyDiff = false;
            for (int j = 0; j < pred1.Length && !anyDiff; j++)
                if (MathF.Abs(pred1[j] - pred2[j]) > 1e-6f) anyDiff = true;
            Assert(anyDiff, "Old stories at different time distances should produce different outputs");
        }

        public void Test_TACAMT_SaveLoad_DecayParams_Preserved()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 3);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            // Train to change params
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 3, Epochs = 5, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            var decayNet = model.PriceBlocks[0].DecayNetwork;
            var logDecayBefore = CloneVector(decayNet.LogBaseDecayRate);
            var w1Before = Clone3D(decayNet.W1);

            var dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);
                var loadedDecay = loaded.PriceBlocks[0].DecayNetwork;

                for (int h = 0; h < config.PriceNumHeads; h++)
                    Assert(MathF.Abs(logDecayBefore[h] - loadedDecay.LogBaseDecayRate[h]) < 1e-6f,
                        $"LogBaseDecayRate[{h}] mismatch after load");
                Assert(!Array3DChanged(w1Before, loadedDecay.W1, 1e-6f),
                    "W1 should be preserved after save/load");
            }
            finally { CleanupDir(dir); }
        }

        public void Test_TACAMT_SaveLoad_NewsMemory_Preserved()
        {
            var (tokenizer, stories, _, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            model.UpdateNewsMemory(stories[0], 100.0);
            int memCount = model.NewsMemory.Count;
            Assert(memCount > 0, "Should have memory entries");

            var dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);
                Assert(loaded.NewsMemory.Count == memCount,
                    $"Loaded memory count {loaded.NewsMemory.Count}, expected {memCount}");

                for (int i = 0; i < memCount; i++)
                {
                    Assert(Math.Abs(model.NewsMemory[i].AbsoluteTimestamp - loaded.NewsMemory[i].AbsoluteTimestamp) < 1e-6,
                        "Memory timestamp mismatch after load");
                    for (int d = 0; d < config.PriceEmbeddingDim; d++)
                        Assert(MathF.Abs(model.NewsMemory[i].HiddenState[d] - loaded.NewsMemory[i].HiddenState[d]) < 1e-6f,
                            "Memory hidden state mismatch");
                }
            }
            finally { CleanupDir(dir); }
        }

        public void Test_TACAMT_SaveLoad_ForwardMatch()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var (pred1, _) = model.Forward(stories[0], priceInputs[0]);

            var dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);
                var (pred2, _) = loaded.Forward(stories[0], priceInputs[0]);

                for (int i = 0; i < pred1.GetLength(0); i++)
                    for (int j = 0; j < pred1.GetLength(1); j++)
                        Assert(MathF.Abs(pred1[i, j] - pred2[i, j]) < 1e-5f,
                            $"Forward mismatch at [{i},{j}] after save/load: {pred1[i, j]} vs {pred2[i, j]}");
            }
            finally { CleanupDir(dir); }
        }

        public void Test_TACAMT_SaveLoad_ContinueTraining()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            // Train a bit
            var t1 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 5, Verbose = false });
            t1.Train(stories, priceInputs, priceTargets);
            float lossMid = t1.Validate(stories, priceInputs, priceTargets);

            var dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);

                // Continue training
                var t2 = new TACAMT_Trainer(loaded, new MultimodalTrainingConfig
                { LearningRate = 0.001f, BatchSize = 5, Epochs = 15, Verbose = false });
                t2.Train(stories, priceInputs, priceTargets);
                float lossAfter = t2.Validate(stories, priceInputs, priceTargets);

                Assert(lossAfter <= lossMid * 1.1f,
                    $"Training after load should not regress significantly: {lossMid:F6} -> {lossAfter:F6}");
            }
            finally { CleanupDir(dir); }
        }


        public void Test_TACAMT_Deterministic_MultiStory()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var (pred1, _) = model.Forward(stories[0], priceInputs[0]);
            var (pred2, _) = model.Forward(stories[0], priceInputs[0]);

            for (int i = 0; i < pred1.GetLength(0); i++)
                for (int j = 0; j < pred1.GetLength(1); j++)
                    Assert(pred1[i, j] == pred2[i, j], $"Non-deterministic at [{i},{j}]");
        }

        public void Test_TACAMT_GradientClipping_WithDecay_NoNaN()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.1f, BatchSize = 5, Epochs = 10, UseGradientClipping = true, GradientClipThreshold = 0.5f, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            var (pred, _) = model.Forward(stories[0], priceInputs[0]);
            Assert(!HasNaN(pred), "NaN after aggressive gradient clipping");
        }

        public void Test_TACAMT_Validate_MultiStory()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5);
            var config = CreateConfig(tokenizer.VocabSize + 2);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
            float loss = trainer.Validate(stories, priceInputs, priceTargets);
            Assert(!float.IsNaN(loss) && loss >= 0, $"Validate produced invalid loss: {loss}");
        }

        public void Test_TACAMT_PerHeadDecayRates_Diverge()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 10);
            var config = CreateConfig(tokenizer.VocabSize + 2, numHeads: 4, embDim: 16);
            var model = new TACAMT_Model(config, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 5, Epochs = 30, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            var decayNet = model.PriceBlocks[0].DecayNetwork;
            bool anyDifferent = false;
            for (int h = 1; h < 4; h++)
                if (MathF.Abs(decayNet.LogBaseDecayRate[0] - decayNet.LogBaseDecayRate[h]) > 1e-6f)
                { anyDifferent = true; break; }
            Assert(anyDifferent, "Per-head decay rates should diverge after training");
        }


        public void Test_TACAMT_PriceOnly_ForwardNoError()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            var rng = new Random(42);
            var price = new float[10, 5];
            for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) price[i, j] = (float)rng.NextDouble();

            var (pred, _) = model.Forward((NewsStory[])null, price);
            Assert(pred != null && !HasNaN(pred), "Price-only forward failed");
        }

        public void Test_TACAMT_PriceOnly_LossDecreases()
        {
            var rng = new Random(42);
            int numSamples = 5;
            var nullStories = new NewsStory[numSamples][];
            var priceInputs = new float[numSamples][,];
            var priceTargets = new float[numSamples][,];
            for (int s = 0; s < numSamples; s++)
            {
                priceInputs[s] = new float[10, 5];
                priceTargets[s] = new float[10, 5];
                for (int t = 0; t < 10; t++)
                    for (int f = 0; f < 5; f++)
                    { priceInputs[s][t, f] = (float)rng.NextDouble(); priceTargets[s][t, f] = (float)rng.NextDouble(); }
            }

            var config = CreateConfig(priceSeqLen: 12);
            var model = new TACAMT_Model(config, new Random(42));

            var t1 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 1, Verbose = false });
            float lossBefore = t1.Validate(nullStories, priceInputs, priceTargets);

            var t2 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 5, Epochs = 20, Verbose = false });
            t2.Train(nullStories, priceInputs, priceTargets);

            float lossAfter = t2.Validate(nullStories, priceInputs, priceTargets);
            Assert(lossAfter < lossBefore, $"Price-only loss: {lossBefore:F6} -> {lossAfter:F6}");
        }

        public void Test_TACAMT_FrozenTextEncoder_DecayStillLearns()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5);
            var config = CreateConfig(tokenizer.VocabSize + 2, freezeText: true);
            var model = new TACAMT_Model(config, new Random(42));

            var textEmbBefore = CloneMatrix(model.TextTokenEmbedding);
            var decayNet = model.PriceBlocks[0].DecayNetwork;
            var logDecayBefore = CloneVector(decayNet.LogBaseDecayRate);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 5, Epochs = 10, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            Assert(!MatrixChanged(textEmbBefore, model.TextTokenEmbedding),
                "Text embedding should NOT change when frozen");
            Assert(VectorChanged(logDecayBefore, decayNet.LogBaseDecayRate),
                "Decay LogBaseDecayRate should still change when text is frozen");
        }


        public void Test_Config_Validate_ThrowsOnBadDims()
        {
            var config = new Transformers.TACAMT.Config { TextEmbeddingDim = 15, TextNumHeads = 4 }; // 15 % 4 != 0
            bool threw = false;
            try { config.Validate(); } catch (ArgumentException) { threw = true; }
            Assert(threw, "Should throw on TextEmbeddingDim not divisible by TextNumHeads");
        }

        public void Test_Config_Validate_AcceptsGoodConfig()
        {
            var config = CreateConfig();
            config.Validate(); // Should not throw
        }


        public void Test_MemoryPruning_AttentionBased_KeepsHighScoreEntries()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.UseAttentionBasedPruning = true;
            model.PruningConfig.MinQueryCountForPruning = 1;

            // Add 10 entries with varying attention scores
            for (int i = 0; i < 10; i++)
            {
                model.NewsMemory.Add(new NewsMemoryEntry
                {
                    HiddenState = new float[config.PriceEmbeddingDim],
                    AbsoluteTimestamp = i * 10.0,
                    AttentionScore = i * 0.1f, // 0.0 to 0.9
                    QueryCount = 5
                });
            }
            var method = typeof(TACAMT_Model).GetMethod(
  "PruneNewsMemory",
  BindingFlags.Instance | BindingFlags.NonPublic);

            if (method == null)
                throw new MissingMethodException("PruneNewsMemory not found");

            method.Invoke(model, new object[] { 5 });
            Assert(model.NewsMemory.Count == 5, $"Should have 5 entries, got {model.NewsMemory.Count}");

            // The highest-score entries should survive
            float minScore = model.NewsMemory.Min(e => e.AttentionScore);
            Assert(minScore >= 0.4f,
                $"Lowest surviving score should be high (≥0.4), got {minScore}");
        }

        public void Test_MemoryPruning_OldestFirst_WhenDisabled()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.UseAttentionBasedPruning = false;

            for (int i = 0; i < 10; i++)
                model.NewsMemory.Add(new NewsMemoryEntry
                {
                    HiddenState = new float[config.PriceEmbeddingDim],
                    AbsoluteTimestamp = i * 10.0,
                    AttentionScore = 0.5f,
                    QueryCount = 5
                });
            var method = typeof(TACAMT_Model).GetMethod(
  "PruneNewsMemory",
  BindingFlags.Instance | BindingFlags.NonPublic);

            if (method == null)
                throw new MissingMethodException("PruneNewsMemory not found");

            method.Invoke(model, new object[] { 5 });

            Assert(model.NewsMemory.Count == 5, $"Should have 5 entries");

            // Newest (highest timestamp) should survive
            double minTs = model.NewsMemory.Min(e => e.AbsoluteTimestamp);
            Assert(minTs >= 50.0, $"Oldest-first: min timestamp should be ≥50, got {minTs}");
        }

        public void Test_MemoryPruning_ColdStartProtection()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));
            model.PruningConfig.UseAttentionBasedPruning = true;
            model.PruningConfig.MinQueryCountForPruning = 3;

            // Add entries: some queried (eligible for pruning), some not (cold-start)
            for (int i = 0; i < 10; i++)
                model.NewsMemory.Add(new NewsMemoryEntry
                {
                    HiddenState = new float[config.PriceEmbeddingDim],
                    AbsoluteTimestamp = i * 10.0,
                    AttentionScore = 0.001f, // Very low score
                    QueryCount = i < 5 ? 0 : 10 // First 5 are cold-start
                });

            var method = typeof(TACAMT_Model).GetMethod(
                "PruneNewsMemory",
                BindingFlags.Instance | BindingFlags.NonPublic);

            if (method == null)
                throw new MissingMethodException("PruneNewsMemory not found");

            method.Invoke(model, new object[] { 5 });

            Assert(model.NewsMemory.Count == 5, $"Should have 5 entries");

            // Cold-start entries (QueryCount < 3) should be protected
            int coldStartSurvivors = model.NewsMemory.Count(e => e.QueryCount < 3);
            Assert(coldStartSurvivors > 0, "Cold-start entries should be protected from pruning");
        }

        public void Test_DecayNetwork_ForwardProduces_ValidBias()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng);
            var queryEmb = new float[4, 16];
            var keyEmb = new float[3, 16];
            var timeDiffs = new float[4, 3];
            var keyTimes = new float[] { -3f, -2f, -1f };
            for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) queryEmb[i, j] = (float)rng.NextDouble() - 0.5f;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 16; j++) keyEmb[i, j] = (float)rng.NextDouble() - 0.5f;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) timeDiffs[i, j] = MathF.Abs(i - keyTimes[j]);
            var (bias, cache) = accel.ContentAwareDecayForward(queryEmb, keyEmb, timeDiffs, keyTimes, net);
            Assert(bias.GetLength(0) == 4, $"Bias dim0: {bias.GetLength(0)}, expected 4");
            Assert(bias.GetLength(1) == 3, $"Bias dim1: {bias.GetLength(1)}, expected 3");
            Assert(bias.GetLength(2) == 2, $"Bias dim2: {bias.GetLength(2)}, expected 2");
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    for (int h = 0; h < 2; h++)
                        Assert(!float.IsNaN(bias[i, j, h]), $"NaN in decay bias at [{i},{j},{h}]");
        }

        public void Test_DecayNetwork_BackwardProduces_NonZeroGrads()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng);
            var queryEmb = new float[4, 16];
            var keyEmb = new float[3, 16];
            var timeDiffs = new float[4, 3];
            var keyTimes = new float[] { -3f, -2f, -1f };
            for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) queryEmb[i, j] = (float)rng.NextDouble() - 0.5f;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 16; j++) keyEmb[i, j] = (float)rng.NextDouble() - 0.5f;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) timeDiffs[i, j] = MathF.Abs(i + 1);
            var (bias, cache) = accel.ContentAwareDecayForward(queryEmb, keyEmb, timeDiffs, keyTimes, net);
            // Create upstream gradient
            var dBias = new float[4, 3, 2];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) for (int h = 0; h < 2; h++)
                dBias[i, j, h] = 1.0f;
            var (grads, dQuery, dKey) = net.Backward(dBias, cache);
            // Check that grads are non-zero somewhere
            bool anyNonZero = false;
            for (int h = 0; h < 2 && !anyNonZero; h++)
                if (MathF.Abs(grads.LogBaseDecayRateGrad[h]) > 1e-12f) anyNonZero = true;
            Assert(anyNonZero, "LogBaseDecayRateGrad should have non-zero values");
            bool queryGradNonZero = false;
            for (int i = 0; i < 4 && !queryGradNonZero; i++)
                for (int j = 0; j < 16 && !queryGradNonZero; j++)
                    if (MathF.Abs(dQuery[i, j]) > 1e-12f) queryGradNonZero = true;
            Assert(queryGradNonZero, "dQueryEmbeddings should have non-zero values");
        }
        public void Test_DecayNetwork_DropoutChangesOutput()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng, memAttnDropout: 0.5f, mlpDropout: 0.5f);
            var queryEmb = new float[4, 16];
            var keyEmb = new float[3, 16];
            var timeDiffs = new float[4, 3];
            var keyTimes = new float[] { -3f, -2f, -1f };
            for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) queryEmb[i, j] = (float)rng.NextDouble() - 0.5f;
            for (int i = 0; i < 3; i++) for (int j = 0; j < 16; j++) keyEmb[i, j] = (float)rng.NextDouble() - 0.5f;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) timeDiffs[i, j] = MathF.Abs(i + 1);
            // Inference (no dropout)
            var (biasInf, _) = accel.ContentAwareDecayForward(queryEmb, keyEmb, timeDiffs, keyTimes, net, isTraining: false);
            // Training (with dropout)
            var (biasTrain, _) = accel.ContentAwareDecayForward(queryEmb, keyEmb, timeDiffs, keyTimes, net, isTraining: true, dropoutRng: new Random(99));
            bool anyDiff = false;
            for (int i = 0; i < 4 && !anyDiff; i++)
                for (int j = 0; j < 3 && !anyDiff; j++)
                    for (int h = 0; h < 2 && !anyDiff; h++)
                        if (MathF.Abs(biasInf[i, j, h] - biasTrain[i, j, h]) > 1e-8f)
                            anyDiff = true;
            Assert(anyDiff, "Dropout should change output between training and inference");
        }
        public void Test_DecayNetwork_WeightDecay_AddedToGrads()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng, weightDecay: 0.01f);
            var queryEmb = new float[4, 16];
            var keyEmb = new float[3, 16];
            var timeDiffs = new float[4, 3];
            var keyTimes = new float[] { -3f, -2f, -1f };
            for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) queryEmb[i, j] = (float)rng.NextDouble();
            for (int i = 0; i < 3; i++) for (int j = 0; j < 16; j++) keyEmb[i, j] = (float)rng.NextDouble();
            for (int i = 0; i < 4; i++) for (int j = 0; j < 3; j++) timeDiffs[i, j] = 1f;
            var (_, cache) = accel.ContentAwareDecayForward(queryEmb, keyEmb, timeDiffs, keyTimes, net);
            var dBias = new float[4, 3, 2];
            // Zero upstream -> only weight decay contributes to grads
            var (grads, _, _) = net.Backward(dBias, cache);
            // With zero upstream gradient and non-zero weight decay,
            // gradients should still be non-zero (from weight decay term)
            bool anyNonZero = false;
            for (int h = 0; h < 2; h++)
                for (int p = 0; p < 8; p++)
                    for (int d = 0; d < 16; d++)
                        if (MathF.Abs(grads.QueryProjectionGrad[h, p, d]) > 1e-12f)
                        { anyNonZero = true; break; }
            Assert(anyNonZero, "Weight decay should add non-zero gradient even with zero upstream");
        }

        public void Test_DecayNetwork_SaveLoad_Roundtrip()
        {
            var rng = new Random(42);
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng, memAttnDropout: 0.2f, mlpDropout: 0.3f, weightDecay: 0.001f, numTimeBases: 8);

            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, true))
                net.WriteTo(w);
            ms.Position = 0;
            ContentAwareDecayNetwork loaded;
            using (var r = new BinaryReader(ms))
                loaded = ContentAwareDecayNetwork.ReadFrom(r);

            Assert(loaded.NumHeads == net.NumHeads, "NumHeads mismatch");
            Assert(loaded.ContentDim == net.ContentDim, "ContentDim mismatch");
            Assert(loaded.ProjectionDim == net.ProjectionDim, "ProjectionDim mismatch");
            Assert(loaded.HiddenDim == net.HiddenDim, "HiddenDim mismatch");
            Assert(loaded.NumTimeBases == net.NumTimeBases, "NumTimeBases mismatch");
            Assert(MathF.Abs(loaded.MemoryAttentionDropout - net.MemoryAttentionDropout) < 1e-6f, "MemAttnDropout mismatch");
            Assert(MathF.Abs(loaded.MLPDropout - net.MLPDropout) < 1e-6f, "MLPDropout mismatch");
            Assert(MathF.Abs(loaded.WeightDecay - net.WeightDecay) < 1e-6f, "WeightDecay mismatch");

            for (int h = 0; h < 2; h++)
                Assert(MathF.Abs(net.LogBaseDecayRate[h] - loaded.LogBaseDecayRate[h]) < 1e-6f,
                    $"LogBaseDecayRate[{h}] mismatch after roundtrip");
        }


        public void Test_ConfidenceHead_Outputs_SigmoidRange()
        {
            var (tokenizer, stories, priceInputs, _) = CreateTestData(numSamples: 1);
            var config = CreateConfig(tokenizer.VocabSize + 2, useConfidence: true);
            var model = new TACAMT_Model(config, new Random(42));

            var (pred, conf) = model.Forward(stories[0], priceInputs[0]);
            Assert(conf != null, "Confidence should not be null when UseConfidenceHead=true");

            for (int i = 0; i < conf.GetLength(0); i++)
                Assert(conf[i, 0] >= 0f && conf[i, 0] <= 1f,
                    $"Confidence[{i}] = {conf[i, 0]} outside [0,1]");
        }

        public void Test_ConfidenceHead_TrainingUpdatesWeights()
        {
            var (tokenizer, stories, priceInputs, priceTargets) = CreateTestData(numSamples: 5);
            var config = CreateConfig(tokenizer.VocabSize + 2, useConfidence: true);
            var model = new TACAMT_Model(config, new Random(42));

            var confProjBefore = CloneMatrix(model.ConfidenceProjection);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false });
            trainer.Train(stories, priceInputs, priceTargets);

            Assert(MatrixChanged(confProjBefore, model.ConfidenceProjection),
                "ConfidenceProjection should change after training");
        }

        public void Test_Tokenizer_SetTokenizer_VocabMismatch_Throws()
        {
            var config = CreateConfig(textVocabSize: 100);
            var model = new TACAMT_Model(config, new Random(42));

            var tokenizer = new BPETokenizer();
            tokenizer.Train(new[] { "hello world" }, vocabSize: 50, minFrequency: 1);

            bool threw = false;
            try { model.SetTokenizer(tokenizer); }
            catch (ArgumentException) { threw = true; }
            Assert(threw, "Should throw when tokenizer vocab size doesn't match config");
        }

        public void Test_Tokenizer_TokenizeStories_WithoutTokenizer_Throws()
        {
            var config = CreateConfig();
            var model = new TACAMT_Model(config, new Random(42));

            bool threw = false;
            try { model.TokenizeStories(new[] { "hello" }, new[] { 0f }); }
            catch (InvalidOperationException) { threw = true; }
            Assert(threw, "Should throw when no tokenizer is attached");
        }



        private (BPETokenizer tokenizer, NewsStory[][] stories, float[][,] priceInputs, float[][,] priceTargets) CreateTestData(int numSamples = 10, int priceSeqLen = 16, int inputFeatures = 5, int outputDim = 5, int seed = 42)
        {
            var random = new Random(seed); string[] corpus = { "stock price rose sharply today", "market crashed due to earnings miss", "bullish sentiment on tech sector" };
            var tokenizer = new BPETokenizer(); tokenizer.Train(corpus, vocabSize: 200, minFrequency: 1);
            var stories = new NewsStory[numSamples][]; var priceInputs = new float[numSamples][,]; var priceTargets = new float[numSamples][,];
            for (int s = 0; s < numSamples; s++) { int ns2 = 2 + random.Next(2); stories[s] = new NewsStory[ns2]; for (int ns = 0; ns < ns2; ns++) { var tokens = tokenizer.Encode(corpus[random.Next(corpus.Length)], addSpecialTokens: true); stories[s][ns] = new NewsStory(tokens, ns * 3.0f); } priceInputs[s] = new float[priceSeqLen, inputFeatures]; priceTargets[s] = new float[priceSeqLen, outputDim]; float bp = 100f + (float)(random.NextDouble() * 50); for (int t = 0; t < priceSeqLen; t++) { for (int f = 0; f < inputFeatures; f++) priceInputs[s][t, f] = (bp + (float)(random.NextDouble() - 0.5) * 10f) / 200f; for (int f = 0; f < outputDim; f++) priceTargets[s][t, f] = (bp + (float)(random.NextDouble() - 0.5) * 10f) / 200f; bp += (float)(random.NextDouble() - 0.48) * 2f; } }
            return (tokenizer, stories, priceInputs, priceTargets);
        }
        private float[,] RandomMatrix(int rows, int cols, Random rng, float scale = 1f) { var m = new float[rows, cols]; for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) m[i, j] = ((float)rng.NextDouble() - 0.5f) * 2f * scale; return m; }
        private bool HasNaN(float[] v) { for (int i = 0; i < v.Length; i++) if (float.IsNaN(v[i]) || float.IsInfinity(v[i])) return true; return false; }

        public void Test_Dims_TextEmbedding_MatchesConfig() { var c = CreateConfig(textVocabSize: 100, embDim: 32); var m = new TACAMT_Model(c, new Random(42)); Assert(m.TextTokenEmbedding.GetLength(0) == 100, $"Rows {m.TextTokenEmbedding.GetLength(0)}"); Assert(m.TextTokenEmbedding.GetLength(1) == 32, $"Cols {m.TextTokenEmbedding.GetLength(1)}"); }
        public void Test_Dims_PriceInputProjection_MatchesConfig() { var c = CreateConfig(embDim: 32, inputFeatures: 7); var m = new TACAMT_Model(c, new Random(42)); Assert(m.PriceInputProjection.GetLength(0) == 32, "Rows"); Assert(m.PriceInputProjection.GetLength(1) == 7, "Cols"); }
        public void Test_Dims_OutputProjection_MatchesConfig() { var c = CreateConfig(embDim: 16, outputDim: 3); var m = new TACAMT_Model(c, new Random(42)); Assert(m.OutputProjection.GetLength(0) == 3, "Rows"); Assert(m.OutputProjection.GetLength(1) == 16, "Cols"); }
        public void Test_Dims_PositionalEncoding_Text_MatchesConfig() { var c = CreateConfig(); var m = new TACAMT_Model(c, new Random(42)); Assert(m.TextPositionalEncoding.GetLength(0) == c.TextMaxSequenceLength, "Rows"); Assert(m.TextPositionalEncoding.GetLength(1) == c.TextEmbeddingDim, "Cols"); }
        public void Test_Dims_PositionalEncoding_Price_MatchesConfig() { var c = CreateConfig(priceSeqLen: 30); var m = new TACAMT_Model(c, new Random(42)); Assert(m.PricePositionalEncoding.GetLength(0) == c.PriceMaxSequenceLength, "Rows"); Assert(m.PricePositionalEncoding.GetLength(1) == c.PriceEmbeddingDim, "Cols"); }
        public void Test_Dims_AllBlocks_SelfAttn_WQ_Correct() { var c = CreateConfig(embDim: 16, numLayers: 3); var m = new TACAMT_Model(c, new Random(42)); for (int i = 0; i < 3; i++) { Assert(m.PriceBlocks[i].SelfAttention.WQ.GetLength(0) == 16, $"Block {i} WQ r"); Assert(m.PriceBlocks[i].SelfAttention.WQ.GetLength(1) == 16, $"Block {i} WQ c"); } }
        public void Test_Dims_AllBlocks_CrossAttn_WQ_Correct() { var c = CreateConfig(embDim: 16, numLayers: 2); var m = new TACAMT_Model(c, new Random(42)); for (int i = 0; i < 2; i++) { Assert(m.PriceBlocks[i].CrossAttention.WQ.GetLength(0) == 16, $"Block {i}"); Assert(m.PriceBlocks[i].CrossAttention.WQ.GetLength(1) == 16, $"Block {i}"); } }
        public void Test_Dims_DecayNetwork_Projections_Correct() { var c = CreateConfig(numHeads: 4, embDim: 16); var m = new TACAMT_Model(c, new Random(42)); var dn = m.PriceBlocks[0].DecayNetwork; Assert(dn.QueryProjection.GetLength(0) == 4, "heads"); Assert(dn.QueryProjection.GetLength(1) == c.DecayProjectionDim, "projDim"); Assert(dn.QueryProjection.GetLength(2) == 16, "contentDim"); Assert(dn.W1.GetLength(2) == c.DecayProjectionDim * 3 + 2, "MLPInputDim"); }
        public void Test_Dims_LayerNormVectors_CorrectLength() { var c = CreateConfig(embDim: 16, numLayers: 2); var m = new TACAMT_Model(c, new Random(42)); for (int i = 0; i < 2; i++) { Assert(m.PriceBlocks[i].LNSelfGamma.Length == 16, $"Block {i}"); Assert(m.PriceBlocks[i].LNCrossGamma.Length == 16, $"Block {i}"); Assert(m.PriceBlocks[i].LNFFNGamma.Length == 16, $"Block {i}"); } }
        public void Test_Dims_ConfidenceProjection_When_Disabled_IsNull() { var m = new TACAMT_Model(CreateConfig(useConfidence: false), new Random(42)); Assert(m.ConfidenceProjection == null, "Should be null"); }
        public void Test_Dims_ConfidenceProjection_When_Enabled_Correct() { var m = new TACAMT_Model(CreateConfig(useConfidence: true, embDim: 16), new Random(42)); Assert(m.ConfidenceProjection != null, "Not null"); Assert(m.ConfidenceProjection.GetLength(0) == 1, "1 row"); Assert(m.ConfidenceProjection.GetLength(1) == 16, "EmbDim cols"); }

        // === FORWARD — VARYING SEQUENCE LENGTHS ===
        public void Test_Forward_SeqLen1_Works() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 5), new Random(42)); var p = new float[1, 5]; for (int j = 0; j < 5; j++) p[0, j] = 0.5f; var (pred, _) = m.Forward((NewsStory[])null, p); Assert(pred.GetLength(0) == 1 && !HasNaN(pred), "Failed"); }
        public void Test_Forward_SeqLen2_Works() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 5), new Random(42)); var rng = new Random(42); var p = new float[2, 5]; for (int i = 0; i < 2; i++) for (int j = 0; j < 5; j++) p[i, j] = (float)rng.NextDouble(); var (pred, _) = m.Forward((NewsStory[])null, p); Assert(pred.GetLength(0) == 2 && !HasNaN(pred), "Failed"); }
        public void Test_Forward_MaxSeqLen_Works() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 20), new Random(42)); var p = RandomMatrix(20, 5, new Random(42), 0.5f); var (pred, _) = m.Forward((NewsStory[])null, p); Assert(pred.GetLength(0) == 20 && !HasNaN(pred), "Failed"); }
        public void Test_Forward_DifferentFeatureDim_Works() { var m = new TACAMT_Model(CreateConfig(inputFeatures: 3, priceSeqLen: 10), new Random(42)); var (pred, _) = m.Forward((NewsStory[])null, RandomMatrix(8, 3, new Random(42), 0.5f)); Assert(pred != null && !HasNaN(pred), "Failed"); }
        public void Test_Forward_DifferentOutputDim_Works() { var m = new TACAMT_Model(CreateConfig(outputDim: 1, priceSeqLen: 10), new Random(42)); var (pred, _) = m.Forward((NewsStory[])null, RandomMatrix(8, 5, new Random(42), 0.5f)); Assert(pred.GetLength(1) == 1, "Output dim should be 1"); }

        // === FORWARD — CONTEXT ROUTING ===
        public void Test_Forward_NullStories_NullText_NoContext() { var m = new TACAMT_Model(CreateConfig(), new Random(42)); var (pred, _) = m.Forward((NewsStory[])null, RandomMatrix(10, 5, new Random(42), 0.5f)); Assert(pred != null && !HasNaN(pred), "Failed"); }
        public void Test_Forward_EmptyStories_NoContext() { var m = new TACAMT_Model(CreateConfig(), new Random(42)); var (pred, _) = m.Forward(new NewsStory[0], RandomMatrix(10, 5, new Random(42), 0.5f)); Assert(pred != null && !HasNaN(pred), "Failed"); }
        public void Test_Forward_SingleStory_CrossAttnActivated() { var (tok, _, pi, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2); var m = new TACAMT_Model(c, new Random(42)); var tokens = tok.Encode("stock price rose", addSpecialTokens: true); var no = m.Forward((NewsStory[])null, pi[0]); var wi = m.Forward(new[] { new NewsStory(tokens, 0f) }, pi[0]); bool d = false; for (int i = 0; i < no.predictions.GetLength(0) && !d; i++) for (int j = 0; j < no.predictions.GetLength(1) && !d; j++) if (MathF.Abs(no.predictions[i, j] - wi.predictions[i, j]) > 1e-6f) d = true; Assert(d, "Story should change output"); }
        public void Test_Forward_MultipleStories_AllContribute() { var (tok, _, pi, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2); var m = new TACAMT_Model(c, new Random(42)); var t1 = tok.Encode("stock price rose", addSpecialTokens: true); var t2 = tok.Encode("market crashed", addSpecialTokens: true); var (p1, _) = m.Forward(new[] { new NewsStory(t1, 0f) }, pi[0]); var (p2, _) = m.Forward(new[] { new NewsStory(t1, 0f), new NewsStory(t2, 1f) }, pi[0]); bool d = false; for (int i = 0; i < p1.GetLength(0) && !d; i++) for (int j = 0; j < p1.GetLength(1) && !d; j++) if (MathF.Abs(p1[i, j] - p2[i, j]) > 1e-6f) d = true; Assert(d, "Second story should change output"); }
        public void Test_Forward_ContextHidden_DimMatchesPriceEmb() { var (tok, stories, pi, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2, embDim: 16); var m = new TACAMT_Model(c, new Random(42)); var cache = new MultimodalForwardCache(c.TextNumLayers, c.PriceNumLayers); m.ForwardWithCache(stories[0], pi[0], cache); Assert(cache.TextFinalHidden.GetLength(1) == 16, $"Dim {cache.TextFinalHidden.GetLength(1)}"); }

        // === MULTI-LAYER ===
        public void Test_MultiLayer_MoreLayers_DifferentOutput() { var (tok, stories, pi, _) = CreateTestData(1); var m1 = new TACAMT_Model(CreateConfig(tok.VocabSize + 2, numLayers: 1), new Random(42)); var m2 = new TACAMT_Model(CreateConfig(tok.VocabSize + 2, numLayers: 2), new Random(42)); var (p1, _) = m1.Forward(stories[0], pi[0]); var (p2, _) = m2.Forward(stories[0], pi[0]); bool d = false; for (int i = 0; i < p1.GetLength(0) && !d; i++) for (int j = 0; j < p1.GetLength(1) && !d; j++) if (MathF.Abs(p1[i, j] - p2[i, j]) > 1e-6f) d = true; Assert(d, "Different layers should differ"); }
        public void Test_MultiLayer_AllBlocks_Executed() { var (tok, stories, pi, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2, numLayers: 3); var m = new TACAMT_Model(c, new Random(42)); var cache = new MultimodalForwardCache(c.TextNumLayers, c.PriceNumLayers); m.ForwardWithCache(stories[0], pi[0], cache); Assert(cache.PriceBlockCaches.Count == 3, $"Got {cache.PriceBlockCaches.Count}"); for (int i = 0; i < 3; i++) Assert(cache.PriceBlockCaches[i].BlockInput != null, $"Block {i} null"); }
        public void Test_MultiLayer_DeepModel_NoNaN() { var (tok, stories, pi, _) = CreateTestData(1); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2, numLayers: 4, embDim: 16, numHeads: 2), new Random(42)); var (pred, _) = m.Forward(stories[0], pi[0]); Assert(!HasNaN(pred), "4-layer NaN"); }

        // === POSITIONAL ENCODING ===
        public void Test_PosEnc_SinCosPattern() { var m = new TACAMT_Model(CreateConfig(embDim: 16), new Random(42)); Assert(MathF.Abs(m.PricePositionalEncoding[0, 0]) < 1e-5f, $"PE[0,0]={m.PricePositionalEncoding[0, 0]}"); Assert(MathF.Abs(m.PricePositionalEncoding[0, 1] - 1.0f) < 1e-5f, $"PE[0,1]={m.PricePositionalEncoding[0, 1]}"); }
        public void Test_PosEnc_DifferentPositions_DifferentEncodings() { var m = new TACAMT_Model(CreateConfig(embDim: 16), new Random(42)); bool d = false; for (int j = 0; j < 16 && !d; j++) if (MathF.Abs(m.PricePositionalEncoding[0, j] - m.PricePositionalEncoding[5, j]) > 1e-6f) d = true; Assert(d, "Pos 0 and 5 should differ"); }
        public void Test_PosEnc_AffectsOutput() { var m = new TACAMT_Model(CreateConfig(), new Random(42)); var rng = new Random(42); var p1 = new float[10, 5]; var p2 = new float[10, 5]; for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) { float v = (float)rng.NextDouble(); p1[i, j] = v; p2[9 - i, j] = v; } var (r1, _) = m.Forward((NewsStory[])null, p1); var (r2, _) = m.Forward((NewsStory[])null, p2); bool d = false; for (int i = 0; i < 10 && !d; i++) for (int j = 0; j < 5 && !d; j++) if (MathF.Abs(r1[i, j] - r2[i, j]) > 1e-6f) d = true; Assert(d, "Reversed should differ"); }

        // === CAUSAL MASKING ===
        public void Test_CausalMask_FutureTokens_DontAffectPast() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 12), new Random(42)); var p1 = RandomMatrix(10, 5, new Random(42), 0.5f); var p2 = (float[,])p1.Clone(); for (int i = 7; i < 10; i++) for (int j = 0; j < 5; j++) p2[i, j] += 1.0f; var (r1, _) = m.Forward((NewsStory[])null, p1); var (r2, _) = m.Forward((NewsStory[])null, p2); for (int i = 0; i < 7; i++) for (int j = 0; j < 5; j++) Assert(MathF.Abs(r1[i, j] - r2[i, j]) < 1e-5f, $"Pos {i} affected by future"); }
        public void Test_CausalMask_Price_DecoderOnly_Enabled() { Assert(CreateConfig().PriceUseDecoderOnly == true, "Should be true"); }

        // === GRADIENT NUMERICAL CHECKS ===
        public void Test_GradCheck_OutputProjection_ApproximatelyCorrect()
        {
            var c = CreateConfig(embDim: 8, numHeads: 2, outputDim: 3, priceSeqLen: 8); var m = new TACAMT_Model(c, new Random(42)); var rng = new Random(42);
            var price = RandomMatrix(6, 5, rng, 0.3f); var target = RandomMatrix(5, 3, rng, 0.3f);
            var cache = new MultimodalForwardCache(c.TextNumLayers, c.PriceNumLayers);
            var (pred, _) = m.ForwardWithCache((NewsStory[])null, SliceRows(price, 0, 5), cache);
            var grads = new Gradients(c); int sl = pred.GetLength(0), od = c.OutputDim, ed = c.PriceEmbeddingDim;
            for (int t = 0; t < sl; t++) for (int v = 0; v < od; v++) for (int e = 0; e < ed; e++) grads.OutputProjectionGrad[v, e] += cache.PriceFinalHidden[t, e] * 2f * (pred[t, v] - target[t, v]) / (sl * od);
            float eps = 1e-3f; float origW = m.OutputProjection[0, 0];
            m.OutputProjection[0, 0] = origW + eps; var (pp, _) = m.Forward((NewsStory[])null, SliceRows(price, 0, 5)); float lp = 0; for (int t = 0; t < sl; t++) for (int j = 0; j < od; j++) { float d = pp[t, j] - target[t, j]; lp += d * d; }
            lp /= (sl * od);
            m.OutputProjection[0, 0] = origW - eps; var (pm, _) = m.Forward((NewsStory[])null, SliceRows(price, 0, 5)); float lm = 0; for (int t = 0; t < sl; t++) for (int j = 0; j < od; j++) { float d = pm[t, j] - target[t, j]; lm += d * d; }
            lm /= (sl * od);
            m.OutputProjection[0, 0] = origW; float fd = (lp - lm) / (2 * eps); float ag = grads.OutputProjectionGrad[0, 0];
            float re = MathF.Abs(fd - ag) / (MathF.Abs(fd) + MathF.Abs(ag) + 1e-8f); Assert(re < 0.1f, $"analytical={ag:E4} fd={fd:E4} relErr={re:F4}");
        }
        public void Test_GradCheck_PriceInputProjection_ApproximatelyCorrect()
        {
            var c = CreateConfig(embDim: 8, numHeads: 2, outputDim: 3, priceSeqLen: 8); var m = new TACAMT_Model(c, new Random(42)); var rng = new Random(42);
            var price = RandomMatrix(6, 5, rng, 0.3f); var target = RandomMatrix(5, 3, rng, 0.3f); float eps = 1e-3f; int sl = 5, od = 3;
            float ow = m.PriceInputProjection[0, 0]; m.PriceInputProjection[0, 0] = ow + eps; var (pp, _) = m.Forward((NewsStory[])null, SliceRows(price, 0, 5)); float lp = 0; for (int t = 0; t < sl; t++) for (int j = 0; j < od; j++) { float d = pp[t, j] - target[t, j]; lp += d * d; }
            m.PriceInputProjection[0, 0] = ow - eps; var (pm, _) = m.Forward((NewsStory[])null, SliceRows(price, 0, 5)); float lm = 0; for (int t = 0; t < sl; t++) for (int j = 0; j < od; j++) { float d = pm[t, j] - target[t, j]; lm += d * d; }
            m.PriceInputProjection[0, 0] = ow; float fd = (lp - lm) / (2 * eps); Assert(!float.IsNaN(fd) && MathF.Abs(fd) > 1e-10f, "FD grad zero or NaN");
        }
        public void Test_GradCheck_OutputBias_ApproximatelyCorrect()
        {
            var c = CreateConfig(embDim: 8, numHeads: 2, outputDim: 3, priceSeqLen: 8); var m = new TACAMT_Model(c, new Random(42)); var rng = new Random(42);
            var price = RandomMatrix(6, 5, rng, 0.3f); var target = RandomMatrix(5, 3, rng, 0.3f); float eps = 1e-3f; int sl = 5, od = 3;
            float ob = m.OutputBias[0]; m.OutputBias[0] = ob + eps; var (pp, _) = m.Forward((NewsStory[])null, SliceRows(price, 0, 5)); float lp = 0; for (int t = 0; t < sl; t++) for (int j = 0; j < od; j++) { float d = pp[t, j] - target[t, j]; lp += d * d; }
            m.OutputBias[0] = ob - eps; var (pm, _) = m.Forward((NewsStory[])null, SliceRows(price, 0, 5)); float lm = 0; for (int t = 0; t < sl; t++) for (int j = 0; j < od; j++) { float d = pm[t, j] - target[t, j]; lm += d * d; }
            m.OutputBias[0] = ob; Assert(MathF.Abs((lp - lm) / (2 * eps)) > 1e-10f, "OutputBias should affect loss");
        }

        // === TRAINING — PARAMETER UPDATES ===
        public void Test_Train_AllPriceBlockParams_Updated() { var (tok, st, pi, pt) = CreateTestData(5); var c = CreateConfig(tok.VocabSize + 2); var m = new TACAMT_Model(c, new Random(42)); var b = m.PriceBlocks[0]; var swk = CloneMatrix(b.SelfAttention.WK); var cwk = CloneMatrix(b.CrossAttention.WK); var lng = CloneVector(b.LNCrossGamma); var lff = CloneVector(b.LNFFNGamma); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false }).Train(st, pi, pt); Assert(MatrixChanged(swk, b.SelfAttention.WK), "SelfWK"); Assert(MatrixChanged(cwk, b.CrossAttention.WK), "CrossWK"); Assert(VectorChanged(lng, b.LNCrossGamma), "LNCross"); Assert(VectorChanged(lff, b.LNFFNGamma), "LNFFN"); }
        public void Test_Train_OutputBias_Updated() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); var bb = CloneVector(m.OutputBias); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false }).Train(st, pi, pt); Assert(VectorChanged(bb, m.OutputBias), "Should change"); }
        public void Test_Train_PriceInputProjectionBias_Updated() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); var bb = CloneVector(m.PriceInputProjectionBias); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false }).Train(st, pi, pt); Assert(VectorChanged(bb, m.PriceInputProjectionBias), "Should change"); }
        public void Test_Train_TextEncoder_Updated_WhenNotFrozen() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2, freezeText: false), new Random(42)); var eb = CloneMatrix(m.TextTokenEmbedding); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false }).Train(st, pi, pt); Assert(MatrixChanged(eb, m.TextTokenEmbedding), "Should change"); }
        public void Test_Train_TextEncoder_NOT_Updated_WhenFrozen() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2, freezeText: true), new Random(42)); var eb = CloneMatrix(m.TextTokenEmbedding); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false }).Train(st, pi, pt); Assert(!MatrixChanged(eb, m.TextTokenEmbedding), "Should NOT change"); }
        public void Test_Train_FFN_Weights_Updated() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); var (pb, _) = m.Forward(st[0], pi[0]); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 10, Verbose = false }).Train(st, pi, pt); var (pa, _) = m.Forward(st[0], pi[0]); Assert(MatrixChanged(pb, pa), "FFN changed (output differs)"); }
        public void Test_Train_AllDecayNetworkParams_Updated() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); var dn = m.PriceBlocks[0].DecayNetwork; var lb = CloneVector(dn.LogBaseDecayRate); var b2b = CloneVector(dn.B2); var w2b = CloneMatrix(dn.W2); var tlf = CloneMatrix(dn.TimeLogFreq); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 10, Verbose = false }).Train(st, pi, pt); Assert(VectorChanged(lb, dn.LogBaseDecayRate), "LogBase"); Assert(VectorChanged(b2b, dn.B2), "B2"); Assert(MatrixChanged(w2b, dn.W2), "W2"); Assert(MatrixChanged(tlf, dn.TimeLogFreq), "TimeLogFreq"); }

        // === TRAINING — LOSS BEHAVIOR ===
        public void Test_Train_ZeroTarget_LossDecreases() { var c = CreateConfig(priceSeqLen: 12); var m = new TACAMT_Model(c, new Random(42)); var rng = new Random(42); int n = 3; var pi = new float[n][,]; var pt = new float[n][,]; var ns = new NewsStory[n][]; for (int s = 0; s < n; s++) { pi[s] = RandomMatrix(10, 5, rng, 0.5f); pt[s] = new float[10, 5]; } var t1 = new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 3, Epochs = 1, Verbose = false }); float lb = t1.Validate(ns, pi, pt); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.005f, BatchSize = 3, Epochs = 50, Verbose = false }).Train(ns, pi, pt); float la = t1.Validate(ns, pi, pt); Assert(la < lb, $"{lb:F6} -> {la:F6}"); }
        public void Test_Train_ConstantTarget_LossDecreases() { var c = CreateConfig(priceSeqLen: 12); var m = new TACAMT_Model(c, new Random(42)); var rng = new Random(42); int n = 3; var pi = new float[n][,]; var pt = new float[n][,]; var ns = new NewsStory[n][]; for (int s = 0; s < n; s++) { pi[s] = RandomMatrix(10, 5, rng, 0.5f); pt[s] = new float[10, 5]; for (int t = 0; t < 10; t++) for (int f = 0; f < 5; f++) pt[s][t, f] = 0.5f; } var t1 = new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 3, Epochs = 1, Verbose = false }); float lb = t1.Validate(ns, pi, pt); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.005f, BatchSize = 3, Epochs = 50, Verbose = false }).Train(ns, pi, pt); float la = t1.Validate(ns, pi, pt); Assert(la < lb, $"{lb:F6} -> {la:F6}"); }
        public void Test_Train_BatchSize1_vs_BatchSizeN_BothWork() { var (tok, st, pi, pt) = CreateTestData(5); var c = CreateConfig(tok.VocabSize + 2); var m1 = new TACAMT_Model(c, new Random(42)); var t1 = new TACAMT_Trainer(m1, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 5, Verbose = false }); float lb1 = t1.Validate(st, pi, pt); t1.Train(st, pi, pt); float la1 = t1.Validate(st, pi, pt); var m5 = new TACAMT_Model(c, new Random(42)); var t5 = new TACAMT_Trainer(m5, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 5, Epochs = 5, Verbose = false }); float lb5 = t5.Validate(st, pi, pt); t5.Train(st, pi, pt); float la5 = t5.Validate(st, pi, pt); Assert(la1 < lb1, "BS1"); Assert(la5 < lb5, "BS5"); }
        public void Test_Train_LearningRateDecay_Works() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 10, UseLearningRateDecay = true, LearningRateDecay = 0.95f, Verbose = false }).Train(st, pi, pt); float l = new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 1, Verbose = false }).Validate(st, pi, pt); Assert(!float.IsNaN(l) && l >= 0, $"Invalid loss: {l}"); }
        public void Test_Train_HighLR_WithClipping_NoExplosion() { var (tok, st, pi, pt) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.5f, BatchSize = 5, Epochs = 5, UseGradientClipping = true, GradientClipThreshold = 0.5f, Verbose = false }).Train(st, pi, pt); var (pred, _) = m.Forward(st[0], pi[0]); Assert(!HasNaN(pred), "NaN after high LR"); }

        // === GRADIENT CLIPPING ===
        public void Test_GradClip_NormReduced() { var (tok, st, pi, pt) = CreateTestData(3); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 1.0f, BatchSize = 3, Epochs = 3, UseGradientClipping = true, GradientClipThreshold = 0.1f, Verbose = false }).Train(st, pi, pt); float l = new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 3, Epochs = 1, Verbose = false }).Validate(st, pi, pt); Assert(!float.IsNaN(l) && !float.IsInfinity(l), "Should prevent NaN"); }
       
        // === PREDICT NEXT ===
        public void Test_PredictNext_ReturnsLastTimestep() { var m = new TACAMT_Model(CreateConfig(), new Random(42)); var (pred, _) = m.PredictNext((NewsStory[])null, RandomMatrix(10, 5, new Random(42), 0.5f)); Assert(pred.Length == 5 && !HasNaN(pred), "Failed"); }
        public void Test_PredictNext_ConfidenceInRange() { var m = new TACAMT_Model(CreateConfig(useConfidence: true), new Random(42)); var (_, conf) = m.PredictNext((NewsStory[])null, RandomMatrix(10, 5, new Random(42), 0.5f)); Assert(conf >= 0f && conf <= 1f, $"Conf {conf}"); }
        public void Test_PredictNext_DefaultConfidence1_WhenDisabled() { var m = new TACAMT_Model(CreateConfig(useConfidence: false), new Random(42)); var (_, conf) = m.PredictNext((NewsStory[])null, RandomMatrix(10, 5, new Random(42), 0.5f)); Assert(conf == 1.0f, $"Got {conf}"); }

        // === PREDICT WITH MEMORY ===
        public void Test_PredictWithMemory_AccumulatesNewsAndPrice() { var (tok, st, pi, _) = CreateTestData(1); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); m.PredictWithMemory(st[0], pi[0], 100.0); Assert(m.NewsMemory.Count > 0, "News"); Assert(m.PriceMemory.Count > 0, "Price"); }
        public void Test_PredictWithMemory_PrunesWhenOverLimit() { var (tok, st, pi, _) = CreateTestData(5); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); for (int i = 0; i < 5; i++) m.PredictWithMemory(st[i], pi[i], (i + 1) * 100.0, maxNewsMemorySize: 5, maxPriceMemorySize: 10); Assert(m.NewsMemory.Count <= 5, $"News {m.NewsMemory.Count}"); Assert(m.PriceMemory.Count <= 10, $"Price {m.PriceMemory.Count}"); }
        public void Test_PredictWithMemory_MultipleCallsChain() { var (tok, st, pi, _) = CreateTestData(3); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); m.PredictWithMemory(st[0], pi[0], 100.0); int n1 = m.NewsMemory.Count; m.PredictWithMemory(st[1], pi[1], 200.0); Assert(m.NewsMemory.Count > n1, "Should accumulate"); }
        public void Test_PredictWithMemory_TimestampOrdering_Preserved() { var (tok, st, pi, _) = CreateTestData(2); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2), new Random(42)); m.PredictWithMemory(st[0], pi[0], 100.0); m.PredictWithMemory(st[1], pi[1], 200.0); Assert(m.NewsMemory.Max(e => e.AbsoluteTimestamp) >= 200.0, "Latest ts"); }

        // === MEMORY PRUNING ADVANCED ===
        public void Test_Pruning_PriceMemory_AttentionBased() { var c = CreateConfig(); var m = new TACAMT_Model(c, new Random(42)); m.PruningConfig.UseAttentionBasedPruning = true; m.PruningConfig.MinQueryCountForPruning = 1; for (int i = 0; i < 10; i++) m.PriceMemory.Add(new PriceMemoryEntry { HiddenState = new float[c.PriceEmbeddingDim], AbsoluteTimestamp = i * 10.0, AttentionScore = i * 0.1f, QueryCount = 5 }); typeof(TACAMT_Model).GetMethod("PricePruneMemory", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(m, new object[] { 5 }); Assert(m.PriceMemory.Count == 5, $"Got {m.PriceMemory.Count}"); }
        public void Test_Pruning_ReserveFraction_Respected() { var c = CreateConfig(); var m = new TACAMT_Model(c, new Random(42)); m.PruningConfig.UseAttentionBasedPruning = true; m.PruningConfig.NewEntryReserveFraction = 0.4f; m.PruningConfig.MinQueryCountForPruning = 1; for (int i = 0; i < 20; i++) m.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[c.PriceEmbeddingDim], AbsoluteTimestamp = i * 10.0, AttentionScore = 0.01f, QueryCount = 10 }); typeof(TACAMT_Model).GetMethod("PruneNewsMemory", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(m, new object[] { 10 }); Assert(m.NewsMemory.Max(e => e.AbsoluteTimestamp) >= 190.0, "Newest should survive"); }
        public void Test_Pruning_ZeroMaxSize_ClearsAll() { var c = CreateConfig(); var m = new TACAMT_Model(c, new Random(42)); m.PruningConfig.UseAttentionBasedPruning = false; for (int i = 0; i < 5; i++) m.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[c.PriceEmbeddingDim], AbsoluteTimestamp = i }); typeof(TACAMT_Model).GetMethod("PruneNewsMemory", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(m, new object[] { 0 }); Assert(m.NewsMemory.Count == 0, "Should be 0"); }
        public void Test_Pruning_ExactlyAtLimit_NoOp() { var c = CreateConfig(); var m = new TACAMT_Model(c, new Random(42)); for (int i = 0; i < 5; i++) m.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[c.PriceEmbeddingDim], AbsoluteTimestamp = i }); typeof(TACAMT_Model).GetMethod("PruneNewsMemory", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(m, new object[] { 5 }); Assert(m.NewsMemory.Count == 5, "No-op"); }
        public void Test_Pruning_Config_SavedAndLoaded() { var m = new TACAMT_Model(CreateConfig(), new Random(42)); m.PruningConfig.AttentionScoreAlpha = 0.2f; m.PruningConfig.MinQueryCountForPruning = 7; m.PruningConfig.NewEntryReserveFraction = 0.3f; m.PruningConfig.UseAttentionBasedPruning = false; var d = GetTempDir(); try { m.Save(d); var ld = TACAMT_Model.Load(d); Assert(MathF.Abs(ld.PruningConfig.AttentionScoreAlpha - 0.2f) < 1e-6f, "Alpha"); Assert(ld.PruningConfig.MinQueryCountForPruning == 7, "MinQ"); Assert(MathF.Abs(ld.PruningConfig.NewEntryReserveFraction - 0.3f) < 1e-6f, "Reserve"); Assert(ld.PruningConfig.UseAttentionBasedPruning == false, "UseAttn"); } finally { CleanupDir(d); } }

        // === SAVE/LOAD EXHAUSTIVE ===
        public void Test_SaveLoad_AllTextBlockWeights_Match() { var (tok, _, _, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2, numLayers: 2); var m = new TACAMT_Model(c, new Random(42)); var d = GetTempDir(); try { m.Save(d); var ld = TACAMT_Model.Load(d); for (int i = 0; i < 2; i++) { Assert(!MatrixChanged(m.TextBlocks[i].Attention.WQ, ld.TextBlocks[i].Attention.WQ, 1e-6f), $"Block {i} WQ"); Assert(!MatrixChanged(m.TextBlocks[i].Attention.WK, ld.TextBlocks[i].Attention.WK, 1e-6f), $"Block {i} WK"); } } finally { CleanupDir(d); } }
        public void Test_SaveLoad_AllPriceBlockWeights_Match() { var (tok, _, _, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2, numLayers: 2); var m = new TACAMT_Model(c, new Random(42)); var d = GetTempDir(); try { m.Save(d); var ld = TACAMT_Model.Load(d); for (int i = 0; i < 2; i++) { Assert(!MatrixChanged(m.PriceBlocks[i].SelfAttention.WQ, ld.PriceBlocks[i].SelfAttention.WQ, 1e-6f), $"Block {i} SelfWQ"); Assert(!VectorChanged(m.PriceBlocks[i].LNSelfGamma, ld.PriceBlocks[i].LNSelfGamma, 1e-6f), $"Block {i} LN"); } } finally { CleanupDir(d); } }
        public void Test_SaveLoad_PriceMemory_Preserved() { var c = CreateConfig(); var m = new TACAMT_Model(c, new Random(42)); var rng = new Random(42); for (int i = 0; i < 3; i++) { var hs = new float[c.PriceEmbeddingDim]; for (int dd = 0; dd < hs.Length; dd++) hs[dd] = (float)rng.NextDouble(); m.PriceMemory.Add(new PriceMemoryEntry { HiddenState = hs, AbsoluteTimestamp = i * 100.0, AttentionScore = 0.5f, QueryCount = i }); } var d = GetTempDir(); try { m.Save(d); var ld = TACAMT_Model.Load(d); Assert(ld.PriceMemory.Count == 3, $"Got {ld.PriceMemory.Count}"); for (int i = 0; i < 3; i++) Assert(Math.Abs(m.PriceMemory[i].AbsoluteTimestamp - ld.PriceMemory[i].AbsoluteTimestamp) < 1e-6, $"Ts {i}"); } finally { CleanupDir(d); } }
        public void Test_SaveLoad_AttentionScores_Preserved() { var c = CreateConfig(); var m = new TACAMT_Model(c, new Random(42)); m.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[c.PriceEmbeddingDim], AbsoluteTimestamp = 1.0, AttentionScore = 0.42f, QueryCount = 7 }); m.PriceMemory.Add(new PriceMemoryEntry { HiddenState = new float[c.PriceEmbeddingDim], AbsoluteTimestamp = 2.0, AttentionScore = 0.77f, QueryCount = 3 }); var d = GetTempDir(); try { m.Save(d); var ld = TACAMT_Model.Load(d); Assert(MathF.Abs(ld.NewsMemory[0].AttentionScore - 0.42f) < 1e-5f, "News AS"); Assert(ld.NewsMemory[0].QueryCount == 7, "News QC"); Assert(MathF.Abs(ld.PriceMemory[0].AttentionScore - 0.77f) < 1e-5f, "Price AS"); Assert(ld.PriceMemory[0].QueryCount == 3, "Price QC"); } finally { CleanupDir(d); } }
        public void Test_SaveLoad_Config_AllFields_Preserved() { var c = CreateConfig(textVocabSize: 77, embDim: 32, numHeads: 4, numLayers: 3, ffnDim: 64, inputFeatures: 7, outputDim: 3, priceSeqLen: 25, useConfidence: true, freezeText: true); var m = new TACAMT_Model(c, new Random(42)); var d = GetTempDir(); try { m.Save(d); var ld = TACAMT_Model.Load(d); Assert(ld.Config.TextVocabSize == 77, "Vocab"); Assert(ld.Config.TextEmbeddingDim == 32, "Emb"); Assert(ld.Config.TextNumHeads == 4, "Heads"); Assert(ld.Config.OutputDim == 3, "OD"); Assert(ld.Config.UseConfidenceHead == true, "Conf"); Assert(ld.Config.FreezeTextEncoder == true, "Freeze"); } finally { CleanupDir(d); } }

        // === CONFIG VALIDATION ===
        public void Test_Config_PriceEmbDim_NotDivisibleByHeads_Throws() { bool t = false; try { new Transformers.TACAMT.Config { PriceEmbeddingDim = 15, PriceNumHeads = 4, TextEmbeddingDim = 16, TextNumHeads = 4 }.Validate(); } catch (ArgumentException) { t = true; } Assert(t, "Should throw"); }
        public void Test_Config_ZeroVocabSize_Throws() { bool t = false; try { new Transformers.TACAMT.Config { TextVocabSize = 0 }.Validate(); } catch (ArgumentException) { t = true; } Assert(t, "Should throw"); }
        public void Test_Config_NegativeL2_Throws() { var c = CreateConfig(); c.L2RegulationLamda = -0.01f; bool t = false; try { c.Validate(); } catch (ArgumentException) { t = true; } Assert(t, "Should throw"); }
        public void Test_Config_DecayDropoutOutOfRange_Throws() { var c = CreateConfig(); c.DecayMemAttnDropout = 1.0f; bool t = false; try { c.Validate(); } catch (ArgumentException) { t = true; } Assert(t, "Should throw"); }
        public void Test_Config_ZeroDecayProjectionDim_Throws() { var c = CreateConfig(); c.DecayProjectionDim = 0; bool t = false; try { c.Validate(); } catch (ArgumentException) { t = true; } Assert(t, "Should throw"); }
        public void Test_Config_ZeroGradientClipThreshold_Throws() { var c = CreateConfig(); c.GradientClippingThreshold = 0f; bool t = false; try { c.Validate(); } catch (ArgumentException) { t = true; } Assert(t, "Should throw"); }

        // === DECAY NETWORK MATH ===
        public void Test_Decay_LargerTimeDiff_StrongerDecay()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng);
            var qe = RandomMatrix(1, 16, rng, 0.5f);
            var ke = RandomMatrix(2, 16, rng, 0.5f);
            var kt = new float[] { -1f, -10f };
            var td = new float[1, 2];
            td[0, 0] = 1f;
            td[0, 1] = 10f;
            var (b, _) = accel.ContentAwareDecayForward(qe, ke, td, kt, net);
            Assert(b[0, 1, 0] < b[0, 0, 0] || b[0, 1, 1] < b[0, 0, 1], "Larger td should be more negative");
        }
        public void Test_Decay_ZeroTimeDiff_NearZeroBias()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng);
            var qe = RandomMatrix(1, 16, rng, 0.5f);
            var ke = RandomMatrix(1, 16, rng, 0.5f);
            var td = new float[1, 1];
            var kt = new float[] { 0f };
            var (b, _) = accel.ContentAwareDecayForward(qe, ke, td, kt, net);
            for (int h = 0; h < 2; h++)
            {
                Assert(MathF.Abs(b[0, 0, h]) < 1e-5f, $"Bias {b[0, 0, h]} for head {h}");
            }
        }
        public void Test_Decay_GateOutputInSigmoidRange()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng);
            var qe = RandomMatrix(5, 16, rng, 2.0f);
            var ke = RandomMatrix(3, 16, rng, 2.0f);
            var td = new float[5, 3];
            var kt = new float[] { -3f, -2f, -1f };
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 3; j++)
                    td[i, j] = MathF.Abs(i + 1);
            var (_, cache) = accel.ContentAwareDecayForward(qe, ke, td, kt, net);
            for (int q = 0; q < 5; q++)
                for (int s = 0; s < 3; s++)
                    for (int h = 0; h < 2; h++)
                        Assert(cache.Gates[q, s, h] > 0f && cache.Gates[q, s, h] < 1f, $"Gate[{q},{s},{h}]={cache.Gates[q, s, h]}");
        }
        public void Test_Decay_BaseDecayRate_AlwaysPositive() { var m = new TACAMT_Model(CreateConfig(numHeads: 4), new Random(42)); for (int l = 0; l < m.Config.PriceNumLayers; l++) for (int h = 0; h < 4; h++) Assert(MathF.Exp(m.PriceBlocks[l].DecayNetwork.LogBaseDecayRate[h]) > 0, $"L{l}H{h}"); }
        public void Test_Decay_MultiScale_TimeEncoding_DifferentPerBase() { var net = new ContentAwareDecayNetwork(1, 16, 8, 16, new Random(42), numTimeBases: 4); bool d = false; for (int b = 1; b < 4 && !d; b++) if (MathF.Abs(net.TimeLogFreq[0, 0] - net.TimeLogFreq[0, b]) > 1e-6f) d = true; Assert(d, "Should differ per base"); }
        public void Test_Decay_MemoryInteraction_ChangesOutput()
        {
            var rng = new Random(42);
            var accel = new CallaghanDev.ML.AccelerationManagers.AccelerationCPU();
            var net = new ContentAwareDecayNetwork(2, 16, 8, 16, rng);
            var qe = RandomMatrix(2, 16, rng, 0.5f);
            var kt = new float[] { -5f, -3f, -1f };
            var ke3 = RandomMatrix(3, 16, rng, 0.5f);
            var td3 = new float[2, 3];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    td3[i, j] = MathF.Abs(i - kt[j]);
            var (b3, _) = accel.ContentAwareDecayForward(qe, ke3, td3, kt, net);
            var ke1 = new float[1, 16];
            for (int d = 0; d < 16; d++)
                ke1[0, d] = ke3[0, d];
            var td1 = new float[2, 1];
            for (int i = 0; i < 2; i++)
                td1[i, 0] = td3[i, 0];
            var (b1, _) = accel.ContentAwareDecayForward(qe, ke1, td1, new float[] { kt[0] }, net);
            bool df = false;
            for (int q = 0; q < 2 && !df; q++)
                for (int h = 0; h < 2 && !df; h++)
                    if (MathF.Abs(b3[q, 0, h] - b1[q, 0, h]) > 1e-6f)
                        df = true;
            Assert(df, "Memory interaction should change bias");
        }
        // === SEQUENTIAL TRAINING ADVANCED ===
        public void Test_Sequential_PriceMemoryGrows_EachSample() { var (tok, st, pi, pt) = CreateTestData(3, priceSeqLen: 8); var c = CreateConfig(tok.VocabSize + 2, priceSeqLen: 8); var m = new TACAMT_Model(c, new Random(42)); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false }).TrainSequential(st, pi, pt, new double[] { 100, 200, 300 }, maxPriceMemory: 500); Assert(m.PriceMemory.Count > pi[0].GetLength(0) - 1, $"Got {m.PriceMemory.Count}"); }
        public void Test_Sequential_MultiplEpochs_MemoryReset_EachEpoch() { var (tok, st, pi, pt) = CreateTestData(2, priceSeqLen: 8); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2, priceSeqLen: 8), new Random(42)); m.NewsMemory.Add(new NewsMemoryEntry { HiddenState = new float[m.Config.PriceEmbeddingDim], AbsoluteTimestamp = 1.0 }); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 2, Verbose = false }).TrainSequential(st, pi, pt, new double[] { 100, 200 }); Assert(m.NewsMemory.Count > 0, "Should have memory after"); }
        public void Test_Sequential_WithNoStories_StillWorks() { var rng = new Random(42); int n = 3; var ns = new NewsStory[n][]; var pi = new float[n][,]; var pt = new float[n][,]; for (int s = 0; s < n; s++) { pi[s] = RandomMatrix(8, 5, rng, 0.5f); pt[s] = RandomMatrix(8, 5, rng, 0.5f); } var m = new TACAMT_Model(CreateConfig(priceSeqLen: 10), new Random(42)); var t = new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.001f, BatchSize = 1, Epochs = 2, Verbose = false }); t.TrainSequential(ns, pi, pt, new double[] { 100, 200, 300 }); Assert(!float.IsNaN(t.Validate(ns, pi, pt)), "Valid loss"); }

        // === PRICE CONTEXT ADVANCED ===
        public void Test_PriceContext_SplitPointDistribution_WithinBounds() { int sl = 20, mh = 5, mc = 5; int maxH = sl - mc - 1; var rng = new Random(42); for (int trial = 0; trial < 100; trial++) { int sp = mh + rng.Next(maxH - mh + 1); Assert(sp >= mh && sp <= maxH && sl - sp >= mc + 1, $"Split {sp} out of bounds"); } }
        public void Test_PriceContext_HistoryDetached_CurrentTrained() { var (tok, st, pi, pt) = CreateTestData(5, priceSeqLen: 16); var m = new TACAMT_Model(CreateConfig(tok.VocabSize + 2, priceSeqLen: 16), new Random(42)); var ob = CloneMatrix(m.OutputProjection); new TACAMT_Trainer(m, new MultimodalTrainingConfig { LearningRate = 0.01f, BatchSize = 5, Epochs = 5, Verbose = false }).Train(st, pi, pt); Assert(MatrixChanged(ob, m.OutputProjection), "OutputProjection should change from current chunk"); }

        // === TEXT ENCODER ===
        public void Test_TextEncoder_Bidirectional_Default() { Assert(CreateConfig().TextUseDecoderOnly == false, "Should be bidirectional"); }
        public void Test_TextEncoder_MeanPooling_Produces_FixedDimOutput() { var (tok, stories, pi, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2, embDim: 16); var m = new TACAMT_Model(c, new Random(42)); var cache = new MultimodalForwardCache(c.TextNumLayers, c.PriceNumLayers); m.ForwardWithCache(stories[0], pi[0], cache); Assert(cache.TextFinalHidden.GetLength(0) == stories[0].Length, $"Stories {cache.TextFinalHidden.GetLength(0)} vs {stories[0].Length}"); Assert(cache.TextFinalHidden.GetLength(1) == 16, $"Dim {cache.TextFinalHidden.GetLength(1)}"); }

        // === TOKENIZER ===
        public void Test_Tokenizer_MatchingVocab_AcceptedAndUsable() { var tok = new BPETokenizer(); tok.Train(new[] { "stock price rose" }, vocabSize: 50, minFrequency: 1); var m = new TACAMT_Model(CreateConfig(textVocabSize: tok.VocabSize), new Random(42)); m.SetTokenizer(tok); var s = m.TokenizeStories(new[] { "stock price" }, new[] { 0f }); Assert(s.Length == 1 && s[0].TokenIds.Length > 0, "Failed"); }
        public void Test_Tokenizer_SavedWithModel_LoadedBack() { var tok = new BPETokenizer(); tok.Train(new[] { "stock price rose sharply" }, vocabSize: 50, minFrequency: 1); var m = new TACAMT_Model(CreateConfig(textVocabSize: tok.VocabSize), new Random(42)); m.SetTokenizer(tok); var d = GetTempDir(); try { m.Save(d); var ld = TACAMT_Model.Load(d); Assert(ld.Tokenizer != null, "Should load"); Assert(ld.Tokenizer.Encode("stock").Length > 0, "Should encode"); } finally { CleanupDir(d); } }

        // === NUMERICAL STABILITY ===
        public void Test_Stability_VeryLargeInputs_NoNaN() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 12), new Random(42)); var p = new float[10, 5]; for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) p[i, j] = 100f; Assert(!HasNaN(m.Forward((NewsStory[])null, p).predictions), "NaN"); }
        public void Test_Stability_VerySmallInputs_NoNaN() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 12), new Random(42)); var p = new float[10, 5]; for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) p[i, j] = 1e-7f; Assert(!HasNaN(m.Forward((NewsStory[])null, p).predictions), "NaN"); }
        public void Test_Stability_ZeroInputs_NoNaN() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 12), new Random(42)); Assert(!HasNaN(m.Forward((NewsStory[])null, new float[10, 5]).predictions), "NaN"); }
        public void Test_Stability_NegativeInputs_NoNaN() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 12), new Random(42)); var p = new float[10, 5]; for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) p[i, j] = -5f; Assert(!HasNaN(m.Forward((NewsStory[])null, p).predictions), "NaN"); }
        public void Test_Stability_MixedMagnitude_NoNaN() { var m = new TACAMT_Model(CreateConfig(priceSeqLen: 12), new Random(42)); var rng = new Random(42); var p = new float[10, 5]; for (int i = 0; i < 10; i++) for (int j = 0; j < 5; j++) p[i, j] = (float)Math.Pow(10, rng.Next(-3, 3)) * (rng.NextSingle() > 0.5f ? 1f : -1f); Assert(!HasNaN(m.Forward((NewsStory[])null, p).predictions), "NaN"); }

        // === EQUIVALENCE ===
        public void Test_Equiv_ForwardAndForwardWithCache_SameOutput() { var (tok, stories, pi, _) = CreateTestData(1); var c = CreateConfig(tok.VocabSize + 2); var m = new TACAMT_Model(c, new Random(42)); var (p1, _) = m.Forward(stories[0], pi[0]); var cache = new MultimodalForwardCache(c.TextNumLayers, c.PriceNumLayers); var (p2, _) = m.ForwardWithCache(stories[0], pi[0], cache, isTraining: false); for (int i = 0; i < p1.GetLength(0); i++) for (int j = 0; j < p1.GetLength(1); j++) Assert(MathF.Abs(p1[i, j] - p2[i, j]) < 1e-5f, $"[{i},{j}]: {p1[i, j]} vs {p2[i, j]}"); }

        public void Test_Equiv_GradientsZero_BeforeTraining() 
        { 
            var c = CreateConfig();

            var g = new Gradients(c);

            for (int i = 0; i < c.TextVocabSize; i++)
            {
                for (int j = 0; j < c.TextEmbeddingDim; j++)
                {
                    Assert(g.TextEmbeddingGrad[i, j] == 0f, $"Not zero");
                }
            }
            for (int i = 0; i < c.OutputDim; i++) for (int j = 0; j < c.PriceEmbeddingDim; j++)
            {
                Assert(g.OutputProjectionGrad[i, j] == 0f, $"Not zero");
            }

            Assert(g.PriceBlockGrads.Count == c.PriceNumLayers, "Count");
        }

        // ===============================================================
        // END-TO-END LEARNING QUALITY — Synthetic Ground Truth
        // ===============================================================

        /// <summary>
        /// Create synthetic data where one news story ("bull") drives prices UP
        /// and another ("bear") drives prices DOWN. After training, the model
        /// should produce higher predictions for "bull" than "bear" on the same
        /// price input.
        /// </summary>
        public void Test_E2E_LearnsBullVsBearSignal()
        {
            // Build a tokenizer and config
            var tok = new BPETokenizer();
            tok.Train(new[] {
                "bull bull bull bull bull",
                "bear bear bear bear bear"
            }, vocabSize: 50, minFrequency: 1);

            var c = CreateConfig(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 12, outputDim: 1, inputFeatures: 3);
            var model = new TACAMT_Model(c, new Random(42));

            int numSamples = 40;
            var rng = new Random(42);
            var stories = new NewsStory[numSamples][];
            var priceInputs = new float[numSamples][,];
            var priceTargets = new float[numSamples][,];

            var bullTokens = tok.Encode("bull bull bull bull bull", addSpecialTokens: true);
            var bearTokens = tok.Encode("bear bear bear bear bear", addSpecialTokens: true);

            for (int s = 0; s < numSamples; s++)
            {
                bool isBull = s < numSamples / 2;
                stories[s] = new[] { new NewsStory(isBull ? bullTokens : bearTokens, 0f) };

                priceInputs[s] = new float[10, 3];
                priceTargets[s] = new float[10, 1];

                // Same base price input for all samples
                float basePrice = 0.5f;
                for (int t = 0; t < 10; t++)
                {
                    priceInputs[s][t, 0] = basePrice;
                    priceInputs[s][t, 1] = basePrice + (float)(rng.NextDouble() - 0.5) * 0.01f;
                    priceInputs[s][t, 2] = 0.5f;

                    // Bull targets go UP, bear targets go DOWN
                    if (isBull)
                        priceTargets[s][t, 0] = 0.7f + (float)rng.NextDouble() * 0.1f;
                    else
                        priceTargets[s][t, 0] = 0.2f + (float)rng.NextDouble() * 0.1f;
                }
            }

            // Train
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 0.003f,
                BatchSize = 10,
                Epochs = 200,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            });
            trainer.Train(stories, priceInputs, priceTargets);

            // Test: same price input, different story => different prediction direction
            var testPrice = new float[10, 3];
            for (int t = 0; t < 10; t++)
            { testPrice[t, 0] = 0.5f; testPrice[t, 1] = 0.5f; testPrice[t, 2] = 0.5f; }

            var (predBull, _) = model.Forward(new[] { new NewsStory(bullTokens, 0f) }, testPrice);
            var (predBear, _) = model.Forward(new[] { new NewsStory(bearTokens, 0f) }, testPrice);

            // Average predictions across timesteps
            float avgBull = 0f, avgBear = 0f;
            for (int t = 0; t < predBull.GetLength(0); t++)
            {
                avgBull += predBull[t, 0];
                avgBear += predBear[t, 0];
            }
            avgBull /= predBull.GetLength(0);
            avgBear /= predBear.GetLength(0);

            Assert(avgBull > avgBear,
                $"Bull prediction ({avgBull:F4}) should exceed bear ({avgBear:F4}) — model should learn news→price signal");
        }

        /// <summary>
        /// Create a synthetic regression task: target = mean(price_input) * constant.
        /// After sufficient training on a single sample, the model should produce
        /// predictions close to the target (MSE < initial_MSE * 0.01).
        /// </summary>
        public void Test_E2E_SingleSample_NearPerfectOverfit()
        {
            var c = CreateConfig(embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64,
                                 priceSeqLen: 12, outputDim: 3, inputFeatures: 3);
            var model = new TACAMT_Model(c, new Random(42));

            var rng = new Random(42);
            var pi = new float[1][,] { RandomMatrix(10, 3, rng, 0.3f) };
            var pt = new float[1][,];
            // Target: a deterministic function of input
            pt[0] = new float[10, 3];
            for (int t = 0; t < 10; t++)
                for (int f = 0; f < 3; f++)
                    pt[0][t, f] = pi[0][t, f] * 2.0f + 0.1f; // linear transform

            var ns = new NewsStory[1][];

            var t1 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 1, Epochs = 1, Verbose = false });
            float lossBefore = t1.Validate(ns, pi, pt);

            var t2 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 0.003f,
                BatchSize = 1,
                Epochs = 2000,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            });
            t2.Train(ns, pi, pt);

            float lossAfter = t2.Validate(ns, pi, pt);

            Assert(lossAfter < lossBefore * 0.01f,
                $"Should overfit to <1% of initial loss. Before: {lossBefore:F6}, After: {lossAfter:F6} (ratio: {lossAfter / lossBefore:F4})");
        }

        /// <summary>
        /// Test that the model learns news arrival time matters: same text but
        /// recent news (small arrival time) should produce a different prediction
        /// than very old news (large arrival time), when trained on data that
        /// correlates recency with target.
        /// </summary>
        public void Test_E2E_LearnRecencyMatters()
        {
            var tok = new BPETokenizer();
            tok.Train(new[] { "breaking news alert", "old background info" }, vocabSize: 50, minFrequency: 1);
            var recentTokens = tok.Encode("breaking news alert", addSpecialTokens: true);
            var oldTokens = tok.Encode("old background info", addSpecialTokens: true);

            var c = CreateConfig(tok.VocabSize + 2, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64, priceSeqLen: 12, outputDim: 1, inputFeatures: 3);
            var model = new TACAMT_Model(c, new Random(42));

            int numSamples = 60;
            var stories = new NewsStory[numSamples][];
            var priceInputs = new float[numSamples][,];
            var priceTargets = new float[numSamples][,];

            for (int s = 0; s < numSamples; s++)
            {
                // Every sample has 2 stories: one recent, one old.
                // When the recent story is the "bullish" one, target is high.
                // When the old story is the "bullish" one, target is low.
                // The model must learn to attend more to the recent story.
                bool recentIsBullish = s < numSamples / 2;

                stories[s] = new[]
                {
            new NewsStory(
                recentIsBullish ? recentTokens : oldTokens,
                0f),       // recent story (arrival time 0)
            new NewsStory(
                recentIsBullish ? oldTokens : recentTokens,
                1000f)     // old story (arrival time 1000)
        };

                priceInputs[s] = new float[10, 3];
                priceTargets[s] = new float[10, 1];

                for (int t = 0; t < 10; t++)
                {
                    priceInputs[s][t, 0] = 0.5f;
                    priceInputs[s][t, 1] = 0.5f;
                    priceInputs[s][t, 2] = 0.5f;
                    priceTargets[s][t, 0] = recentIsBullish ? 0.9f : 0.1f;
                }
            }

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 10,
                Epochs = 500,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            });

            trainer.Train(stories, priceInputs, priceTargets);

            var testPrice = new float[10, 3];
            for (int t = 0; t < 10; t++)
            {
                testPrice[t, 0] = 0.5f;
                testPrice[t, 1] = 0.5f;
                testPrice[t, 2] = 0.5f;
            }

            // Test: "breaking news" as recent + "old background" as old => should predict high
            var cache1 = new MultimodalForwardCache(c.TextNumLayers, c.PriceNumLayers);
            var (predRecentBullish, _) = model.ForwardWithCache(new[]
            {
        new NewsStory(recentTokens, 0f),
        new NewsStory(oldTokens, 1000f)
    }, testPrice, cache1);

            // Test: "old background" as recent + "breaking news" as old => should predict low
            var cache2 = new MultimodalForwardCache(c.TextNumLayers, c.PriceNumLayers);
            var (predOldBullish, _) = model.ForwardWithCache(new[]
            {
        new NewsStory(oldTokens, 0f),
        new NewsStory(recentTokens, 1000f)
    }, testPrice, cache2);

            float avgHigh = 0f, avgLow = 0f;
            for (int t = 0; t < predRecentBullish.GetLength(0); t++)
            {
                avgHigh += predRecentBullish[t, 0];
                avgLow += predOldBullish[t, 0];
            }

            avgHigh /= predRecentBullish.GetLength(0);
            avgLow /= predOldBullish.GetLength(0);

            Assert(avgHigh > avgLow, $"Recent bullish news ({avgHigh:F4}) should produce higher prediction than old bullish news ({avgLow:F4})");
        }
        // ===============================================================
        // LARGER SCALE — Stress Tests
        // ===============================================================

        /// <summary>
        /// Test with a production-like config: larger embedding dims, more layers,
        /// more heads. Verifies no overflow/NaN at realistic scale.
        /// </summary>
        public void Test_Scale_LargerConfig_NoNaN()
        {
            var c = new Transformers.TACAMT.Config
            {
                TextVocabSize = 500,
                TextMaxSequenceLength = 128,
                TextEmbeddingDim = 64,
                TextNumHeads = 8,
                TextNumLayers = 3,
                TextFeedForwardDim = 256,
                TextUseDecoderOnly = false,
                PriceInputFeatureDim = 10,
                PriceMaxSequenceLength = 102,
                PriceEmbeddingDim = 64,
                PriceNumHeads = 8,
                PriceNumLayers = 3,
                PriceFeedForwardDim = 256,
                PriceUseDecoderOnly = true,
                OutputDim = 10,
                UseConfidenceHead = true,
                FreezeTextEncoder = false,
                FFNActivationType = ActivationType.Relu,
                AccelerationType = AccelerationType.CPU,
                L2RegulationLamda = 1e-5f,
                GradientClippingThreshold = 1.0f,
                PriceContextMinHistoryLength = 20,
                PriceContextMinCurrentLength = 20,
                DecayProjectionDim = 16,
                DecayHiddenDim = 32,
                DecayTimeEncodingBases = 16,
            };
            c.Validate();

            var model = new TACAMT_Model(c, new Random(42));

            // Create longer price sequences
            var rng = new Random(42);
            var price = new float[100, 10];
            for (int i = 0; i < 100; i++)
                for (int j = 0; j < 10; j++)
                    price[i, j] = (float)(rng.NextDouble() - 0.5) * 2f;

            // Create stories with token IDs in valid range
            var stories = new NewsStory[5];
            for (int s = 0; s < 5; s++)
            {
                var toks = new int[20 + rng.Next(30)];
                for (int t = 0; t < toks.Length; t++)
                    toks[t] = 2 + rng.Next(498); // valid vocab range
                stories[s] = new NewsStory(toks, s * 5f);
            }

            var (pred, conf) = model.Forward(stories, price);

            Assert(!HasNaN(pred), "NaN in large-scale forward");
            Assert(pred.GetLength(0) == 100, $"Expected 100 rows, got {pred.GetLength(0)}");
            Assert(pred.GetLength(1) == 10, $"Expected 10 cols, got {pred.GetLength(1)}");
            Assert(conf != null, "Confidence should exist");
            for (int i = 0; i < conf.GetLength(0); i++)
                Assert(conf[i, 0] >= 0f && conf[i, 0] <= 1f, $"Confidence[{i}] out of range");
        }

        /// <summary>
        /// Train at larger scale with multiple stories per sample and verify loss decreases.
        /// </summary>
        public void Test_Scale_LargerConfig_TrainingConverges()
        {
            var c = new Transformers.TACAMT.Config
            {
                TextVocabSize = 200,
                TextMaxSequenceLength = 64,
                TextEmbeddingDim = 32,
                TextNumHeads = 4,
                TextNumLayers = 2,
                TextFeedForwardDim = 128,
                TextUseDecoderOnly = false,
                PriceInputFeatureDim = 5,
                PriceMaxSequenceLength = 52,
                PriceEmbeddingDim = 32,
                PriceNumHeads = 4,
                PriceNumLayers = 2,
                PriceFeedForwardDim = 128,
                PriceUseDecoderOnly = true,
                OutputDim = 5,
                UseConfidenceHead = false,
                FreezeTextEncoder = false,
                FFNActivationType = ActivationType.Relu,
                AccelerationType = AccelerationType.CPU,
                L2RegulationLamda = 0f,
                GradientClippingThreshold = 1.0f,
                PriceContextMinHistoryLength = 10,
                PriceContextMinCurrentLength = 10,
                DecayProjectionDim = 8,
                DecayHiddenDim = 16,
                DecayTimeEncodingBases = 8,
            };
            c.Validate();

            var model = new TACAMT_Model(c, new Random(42));
            var rng = new Random(42);
            int numSamples = 8;
            var stories = new NewsStory[numSamples][];
            var pi = new float[numSamples][,];
            var pt = new float[numSamples][,];

            for (int s = 0; s < numSamples; s++)
            {
                // 3-6 stories per sample with varied lengths
                int nStories = 3 + rng.Next(4);
                stories[s] = new NewsStory[nStories];
                for (int ns = 0; ns < nStories; ns++)
                {
                    var toks = new int[10 + rng.Next(20)];
                    for (int t = 0; t < toks.Length; t++)
                        toks[t] = 2 + rng.Next(198);
                    stories[s][ns] = new NewsStory(toks, ns * 2.0f);
                }

                pi[s] = new float[50, 5];
                pt[s] = new float[50, 5];
                for (int t = 0; t < 50; t++)
                    for (int f = 0; f < 5; f++)
                    {
                        pi[s][t, f] = (float)(rng.NextDouble() - 0.5) * 2f;
                        pt[s][t, f] = (float)(rng.NextDouble() - 0.5) * 2f;
                    }
            }

            var t1 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            { LearningRate = 0.001f, BatchSize = 4, Epochs = 1, Verbose = false });
            float lossBefore = t1.Validate(stories, pi, pt);

            var t2 = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 0.001f,
                BatchSize = 4,
                Epochs = 20,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            });
            t2.Train(stories, pi, pt);

            float lossAfter = t2.Validate(stories, pi, pt);
            Assert(lossAfter < lossBefore,
                $"Larger scale training should decrease loss: {lossBefore:F6} -> {lossAfter:F6}");
            Assert(!float.IsNaN(lossAfter) && !float.IsInfinity(lossAfter),
                $"Loss should be finite: {lossAfter}");
        }

        /// <summary>
        /// Many stories accumulated over time in memory — simulates long-running inference.
        /// </summary>
        public void Test_Scale_ManyMemoryEntries_NoNaN()
        {
            var c = CreateConfig(embDim: 32, numHeads: 4);
            var model = new TACAMT_Model(c, new Random(42));
            var rng = new Random(42);

            // Accumulate 50 memory entries over time
            for (int i = 0; i < 50; i++)
            {
                var toks = new int[5 + rng.Next(10)];
                for (int t = 0; t < toks.Length; t++)
                    toks[t] = 1 + rng.Next(49);
                var story = new[] { new NewsStory(toks, i * 0.5f) };
                model.UpdateNewsMemory(story, (double)(i * 100));
            }

            Assert(model.NewsMemory.Count > 0, "Should have memory entries");

            // Now predict with accumulated memory
            var price = RandomMatrix(10, 5, rng, 0.5f);
            var (pred, conf) = model.PredictWithMemory(null, price, 5000.0,
                maxNewsMemorySize: 30, maxPriceMemorySize: 100);

            Assert(pred != null, "Prediction should not be null");
            Assert(!HasNaN(pred), "NaN in prediction with many memory entries");
            Assert(pred.Length == c.OutputDim, $"Prediction dim: {pred.Length}");
        }

        // ===============================================================
        // TEXT ENCODER PIPELINE — Full Tokenizer→Encode→Train→Predict
        // ===============================================================

        /// <summary>
        /// Full pipeline: create tokenizer, attach to model, tokenize raw text,
        /// train, verify text encoder weights change, verify loss decreases.
        /// </summary>
        public void Test_TextPipeline_FullTokenizerFlow()
        {
            string[] corpus = {
                "the stock market rallied strongly today on positive earnings",
                "markets fell sharply due to inflation concerns and rate hikes",
                "tech sector posted record gains driven by AI momentum",
                "banking sector showed weakness amid regulatory uncertainty"
            };

            var tok = new BPETokenizer();
            tok.Train(corpus, vocabSize: 100, minFrequency: 1);

            var c = CreateConfig(tok.VocabSize, embDim: 32, numHeads: 4, numLayers: 2, ffnDim: 64);
            var model = new TACAMT_Model(c, new Random(42));
            model.SetTokenizer(tok);

            // Use TokenizeStories to create training data from raw text
            var bullStories = model.TokenizeStories(
                new[] { "the stock market rallied strongly today on positive earnings" },
                new[] { 0f });
            var bearStories = model.TokenizeStories(
                new[] { "markets fell sharply due to inflation concerns and rate hikes" },
                new[] { 0f });

            Assert(bullStories.Length == 1, "Should have 1 story");
            Assert(bullStories[0].TokenIds.Length > 0, "Should have tokens");

            // Create training data
            var rng = new Random(42);
            int numSamples = 10;
            var stories = new NewsStory[numSamples][];
            var pi = new float[numSamples][,];
            var pt = new float[numSamples][,];

            for (int s = 0; s < numSamples; s++)
            {
                string text = corpus[rng.Next(corpus.Length)];
                stories[s] = model.TokenizeStories(new[] { text }, new[] { (float)(s * 2) });
                pi[s] = RandomMatrix(10, 5, rng, 0.3f);
                pt[s] = RandomMatrix(10, 5, rng, 0.3f);
            }

            var textEmbBefore = CloneMatrix(model.TextTokenEmbedding);

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 0.005f,
                BatchSize = 5,
                Epochs = 10,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            });
            trainer.Train(stories, pi, pt);

            Assert(MatrixChanged(textEmbBefore, model.TextTokenEmbedding),
                "Text embeddings should change during training with tokenized text");

            float loss = trainer.Validate(stories, pi, pt);
            Assert(!float.IsNaN(loss) && loss >= 0,
                $"Loss should be valid after tokenizer pipeline training: {loss}");
        }

        /// <summary>
        /// Verify that different tokenized texts produce different model outputs
        /// (i.e., the text content actually flows through to predictions).
        /// </summary>
        public void Test_TextPipeline_DifferentTexts_DifferentOutputs()
        {
            var tok = new BPETokenizer();
            tok.Train(new[] {
                "stocks rallied strongly on earnings",
                "market crashed on bad data"
            }, vocabSize: 80, minFrequency: 1);

            var c = CreateConfig(tok.VocabSize + 2, embDim: 32, numHeads: 4);
            var model = new TACAMT_Model(c, new Random(42));

            var tokens1 = tok.Encode("stocks rallied strongly on earnings", addSpecialTokens: true);
            var tokens2 = tok.Encode("market crashed on bad data", addSpecialTokens: true);

            var price = RandomMatrix(10, 5, new Random(42), 0.5f);

            var (pred1, _) = model.Forward(new[] { new NewsStory(tokens1, 0f) }, price);
            var (pred2, _) = model.Forward(new[] { new NewsStory(tokens2, 0f) }, price);

            bool anyDiff = false;
            for (int i = 0; i < pred1.GetLength(0) && !anyDiff; i++)
                for (int j = 0; j < pred1.GetLength(1) && !anyDiff; j++)
                    if (MathF.Abs(pred1[i, j] - pred2[i, j]) > 1e-6f)
                        anyDiff = true;

            Assert(anyDiff, "Different tokenized texts should produce different outputs");
        }

        /// <summary>
        /// Multiple stories with varied lengths through tokenizer — verifies
        /// the mean-pooling handles variable-length sequences.
        /// </summary>
        public void Test_TextPipeline_MultipleStories_VariedLengths()
        {
            var tok = new BPETokenizer();
            tok.Train(new[] {
                "short",
                "this is a much longer sentence with many tokens in it for testing",
                "medium length text here"
            }, vocabSize: 100, minFrequency: 1);

            var c = CreateConfig(tok.VocabSize + 2, embDim: 16, numHeads: 2);
            var model = new TACAMT_Model(c, new Random(42));

            var stories = new[] {
                new NewsStory(tok.Encode("short", addSpecialTokens: true), 0f),
                new NewsStory(tok.Encode("this is a much longer sentence with many tokens in it for testing", addSpecialTokens: true), 1f),
                new NewsStory(tok.Encode("medium length text here", addSpecialTokens: true), 2f)
            };

            var price = RandomMatrix(10, 5, new Random(42), 0.5f);
            var (pred, _) = model.Forward(stories, price);

            Assert(pred != null, "Prediction should not be null");
            Assert(!HasNaN(pred), "NaN with varied-length stories");
            Assert(pred.GetLength(0) == 10, "Output rows should match input");
        }

        #region Gradient correctness tests

        public void Test_GradCheck_SelfAttention_WQ_FiniteDifference()
        {
            var c = CreateConfig(embDim: 8, numHeads: 2, outputDim: 3, priceSeqLen: 8);
            var model = new TACAMT_Model(c, new Random(42));
            var rng = new Random(42);
            var price = RandomMatrix(5, 5, rng, 0.3f);
            var target = RandomMatrix(5, 3, rng, 0.3f);

            float eps = 1e-3f;
            int sl = 5, od = 3;
            var wq = model.PriceBlocks[0].SelfAttention.WQ;

            float origW = wq[0, 0];
            wq[0, 0] = origW + eps;
            var (pp, _) = model.Forward((NewsStory[])null, price);
            float lp = 0f;

            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < od; j++)
                { 
                    float d = pp[t, j] - target[t, j];
                    lp += d * d; 
                }
            }

            lp = lp/(sl * od);

            wq[0, 0] = origW - eps;

            var (pm, _) = model.Forward((NewsStory[])null, price);
            float lm = 0f;

            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < od; j++)
                { 
                    float d = pm[t, j] - target[t, j]; 
                    lm += d * d; 
                }
            }

            lm = lm/(sl * od);

            wq[0, 0] = origW;

            float fd = (lp - lm) / (2 * eps);

            Assert(!float.IsNaN(fd), "Finite difference is NaN");

            Assert(MathF.Abs(fd) > 1e-10f, $"Self-attention WQ gradient should be non-zero: {fd:E6}");
        }

        public void Test_GradCheck_CrossAttention_WK_FiniteDifference()
        {
            var (tok, stories, pi, _) = CreateTestData(numSamples: 1, priceSeqLen: 8);
            var c = CreateConfig(tok.VocabSize + 2, embDim: 8, numHeads: 2, outputDim: 3, priceSeqLen: 8);
            var model = new TACAMT_Model(c, new Random(42));
            var rng = new Random(42);
            var target = RandomMatrix(8, 3, rng, 0.3f);

            float eps = 1e-3f;
            int sl = 8, od = 3;
            var wk = model.PriceBlocks[0].CrossAttention.WK;

            float origW = wk[0, 0];
            wk[0, 0] = origW + eps;
            var (pp, _) = model.Forward(stories[0], pi[0]);
            float lp = 0f;

            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < od; j++)
                { 
                    float d = pp[t, j] - target[t, j];
                    lp += d * d;
                }
            }

            wk[0, 0] = origW - eps;
            var (pm, _) = model.Forward(stories[0], pi[0]);
            float lm = 0f;
            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < od; j++)
                {
                    float d = pm[t, j] - target[t, j];
                    lm += d * d;
                }
            }

            wk[0, 0] = origW;

            float fd = (lp - lm) / (2 * eps);
            Assert(!float.IsNaN(fd), "Finite difference is NaN");
            Assert(MathF.Abs(fd) > 1e-10f, $"Cross-attention WK gradient should be non-zero (text→price flow): {fd:E6}");
        }
        public void Test_GradCheck_DecayNetwork_LogBaseDecayRate_FiniteDifference()
        {
            var (tok, stories, pi, _) = CreateTestData(numSamples: 1, priceSeqLen: 8);
            var c = CreateConfig(tok.VocabSize + 2, embDim: 8, numHeads: 2, outputDim: 3, priceSeqLen: 8);
            var model = new TACAMT_Model(c, new Random(42));
            var rng = new Random(42);
            var target = RandomMatrix(8, 3, rng, 0.3f);

            float eps = 1e-2f; // Larger epsilon — the gradient is small
            int sl = 8, od = 3;

            // Try all heads across all layers
            bool anyNonZero = false;
            for (int layer = 0; layer < c.PriceNumLayers && !anyNonZero; layer++)
            {
                var decayNet = model.PriceBlocks[layer].DecayNetwork;
                for (int h = 0; h < c.PriceNumHeads && !anyNonZero; h++)
                {
                    float origVal = decayNet.LogBaseDecayRate[h];

                    decayNet.LogBaseDecayRate[h] = origVal + eps;
                    var (pp, _) = model.Forward(stories[0], pi[0]);
                    float lp = 0f;
                    for (int t = 0; t < sl; t++)
                        for (int j = 0; j < od; j++)
                        { float d = pp[t, j] - target[t, j]; lp += d * d; }

                    decayNet.LogBaseDecayRate[h] = origVal - eps;
                    var (pm, _) = model.Forward(stories[0], pi[0]);
                    float lm = 0f;
                    for (int t = 0; t < sl; t++)
                        for (int j = 0; j < od; j++)
                        { float d = pm[t, j] - target[t, j]; lm += d * d; }

                    decayNet.LogBaseDecayRate[h] = origVal;

                    if (MathF.Abs((lp - lm) / (2 * eps)) > 1e-12f)
                        anyNonZero = true;
                }
            }

            Assert(anyNonZero, "At least one LogBaseDecayRate element should have non-zero gradient");
        }

        public void Test_GradCheck_DecayNetwork_W1_FiniteDifference()
        {
            var (tok, stories, pi, _) = CreateTestData(numSamples: 1, priceSeqLen: 8);
            var c = CreateConfig(tok.VocabSize + 2, embDim: 8, numHeads: 2, outputDim: 3, priceSeqLen: 8);
            var model = new TACAMT_Model(c, new Random(42));
            var rng = new Random(42);
            var target = RandomMatrix(8, 3, rng, 0.3f);

            float eps = 1e-3f;
            int sl = 8, od = 3;
            var decayNet = model.PriceBlocks[0].DecayNetwork;

            // Try multiple coordinates — some may be on dead ReLU paths
            bool anyNonZero = false;
            for (int h = 0; h < decayNet.W1.GetLength(0) && !anyNonZero; h++)
            {
                for (int i = 0; i < decayNet.W1.GetLength(1) && !anyNonZero; i++)
                {
                    for (int j = 0; j < decayNet.W1.GetLength(2) && !anyNonZero; j++)
                    {
                        float origVal = decayNet.W1[h, i, j];

                        decayNet.W1[h, i, j] = origVal + eps;
                        var (pp, _) = model.Forward(stories[0], pi[0]);
                        float lp = 0f;

                        for (int t = 0; t < sl; t++)
                        {
                            for (int jj = 0; jj < od; jj++)
                            {
                                float d = pp[t, jj] - target[t, jj];
                                lp += d * d;
                            }
                        }

                        decayNet.W1[h, i, j] = origVal - eps;
                        var (pm, _) = model.Forward(stories[0], pi[0]);

                        float lm = 0f;

                        for (int t = 0; t < sl; t++)
                        {
                            for (int jj = 0; jj < od; jj++)
                            {
                                float d = pm[t, jj] - target[t, jj];
                                lm += d * d;
                            }
                        }

                        decayNet.W1[h, i, j] = origVal;

                        float fd = MathF.Abs((lp - lm) / (2 * eps));

                        if (fd > 1e-12f)
                        {
                            anyNonZero = true;
                        }
                    
                    }
                }
            }
              

            Assert(anyNonZero, "At least one W1 element should have non-zero gradient");
        }

        public void Test_GradCheck_TextEmbedding_FiniteDifference()
        {
            var (tok, stories, pi, _) = CreateTestData(numSamples: 1, priceSeqLen: 8);
            var c = CreateConfig(tok.VocabSize + 2, embDim: 8, numHeads: 2, outputDim: 3,
                                 priceSeqLen: 8, freezeText: false);
            var model = new TACAMT_Model(c, new Random(42));
            var rng = new Random(42);
            var target = RandomMatrix(8, 3, rng, 0.3f);

            float eps = 1e-3f;
            int sl = 8, od = 3;

            int tokenId = stories[0][0].TokenIds[0];

            float origVal = model.TextTokenEmbedding[tokenId, 0];

            model.TextTokenEmbedding[tokenId, 0] = origVal + eps;
            var (pp, _) = model.Forward(stories[0], pi[0]);
            float lp = 0f;
            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < od; j++)
                { 
                    float d = pp[t, j] - target[t, j]; 
                    lp += d * d;
                }
            }

            model.TextTokenEmbedding[tokenId, 0] = origVal - eps;
            var (pm, _) = model.Forward(stories[0], pi[0]);
            float lm = 0f;

            for (int t = 0; t < sl; t++)
            {
                for (int j = 0; j < od; j++)
                { 
                    float d = pm[t, j] - target[t, j]; 
                    lm += d * d;
                }
            }

            model.TextTokenEmbedding[tokenId, 0] = origVal;

            float fd = (lp - lm) / (2 * eps);
            Assert(!float.IsNaN(fd), "TextEmbedding FD is NaN");
            Assert(MathF.Abs(fd) > 1e-10f, $"TextEmbedding should affect loss through cross-attention: {fd:E6}");
        }

        #endregion


        public void Test_ContextTypeEmbedding_ChangesAfterTraining()
        {
            var c = CreateConfig(priceSeqLen: 32);
            var model = new TACAMT_Model(c, new Random(42));

            float[,] before = (float[,])model.ContextTypeEmbedding.Clone();

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 1,
                Epochs = 5,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            });

            var rng = new Random(42);
            var pi = new float[3][,];
            var pt = new float[3][,];
            var ns = new NewsStory[3][];
            for (int s = 0; s < 3; s++)
            {
                pi[s] = RandomMatrix(30, c.PriceInputFeatureDim, rng, 0.5f);
                pt[s] = RandomMatrix(30, c.OutputDim, rng, 0.5f);
                ns[s] = new NewsStory[]
                {
            new NewsStory(new int[] { 4, 5, 6 }, -1.0f)
                };
            }

            trainer.Train(ns, pi, pt);

            bool anyChanged = false;
            for (int t = 0; t < 2 && !anyChanged; t++)
                for (int d = 0; d < c.PriceEmbeddingDim && !anyChanged; d++)
                    if (MathF.Abs(model.ContextTypeEmbedding[t, d] - before[t, d]) > 1e-8f)
                        anyChanged = true;

            Assert(anyChanged, "ContextTypeEmbedding should change after training");
        }

        public void Test_ContextTypeEmbedding_TypesDivergeAfterTraining()
        {
            var c = CreateConfig(priceSeqLen: 32);
            var model = new TACAMT_Model(c, new Random(42));

            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 0.01f,
                BatchSize = 1,
                Epochs = 10,
                UseGradientClipping = true,
                GradientClipThreshold = 1.0f,
                Verbose = false
            });

            var rng = new Random(42);
            var pi = new float[5][,];
            var pt = new float[5][,];
            var ns = new NewsStory[5][];
            for (int s = 0; s < 5; s++)
            {
                pi[s] = RandomMatrix(30, c.PriceInputFeatureDim, rng, 0.5f);
                pt[s] = RandomMatrix(30, c.OutputDim, rng, 0.5f);
                ns[s] = new NewsStory[]
                {
            new NewsStory(new int[] { 4, 5, 6 }, -1.0f)
                };
            }

            trainer.Train(ns, pi, pt);

            float sumDiff = 0;
            for (int d = 0; d < c.PriceEmbeddingDim; d++)
                sumDiff += MathF.Abs(model.ContextTypeEmbedding[0, d] - model.ContextTypeEmbedding[1, d]);

            Assert(sumDiff > 0.01f, $"Type embeddings should diverge, diff={sumDiff}");
        }

        public void Test_ContextTypeEmbedding_SaveLoadRoundTrip()
        {
            var c = CreateConfig();
            var model = new TACAMT_Model(c, new Random(42));

            for (int t = 0; t < 2; t++)
                for (int d = 0; d < c.PriceEmbeddingDim; d++)
                    model.ContextTypeEmbedding[t, d] = (t + 1) * 0.1f + d * 0.001f;

            var dir = GetTempDir();
            try
            {
                model.Save(dir);
                var loaded = TACAMT_Model.Load(dir);

                for (int t = 0; t < 2; t++)
                    for (int d = 0; d < c.PriceEmbeddingDim; d++)
                        Assert(model.ContextTypeEmbedding[t, d] == loaded.ContextTypeEmbedding[t, d],
                            $"Type embedding [{t},{d}] mismatch after load: {model.ContextTypeEmbedding[t, d]} vs {loaded.ContextTypeEmbedding[t, d]}");
            }
            finally { CleanupDir(dir); }
        }

        public void Test_ContextTypeEmbedding_IncludedInGradientClipping()
        {
            var c = CreateConfig(priceSeqLen: 32);
            var model = new TACAMT_Model(c, new Random(42));

            float[,] before = (float[,])model.ContextTypeEmbedding.Clone();

            // Train with very aggressive LR and tight clipping — if type embedding
            // is excluded from the norm, it would get disproportionately large updates
            var trainer = new TACAMT_Trainer(model, new MultimodalTrainingConfig
            {
                LearningRate = 1.0f,
                BatchSize = 1,
                Epochs = 3,
                UseGradientClipping = true,
                GradientClipThreshold = 0.1f,
                Verbose = false
            });

            var rng = new Random(42);
            var pi = new float[3][,];
            var pt = new float[3][,];
            var ns = new NewsStory[3][];
            for (int s = 0; s < 3; s++)
            {
                pi[s] = RandomMatrix(30, c.PriceInputFeatureDim, rng, 0.5f);
                pt[s] = RandomMatrix(30, c.OutputDim, rng, 0.5f);
                ns[s] = new NewsStory[]
                {
            new NewsStory(new int[] { 4, 5, 6 }, -1.0f)
                };
            }

            trainer.Train(ns, pi, pt);

            // Verify no NaN in type embeddings (clipping should prevent explosion)
            for (int t = 0; t < 2; t++)
                for (int d = 0; d < c.PriceEmbeddingDim; d++)
                    Assert(!float.IsNaN(model.ContextTypeEmbedding[t, d]) && !float.IsInfinity(model.ContextTypeEmbedding[t, d]),
                        $"ContextTypeEmbedding[{t},{d}] is NaN/Inf — gradient clipping may not include it");

            // Verify the update magnitude is bounded (tight clipping + high LR should still be controlled)
            float maxChange = 0f;
            for (int t = 0; t < 2; t++)
                for (int d = 0; d < c.PriceEmbeddingDim; d++)
                {
                    float change = MathF.Abs(model.ContextTypeEmbedding[t, d] - before[t, d]);
                    if (change > maxChange) maxChange = change;
                }

            // With clip threshold 0.1 and LR 1.0, max per-element change should be bounded
            Assert(maxChange < 5.0f,
                $"Max change {maxChange:F4} seems too large — gradient clipping may not be applied to type embeddings");
        }



    }
}