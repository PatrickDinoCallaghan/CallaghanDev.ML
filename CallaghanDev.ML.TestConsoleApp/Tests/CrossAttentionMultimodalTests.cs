using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CrossModel = CallaghanDev.ML.Transformers.CrossAttentionMultimodal.Model;
using CrossTrainer = CallaghanDev.ML.Transformers.CrossAttentionMultimodal.Trainer;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    /// <summary>
    /// High-coverage confidence suite for CrossAttentionMultimodal.Model + Trainer.
    ///
    /// This deliberately tests more than "does Forward return the right shape":
    /// - cached forward path must match inference forward path, because training uses the cached path;
    /// - causal masking must stop future price rows leaking into earlier predictions;
    /// - confidence loss weight 0 must not update confidence-head params;
    /// - invalid data should throw instead of being silently skipped;
    /// - text, cross-attention, price, output and confidence parameters are all checked for expected update/no-update paths;
    /// - save/load must preserve trained predictions and config switches.
    ///
    /// Some guard tests are expected to fail on the older implementation that swallowed TrainBatch exceptions
    /// or did not validate shapes/token ids. Those failures are useful: they tell you exactly where the model is not safe yet.
    /// </summary>
    internal sealed class CrossAttentionMultimodalTests : TestBase
    {
        private const float StrictTol = 1e-6f;
        private const float SaveLoadTol = 1e-5f;
        private const float ChangedTol = 1e-8f;

        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * CrossAttentionMultimodal confidence suite");
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            (Test_Save_CreatesExpectedFiles,                                      "Save: expected files/directories are created"),
            // Construction / config guards
            (Test_ModelConstruction_Default,                                      "Construction: default small model initialises"),
            (Test_ModelConstruction_ParameterShapes,                              "Construction: public parameter shapes match config"),
            (Test_ModelConstruction_MultipleLayers,                               "Construction: multiple text/price layers initialise"),
            (Test_Config_EmbeddingDimMismatchThrows,                              "Config: text/price embedding mismatch throws"),
            (Test_Config_EmbeddingDimNotDivisibleByHeadsThrows,                   "Config: embedding dim not divisible by heads throws"),
            (Test_Config_InvalidVocabSizeThrows,                                  "Config: invalid vocab size throws"),
            (Test_Config_InvalidOutputDimThrows,                                  "Config: invalid output dim throws"),

            // Forward / inference
            (Test_Forward_WithTextShapeAndFinite,                                 "Forward: with text returns finite [seqLen, outputDim]"),
            (Test_Forward_PriceOnlyShapeAndFinite,                                "Forward: null text price-only path works"),
            (Test_Forward_EmptyTextEqualsNullText,                                "Forward: empty text tokens equal null text path"),
            (Test_Forward_Deterministic,                                          "Forward: deterministic for repeated calls"),
            (Test_Forward_CacheMatchesForward_WithText,                           "ForwardWithCache: matches Forward with text"),
            (Test_Forward_CacheMatchesForward_PriceOnly,                          "ForwardWithCache: matches Forward with null text"),
            (Test_Forward_ConfidenceRange,                                        "Forward: confidence values are in [0, 1]"),
            (Test_Forward_NoConfidenceHeadReturnsNull,                            "Forward: confidence is null when disabled"),
            (Test_PredictNext_EqualsLastForwardRow_WithConfidence,                "PredictNext: equals final Forward row and confidence"),
            (Test_PredictNext_NoConfidenceHeadReturnsOneConfidence,               "PredictNext: confidence defaults to 1 when head disabled"),
            (Test_Forward_TextVsNoTextDiffer,                                     "Forward: text-conditioned output differs from price-only output"),
            (Test_Forward_DifferentTextsUsuallyDiffer,                            "Forward: different text inputs affect predictions"),
            (Test_Forward_VaryingTextLengths,                                     "Forward: varying text lengths work"),
            (Test_Forward_VaryingPriceSequenceLengths,                            "Forward: varying price sequence lengths work"),
            (Test_Forward_MaxConfiguredPriceLengthWorks,                          "Forward: max configured price length works"),
            (Test_Forward_CausalPriceMaskBlocksFutureLeakage,                     "Forward: decoder-only price mask blocks future leakage"),
            (Test_Forward_InvalidTokenIdThrows,                                   "Forward guard: invalid token id throws"),
            (Test_Forward_TextTooLongThrows,                                      "Forward guard: text longer than max throws"),
            (Test_Forward_PriceTooLongThrows,                                     "Forward guard: price sequence longer than max throws"),
            (Test_Forward_PriceFeatureMismatchThrows,                             "Forward guard: price feature mismatch throws"),
            (Test_PredictNext_EmptyPriceSequenceThrows,                           "PredictNext guard: empty price sequence throws"),
            (Test_Forward_NonFiniteInputThrows,                                   "Forward guard: NaN/Infinity input throws"),

            // Validation
            (Test_Validate_ReturnsFiniteNonNegativeLoss,                          "Validate: returns finite non-negative loss"),
            (Test_Validate_DoesNotMutateWeights,                                  "Validate: does not mutate model weights"),
            (Test_Validate_ConfidenceTargetsAffectLossWhenWeighted,               "Validate: confidence targets affect loss when confidence weight > 0"),
            (Test_Validate_ConfidenceTargetsIgnoredWhenWeightZero,                "Validate: confidence targets ignored when confidence weight = 0"),
            (Test_Validate_LengthMismatchThrows,                                  "Validate guard: dataset length mismatch throws"),

            // Training / learning behaviour
            (Test_Train_LossDecreases_OnSmoothNextStepData,                       "Train: loss decreases on smooth next-step data"),
            (Test_Train_PriceOnlyLossDecreases,                                   "Train: price-only loss decreases"),
            (Test_Train_SingleSampleOverfitsConstantTarget,                       "Train: single sample overfits simple constant target"),
            (Test_Train_BatchSizeOneWorks,                                        "Train: batch size 1 works"),
            (Test_Train_BatchSizeGreaterThanSampleCountWorks,                     "Train: batch size greater than sample count works"),
            (Test_Train_ZeroEpochsDoesNotChangeWeights,                           "Train: zero epochs leaves weights unchanged"),
            (Test_Train_LearningRateDecayRunsFinite,                              "Train: learning-rate decay runs without NaN"),
            (Test_Train_GradientClippingHighLearningRateStaysFinite,              "Train: gradient clipping keeps high-LR run finite"),
            (Test_Train_MixedNullTextBatchWorks,                                  "Train: mixed null/non-null text batch works"),
            (Test_Train_NoConfidenceHeadWorks,                                    "Train: no-confidence-head model trains"),
            (Test_Train_OutputHeadUpdates,                                        "Train: output head updates"),
            (Test_Train_ConfidenceHeadUpdatesWhenWeighted,                        "Train: confidence head updates when confidence loss is weighted"),
            (Test_Train_ConfidenceLossWeightZeroDoesNotUpdateConfidenceHead,      "Train: confidence weight 0 does not update confidence head"),
            (Test_Train_ExplicitConfidenceTargetsCanIncreaseConfidence,           "Train: explicit confidence targets can increase confidence"),
            (Test_Train_FrozenTextEncoderUnchanged,                               "Train: frozen text encoder remains unchanged"),
            (Test_Train_UnfrozenTextEncoderUpdatesWithText,                       "Train: unfrozen text encoder updates when text is present"),
            (Test_Train_UnfrozenTextEncoderUnchangedWhenAllTextNull,              "Train: text encoder unchanged when all text is null"),
            (Test_Train_CrossAttentionUpdatesWhenTextPresent,                     "Train: cross-attention weights update when text is present"),
            (Test_Train_CrossAttentionWeightsUnchangedWhenAllTextNull,            "Train: cross-attention weights unchanged when all text is null"),
            (Test_Train_PriceSelfAttentionUpdates,                                "Train: price self-attention weights update"),
            (Test_Train_PriceInputProjectionUpdates,                              "Train: price input projection updates"),
            (Test_Train_PriceLayerNormUpdates,                                    "Train: price layernorm parameters update"),
            (Test_Train_TextSignalCanBeLearnedForIdenticalPriceInputs,            "Train: text signal can disambiguate identical price inputs"),
            (Test_Train_RepeatedTrainingKeepsPredictionsFinite,                   "Train: repeated training keeps predictions finite"),
            (Test_Train_ArrayLengthMismatchThrows,                                "Train guard: dataset length mismatch throws"),
            (Test_Train_InvalidTokenIdThrows,                                     "Train guard: invalid token id throws, not silently skipped"),
            (Test_Train_PriceFeatureMismatchThrows,                               "Train guard: price feature mismatch throws, not silently skipped"),
            (Test_Train_TargetOutputDimMismatchThrows,                            "Train guard: target output dim mismatch throws, not silently skipped"),
            (Test_Train_ConfidenceTargetLengthMismatchThrows,                     "Train guard: confidence target length mismatch throws"),
            (Test_Train_NonFiniteInputThrows,                                     "Train guard: NaN/Infinity input throws"),

            // Save / load
            (Test_SaveLoad_TrainedWeightsAndForwardIdentical,                     "SaveLoad: trained model forward output is preserved"),
            (Test_SaveLoad_NoConfidenceHeadRoundTrips,                            "SaveLoad: no-confidence model round-trips"),
            (Test_SaveLoad_PredictNextIdentical,                                  "SaveLoad: PredictNext output is preserved"),
            (Test_SaveLoad_ConfigSwitchesPreserved,                               "SaveLoad: important config switches are preserved"),
            (Test_Load_MissingWeightsThrows,                                      "Load guard: missing weights file throws"),
            (Test_Load_CorruptWeightsThrows,                                      "Load guard: corrupt weight shape throws"),
        };

        // ---------------------------------------------------------------------
        // Construction / config
        // ---------------------------------------------------------------------

        private void Test_ModelConstruction_Default()
        {
            var cfg = MakeCfg(vocabSize: 64);
            var model = new CrossModel(cfg, new Random(42));

            Assert(model.Config == cfg, "model should keep config reference");
            Assert(model.TextBlocks.Length == cfg.Text.NumLayers, "wrong text layer count");
            Assert(model.PriceBlocks.Length == cfg.Price.NumLayers, "wrong price layer count");
            Assert(model.AccelerationManager != null, "acceleration manager is null");
        }

        private void Test_ModelConstruction_ParameterShapes()
        {
            var cfg = MakeCfg(vocabSize: 77, embDim: 12, numHeads: 3, numLayers: 2, ffnDim: 24, priceFeatures: 4, outputDim: 3, priceSeqLen: 9);
            var model = new CrossModel(cfg, new Random(7));

            AssertDims(model.TextTokenEmbedding, cfg.Text.VocabSize, cfg.Text.EmbeddingDim, "TextTokenEmbedding");
            AssertDims(model.PriceInputProjection, cfg.Price.EmbeddingDim, cfg.Price.InputFeatureDim, "PriceInputProjection");
            Assert(model.PriceInputProjectionBias.Length == cfg.Price.EmbeddingDim, "PriceInputProjectionBias length");
            AssertDims(model.OutputProjection, cfg.Output.OutputDim, cfg.Price.EmbeddingDim, "OutputProjection");
            Assert(model.OutputBias.Length == cfg.Output.OutputDim, "OutputBias length");
            AssertDims(model.ConfidenceProjection, 1, cfg.Price.EmbeddingDim, "ConfidenceProjection");
            Assert(model.ConfidenceBias.Length == 1, "ConfidenceBias length");

            foreach (var block in model.PriceBlocks)
            {
                AssertDims(block.SelfAttention.WQ, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "SelfAttention.WQ");
                AssertDims(block.SelfAttention.WK, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "SelfAttention.WK");
                AssertDims(block.SelfAttention.WV, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "SelfAttention.WV");
                AssertDims(block.SelfAttention.WO, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "SelfAttention.WO");
                AssertDims(block.CrossAttention.WQ, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "CrossAttention.WQ");
                AssertDims(block.CrossAttention.WK, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "CrossAttention.WK");
                AssertDims(block.CrossAttention.WV, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "CrossAttention.WV");
                AssertDims(block.CrossAttention.WO, cfg.Price.EmbeddingDim, cfg.Price.EmbeddingDim, "CrossAttention.WO");
                Assert(block.LNSelfGamma.Length == cfg.Price.EmbeddingDim, "LNSelfGamma length");
                Assert(block.LNCrossGamma.Length == cfg.Price.EmbeddingDim, "LNCrossGamma length");
                Assert(block.LNFFNGamma.Length == cfg.Price.EmbeddingDim, "LNFFNGamma length");
            }
        }

        private void Test_ModelConstruction_MultipleLayers()
        {
            var cfg = MakeCfg(vocabSize: 80, embDim: 16, numHeads: 4, numLayers: 3, ffnDim: 32);
            var model = new CrossModel(cfg, new Random(123));

            Assert(model.TextBlocks.Length == 3, "text block count");
            Assert(model.PriceBlocks.Length == 3, "price block count");
        }

        private void Test_Config_EmbeddingDimMismatchThrows()
        {
            ExpectThrowsAny(() =>
            {
                var cfg = MakeCfgNoValidate(vocabSize: 50, textEmbDim: 16, priceEmbDim: 32, numHeads: 2);
                cfg.Validate();
            }, "expected shared embedding-dim mismatch to throw");
        }

        private void Test_Config_EmbeddingDimNotDivisibleByHeadsThrows()
        {
            ExpectThrowsAny(() =>
            {
                var cfg = MakeCfgNoValidate(vocabSize: 50, textEmbDim: 15, priceEmbDim: 15, numHeads: 4);
                cfg.Validate();
            }, "expected embedding dim not divisible by heads to throw");
        }

        private void Test_Config_InvalidVocabSizeThrows()
        {
            ExpectThrowsAny(() =>
            {
                var cfg = MakeCfgNoValidate(vocabSize: 0, textEmbDim: 16, priceEmbDim: 16, numHeads: 2);
                cfg.Validate();
            }, "expected invalid vocab size to throw");
        }

        private void Test_Config_InvalidOutputDimThrows()
        {
            ExpectThrowsAny(() =>
            {
                var cfg = MakeCfgNoValidate(vocabSize: 50, textEmbDim: 16, priceEmbDim: 16, numHeads: 2, outputDim: 0);
                cfg.Validate();
            }, "expected invalid output dim to throw");
        }

        // ---------------------------------------------------------------------
        // Forward / inference
        // ---------------------------------------------------------------------

        private void Test_Forward_WithTextShapeAndFinite()
        {
            var data = SmoothData(n: 1, seqLen: 8, features: 5, outputDim: 5, seed: 1);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 8), new Random(42));

            var (pred, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            AssertDims(pred, 8, 5, "prediction");
            AssertDims(conf, 8, 1, "confidence");
            AssertFinite(pred, "prediction");
            AssertFinite(conf, "confidence");
        }

        private void Test_Forward_PriceOnlyShapeAndFinite()
        {
            var data = SmoothData(n: 1, seqLen: 7, features: 5, outputDim: 5, seed: 2);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 7), new Random(42));

            var (pred, conf) = model.Forward(null, data.PriceInputs[0]);

            AssertDims(pred, 7, 5, "prediction");
            AssertDims(conf, 7, 1, "confidence");
            AssertFinite(pred, "prediction");
            AssertFinite(conf, "confidence");
        }

        private void Test_Forward_EmptyTextEqualsNullText()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 3);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            var (pNull, cNull) = model.Forward(null, data.PriceInputs[0]);
            var (pEmpty, cEmpty) = model.Forward(Array.Empty<int>(), data.PriceInputs[0]);

            AssertMatrixClose(pNull, pEmpty, StrictTol, "empty text should match null text predictions");
            AssertMatrixClose(cNull, cEmpty, StrictTol, "empty text should match null text confidence");
        }

        private void Test_Forward_Deterministic()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 4);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            var (p1, c1) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            var (p2, c2) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            AssertMatrixClose(p1, p2, 0f, "predictions should be exactly deterministic");
            AssertMatrixClose(c1, c2, 0f, "confidence should be exactly deterministic");
        }

        private void Test_Forward_CacheMatchesForward_WithText()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 5);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 2, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));
            var cache = new MultimodalForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers);

            var (pForward, cForward) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            var (pCache, cCache) = model.ForwardWithCache(data.Texts[0], data.PriceInputs[0], cache);

            AssertMatrixClose(pForward, pCache, StrictTol, "cached predictions differ from normal Forward");
            AssertMatrixClose(cForward, cCache, StrictTol, "cached confidence differs from normal Forward");
            Assert(cache.TextFinalHidden != null, "expected text cache to be populated");
            Assert(cache.PriceFinalHidden != null, "expected price cache to be populated");
        }

        private void Test_Forward_CacheMatchesForward_PriceOnly()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 6);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 2, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));
            var cache = new MultimodalForwardCache(cfg.Text.NumLayers, cfg.Price.NumLayers);

            var (pForward, cForward) = model.Forward(null, data.PriceInputs[0]);
            var (pCache, cCache) = model.ForwardWithCache(null, data.PriceInputs[0], cache);

            AssertMatrixClose(pForward, pCache, StrictTol, "cached price-only predictions differ from normal Forward");
            AssertMatrixClose(cForward, cCache, StrictTol, "cached price-only confidence differs from normal Forward");
            Assert(cache.TextFinalHidden == null, "text cache should be null for price-only forward");
            Assert(cache.PriceBlockCaches[0].CrossQ == null, "cross-attention cache should be null when text is absent");
        }

        private void Test_Forward_ConfidenceRange()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 7);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));

            for (int s = 0; s < data.Texts.Length; s++)
            {
                var (_, conf) = model.Forward(data.Texts[s], data.PriceInputs[s]);
                Assert(conf != null, "confidence should not be null");
                for (int i = 0; i < conf.GetLength(0); i++)
                    Assert(conf[i, 0] >= 0f && conf[i, 0] <= 1f, $"confidence[{i}] out of range: {conf[i, 0]}");
            }
        }

        private void Test_Forward_NoConfidenceHeadReturnsNull()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 8);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: false), new Random(42));

            var (_, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            Assert(conf == null, "confidence should be null when confidence head is disabled");
            Assert(model.ConfidenceProjection == null, "confidence projection should be null when disabled");
            Assert(model.ConfidenceBias == null, "confidence bias should be null when disabled");
        }

        private void Test_PredictNext_EqualsLastForwardRow_WithConfidence()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 9);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));

            var (matrixPred, matrixConf) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            var (nextPred, nextConf) = model.PredictNext(data.Texts[0], data.PriceInputs[0]);
            int last = matrixPred.GetLength(0) - 1;

            Assert(nextPred.Length == matrixPred.GetLength(1), "PredictNext output dim");
            for (int j = 0; j < nextPred.Length; j++)
                AssertClose(matrixPred[last, j], nextPred[j], StrictTol, $"PredictNext pred[{j}] mismatch");
            AssertClose(matrixConf[last, 0], nextConf, StrictTol, "PredictNext confidence mismatch");
        }

        private void Test_PredictNext_NoConfidenceHeadReturnsOneConfidence()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 10);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: false), new Random(42));

            var (prediction, confidence) = model.PredictNext(data.Texts[0], data.PriceInputs[0]);

            Assert(prediction.Length == 5, "PredictNext wrong output dim");
            AssertClose(1f, confidence, 0f, "confidence should be 1 when confidence head is disabled");
        }

        private void Test_Forward_TextVsNoTextDiffer()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 11);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            var (pWithText, _) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            var (pNoText, _) = model.Forward(null, data.PriceInputs[0]);

            Assert(MatrixChanged(pWithText, pNoText, 1e-7f), "text-conditioned and price-only outputs were identical");
        }

        private void Test_Forward_DifferentTextsUsuallyDiffer()
        {
            string[] corpus = { "bullish earnings growth", "bearish recession warning" };
            var tok = NewTokenizer(corpus);
            var bullish = tok.Encode(corpus[0], addSpecialTokens: true);
            var bearish = tok.Encode(corpus[1], addSpecialTokens: true);
            var price = MakeSmoothMatrix(seqLen: 6, features: 5, sampleIndex: 0, seed: 12);
            var model = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 6), new Random(42));

            var (pBull, _) = model.Forward(bullish, price);
            var (pBear, _) = model.Forward(bearish, price);

            Assert(MatrixChanged(pBull, pBear, 1e-7f), "different text inputs produced identical predictions");
        }

        private void Test_Forward_VaryingTextLengths()
        {
            string[] corpus =
            {
                "hi",
                "the stock market rallied strongly today on positive earnings news",
                "bullish"
            };
            var tok = NewTokenizer(corpus);
            var model = new CrossModel(MakeCfg(tok.VocabSize + 2, priceSeqLen: 6, textSeqLen: 32), new Random(42));
            var price = MakeSmoothMatrix(seqLen: 6, features: 5, sampleIndex: 0, seed: 13);

            foreach (string sentence in corpus)
            {
                var (pred, conf) = model.Forward(tok.Encode(sentence, addSpecialTokens: true), price);
                AssertDims(pred, 6, 5, $"prediction for '{sentence}'");
                AssertFinite(pred, "prediction");
                AssertFinite(conf, "confidence");
            }
        }

        private void Test_Forward_VaryingPriceSequenceLengths()
        {
            var data = SmoothData(n: 1, seqLen: 9, seed: 14);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 9), new Random(42));

            foreach (int len in new[] { 2, 3, 5, 9 })
            {
                var price = SliceRows(data.PriceInputs[0], 0, len);
                var (pred, conf) = model.Forward(data.Texts[0], price);
                AssertDims(pred, len, 5, $"prediction len {len}");
                AssertDims(conf, len, 1, $"confidence len {len}");
                AssertFinite(pred, $"prediction len {len}");
            }
        }

        private void Test_Forward_MaxConfiguredPriceLengthWorks()
        {
            var data = SmoothData(n: 1, seqLen: 10, seed: 15);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 10), new Random(42));

            var (pred, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            AssertDims(pred, 10, 5, "prediction at max price length");
            AssertDims(conf, 10, 1, "confidence at max price length");
        }

        private void Test_Forward_CausalPriceMaskBlocksFutureLeakage()
        {
            var data = SmoothData(n: 1, seqLen: 8, seed: 16);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 8, priceDecoderOnly: true);
            var model = new CrossModel(cfg, new Random(42));

            var priceA = CloneMatrix(data.PriceInputs[0]);
            var priceB = CloneMatrix(priceA);
            for (int t = 4; t < priceB.GetLength(0); t++)
                for (int f = 0; f < priceB.GetLength(1); f++)
                    priceB[t, f] += 1000f + 10f * t + f;

            var (predA, confA) = model.Forward(data.Texts[0], priceA);
            var (predB, confB) = model.Forward(data.Texts[0], priceB);

            for (int t = 0; t < 4; t++)
            {
                for (int j = 0; j < predA.GetLength(1); j++)
                    AssertClose(predA[t, j], predB[t, j], 1e-5f, $"future price leaked into pred[{t},{j}]");
                AssertClose(confA[t, 0], confB[t, 0], 1e-5f, $"future price leaked into conf[{t}]");
            }
        }

        private void Test_Forward_InvalidTokenIdThrows()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 17);
            var cfg = MakeCfg(vocabSize: 20, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));
            int[] badTokens = { 0, 1, cfg.Text.VocabSize + 5 };

            ExpectThrowsAny(() => model.Forward(badTokens, data.PriceInputs[0]), "invalid token id should throw");
        }

        private void Test_Forward_TextTooLongThrows()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 18);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, textSeqLen: 4);
            var model = new CrossModel(cfg, new Random(42));
            int[] tooLong = Enumerable.Repeat(1, cfg.Text.MaxSequenceLength + 1).ToArray();

            ExpectThrowsAny(() => model.Forward(tooLong, data.PriceInputs[0]), "text longer than configured max should throw");
        }

        private void Test_Forward_PriceTooLongThrows()
        {
            var data = SmoothData(n: 1, seqLen: 9, seed: 19);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));

            ExpectThrowsAny(() => model.Forward(data.Texts[0], data.PriceInputs[0]), "price sequence longer than configured max should throw");
        }

        private void Test_Forward_PriceFeatureMismatchThrows()
        {
            var data = SmoothData(n: 1, seqLen: 6, features: 4, outputDim: 5, seed: 20);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceFeatures: 5, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));

            ExpectThrowsAny(() => model.Forward(data.Texts[0], data.PriceInputs[0]), "price feature dimension mismatch should throw");
        }

        private void Test_PredictNext_EmptyPriceSequenceThrows()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 21);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));
            var emptyPrice = new float[0, cfg.Price.InputFeatureDim];

            ExpectThrowsAny(() => model.PredictNext(data.Texts[0], emptyPrice), "PredictNext should reject empty price sequence");
        }

        private void Test_Forward_NonFiniteInputThrows()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 22);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));
            data.PriceInputs[0][2, 1] = float.NaN;

            ExpectThrowsAny(() => model.Forward(data.Texts[0], data.PriceInputs[0]), "Forward should reject NaN price input");
        }

        // ---------------------------------------------------------------------
        // Validation
        // ---------------------------------------------------------------------

        private void Test_Validate_ReturnsFiniteNonNegativeLoss()
        {
            var data = SmoothData(n: 5, seqLen: 7, seed: 23);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 7), new Random(42));
            var trainer = new CrossTrainer(model, TC(epochs: 1));

            float loss = trainer.Validate(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertFinite(loss, "validation loss");
            Assert(loss >= 0f, $"validation loss should be non-negative: {loss}");
        }

        private void Test_Validate_DoesNotMutateWeights()
        {
            var data = SmoothData(n: 3, seqLen: 6, seed: 24);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var beforeOutput = CloneMatrix(model.OutputProjection);
            var beforePriceInput = CloneMatrix(model.PriceInputProjection);
            var trainer = new CrossTrainer(model, TC(epochs: 1));

            _ = trainer.Validate(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertMatrixClose(beforeOutput, model.OutputProjection, 0f, "Validate changed OutputProjection");
            AssertMatrixClose(beforePriceInput, model.PriceInputProjection, 0f, "Validate changed PriceInputProjection");
        }

        private void Test_Validate_ConfidenceTargetsAffectLossWhenWeighted()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 25);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));
            var trainer = new CrossTrainer(model, TC(epochs: 1, confWeight: 1.0f));
            var zeros = MakeConfidenceTargets(data.PriceInputs.Length, data.PriceInputs[0].GetLength(0), 0f);
            var ones = MakeConfidenceTargets(data.PriceInputs.Length, data.PriceInputs[0].GetLength(0), 1f);

            float lossZeros = trainer.Validate(data.Texts, data.PriceInputs, data.PriceTargets, zeros);
            float lossOnes = trainer.Validate(data.Texts, data.PriceInputs, data.PriceTargets, ones);

            Assert(MathF.Abs(lossZeros - lossOnes) > 1e-6f, $"confidence targets should affect validation loss when weighted, zero={lossZeros}, one={lossOnes}");
        }

        private void Test_Validate_ConfidenceTargetsIgnoredWhenWeightZero()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 26);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));
            var trainer = new CrossTrainer(model, TC(epochs: 1, confWeight: 0.0f));
            var zeros = MakeConfidenceTargets(data.PriceInputs.Length, data.PriceInputs[0].GetLength(0), 0f);
            var ones = MakeConfidenceTargets(data.PriceInputs.Length, data.PriceInputs[0].GetLength(0), 1f);

            float lossZeros = trainer.Validate(data.Texts, data.PriceInputs, data.PriceTargets, zeros);
            float lossOnes = trainer.Validate(data.Texts, data.PriceInputs, data.PriceTargets, ones);

            AssertClose(lossZeros, lossOnes, 1e-6f, "confidence targets should not affect loss when confidence weight is zero");
        }

        private void Test_Validate_LengthMismatchThrows()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 27);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var trainer = new CrossTrainer(model, TC(epochs: 1));
            var shorterTargets = data.PriceTargets.Take(3).ToArray();

            ExpectThrowsAny(() => trainer.Validate(data.Texts, data.PriceInputs, shorterTargets, data.ConfidenceTargets), "Validate should reject mismatched dataset lengths");
        }

        // ---------------------------------------------------------------------
        // Training / learning behaviour
        // ---------------------------------------------------------------------

        private void Test_Train_LossDecreases_OnSmoothNextStepData()
        {
            var data = SmoothData(n: 16, seqLen: 8, seed: 28);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 1, priceSeqLen: 8), new Random(42));

            float before = new CrossTrainer(model, TC(epochs: 1)).Validate(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            new CrossTrainer(model, TC(lr: 0.003f, bs: 4, epochs: 30)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            float after = new CrossTrainer(model, TC(epochs: 1)).Validate(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertLossLower(before, after, "smooth next-step training should reduce validation loss");
        }

        private void Test_Train_PriceOnlyLossDecreases()
        {
            var data = SmoothData(n: 16, seqLen: 8, seed: 29);
            var nullTexts = new int[data.Texts.Length][];
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 1, priceSeqLen: 8), new Random(42));

            float before = new CrossTrainer(model, TC(epochs: 1)).Validate(nullTexts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            new CrossTrainer(model, TC(lr: 0.003f, bs: 4, epochs: 30)).Train(nullTexts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            float after = new CrossTrainer(model, TC(epochs: 1)).Validate(nullTexts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertLossLower(before, after, "price-only training should reduce validation loss");
        }

        private void Test_Train_SingleSampleOverfitsConstantTarget()
        {
            var data = ConstantTargetData(n: 1, seqLen: 5, features: 3, outputDim: 2, targetValue: 0.25f, seed: 30);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, priceFeatures: 3, outputDim: 2, priceSeqLen: 5, useConf: false);
            var model = new CrossModel(cfg, new Random(42));

            float before = new CrossTrainer(model, TC(epochs: 1)).Validate(data.Texts, data.PriceInputs, data.PriceTargets);
            new CrossTrainer(model, TC(lr: 0.01f, bs: 1, epochs: 120, confWeight: 0f)).Train(data.Texts, data.PriceInputs, data.PriceTargets);
            float after = new CrossTrainer(model, TC(epochs: 1)).Validate(data.Texts, data.PriceInputs, data.PriceTargets);

            Assert(after < before * 0.60f || after < 0.01f, $"single-sample overfit did not improve enough: before={before}, after={after}");
        }

        private void Test_Train_BatchSizeOneWorks()
        {
            var data = SmoothData(n: 5, seqLen: 6, seed: 31);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            new CrossTrainer(model, TC(lr: 0.002f, bs: 1, epochs: 3)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            var (pred, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            AssertFinite(pred, "prediction after batch-size-1 training");
            AssertFinite(conf, "confidence after batch-size-1 training");
        }

        private void Test_Train_BatchSizeGreaterThanSampleCountWorks()
        {
            var data = SmoothData(n: 3, seqLen: 6, seed: 32);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            new CrossTrainer(model, TC(lr: 0.002f, bs: 50, epochs: 3)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            var (pred, _) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            AssertFinite(pred, "prediction after oversized batch training");
        }

        private void Test_Train_ZeroEpochsDoesNotChangeWeights()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 33);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var before = CloneMatrix(model.OutputProjection);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 0)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertMatrixClose(before, model.OutputProjection, 0f, "zero epochs should not update OutputProjection");
        }

        private void Test_Train_LearningRateDecayRunsFinite()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 34);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var tc = TC(lr: 0.01f, bs: 4, epochs: 8);
            tc.UseLearningRateDecay = true;
            tc.LearningRateDecay = 0.85f;

            new CrossTrainer(model, tc).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            var (pred, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            AssertFinite(pred, "prediction after LR decay training");
            AssertFinite(conf, "confidence after LR decay training");
        }

        private void Test_Train_GradientClippingHighLearningRateStaysFinite()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 35);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            new CrossTrainer(model, TC(lr: 0.15f, bs: 4, epochs: 5, clip: 0.25f)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            var (pred, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            AssertFinite(pred, "prediction after high-LR clipped training");
            AssertFinite(conf, "confidence after high-LR clipped training");
        }

        private void Test_Train_MixedNullTextBatchWorks()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 36);
            var mixed = (int[][])data.Texts.Clone();
            for (int i = 0; i < mixed.Length; i += 2)
                mixed[i] = null;
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            new CrossTrainer(model, TC(lr: 0.002f, bs: 4, epochs: 4)).Train(mixed, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            float loss = new CrossTrainer(model, TC(epochs: 1)).Validate(mixed, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertFinite(loss, "loss after mixed null text training");
        }

        private void Test_Train_NoConfidenceHeadWorks()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 37);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: false), new Random(42));

            new CrossTrainer(model, TC(lr: 0.003f, bs: 4, epochs: 5, confWeight: 0f)).Train(data.Texts, data.PriceInputs, data.PriceTargets);
            var (pred, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);

            Assert(conf == null, "confidence should remain null after no-confidence training");
            AssertFinite(pred, "prediction after no-confidence training");
        }

        private void Test_Train_OutputHeadUpdates()
        {
            var data = SmoothData(n: 6, seqLen: 6, seed: 38);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var beforeW = CloneMatrix(model.OutputProjection);
            var beforeB = CloneVector(model.OutputBias);

            new CrossTrainer(model, TC(lr: 0.005f, bs: 3, epochs: 4)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            Assert(MatrixChanged(beforeW, model.OutputProjection, ChangedTol), "OutputProjection did not update");
            Assert(VectorChanged(beforeB, model.OutputBias, ChangedTol), "OutputBias did not update");
        }

        private void Test_Train_ConfidenceHeadUpdatesWhenWeighted()
        {
            var data = SmoothData(n: 6, seqLen: 6, seed: 39);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));
            var beforeW = CloneMatrix(model.ConfidenceProjection);
            var beforeB = CloneVector(model.ConfidenceBias);
            var confTargets = MakeConfidenceTargets(data.PriceInputs.Length, data.PriceInputs[0].GetLength(0), 1f);

            new CrossTrainer(model, TC(lr: 0.005f, bs: 3, epochs: 4, confWeight: 1f)).Train(data.Texts, data.PriceInputs, data.PriceTargets, confTargets);

            Assert(MatrixChanged(beforeW, model.ConfidenceProjection, ChangedTol), "ConfidenceProjection did not update");
            Assert(VectorChanged(beforeB, model.ConfidenceBias, ChangedTol), "ConfidenceBias did not update");
        }

        private void Test_Train_ConfidenceLossWeightZeroDoesNotUpdateConfidenceHead()
        {
            var data = SmoothData(n: 6, seqLen: 6, seed: 40);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));
            var beforeW = CloneMatrix(model.ConfidenceProjection);
            var beforeB = CloneVector(model.ConfidenceBias);
            var confTargets = MakeConfidenceTargets(data.PriceInputs.Length, data.PriceInputs[0].GetLength(0), 0f);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 3, epochs: 5, confWeight: 0f)).Train(data.Texts, data.PriceInputs, data.PriceTargets, confTargets);

            AssertMatrixClose(beforeW, model.ConfidenceProjection, 0f, "ConfidenceProjection changed even though ConfidenceLossWeight=0");
            AssertVectorClose(beforeB, model.ConfidenceBias, 0f, "ConfidenceBias changed even though ConfidenceLossWeight=0");
        }

        private void Test_Train_ExplicitConfidenceTargetsCanIncreaseConfidence()
        {
            var data = ConstantTargetData(n: 8, seqLen: 6, features: 5, outputDim: 5, targetValue: 0.0f, seed: 41);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));
            var confTargets = MakeConfidenceTargets(data.PriceInputs.Length, data.PriceInputs[0].GetLength(0), 1f);

            float before = MeanConfidence(model, data.Texts, data.PriceInputs);
            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 12, confWeight: 2f)).Train(data.Texts, data.PriceInputs, data.PriceTargets, confTargets);
            float after = MeanConfidence(model, data.Texts, data.PriceInputs);

            Assert(after > before + 0.01f, $"confidence did not increase enough: before={before}, after={after}");
        }

        private void Test_Train_FrozenTextEncoderUnchanged()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 42);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, freezeText: true), new Random(42));
            var beforeEmb = CloneMatrix(model.TextTokenEmbedding);
            var beforeWQ = CloneMatrix(model.TextBlocks[0].Attention.WQ);
            var beforeGamma = CloneVector(model.TextBlocks[0].LN1Gamma);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertMatrixClose(beforeEmb, model.TextTokenEmbedding, 0f, "frozen TextTokenEmbedding changed");
            AssertMatrixClose(beforeWQ, model.TextBlocks[0].Attention.WQ, 0f, "frozen text attention changed");
            AssertVectorClose(beforeGamma, model.TextBlocks[0].LN1Gamma, 0f, "frozen text layernorm changed");
        }

        private void Test_Train_UnfrozenTextEncoderUpdatesWithText()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 43);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, freezeText: false), new Random(42));
            var beforeEmb = CloneMatrix(model.TextTokenEmbedding);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            Assert(MatrixChanged(beforeEmb, model.TextTokenEmbedding, ChangedTol), "unfrozen TextTokenEmbedding did not update");
        }

        private void Test_Train_UnfrozenTextEncoderUnchangedWhenAllTextNull()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 44);
            var nullTexts = new int[data.Texts.Length][];
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, freezeText: false), new Random(42));
            var beforeEmb = CloneMatrix(model.TextTokenEmbedding);
            var beforeWQ = CloneMatrix(model.TextBlocks[0].Attention.WQ);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(nullTexts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertMatrixClose(beforeEmb, model.TextTokenEmbedding, 0f, "text embedding changed even though all text was null");
            AssertMatrixClose(beforeWQ, model.TextBlocks[0].Attention.WQ, 0f, "text attention changed even though all text was null");
        }

        private void Test_Train_CrossAttentionUpdatesWhenTextPresent()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 45);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var beforeWQ = CloneMatrix(model.PriceBlocks[0].CrossAttention.WQ);
            var beforeWK = CloneMatrix(model.PriceBlocks[0].CrossAttention.WK);
            var beforeWV = CloneMatrix(model.PriceBlocks[0].CrossAttention.WV);
            var beforeWO = CloneMatrix(model.PriceBlocks[0].CrossAttention.WO);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            Assert(MatrixChanged(beforeWQ, model.PriceBlocks[0].CrossAttention.WQ, ChangedTol), "CrossAttention.WQ did not update");
            Assert(MatrixChanged(beforeWK, model.PriceBlocks[0].CrossAttention.WK, ChangedTol), "CrossAttention.WK did not update");
            Assert(MatrixChanged(beforeWV, model.PriceBlocks[0].CrossAttention.WV, ChangedTol), "CrossAttention.WV did not update");
            Assert(MatrixChanged(beforeWO, model.PriceBlocks[0].CrossAttention.WO, ChangedTol), "CrossAttention.WO did not update");
        }

        private void Test_Train_CrossAttentionWeightsUnchangedWhenAllTextNull()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 46);
            var nullTexts = new int[data.Texts.Length][];
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var beforeWQ = CloneMatrix(model.PriceBlocks[0].CrossAttention.WQ);
            var beforeWK = CloneMatrix(model.PriceBlocks[0].CrossAttention.WK);
            var beforeWV = CloneMatrix(model.PriceBlocks[0].CrossAttention.WV);
            var beforeWO = CloneMatrix(model.PriceBlocks[0].CrossAttention.WO);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(nullTexts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            AssertMatrixClose(beforeWQ, model.PriceBlocks[0].CrossAttention.WQ, 0f, "CrossAttention.WQ changed without text");
            AssertMatrixClose(beforeWK, model.PriceBlocks[0].CrossAttention.WK, 0f, "CrossAttention.WK changed without text");
            AssertMatrixClose(beforeWV, model.PriceBlocks[0].CrossAttention.WV, 0f, "CrossAttention.WV changed without text");
            AssertMatrixClose(beforeWO, model.PriceBlocks[0].CrossAttention.WO, 0f, "CrossAttention.WO changed without text");
        }

        private void Test_Train_PriceSelfAttentionUpdates()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 47);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var before = CloneMatrix(model.PriceBlocks[0].SelfAttention.WQ);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            Assert(MatrixChanged(before, model.PriceBlocks[0].SelfAttention.WQ, ChangedTol), "SelfAttention.WQ did not update");
        }

        private void Test_Train_PriceInputProjectionUpdates()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 48);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var beforeW = CloneMatrix(model.PriceInputProjection);
            var beforeB = CloneVector(model.PriceInputProjectionBias);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            Assert(MatrixChanged(beforeW, model.PriceInputProjection, ChangedTol), "PriceInputProjection did not update");
            Assert(VectorChanged(beforeB, model.PriceInputProjectionBias, ChangedTol), "PriceInputProjectionBias did not update");
        }

        private void Test_Train_PriceLayerNormUpdates()
        {
            var data = SmoothData(n: 8, seqLen: 6, seed: 49);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var beforeSelfGamma = CloneVector(model.PriceBlocks[0].LNSelfGamma);
            var beforeCrossGamma = CloneVector(model.PriceBlocks[0].LNCrossGamma);
            var beforeFfnGamma = CloneVector(model.PriceBlocks[0].LNFFNGamma);

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 5)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            Assert(VectorChanged(beforeSelfGamma, model.PriceBlocks[0].LNSelfGamma, ChangedTol), "LNSelfGamma did not update");
            Assert(VectorChanged(beforeCrossGamma, model.PriceBlocks[0].LNCrossGamma, ChangedTol), "LNCrossGamma did not update");
            Assert(VectorChanged(beforeFfnGamma, model.PriceBlocks[0].LNFFNGamma, ChangedTol), "LNFFNGamma did not update");
        }

        private void Test_Train_TextSignalCanBeLearnedForIdenticalPriceInputs()
        {
            var data = TextSignalData(repeatsPerClass: 8, seqLen: 5, seed: 50);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 1, ffnDim: 32, priceFeatures: 1, outputDim: 1, priceSeqLen: 5, useConf: false);
            var model = new CrossModel(cfg, new Random(42));

            new CrossTrainer(model, TC(lr: 0.01f, bs: 4, epochs: 120, confWeight: 0f)).Train(data.Texts, data.PriceInputs, data.PriceTargets);

            var (bullPred, _) = model.Forward(data.BullishTokens, data.SharedPrice);
            var (bearPred, _) = model.Forward(data.BearishTokens, data.SharedPrice);
            float bullMean = MeanMatrix(bullPred);
            float bearMean = MeanMatrix(bearPred);

            Assert(bullMean > bearMean + 0.15f, $"text signal was not learned strongly enough: bullish={bullMean}, bearish={bearMean}");
        }

        private void Test_Train_RepeatedTrainingKeepsPredictionsFinite()
        {
            var data = SmoothData(n: 10, seqLen: 6, seed: 51);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));

            for (int i = 0; i < 3; i++)
                new CrossTrainer(model, TC(lr: 0.005f, bs: 5, epochs: 8)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);

            var (pred, conf) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            AssertFinite(pred, "prediction after repeated training");
            AssertFinite(conf, "confidence after repeated training");
        }

        private void Test_Train_ArrayLengthMismatchThrows()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 52);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            var trainer = new CrossTrainer(model, TC(epochs: 1));
            var fewerPriceInputs = data.PriceInputs.Take(3).ToArray();

            ExpectThrowsAny(() => trainer.Train(data.Texts, fewerPriceInputs, data.PriceTargets, data.ConfidenceTargets), "Train should reject mismatched dataset lengths");
        }

        private void Test_Train_InvalidTokenIdThrows()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 53);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));
            data.Texts[0] = new[] { cfg.Text.VocabSize + 10 };

            ExpectThrowsAny(() => new CrossTrainer(model, TC(epochs: 1)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets), "Train should reject invalid token id and not silently skip batch");
        }

        private void Test_Train_PriceFeatureMismatchThrows()
        {
            var data = SmoothData(n: 4, seqLen: 6, features: 4, outputDim: 5, seed: 54);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceFeatures: 5, outputDim: 5, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));

            ExpectThrowsAny(() => new CrossTrainer(model, TC(epochs: 1)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets), "Train should reject price feature mismatch and not silently skip batch");
        }

        private void Test_Train_TargetOutputDimMismatchThrows()
        {
            var data = SmoothData(n: 4, seqLen: 6, features: 5, outputDim: 4, seed: 55);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceFeatures: 5, outputDim: 5, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));

            ExpectThrowsAny(() => new CrossTrainer(model, TC(epochs: 1)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets), "Train should reject target output dim mismatch and not silently skip batch");
        }

        private void Test_Train_ConfidenceTargetLengthMismatchThrows()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 56);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true);
            var model = new CrossModel(cfg, new Random(42));
            data.ConfidenceTargets[0] = new float[1];

            ExpectThrowsAny(() => new CrossTrainer(model, TC(epochs: 1, confWeight: 1f)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets), "Train should reject short confidence target arrays");
        }

        private void Test_Train_NonFiniteInputThrows()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 57);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6);
            var model = new CrossModel(cfg, new Random(42));
            data.PriceInputs[0][1, 2] = float.PositiveInfinity;

            ExpectThrowsAny(() => new CrossTrainer(model, TC(epochs: 1)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets), "Train should reject non-finite input and not silently skip batch");
        }

        // ---------------------------------------------------------------------
        // Save / load
        // ---------------------------------------------------------------------

        private void Test_SaveLoad_TrainedWeightsAndForwardIdentical()
        {
            var data = SmoothData(n: 6, seqLen: 6, seed: 58);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 2, priceSeqLen: 6, useConf: true);
            var model = new CrossModel(cfg, new Random(42));
            new CrossTrainer(model, TC(lr: 0.003f, bs: 3, epochs: 5)).Train(data.Texts, data.PriceInputs, data.PriceTargets, data.ConfidenceTargets);
            var (pBefore, cBefore) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            string dir = NewTempDir();

            try
            {
                model.Save(dir);
                var loaded = CrossModel.Load(dir);
                var (pAfter, cAfter) = loaded.Forward(data.Texts[0], data.PriceInputs[0]);

                AssertMatrixClose(pBefore, pAfter, SaveLoadTol, "predictions changed after load");
                AssertMatrixClose(cBefore, cAfter, SaveLoadTol, "confidence changed after load");
            }
            finally
            {
                DeleteTempDir(dir);
            }
        }

        private void Test_SaveLoad_NoConfidenceHeadRoundTrips()
        {
            var data = SmoothData(n: 3, seqLen: 6, seed: 59);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: false), new Random(42));
            var (pBefore, confBefore) = model.Forward(data.Texts[0], data.PriceInputs[0]);
            Assert(confBefore == null, "confidence should be null before save");
            string dir = NewTempDir();

            try
            {
                model.Save(dir);
                var loaded = CrossModel.Load(dir);
                var (pAfter, confAfter) = loaded.Forward(data.Texts[0], data.PriceInputs[0]);

                Assert(confAfter == null, "confidence should be null after load");
                AssertMatrixClose(pBefore, pAfter, SaveLoadTol, "no-confidence predictions changed after load");
            }
            finally
            {
                DeleteTempDir(dir);
            }
        }

        private void Test_SaveLoad_PredictNextIdentical()
        {
            var data = SmoothData(n: 4, seqLen: 6, seed: 60);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6, useConf: true), new Random(42));
            var (beforePred, beforeConf) = model.PredictNext(data.Texts[0], data.PriceInputs[0]);
            string dir = NewTempDir();

            try
            {
                model.Save(dir);
                var loaded = CrossModel.Load(dir);
                var (afterPred, afterConf) = loaded.PredictNext(data.Texts[0], data.PriceInputs[0]);

                AssertVectorClose(beforePred, afterPred, SaveLoadTol, "PredictNext prediction changed after load");
                AssertClose(beforeConf, afterConf, SaveLoadTol, "PredictNext confidence changed after load");
            }
            finally
            {
                DeleteTempDir(dir);
            }
        }

        private void Test_SaveLoad_ConfigSwitchesPreserved()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 61);
            var cfg = MakeCfg(data.Tokenizer.VocabSize + 2, embDim: 16, numHeads: 2, numLayers: 2, priceSeqLen: 6, useConf: true, freezeText: true, priceDecoderOnly: true, textDecoderOnly: true);
            var model = new CrossModel(cfg, new Random(42));
            string dir = NewTempDir();

            try
            {
                model.Save(dir);
                var loaded = CrossModel.Load(dir);

                Assert(loaded.Config.Text.Freeze == true, "FreezeTextEncoder not preserved");
                Assert(loaded.Config.Text.UseDecoderOnly == true, "Text.UseDecoderOnly not preserved");
                Assert(loaded.Config.Price.UseDecoderOnly == true, "Price.UseDecoderOnly not preserved");
                Assert(loaded.Config.Output.UseConfidenceHead == true, "UseConfidenceHead not preserved");
                Assert(loaded.Config.Text.NumLayers == 2, "Text.NumLayers not preserved");
                Assert(loaded.Config.Price.NumLayers == 2, "Price.NumLayers not preserved");
            }
            finally
            {
                DeleteTempDir(dir);
            }
        }

        private void Test_Save_CreatesExpectedFiles()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 62);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, numLayers: 2, priceSeqLen: 6), new Random(42));
            string dir = NewTempDir();

            try
            {
                model.Save(dir);

                Assert(File.Exists(Path.Combine(dir, "config.json")), "config.json not created");
                Assert(File.Exists(Path.Combine(dir, "weights.bin")), "weights.bin not created");
                Assert(Directory.Exists(Path.Combine(dir, "text_ffn_0")), "text_ffn_0 not created");
                Assert(Directory.Exists(Path.Combine(dir, "text_ffn_1")), "text_ffn_1 not created");
                Assert(Directory.Exists(Path.Combine(dir, "price_ffn_0")), "price_ffn_0 not created");
                Assert(Directory.Exists(Path.Combine(dir, "price_ffn_1")), "price_ffn_1 not created");
            }
            finally
            {
                DeleteTempDir(dir);
            }
        }

        private void Test_Load_MissingWeightsThrows()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 63);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            string dir = NewTempDir();

            try
            {
                model.Save(dir);
                File.Delete(Path.Combine(dir, "weights.bin"));
                ExpectThrowsAny(() => CrossModel.Load(dir), "Load should throw when weights.bin is missing");
            }
            finally
            {
                DeleteTempDir(dir);
            }
        }

        private void Test_Load_CorruptWeightsThrows()
        {
            var data = SmoothData(n: 1, seqLen: 6, seed: 64);
            var model = new CrossModel(MakeCfg(data.Tokenizer.VocabSize + 2, priceSeqLen: 6), new Random(42));
            string dir = NewTempDir();

            try
            {
                model.Save(dir);
                string weightsPath = Path.Combine(dir, "weights.bin");
                using (var stream = new FileStream(weightsPath, FileMode.Open, FileAccess.ReadWrite))
                using (var writer = new BinaryWriter(stream))
                {
                    writer.Write(model.TextTokenEmbedding.GetLength(0) + 12345); // corrupt first matrix row count
                }

                ExpectThrowsAny(() => CrossModel.Load(dir), "Load should throw on corrupt weight matrix shape");
            }
            finally
            {
                DeleteTempDir(dir);
            }
        }

        // ---------------------------------------------------------------------
        // Config / data helpers
        // ---------------------------------------------------------------------

        private MultimodalTransformerConfig MakeCfg(
            int vocabSize,
            int embDim = 16,
            int numHeads = 2,
            int numLayers = 1,
            int ffnDim = 32,
            int priceFeatures = 5,
            int outputDim = 5,
            int priceSeqLen = 10,
            int textSeqLen = 32,
            bool useConf = true,
            bool freezeText = false,
            bool priceDecoderOnly = false,
            bool textDecoderOnly = false)
        {
            var cfg = new MultimodalTransformerConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = vocabSize,
                    MaxSequenceLength = textSeqLen,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    UseDecoderOnly = textDecoderOnly,
                    Freeze = freezeText
                },
                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = priceFeatures,
                    MaxSequenceLength = priceSeqLen,
                    EmbeddingDim = embDim,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    FeedForwardDim = ffnDim,
                    UseDecoderOnly = priceDecoderOnly
                },
                Output = new OutputHeadConfig
                {
                    OutputDim = outputDim,
                    UseConfidenceHead = useConf
                },
                Runtime = new RuntimeConfig
                {
                    FFNActivationType = ActivationType.Relu,
                    AccelerationType = AccelerationType.CPU
                },
                Regularization = new RegularizationConfig
                {
                    L2RegulationLamda = 0f,
                    GradientClippingThreshold = 1f
                },
                RequireSharedCrossAttentionEmbeddingDim = true,
            };

            cfg.Validate();
            return cfg;
        }

        private MultimodalTransformerConfig MakeCfgNoValidate(
            int vocabSize,
            int textEmbDim,
            int priceEmbDim,
            int numHeads,
            int outputDim = 5)
        {
            return new MultimodalTransformerConfig
            {
                Text = new TextEncoderConfig
                {
                    VocabSize = vocabSize,
                    MaxSequenceLength = 32,
                    EmbeddingDim = textEmbDim,
                    NumHeads = numHeads,
                    NumLayers = 1,
                    FeedForwardDim = 32,
                },
                Price = new PriceDecoderConfig
                {
                    InputFeatureDim = 5,
                    MaxSequenceLength = 10,
                    EmbeddingDim = priceEmbDim,
                    NumHeads = numHeads,
                    NumLayers = 1,
                    FeedForwardDim = 32,
                },
                Output = new OutputHeadConfig
                {
                    OutputDim = outputDim,
                    UseConfidenceHead = true
                },
                Runtime = new RuntimeConfig
                {
                    FFNActivationType = ActivationType.Relu,
                    AccelerationType = AccelerationType.CPU
                },
                Regularization = new RegularizationConfig
                {
                    L2RegulationLamda = 0f,
                    GradientClippingThreshold = 1f
                },
                RequireSharedCrossAttentionEmbeddingDim = true,
            };
        }

        private TrainingConfig TC(
            float lr = 0.001f,
            int bs = 4,
            int epochs = 10,
            float confWeight = 0.1f,
            float clip = 1f)
        {
            return new TrainingConfig
            {
                LearningRate = lr,
                BatchSize = bs,
                Epochs = epochs,
                UseGradientClipping = true,
                GradientClipThreshold = clip,
                ConfidenceLossWeight = confWeight,
                Verbose = false
            };
        }

        private class DataSet
        {
            public BPETokenizer Tokenizer { get; set; }
            public int[][] Texts { get; set; }
            public float[][,] PriceInputs { get; set; }
            public float[][,] PriceTargets { get; set; }
            public float[][] ConfidenceTargets { get; set; }
        }

        private sealed class TextSignalDataSet : DataSet
        {
            public int[] BullishTokens { get; set; }
            public int[] BearishTokens { get; set; }
            public float[,] SharedPrice { get; set; }
        }

        private DataSet SmoothData(int n = 10, int seqLen = 8, int features = 5, int outputDim = 5, int seed = 42)
        {
            var rng = new Random(seed);
            string[] corpus =
            {
                "stock rose sharply after strong earnings",
                "market crashed on bearish data",
                "bullish outlook for revenue growth",
                "bearish guidance and weak demand",
                "neutral inflation report before open"
            };

            var tok = NewTokenizer(corpus);
            var texts = new int[n][];
            var priceInputs = new float[n][,];
            var priceTargets = new float[n][,];
            var confTargets = new float[n][];

            for (int s = 0; s < n; s++)
            {
                int textIndex = rng.Next(corpus.Length);
                texts[s] = tok.Encode(corpus[textIndex], addSpecialTokens: true);
                priceInputs[s] = MakeSmoothMatrix(seqLen, features, s, seed);
                priceTargets[s] = new float[seqLen, outputDim];
                confTargets[s] = new float[seqLen];

                float textSignal = corpus[textIndex].Contains("bullish") || corpus[textIndex].Contains("rose") ? 0.03f : -0.03f;
                if (corpus[textIndex].Contains("neutral")) textSignal = 0f;

                for (int t = 0; t < seqLen; t++)
                {
                    for (int j = 0; j < outputDim; j++)
                    {
                        float baseValue = priceInputs[s][t, j % features];
                        float neighbour = priceInputs[s][t, (j + 1) % features];
                        priceTargets[s][t, j] = 0.72f * baseValue + 0.18f * neighbour + textSignal + 0.01f * j;
                    }
                    confTargets[s][t] = 0.85f;
                }
            }

            return new DataSet
            {
                Tokenizer = tok,
                Texts = texts,
                PriceInputs = priceInputs,
                PriceTargets = priceTargets,
                ConfidenceTargets = confTargets
            };
        }

        private DataSet ConstantTargetData(int n, int seqLen, int features, int outputDim, float targetValue, int seed)
        {
            string[] corpus = { "bullish", "bearish", "neutral" };
            var tok = NewTokenizer(corpus);
            var texts = new int[n][];
            var priceInputs = new float[n][,];
            var priceTargets = new float[n][,];
            var confTargets = new float[n][];

            for (int s = 0; s < n; s++)
            {
                texts[s] = tok.Encode(corpus[s % corpus.Length], addSpecialTokens: true);
                priceInputs[s] = MakeSmoothMatrix(seqLen, features, s, seed);
                priceTargets[s] = new float[seqLen, outputDim];
                confTargets[s] = new float[seqLen];

                for (int t = 0; t < seqLen; t++)
                {
                    for (int j = 0; j < outputDim; j++)
                        priceTargets[s][t, j] = targetValue;
                    confTargets[s][t] = 1f;
                }
            }

            return new DataSet
            {
                Tokenizer = tok,
                Texts = texts,
                PriceInputs = priceInputs,
                PriceTargets = priceTargets,
                ConfidenceTargets = confTargets
            };
        }

        private TextSignalDataSet TextSignalData(int repeatsPerClass, int seqLen, int seed)
        {
            string bullishText = "bullish earnings growth";
            string bearishText = "bearish recession warning";
            string[] corpus = { bullishText, bearishText };
            var tok = NewTokenizer(corpus);
            var bullishTokens = tok.Encode(bullishText, addSpecialTokens: true);
            var bearishTokens = tok.Encode(bearishText, addSpecialTokens: true);
            var sharedPrice = MakeSmoothMatrix(seqLen, features: 1, sampleIndex: 0, seed: seed);

            int n = repeatsPerClass * 2;
            var texts = new int[n][];
            var priceInputs = new float[n][,];
            var priceTargets = new float[n][,];
            var confTargets = new float[n][];

            for (int s = 0; s < n; s++)
            {
                bool bullish = s < repeatsPerClass;
                texts[s] = bullish ? bullishTokens : bearishTokens;
                priceInputs[s] = CloneMatrix(sharedPrice);
                priceTargets[s] = new float[seqLen, 1];
                confTargets[s] = new float[seqLen];

                float value = bullish ? 0.75f : -0.75f;
                for (int t = 0; t < seqLen; t++)
                {
                    priceTargets[s][t, 0] = value;
                    confTargets[s][t] = 1f;
                }
            }

            return new TextSignalDataSet
            {
                Tokenizer = tok,
                Texts = texts,
                PriceInputs = priceInputs,
                PriceTargets = priceTargets,
                ConfidenceTargets = confTargets,
                BullishTokens = bullishTokens,
                BearishTokens = bearishTokens,
                SharedPrice = sharedPrice
            };
        }

        private BPETokenizer NewTokenizer(string[] corpus)
        {
            var tok = new BPETokenizer(new AccelerationCPU());
            tok.Train(corpus, vocabSize: 200, minFrequency: 1);
            return tok;
        }

        private float[,] MakeSmoothMatrix(int seqLen, int features, int sampleIndex, int seed)
        {
            var matrix = new float[seqLen, features];
            float phase = 0.17f * seed + 0.31f * sampleIndex;
            for (int t = 0; t < seqLen; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    float wave = MathF.Sin(phase + 0.37f * t + 0.19f * f);
                    float trend = 0.02f * t + 0.01f * sampleIndex - 0.015f * f;
                    matrix[t, f] = 0.35f * wave + trend;
                }
            }
            return matrix;
        }

        private float[][] MakeConfidenceTargets(int samples, int seqLen, float value)
        {
            var result = new float[samples][];
            for (int i = 0; i < samples; i++)
            {
                result[i] = new float[seqLen];
                for (int t = 0; t < seqLen; t++)
                    result[i][t] = value;
            }
            return result;
        }

        // ---------------------------------------------------------------------
        // Assertion / numeric helpers
        // ---------------------------------------------------------------------

        private void ExpectThrowsAny(Action action, string message)
        {
            try
            {
                action();
            }
            catch
            {
                return;
            }

            Assert(false, message);
        }

        private void AssertDims(float[,] matrix, int rows, int cols, string name)
        {
            Assert(matrix != null, $"{name} is null");
            Assert(matrix.GetLength(0) == rows, $"{name} rows: expected {rows}, got {matrix.GetLength(0)}");
            Assert(matrix.GetLength(1) == cols, $"{name} cols: expected {cols}, got {matrix.GetLength(1)}");
        }

        private void AssertFinite(float value, string name)
        {
            Assert(!float.IsNaN(value) && !float.IsInfinity(value), $"{name} is not finite: {value}");
        }

        private void AssertFinite(float[,] matrix, string name)
        {
            Assert(matrix != null, $"{name} is null");
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    AssertFinite(matrix[i, j], $"{name}[{i},{j}]");
        }

        private void AssertClose(float expected, float actual, float tol, string message)
        {
            float diff = MathF.Abs(expected - actual);
            Assert(diff <= tol, $"{message}: expected={expected}, actual={actual}, diff={diff}, tol={tol}");
        }

        private void AssertMatrixClose(float[,] expected, float[,] actual, float tol, string message)
        {
            Assert(expected != null && actual != null, $"{message}: one matrix is null");
            Assert(expected.GetLength(0) == actual.GetLength(0), $"{message}: row count mismatch");
            Assert(expected.GetLength(1) == actual.GetLength(1), $"{message}: col count mismatch");

            for (int i = 0; i < expected.GetLength(0); i++)
                for (int j = 0; j < expected.GetLength(1); j++)
                    AssertClose(expected[i, j], actual[i, j], tol, $"{message} at [{i},{j}]");
        }

        private void AssertVectorClose(float[] expected, float[] actual, float tol, string message)
        {
            Assert(expected != null && actual != null, $"{message}: one vector is null");
            Assert(expected.Length == actual.Length, $"{message}: length mismatch");

            for (int i = 0; i < expected.Length; i++)
                AssertClose(expected[i], actual[i], tol, $"{message} at [{i}]");
        }

        private void AssertLossLower(float before, float after, string message)
        {
            AssertFinite(before, "loss before");
            AssertFinite(after, "loss after");
            Assert(after < before, $"{message}: before={before}, after={after}");
        }

        private float[,] CloneMatrix(float[,] matrix)
        {
            var clone = new float[matrix.GetLength(0), matrix.GetLength(1)];
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    clone[i, j] = matrix[i, j];
            return clone;
        }

        private float[] CloneVector(float[] vector)
        {
            var clone = new float[vector.Length];
            Array.Copy(vector, clone, vector.Length);
            return clone;
        }

        private bool MatrixChanged(float[,] before, float[,] after, float tolerance)
        {
            if (before.GetLength(0) != after.GetLength(0) || before.GetLength(1) != after.GetLength(1))
                return true;

            for (int i = 0; i < before.GetLength(0); i++)
                for (int j = 0; j < before.GetLength(1); j++)
                    if (MathF.Abs(before[i, j] - after[i, j]) > tolerance)
                        return true;
            return false;
        }

        private bool VectorChanged(float[] before, float[] after, float tolerance)
        {
            if (before.Length != after.Length)
                return true;

            for (int i = 0; i < before.Length; i++)
                if (MathF.Abs(before[i] - after[i]) > tolerance)
                    return true;
            return false;
        }

        private float MeanConfidence(CrossModel model, int[][] texts, float[][,] prices)
        {
            float sum = 0f;
            int count = 0;
            for (int s = 0; s < prices.Length; s++)
            {
                var (_, conf) = model.Forward(texts[s], prices[s]);
                Assert(conf != null, "confidence should not be null");
                for (int t = 0; t < conf.GetLength(0); t++)
                {
                    sum += conf[t, 0];
                    count++;
                }
            }
            return sum / count;
        }

        private float MeanMatrix(float[,] matrix)
        {
            float sum = 0f;
            int count = 0;
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j];
                    count++;
                }
            }
            return sum / count;
        }

        private float[,] SliceRows(float[,] matrix, int startRow, int endRowExclusive)
        {
            int rows = endRowExclusive - startRow;
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = matrix[startRow + i, j];
            return result;
        }

        private string NewTempDir()
        {
            string dir = Path.Combine(Path.GetTempPath(), "CrossAttentionMultimodalTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            return dir;
        }

        private void DeleteTempDir(string dir)
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
    }
}
