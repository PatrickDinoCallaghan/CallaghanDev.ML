using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Configuration;
using CallaghanDev.ML.Transformers.CrossAttentionMultimodal;
using CrossModel = CallaghanDev.ML.Transformers.CrossAttentionMultimodal.Model;
using CrossTrainer = CallaghanDev.ML.Transformers.CrossAttentionMultimodal.Trainer;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    internal sealed class CrossAttentionMultimodalTests : TestBase
    {
        private MultimodalTransformerConfig MakeCfg(int vocabSize, int embDim = 16, int numHeads = 2, int numLayers = 1, int ffnDim = 32, int priceFeatures = 5, int outputDim = 5, int priceSeqLen = 10, bool useConf = true, bool freezeText = false)
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

        private (BPETokenizer tok, int[][] texts, float[][,] priceIn, float[][,] priceTgt) Data(int n = 10, int seqLen = 8, int features = 5, int outDim = 5, int seed = 42)
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

        public void RunAllTests()
        {
            CountNumber++;
            Run(Tests(), $"{CountNumber} * CrossAttentionMultimodal");
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            //  construction & forward 
            (Test_ModelConstruction,              "Construction: model initialises without error"),
            (Test_ForwardShape,                   "Forward: output shape [seqLen, outputDim]"),
            (Test_ForwardNaN,                     "Forward: no NaN in predictions"),
            (Test_ForwardDeterministic,           "Forward: deterministic"),
            (Test_ConfidenceRange,                "Forward: confidence values in [0, 1]"),
            (Test_NoConfidenceHead_Null,          "Forward: confidence null when head disabled"),
            (Test_TextVsNoText_Differ,            "Forward: text vs no-text produce different outputs"),
            (Test_PriceOnly_Forward,              "Forward: price-only (null text) works"),
            (Test_PredictNext_Shape,              "PredictNext: returns correct output dim"),
            //  training 
            (Test_LossDecreases,                  "Train: loss decreases"),
            (Test_AllPriceParamsUpdated,          "Train: all price-decoder params receive gradients"),
            (Test_FrozenTextEncoder_Unchanged,    "Train: frozen text encoder weights unchanged"),
            (Test_UnfrozenTextEncoder_Changed,    "Train: unfrozen text encoder weights change"),
            (Test_MixedBatch_NullText,            "Train: mixed batch (some null text) works"),
            (Test_GradientClipping_NoNaN,         "Train: high LR + clipping prevents NaN"),
            (Test_LearningRateDecay,              "Train: LR decay runs without crash"),
            (Test_PriceOnly_LossDecreases,        "Train: price-only loss decreases"),
            (Test_SingleSampleOverfit,            "Train: single sample overfits (loss halves)"),
            //  validation 
            (Test_ValidationLossValid,            "Validate: returns finite non-negative value"),
            //  save / load 
            (Test_SaveLoad_WeightsAndForward,     "SaveLoad: weights preserved, forward identical"),
            (Test_SaveLoad_NoConfidenceHead,      "SaveLoad: no-confidence model round-trips"),
            //  dimension & config guards 
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
}
