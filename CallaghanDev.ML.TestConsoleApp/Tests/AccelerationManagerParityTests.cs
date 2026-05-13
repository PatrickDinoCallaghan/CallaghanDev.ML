using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.Cache;
using CallaghanDev.ML.Transformers.TACAMT;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using static CallaghanDev.ML.AccelerationManagers.AccelerationCPU;

namespace CallaghanDev.ML.TestConsoleApp.Tests
{
    /// <summary>
    /// Parity tests for IAccelerationManager implementations.
    /// CPU is the reference implementation. MultiThreadCPU and GPU, when available,
    /// must produce the same results within a small floating point tolerance.
    ///
    /// GPU tests are skipped automatically when ILGPU cannot create a GPU/CUDA device.
    /// Optional TACAMT-only tests are skipped if their constructors are not available in
    /// the referenced build.
    /// </summary>
    internal sealed class AccelerationManagerParityTests : TestBase
    {
        private readonly IAccelerationManager _cpu = new AccelerationCPU();
        private readonly IAccelerationManager _multi = new AccelerationMutliThreadCPU();

        private IAccelerationManager _gpu;
        private string _gpuName;
        private string _gpuSkipReason;
        private bool _gpuInitAttempted;

        private const float CpuTolerance = 1e-5f;
        private const float GpuTolerance = 3e-3f;

        public void RunAllTests()
        {
            CountNumber++;

            try
            {
                Run(Tests(), $"{CountNumber} * Acceleration Manager Parity");
            }
            finally
            {
                _gpu?.Dispose();
                _gpu = null;
                _gpuName = null;
                _gpuSkipReason = null;
                _gpuInitAttempted = false;
            }
        }

        private (Action, string)[] Tests() => new (Action, string)[]
        {
            (Test_AllInterfaceMethods_AreCoveredByThisSuite, "Coverage: every IAccelerationManager method is represented"),
            (Test_Backends_Construct_AndOptionalGpuSkipIsClean, "Backends: CPU/Multi construct and GPU skip is clean"),
            (Test_SharedTensorPrimitives_AllBackendsMatch, "Shared tensor primitives: all backends match CPU"),
            (Test_InPlaceTensorPrimitives_AllBackendsMatch, "In-place tensor primitives: all backends match CPU"),
            (Test_NeuralNetworkHelpers_AllBackendsMatch, "Neural network helpers: all backends match CPU"),
            (Test_TransformerCore_AllBackendsMatch, "Transformer core: all backends match CPU"),
            (Test_ScaledDotProductAttention_AllBackendsMatch, "Scaled dot-product attention: all backends match CPU"),
            (Test_FusedQkvProjection_AllBackendsMatch, "Fused QKV projection/backprop: all backends match CPU"),
            (Test_TransformerCore_EdgeCases_AllBackendsMatch, "Transformer core edge cases: all backends match CPU"),
            (Test_TransformerTraining_AllBackendsMatch, "Transformer training: all backends match CPU"),
            (Test_TransformerSpecific_Mmtac_AllBackendsMatch, "Transformer-specific/MMTAC helpers: all backends match CPU"),
            (Test_ContentAwareCrossAttentionForward_AllBackendsMatch, "Content-aware cross-attention forward: all backends match CPU"),
            (Test_ContentAwareDecayForward_OptionalAllBackendsMatch, "Content-aware decay forward: optional parity"),
            (Test_ContentAwareCrossAttentionWithCache_OptionalAllBackendsMatch, "Content-aware cross-attention with cache: optional parity"),
            (Test_BackpropTimeDecayedAttention_AllBackendsMatch, "Backprop time-decayed attention: all backends match CPU"),
            (Test_TokenizerAcceleration_AllBackendsMatch, "Tokenizer acceleration: all backends match CPU"),
            (Test_RotaryEmbeddings_AllBackendsMatch, "Rotary embeddings: all backends match CPU"),
            (Test_SigmoidAndDispose_AllBackendsMatch, "Sigmoid/Dispose: all backends match CPU"),
            (Test_GpuThresholdSizedSharedPrimitive_OptionalMatchesCpu, "GPU threshold-sized matrix multiply: optional true GPU parity"),
            (Test_TransformerCoreGpuKernelPaths_OptionalMatchCpu, "GPU transformer-core kernel paths: optional true GPU parity"),
            (Test_GpuConcreteHelperKernels_OptionalMatchCpu, "GPU concrete helper kernels: optional true GPU parity"),
        };

        private void Test_AllInterfaceMethods_AreCoveredByThisSuite()
        {
            var covered = new HashSet<string>(StringComparer.Ordinal)
            {
                // Shared tensor primitives
                nameof(IAccelerationManager.MatrixMultiply),
                nameof(IAccelerationManager.MatrixMultiplyTranspose),
                nameof(IAccelerationManager.MatrixScale),
                nameof(IAccelerationManager.MatrixAdd),
                nameof(IAccelerationManager.MatrixAddBias),
                nameof(IAccelerationManager.BatchDotProduct),
                nameof(IAccelerationManager.SliceRows),
                nameof(IAccelerationManager.ExtractRow),
                nameof(IAccelerationManager.SetRow),
                nameof(IAccelerationManager.ZeroMatrix),
                nameof(IAccelerationManager.ZeroVector),
                nameof(IAccelerationManager.MatrixAddInPlace),
                nameof(IAccelerationManager.VectorAccumulate),

                // Neural network
                nameof(IAccelerationManager.CalculateDotProduct),
                nameof(IAccelerationManager.ActivateLayer),
                nameof(IAccelerationManager.CalculateOutputGradients),
                nameof(IAccelerationManager.CalculateHiddenGradients),
                nameof(IAccelerationManager.UpdateWeights),
                nameof(IAccelerationManager.UpdateBias),

                // Transformer core
                nameof(IAccelerationManager.Softmax),
                nameof(IAccelerationManager.LayerNorm),
                nameof(IAccelerationManager.LayerNormForward),
                nameof(IAccelerationManager.LayerNormBackward),
                nameof(IAccelerationManager.CreateCausalMask),
                nameof(IAccelerationManager.MultiHeadAttentionForward),
                nameof(IAccelerationManager.MultiHeadAttentionBackward),
                nameof(IAccelerationManager.FFNForwardBatch),
                nameof(IAccelerationManager.ScaledDotProductAttention),
                nameof(IAccelerationManager.ProjectQKV),
                nameof(IAccelerationManager.BackpropQKV),

                // Transformer training
                nameof(IAccelerationManager.BackpropLinearProjection),
                nameof(IAccelerationManager.BackpropOutputProjection),
                nameof(IAccelerationManager.BackpropInputProjection),
                nameof(IAccelerationManager.AccumulateTokenEmbeddingGrad),
                nameof(IAccelerationManager.CrossEntropyLossAndGradient),
                nameof(IAccelerationManager.MSELossAndGradient),
                nameof(IAccelerationManager.MatrixSquaredNorm),
                nameof(IAccelerationManager.VectorSquaredNorm),
                nameof(IAccelerationManager.MatrixScaleInPlace),
                nameof(IAccelerationManager.VectorScaleInPlace),
                nameof(IAccelerationManager.MatrixUpdate),
                nameof(IAccelerationManager.VectorUpdate),

                // Transformer-specific / MMTAC
                nameof(IAccelerationManager.ApplyContextTypeEmbedding),
                nameof(IAccelerationManager.ComputeTimeDiffMatrix),
                nameof(IAccelerationManager.ComputeMemoryAttentionScores),
                nameof(IAccelerationManager.ProjectOutputBatch),
                nameof(IAccelerationManager.ContentAwareDecayForward),
                nameof(IAccelerationManager.ContentAwareCrossAttentionForward),
                nameof(IAccelerationManager.ContentAwareCrossAttentionWithCache),
                nameof(IAccelerationManager.Matrix3DScaleInPlace),
                nameof(IAccelerationManager.MatrixSquaredNorm3D),
                nameof(IAccelerationManager.ProjectGlobalFeatures),
                nameof(IAccelerationManager.EmbedTokenIds),
                nameof(IAccelerationManager.MeanPoolRows),
                nameof(IAccelerationManager.BuildMmtacContext),
                nameof(IAccelerationManager.ProjectMmtacOutputHeads),
                nameof(IAccelerationManager.SoftmaxVector),
                nameof(IAccelerationManager.BackpropTimeDecayedAttention),

                // Tokenizer acceleration
                nameof(IAccelerationManager.PreTokenize),
                nameof(IAccelerationManager.GetWordFrequencies),
                nameof(IAccelerationManager.BuildCharacterVocabulary),
                nameof(IAccelerationManager.ApplyMerge),
                nameof(IAccelerationManager.EncodeWord),
                nameof(IAccelerationManager.CountPairFrequencies),
                nameof(IAccelerationManager.SelectBestPair),
                nameof(IAccelerationManager.ApplyMergeToVocabulary),
                nameof(IAccelerationManager.DecodeTokens),
                nameof(IAccelerationManager.PadOrTruncate),

                // Rotary
                nameof(IAccelerationManager.ApplyRotaryPositionEmbeddingHeadInPlace),
                nameof(IAccelerationManager.ApplyRotaryPositionEmbeddingInPlace),

                // Other
                nameof(IAccelerationManager.SigmoidInPlace),
                nameof(IAccelerationManager.Dispose),
            };

            var methodNames = typeof(IAccelerationManager)
                .GetMethods(BindingFlags.Instance | BindingFlags.Public)
                .Select(m => m.Name)
                .Distinct(StringComparer.Ordinal)
                .OrderBy(x => x, StringComparer.Ordinal)
                .ToArray();

            var missing = methodNames
                .Where(name => !covered.Contains(name))
                .ToArray();

            Assert(missing.Length == 0, "IAccelerationManager has public methods not represented by this test suite: " + string.Join(", ", missing));
        }

        private void Test_Backends_Construct_AndOptionalGpuSkipIsClean()
        {
            Assert(_cpu != null, "CPU acceleration manager should construct");
            Assert(_multi != null, "MultiThreadCPU acceleration manager should construct");

            if (TryGetGpu(out _, out string name, out string reason))
            {
                Assert(!string.IsNullOrWhiteSpace(name), "GPU backend name should be set");
            }
            else
            {
                Console.WriteLine($"[SKIP] GPU parity tests skipped: {reason}");
            }
        }

        private void Test_SharedTensorPrimitives_AllBackendsMatch()
        {
            var a = Matrix(5, 4, 1);
            var b = Matrix(4, 3, 2);
            var kt = Matrix(6, 4, 3);
            var bias = Vector(3, 4);

            CompareMatrixAll("MatrixMultiply", _cpu.MatrixMultiply(a, b), m => m.MatrixMultiply(Clone(a), Clone(b)));
            CompareMatrixAll("MatrixMultiplyTranspose", _cpu.MatrixMultiplyTranspose(a, kt), m => m.MatrixMultiplyTranspose(Clone(a), Clone(kt)));
            CompareMatrixAll("MatrixScale", _cpu.MatrixScale(a, -0.75f), m => m.MatrixScale(Clone(a), -0.75f));
            CompareMatrixAll("MatrixAdd", _cpu.MatrixAdd(a, Matrix(5, 4, 5)), m => m.MatrixAdd(Clone(a), Matrix(5, 4, 5)));
            CompareMatrixAll("MatrixAddBias", _cpu.MatrixAddBias(_cpu.MatrixMultiply(a, b), bias), m => m.MatrixAddBias(m.MatrixMultiply(Clone(a), Clone(b)), Clone(bias)));
            CompareMatrixAll("BatchDotProduct", _cpu.BatchDotProduct(b, Matrix(7, 3, 6)), m => m.BatchDotProduct(Clone(b), Matrix(7, 3, 6)));
            CompareMatrixAll("BatchDotProduct slice", _cpu.BatchDotProduct(b, Matrix(9, 3, 7), 2, 4), m => m.BatchDotProduct(Clone(b), Matrix(9, 3, 7), 2, 4));
            CompareMatrixAll("SliceRows", _cpu.SliceRows(Matrix(8, 4, 8), 2, 6), m => m.SliceRows(Matrix(8, 4, 8), 2, 6));
            CompareVectorAll("ExtractRow", _cpu.ExtractRow(Matrix(5, 4, 9), 3, 4), m => m.ExtractRow(Matrix(5, 4, 9), 3, 4));

            var setExpected = Matrix(3, 4, 10);
            _cpu.SetRow(setExpected, 1, new[] { 9f, 8f, 7f, 6f }, 4);
            CompareMatrixAll("SetRow", setExpected, m =>
            {
                var x = Matrix(3, 4, 10);
                m.SetRow(x, 1, new[] { 9f, 8f, 7f, 6f }, 4);
                return x;
            });
        }

        private void Test_InPlaceTensorPrimitives_AllBackendsMatch()
        {
            var zeroM = Matrix(4, 5, 11);
            _cpu.ZeroMatrix(zeroM);
            CompareMatrixAll("ZeroMatrix", zeroM, m =>
            {
                var x = Matrix(4, 5, 11);
                m.ZeroMatrix(x);
                return x;
            });

            var zeroV = Vector(7, 12);
            _cpu.ZeroVector(zeroV);
            CompareVectorAll("ZeroVector", zeroV, m =>
            {
                var x = Vector(7, 12);
                m.ZeroVector(x);
                return x;
            });

            var addTarget = Matrix(4, 5, 13);
            _cpu.MatrixAddInPlace(addTarget, Matrix(4, 5, 14));
            CompareMatrixAll("MatrixAddInPlace", addTarget, m =>
            {
                var x = Matrix(4, 5, 13);
                m.MatrixAddInPlace(x, Matrix(4, 5, 14));
                return x;
            });

            var accum = Vector(8, 15);
            _cpu.VectorAccumulate(accum, Vector(8, 16));
            CompareVectorAll("VectorAccumulate", accum, m =>
            {
                var x = Vector(8, 15);
                m.VectorAccumulate(x, Vector(8, 16));
                return x;
            });
        }

        private void Test_NeuralNetworkHelpers_AllBackendsMatch()
        {
            var matrix = Matrix(4, 5, 17);
            var vector = Vector(5, 18);
            var dot = Vector(5, 19);
            var bias = Vector(5, 20);
            var cost = Vector(5, 21);
            var derivative = PositiveVector(5, 22);
            var nextDeltas = Vector(4, 23);

            CompareVectorAll("CalculateDotProduct", _cpu.CalculateDotProduct(matrix, vector), m => m.CalculateDotProduct(Clone(matrix), Clone(vector)));

            var expectedActivation = _cpu.ActivateLayer(dot, bias, ActivationType.Relu);
            foreach (var backend in NonReferenceBackends())
            {
                var actual = backend.Manager.ActivateLayer(Clone(dot), Clone(bias), ActivationType.Relu);
                AssertClose(expectedActivation.activation, actual.activation, backend.Tolerance, backend.Name + ": ActivateLayer.activation");
                AssertClose(expectedActivation.derivative, actual.derivative, backend.Tolerance, backend.Name + ": ActivateLayer.derivative");
            }

            CompareVectorAll("CalculateOutputGradients", _cpu.CalculateOutputGradients(cost, derivative), m => m.CalculateOutputGradients(Clone(cost), Clone(derivative)));
            CompareVectorAll("CalculateHiddenGradients", _cpu.CalculateHiddenGradients(matrix, nextDeltas, derivative), m => m.CalculateHiddenGradients(Clone(matrix), Clone(nextDeltas), Clone(derivative)));
            CompareMatrixAll("UpdateWeights", _cpu.UpdateWeights(matrix, nextDeltas, vector, 0.01f, 0.001f), m => m.UpdateWeights(Clone(matrix), Clone(nextDeltas), Clone(vector), 0.01f, 0.001f));
            CompareVectorAll("UpdateBias", _cpu.UpdateBias(Vector(4, 24), nextDeltas, 0.01f), m => m.UpdateBias(Vector(4, 24), Clone(nextDeltas), 0.01f));
        }

        private void Test_TransformerCore_AllBackendsMatch()
        {
            var scores = Matrix(4, 4, 25);
            var mask = new bool[,]
            {
                { true,  false, false, false },
                { true,  true,  false, false },
                { true,  false, true,  false },
                { true,  true,  true,  true  },
            };

            CompareMatrixAll("Softmax no mask", _cpu.Softmax(scores), m => m.Softmax(Clone(scores)));
            CompareMatrixAll("Softmax mask", _cpu.Softmax(scores, mask), m => m.Softmax(Clone(scores), Clone(mask)));

            var input = Matrix(5, 6, 26);
            var gamma = PositiveVector(6, 27);
            var beta = Vector(6, 28);

            CompareMatrixAll("LayerNorm", _cpu.LayerNorm(input, gamma, beta), m => m.LayerNorm(Clone(input), Clone(gamma), Clone(beta)));

            var expectedLn = _cpu.LayerNormForward(input, gamma, beta);
            foreach (var backend in NonReferenceBackends())
            {
                var actual = backend.Manager.LayerNormForward(Clone(input), Clone(gamma), Clone(beta));
                AssertClose(expectedLn.output, actual.output, backend.Tolerance, backend.Name + ": LayerNormForward.output");
                AssertClose(expectedLn.means, actual.means, backend.Tolerance, backend.Name + ": LayerNormForward.means");
                AssertClose(expectedLn.variances, actual.variances, backend.Tolerance, backend.Name + ": LayerNormForward.variances");
                AssertClose(expectedLn.normalized, actual.normalized, backend.Tolerance, backend.Name + ": LayerNormForward.normalized");
            }

            var dOut = Matrix(5, 6, 29);
            var expectedBack = _cpu.LayerNormBackward(dOut, expectedLn.normalized, gamma, input, expectedLn.means, expectedLn.variances);
            foreach (var backend in NonReferenceBackends())
            {
                var actual = backend.Manager.LayerNormBackward(Clone(dOut), Clone(expectedLn.normalized), Clone(gamma), Clone(input), Clone(expectedLn.means), Clone(expectedLn.variances));
                AssertClose(expectedBack.dInput, actual.dInput, backend.Tolerance, backend.Name + ": LayerNormBackward.dInput");
                AssertClose(expectedBack.dGamma, actual.dGamma, backend.Tolerance, backend.Name + ": LayerNormBackward.dGamma");
                AssertClose(expectedBack.dBeta, actual.dBeta, backend.Tolerance, backend.Name + ": LayerNormBackward.dBeta");
            }

            CompareBoolMatrixAll("CreateCausalMask", _cpu.CreateCausalMask(6), m => m.CreateCausalMask(6));

            var q = Matrix(4, 6, 30);
            var k = Matrix(4, 6, 31);
            var v = Matrix(4, 6, 32);
            var dConcat = Matrix(4, 6, 33);
            int heads = 2;
            float scale = 1.0f / MathF.Sqrt(3f);
            var causal = _cpu.CreateCausalMask(4);

            CompareMatrixAll("MultiHeadAttentionForward", _cpu.MultiHeadAttentionForward(q, k, v, heads, scale, causal), m => m.MultiHeadAttentionForward(Clone(q), Clone(k), Clone(v), heads, scale, Clone(causal)), 5e-3f);

            var expectedMhaBackBool = _cpu.MultiHeadAttentionBackward(q, k, v, dConcat, heads, scale, useDecoderMask: true);
            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.MultiHeadAttentionBackward(Clone(q), Clone(k), Clone(v), Clone(dConcat), heads, scale, useDecoderMask: true);
                AssertClose(expectedMhaBackBool.dQ, actual.dQ, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward(bool).dQ");
                AssertClose(expectedMhaBackBool.dK, actual.dK, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward(bool).dK");
                AssertClose(expectedMhaBackBool.dV, actual.dV, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward(bool).dV");
            }

            var expectedMhaBackMask = _cpu.MultiHeadAttentionBackward(q, k, v, dConcat, heads, scale, causal);
            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.MultiHeadAttentionBackward(Clone(q), Clone(k), Clone(v), Clone(dConcat), heads, scale, Clone(causal));
                AssertClose(expectedMhaBackMask.dQ, actual.dQ, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward(mask).dQ");
                AssertClose(expectedMhaBackMask.dK, actual.dK, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward(mask).dK");
                AssertClose(expectedMhaBackMask.dV, actual.dV, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward(mask).dV");
            }

            CompareMatrixAll("FFNForwardBatch", _cpu.FFNForwardBatch(input, input.GetLength(0), input.GetLength(1), TinyForwardPass), m => m.FFNForwardBatch(Clone(input), input.GetLength(0), input.GetLength(1), TinyForwardPass));
        }

        private void Test_ScaledDotProductAttention_AllBackendsMatch()
        {
            const int seqLen = 4;
            const int keyLen = 5;
            const int embeddingDim = 8;
            const int heads = 2;

            var qSelf = Matrix(seqLen, embeddingDim, 118);
            var kSelf = Matrix(seqLen, embeddingDim, 119);
            var vSelf = Matrix(seqLen, embeddingDim, 120);

            CompareMatrixAll(
                "ScaledDotProductAttention self/no mask",
                _cpu.ScaledDotProductAttention(qSelf, kSelf, vSelf, heads, mask: null, causal: false),
                m => m.ScaledDotProductAttention(Clone(qSelf), Clone(kSelf), Clone(vSelf), heads, mask: null, causal: false),
                5e-3f);

            var selfMask = new bool[,]
            {
                { true,  false, true,  false },
                { true,  true,  false, false },
                { false, true,  true,  false },
                { true,  true,  true,  true  },
            };

            CompareMatrixAll(
                "ScaledDotProductAttention self/explicit mask",
                _cpu.ScaledDotProductAttention(qSelf, kSelf, vSelf, heads, selfMask, causal: false),
                m => m.ScaledDotProductAttention(Clone(qSelf), Clone(kSelf), Clone(vSelf), heads, Clone(selfMask), causal: false),
                5e-3f);

            CompareMatrixAll(
                "ScaledDotProductAttention self/causal flag",
                _cpu.ScaledDotProductAttention(qSelf, kSelf, vSelf, heads, mask: null, causal: true),
                m => m.ScaledDotProductAttention(Clone(qSelf), Clone(kSelf), Clone(vSelf), heads, mask: null, causal: true),
                5e-3f);

            var causalMask = _cpu.CreateCausalMask(seqLen);

            var expectedCausalMask = _cpu.ScaledDotProductAttention(qSelf, kSelf, vSelf, heads, causalMask, causal: false);
            var expectedCausalFlag = _cpu.ScaledDotProductAttention(qSelf, kSelf, vSelf, heads, mask: null, causal: true);

            AssertClose(expectedCausalMask, expectedCausalFlag, CpuTolerance, "CPU: ScaledDotProductAttention causal flag should match explicit causal mask");

            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actualCausalMask = backend.Manager.ScaledDotProductAttention(Clone(qSelf), Clone(kSelf), Clone(vSelf), heads, Clone(causalMask), causal: false);
                var actualCausalFlag = backend.Manager.ScaledDotProductAttention(Clone(qSelf), Clone(kSelf), Clone(vSelf), heads, mask: null, causal: true);

                AssertClose(expectedCausalMask, actualCausalMask, backend.Tolerance, backend.Name + ": ScaledDotProductAttention explicit causal mask");
                AssertClose(expectedCausalFlag, actualCausalFlag, backend.Tolerance, backend.Name + ": ScaledDotProductAttention causal flag");
                AssertClose(actualCausalMask, actualCausalFlag, backend.Tolerance, backend.Name + ": ScaledDotProductAttention causal flag should match explicit causal mask");
            }

            var qCross = Matrix(seqLen, embeddingDim, 121);
            var kCross = Matrix(keyLen, embeddingDim, 122);
            var vCross = Matrix(keyLen, embeddingDim, 123);

            var crossMask = new bool[,]
            {
                { true,  false, true,  false, true  },
                { true,  true,  false, true,  false },
                { false, true,  true,  false, true  },
                { true,  false, false, true,  true  },
            };

            CompareMatrixAll(
                "ScaledDotProductAttention cross/no mask",
                _cpu.ScaledDotProductAttention(qCross, kCross, vCross, heads, mask: null, causal: false),
                m => m.ScaledDotProductAttention(Clone(qCross), Clone(kCross), Clone(vCross), heads, mask: null, causal: false),
                5e-3f);

            CompareMatrixAll(
                "ScaledDotProductAttention cross/non-square mask",
                _cpu.ScaledDotProductAttention(qCross, kCross, vCross, heads, crossMask, causal: false),
                m => m.ScaledDotProductAttention(Clone(qCross), Clone(kCross), Clone(vCross), heads, Clone(crossMask), causal: false),
                5e-3f);

            var allMaskedFirstRow = new bool[,]
            {
                { false, false, false, false, false },
                { true,  true,  false, true,  false },
                { false, true,  true,  false, true  },
                { true,  false, false, true,  true  },
            };

            var expectedAllMasked = _cpu.ScaledDotProductAttention(qCross, kCross, vCross, heads, allMaskedFirstRow, causal: false);

            CompareMatrixAll(
                "ScaledDotProductAttention all-masked row",
                expectedAllMasked,
                m => m.ScaledDotProductAttention(Clone(qCross), Clone(kCross), Clone(vCross), heads, Clone(allMaskedFirstRow), causal: false),
                5e-3f);

            for (int j = 0; j < embeddingDim; j++)
            {
                AssertClose(0.0f, expectedAllMasked[0, j], CpuTolerance, $"CPU: ScaledDotProductAttention all-masked row output[0,{j}] should be zero");
            }

            AssertThrows<ArgumentException>(
                () => _cpu.ScaledDotProductAttention(qCross, kCross, vCross, heads, mask: null, causal: true),
                "CPU: ScaledDotProductAttention causal cross-attention should reject queryLen != keyLen");

            foreach (var backend in NonReferenceBackends())
            {
                AssertThrows<ArgumentException>(
                    () => backend.Manager.ScaledDotProductAttention(Clone(qCross), Clone(kCross), Clone(vCross), heads, mask: null, causal: true),
                    backend.Name + ": ScaledDotProductAttention causal cross-attention should reject queryLen != keyLen");
            }
        }

        private void Test_FusedQkvProjection_AllBackendsMatch()
        {
            const int rows = 6;
            const int inputDim = 5;
            const int outputDim = 8;

            var input = Matrix(rows, inputDim, 124);
            var wq = Matrix(outputDim, inputDim, 125);
            var wk = Matrix(outputDim, inputDim, 126);
            var wv = Matrix(outputDim, inputDim, 127);
            var biasQ = Vector(outputDim, 128);
            var biasK = Vector(outputDim, 129);
            var biasV = Vector(outputDim, 130);

            var expectedForward = ProjectQKVExpected(input, wq, biasQ, wk, biasK, wv, biasV);
            var cpuForward = _cpu.ProjectQKV(Clone(input), Clone(wq), Clone(biasQ), Clone(wk), Clone(biasK), Clone(wv), Clone(biasV));

            AssertClose(expectedForward.Q, cpuForward.Q, CpuTolerance, "CPU: ProjectQKV.Q");
            AssertClose(expectedForward.K, cpuForward.K, CpuTolerance, "CPU: ProjectQKV.K");
            AssertClose(expectedForward.V, cpuForward.V, CpuTolerance, "CPU: ProjectQKV.V");

            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.ProjectQKV(Clone(input), Clone(wq), Clone(biasQ), Clone(wk), Clone(biasK), Clone(wv), Clone(biasV));
                AssertClose(expectedForward.Q, actual.Q, backend.Tolerance, backend.Name + ": ProjectQKV.Q");
                AssertClose(expectedForward.K, actual.K, backend.Tolerance, backend.Name + ": ProjectQKV.K");
                AssertClose(expectedForward.V, actual.V, backend.Tolerance, backend.Name + ": ProjectQKV.V");
            }

            AssertClose(_cpu.MatrixAddBias(_cpu.BatchDotProduct(wq, input), biasQ), cpuForward.Q, CpuTolerance, "CPU: ProjectQKV.Q should match BatchDotProduct + MatrixAddBias");
            AssertClose(_cpu.MatrixAddBias(_cpu.BatchDotProduct(wk, input), biasK), cpuForward.K, CpuTolerance, "CPU: ProjectQKV.K should match BatchDotProduct + MatrixAddBias");
            AssertClose(_cpu.MatrixAddBias(_cpu.BatchDotProduct(wv, input), biasV), cpuForward.V, CpuTolerance, "CPU: ProjectQKV.V should match BatchDotProduct + MatrixAddBias");

            var dQ = Matrix(rows, outputDim, 131);
            var dK = Matrix(rows, outputDim, 132);
            var dV = Matrix(rows, outputDim, 133);

            var wqGradInitial = Matrix(outputDim, inputDim, 134);
            var wkGradInitial = Matrix(outputDim, inputDim, 135);
            var wvGradInitial = Matrix(outputDim, inputDim, 136);
            var biasQGradInitial = Vector(outputDim, 137);
            var biasKGradInitial = Vector(outputDim, 138);
            var biasVGradInitial = Vector(outputDim, 139);

            var expectedWQGrad = Clone(wqGradInitial);
            var expectedWKGrad = Clone(wkGradInitial);
            var expectedWVGrad = Clone(wvGradInitial);
            var expectedBiasQGrad = Clone(biasQGradInitial);
            var expectedBiasKGrad = Clone(biasKGradInitial);
            var expectedBiasVGrad = Clone(biasVGradInitial);

            var expectedDInput = BackpropQKVExpected(input, dQ, dK, dV, wq, wk, wv, expectedWQGrad, expectedBiasQGrad, expectedWKGrad, expectedBiasKGrad, expectedWVGrad, expectedBiasVGrad);

            var cpuWQGrad = Clone(wqGradInitial);
            var cpuWKGrad = Clone(wkGradInitial);
            var cpuWVGrad = Clone(wvGradInitial);
            var cpuBiasQGrad = Clone(biasQGradInitial);
            var cpuBiasKGrad = Clone(biasKGradInitial);
            var cpuBiasVGrad = Clone(biasVGradInitial);

            var cpuDInput = _cpu.BackpropQKV(Clone(input), Clone(dQ), Clone(dK), Clone(dV), Clone(wq), Clone(wk), Clone(wv), cpuWQGrad, cpuBiasQGrad, cpuWKGrad, cpuBiasKGrad, cpuWVGrad, cpuBiasVGrad);

            AssertClose(expectedDInput, cpuDInput, CpuTolerance, "CPU: BackpropQKV.dInput");
            AssertClose(expectedWQGrad, cpuWQGrad, CpuTolerance, "CPU: BackpropQKV.WQGrad");
            AssertClose(expectedWKGrad, cpuWKGrad, CpuTolerance, "CPU: BackpropQKV.WKGrad");
            AssertClose(expectedWVGrad, cpuWVGrad, CpuTolerance, "CPU: BackpropQKV.WVGrad");
            AssertClose(expectedBiasQGrad, cpuBiasQGrad, CpuTolerance, "CPU: BackpropQKV.biasQGrad");
            AssertClose(expectedBiasKGrad, cpuBiasKGrad, CpuTolerance, "CPU: BackpropQKV.biasKGrad");
            AssertClose(expectedBiasVGrad, cpuBiasVGrad, CpuTolerance, "CPU: BackpropQKV.biasVGrad");

            var oldWQGrad = Clone(wqGradInitial);
            var oldWKGrad = Clone(wkGradInitial);
            var oldWVGrad = Clone(wvGradInitial);
            var oldBiasQGrad = Clone(biasQGradInitial);
            var oldBiasKGrad = Clone(biasKGradInitial);
            var oldBiasVGrad = Clone(biasVGradInitial);
            var oldDInput = Zeros(rows, inputDim);
            _cpu.BackpropLinearProjection(input, dQ, wq, oldWQGrad, oldBiasQGrad, oldDInput);
            _cpu.BackpropLinearProjection(input, dK, wk, oldWKGrad, oldBiasKGrad, oldDInput);
            _cpu.BackpropLinearProjection(input, dV, wv, oldWVGrad, oldBiasVGrad, oldDInput);
            AssertClose(oldDInput, cpuDInput, CpuTolerance, "CPU: BackpropQKV.dInput should match three BackpropLinearProjection calls");
            AssertClose(oldWQGrad, cpuWQGrad, CpuTolerance, "CPU: BackpropQKV.WQGrad should match old path");
            AssertClose(oldWKGrad, cpuWKGrad, CpuTolerance, "CPU: BackpropQKV.WKGrad should match old path");
            AssertClose(oldWVGrad, cpuWVGrad, CpuTolerance, "CPU: BackpropQKV.WVGrad should match old path");
            AssertClose(oldBiasQGrad, cpuBiasQGrad, CpuTolerance, "CPU: BackpropQKV.biasQGrad should match old path");
            AssertClose(oldBiasKGrad, cpuBiasKGrad, CpuTolerance, "CPU: BackpropQKV.biasKGrad should match old path");
            AssertClose(oldBiasVGrad, cpuBiasVGrad, CpuTolerance, "CPU: BackpropQKV.biasVGrad should match old path");

            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actualWQGrad = Clone(wqGradInitial);
                var actualWKGrad = Clone(wkGradInitial);
                var actualWVGrad = Clone(wvGradInitial);
                var actualBiasQGrad = Clone(biasQGradInitial);
                var actualBiasKGrad = Clone(biasKGradInitial);
                var actualBiasVGrad = Clone(biasVGradInitial);

                var actualDInput = backend.Manager.BackpropQKV(Clone(input), Clone(dQ), Clone(dK), Clone(dV), Clone(wq), Clone(wk), Clone(wv), actualWQGrad, actualBiasQGrad, actualWKGrad, actualBiasKGrad, actualWVGrad, actualBiasVGrad);

                AssertClose(expectedDInput, actualDInput, backend.Tolerance, backend.Name + ": BackpropQKV.dInput");
                AssertClose(expectedWQGrad, actualWQGrad, backend.Tolerance, backend.Name + ": BackpropQKV.WQGrad");
                AssertClose(expectedWKGrad, actualWKGrad, backend.Tolerance, backend.Name + ": BackpropQKV.WKGrad");
                AssertClose(expectedWVGrad, actualWVGrad, backend.Tolerance, backend.Name + ": BackpropQKV.WVGrad");
                AssertClose(expectedBiasQGrad, actualBiasQGrad, backend.Tolerance, backend.Name + ": BackpropQKV.biasQGrad");
                AssertClose(expectedBiasKGrad, actualBiasKGrad, backend.Tolerance, backend.Name + ": BackpropQKV.biasKGrad");
                AssertClose(expectedBiasVGrad, actualBiasVGrad, backend.Tolerance, backend.Name + ": BackpropQKV.biasVGrad");
            }

            AssertThrows<ArgumentException>(() => _cpu.ProjectQKV(Matrix(rows, inputDim + 1, 140), Clone(wq), Clone(biasQ), Clone(wk), Clone(biasK), Clone(wv), Clone(biasV)), "CPU: ProjectQKV should reject input width mismatch");
            AssertThrows<ArgumentException>(() => _cpu.BackpropQKV(Clone(input), Matrix(rows, outputDim + 1, 141), Clone(dK), Clone(dV), Clone(wq), Clone(wk), Clone(wv), Clone(wqGradInitial), Clone(biasQGradInitial), Clone(wkGradInitial), Clone(biasKGradInitial), Clone(wvGradInitial), Clone(biasVGradInitial)), "CPU: BackpropQKV should reject dQ width mismatch");
        }

        private void Test_TransformerCore_EdgeCases_AllBackendsMatch()
        {
            var edgeScores = new float[,]
            {
                { -1000f, 0f, 1000f },
                { 3f, -2f, 1f },
                { 0f, 0f, 0f },
            };

            var edgeMask = new bool[,]
            {
                { false, false, false },
                { true,  false, true  },
                { false, true,  false },
            };

            CompareMatrixAll("Softmax edge/all-masked rows", _cpu.Softmax(edgeScores, edgeMask), m => m.Softmax(Clone(edgeScores), Clone(edgeMask)), 5e-3f);
            CompareBoolMatrixAll("CreateCausalMask zero length", _cpu.CreateCausalMask(0), m => m.CreateCausalMask(0));
            CompareBoolMatrixAll("CreateCausalMask single", _cpu.CreateCausalMask(1), m => m.CreateCausalMask(1));

            var q = Matrix(3, 4, 101);
            var k = Matrix(5, 4, 102);
            var v = Matrix(5, 4, 103);
            var dConcat = Matrix(3, 4, 104);
            int heads = 2;
            float scale = 1.0f / MathF.Sqrt(2f);

            var crossMask = new bool[,]
            {
                { true,  false, true,  false, true  },
                { true,  true,  false, true,  false },
                { false, true,  true,  false, true  },
            };

            CompareMatrixAll("MultiHeadAttentionForward non-square mask", _cpu.MultiHeadAttentionForward(q, k, v, heads, scale, crossMask), m => m.MultiHeadAttentionForward(Clone(q), Clone(k), Clone(v), heads, scale, Clone(crossMask)), 5e-3f);

            var expectedBack = _cpu.MultiHeadAttentionBackward(q, k, v, dConcat, heads, scale, crossMask);
            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.MultiHeadAttentionBackward(Clone(q), Clone(k), Clone(v), Clone(dConcat), heads, scale, Clone(crossMask));
                AssertClose(expectedBack.dQ, actual.dQ, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward non-square mask.dQ");
                AssertClose(expectedBack.dK, actual.dK, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward non-square mask.dK");
                AssertClose(expectedBack.dV, actual.dV, backend.Tolerance, backend.Name + ": MultiHeadAttentionBackward non-square mask.dV");
            }
        }

        private void Test_TransformerTraining_AllBackendsMatch()
        {
            var input = Matrix(4, 5, 34);
            var dOutput = Matrix(4, 3, 35);
            var weights = Matrix(3, 5, 36);

            var cpuWg = Zeros(3, 5);
            var cpuBg = new float[3];
            var cpuDi = Zeros(4, 5);
            _cpu.BackpropLinearProjection(input, dOutput, weights, cpuWg, cpuBg, cpuDi);

            foreach (var backend in NonReferenceBackends())
            {
                var wg = Zeros(3, 5);
                var bg = new float[3];
                var di = Zeros(4, 5);
                backend.Manager.BackpropLinearProjection(Clone(input), Clone(dOutput), Clone(weights), wg, bg, di);
                AssertClose(cpuWg, wg, backend.Tolerance, backend.Name + ": BackpropLinearProjection.weightGrad");
                AssertClose(cpuBg, bg, backend.Tolerance, backend.Name + ": BackpropLinearProjection.biasGrad");
                AssertClose(cpuDi, di, backend.Tolerance, backend.Name + ": BackpropLinearProjection.dInput");
            }

            var dLogits = Matrix(4, 3, 37);
            var outWg = Zeros(3, 5);
            var outBg = new float[3];
            var expectedOutProj = _cpu.BackpropOutputProjection(dLogits, input, weights, outWg, outBg, 4, 3, 5);

            foreach (var backend in NonReferenceBackends())
            {
                var wg = Zeros(3, 5);
                var bg = new float[3];
                var actual = backend.Manager.BackpropOutputProjection(Clone(dLogits), Clone(input), Clone(weights), wg, bg, 4, 3, 5);
                AssertClose(expectedOutProj, actual, backend.Tolerance, backend.Name + ": BackpropOutputProjection.dHidden");
                AssertClose(outWg, wg, backend.Tolerance, backend.Name + ": BackpropOutputProjection.weightGrad");
                AssertClose(outBg, bg, backend.Tolerance, backend.Name + ": BackpropOutputProjection.biasGrad");
            }

            var continuous = Matrix(4, 2, 38);
            var dX = Matrix(4, 5, 39);
            var cpuInpWg = Zeros(5, 2);
            var cpuInpBg = new float[5];
            _cpu.BackpropInputProjection(dX, continuous, cpuInpWg, cpuInpBg, 4, 5, 2);
            foreach (var backend in NonReferenceBackends())
            {
                var wg = Zeros(5, 2);
                var bg = new float[5];
                backend.Manager.BackpropInputProjection(Clone(dX), Clone(continuous), wg, bg, 4, 5, 2);
                AssertClose(cpuInpWg, wg, backend.Tolerance, backend.Name + ": BackpropInputProjection.weightGrad");
                AssertClose(cpuInpBg, bg, backend.Tolerance, backend.Name + ": BackpropInputProjection.biasGrad");
            }

            var tokenGrad = Zeros(8, 5);
            var tokenIds = new[] { 1, 2, 2, 5 };
            _cpu.AccumulateTokenEmbeddingGrad(tokenGrad, input, tokenIds, 4, 5);
            CompareMatrixAll("AccumulateTokenEmbeddingGrad", tokenGrad, m =>
            {
                var x = Zeros(8, 5);
                m.AccumulateTokenEmbeddingGrad(x, Clone(input), Clone(tokenIds), 4, 5);
                return x;
            });

            var logits = Matrix(4, 5, 40);
            var targets = new[] { 0, 2, 4, 1 };
            var expectedCe = _cpu.CrossEntropyLossAndGradient(logits, targets, 4);
            foreach (var backend in NonReferenceBackends())
            {
                var actual = backend.Manager.CrossEntropyLossAndGradient(Clone(logits), Clone(targets), 4);
                AssertClose(expectedCe.loss, actual.loss, backend.Tolerance, backend.Name + ": CrossEntropyLoss.loss");
                AssertClose(expectedCe.dLogits, actual.dLogits, backend.Tolerance, backend.Name + ": CrossEntropyLoss.gradient");
            }

            var predictions = Matrix(4, 3, 41);
            var targetsMse = Matrix(4, 3, 42);
            var expectedMse = _cpu.MSELossAndGradient(predictions, targetsMse, 4);
            foreach (var backend in NonReferenceBackends())
            {
                var actual = backend.Manager.MSELossAndGradient(Clone(predictions), Clone(targetsMse), 4);
                AssertClose(expectedMse.loss, actual.loss, backend.Tolerance, backend.Name + ": MSE.loss");
                AssertClose(expectedMse.dOutput, actual.dOutput, backend.Tolerance, backend.Name + ": MSE.gradient");
            }

            CompareScalarAll("MatrixSquaredNorm", _cpu.MatrixSquaredNorm(input), m => m.MatrixSquaredNorm(Clone(input)));
            CompareScalarAll("VectorSquaredNorm", _cpu.VectorSquaredNorm(Vector(9, 43)), m => m.VectorSquaredNorm(Vector(9, 43)));

            var scaleM = Matrix(4, 5, 44);
            _cpu.MatrixScaleInPlace(scaleM, -0.25f);
            CompareMatrixAll("MatrixScaleInPlace", scaleM, m =>
            {
                var x = Matrix(4, 5, 44);
                m.MatrixScaleInPlace(x, -0.25f);
                return x;
            });

            var scaleV = Vector(7, 45);
            _cpu.VectorScaleInPlace(scaleV, 0.25f);
            CompareVectorAll("VectorScaleInPlace", scaleV, m =>
            {
                var x = Vector(7, 45);
                m.VectorScaleInPlace(x, 0.25f);
                return x;
            });

            var updM = Matrix(4, 5, 46);
            _cpu.MatrixUpdate(updM, Matrix(4, 5, 47), 0.02f);
            CompareMatrixAll("MatrixUpdate", updM, m =>
            {
                var x = Matrix(4, 5, 46);
                m.MatrixUpdate(x, Matrix(4, 5, 47), 0.02f);
                return x;
            });

            var updV = Vector(7, 48);
            _cpu.VectorUpdate(updV, Vector(7, 49), 0.02f);
            CompareVectorAll("VectorUpdate", updV, m =>
            {
                var x = Vector(7, 48);
                m.VectorUpdate(x, Vector(7, 49), 0.02f);
                return x;
            });
        }

        private void Test_TransformerSpecific_Mmtac_AllBackendsMatch()
        {
            var context = Matrix(4, 6, 50);
            var typeEmb = Matrix(3, 6, 51);
            var typeIndices = new[] { 0, 1, 2, 0 };
            var expectedCtx = Clone(context);
            _cpu.ApplyContextTypeEmbedding(expectedCtx, typeEmb, typeIndices);
            CompareMatrixAll("ApplyContextTypeEmbedding", expectedCtx, m =>
            {
                var x = Clone(context);
                m.ApplyContextTypeEmbedding(x, Clone(typeEmb), Clone(typeIndices));
                return x;
            });

            CompareMatrixAll("ComputeTimeDiffMatrix", _cpu.ComputeTimeDiffMatrix(4, new[] { -2f, 0f, 1.5f }), m => m.ComputeTimeDiffMatrix(4, new[] { -2f, 0f, 1.5f }));
            CompareVectorAll("ComputeMemoryAttentionScores", _cpu.ComputeMemoryAttentionScores(Matrix(4, 6, 52), 3, Matrix(5, 6, 53), 5, 0.25f), m => m.ComputeMemoryAttentionScores(Matrix(4, 6, 52), 3, Matrix(5, 6, 53), 5, 0.25f));
            CompareMatrixAll("ProjectOutputBatch", _cpu.ProjectOutputBatch(Matrix(4, 6, 54), Matrix(3, 6, 55), Vector(3, 56), 4, 3), m => m.ProjectOutputBatch(Matrix(4, 6, 54), Matrix(3, 6, 55), Vector(3, 56), 4, 3));

            var m3 = Tensor3(2, 3, 4, 57);
            _cpu.Matrix3DScaleInPlace(m3, -0.5f);
            CompareTensor3All("Matrix3DScaleInPlace", m3, m =>
            {
                var x = Tensor3(2, 3, 4, 57);
                m.Matrix3DScaleInPlace(x, -0.5f);
                return x;
            });
            CompareScalarAll("MatrixSquaredNorm3D", _cpu.MatrixSquaredNorm3D(Tensor3(2, 3, 4, 58)), m => m.MatrixSquaredNorm3D(Tensor3(2, 3, 4, 58)));

            CompareVectorAll("ProjectGlobalFeatures", _cpu.ProjectGlobalFeatures(Vector(3, 59), Matrix(6, 3, 60), Vector(6, 61)), m => m.ProjectGlobalFeatures(Vector(3, 59), Matrix(6, 3, 60), Vector(6, 61)));
            CompareMatrixAll("EmbedTokenIds", _cpu.EmbedTokenIds(new[] { 0, 2, 4 }, Matrix(6, 5, 62), 5), m => m.EmbedTokenIds(new[] { 0, 2, 4 }, Matrix(6, 5, 62), 5));
            CompareVectorAll("MeanPoolRows", _cpu.MeanPoolRows(Matrix(4, 5, 63)), m => m.MeanPoolRows(Matrix(4, 5, 63)));

            var buildExpected = _cpu.BuildMmtacContext(Matrix(2, 5, 64), new[] { -1.0f, -0.25f }, Vector(5, 65), Matrix(3, 5, 66));
            foreach (var backend in NonReferenceBackends())
            {
                var actual = backend.Manager.BuildMmtacContext(Matrix(2, 5, 64), new[] { -1.0f, -0.25f }, Vector(5, 65), Matrix(3, 5, 66));
                AssertClose(buildExpected.contextHidden, actual.contextHidden, backend.Tolerance, backend.Name + ": BuildMmtacContext.contextHidden");
                AssertClose(buildExpected.contextTimes, actual.contextTimes, backend.Tolerance, backend.Name + ": BuildMmtacContext.contextTimes");
                Assert(buildExpected.numGlobal == actual.numGlobal, backend.Name + ": BuildMmtacContext.numGlobal mismatch");
                Assert(buildExpected.numNews == actual.numNews, backend.Name + ": BuildMmtacContext.numNews mismatch");
            }

            var hidden = Matrix(4, 5, 67);
            var expectedHeads = _cpu.ProjectMmtacOutputHeads(
                hidden,
                Matrix(3, 5, 68), Vector(3, 69),
                Matrix(1, 5, 70), Vector(1, 71),
                Matrix(1, 5, 72), Vector(1, 73),
                Matrix(1, 5, 74), Vector(1, 75),
                Matrix(1, 5, 76), Vector(1, 77),
                Matrix(1, 5, 78), Vector(1, 79),
                useConfidenceHead: true);

            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.ProjectMmtacOutputHeads(
                    Clone(hidden),
                    Matrix(3, 5, 68), Vector(3, 69),
                    Matrix(1, 5, 70), Vector(1, 71),
                    Matrix(1, 5, 72), Vector(1, 73),
                    Matrix(1, 5, 74), Vector(1, 75),
                    Matrix(1, 5, 76), Vector(1, 77),
                    Matrix(1, 5, 78), Vector(1, 79),
                    useConfidenceHead: true);

                AssertClose(expectedHeads.regression, actual.regression, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.regression");
                AssertClose(expectedHeads.range, actual.range, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.range");
                AssertClose(expectedHeads.quality, actual.quality, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.quality");
                AssertClose(expectedHeads.direction, actual.direction, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.direction");
                AssertClose(expectedHeads.midDirection, actual.midDirection, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.midDirection");
                AssertClose(expectedHeads.confidence, actual.confidence, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.confidence");
                AssertClose(expectedHeads.regressionLogits, actual.regressionLogits, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.regressionLogits");
                AssertClose(expectedHeads.rangeLogits, actual.rangeLogits, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.rangeLogits");
                AssertClose(expectedHeads.qualityLogits, actual.qualityLogits, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads.qualityLogits");
            }

            var expectedHeadsNoConfidence = _cpu.ProjectMmtacOutputHeads(
                hidden,
                Matrix(3, 5, 168), Vector(3, 169),
                Matrix(1, 5, 170), Vector(1, 171),
                Matrix(1, 5, 172), Vector(1, 173),
                Matrix(1, 5, 174), Vector(1, 175),
                Matrix(1, 5, 176), Vector(1, 177),
                Matrix(1, 5, 178), Vector(1, 179),
                useConfidenceHead: false);

            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.ProjectMmtacOutputHeads(
                    Clone(hidden),
                    Matrix(3, 5, 168), Vector(3, 169),
                    Matrix(1, 5, 170), Vector(1, 171),
                    Matrix(1, 5, 172), Vector(1, 173),
                    Matrix(1, 5, 174), Vector(1, 175),
                    Matrix(1, 5, 176), Vector(1, 177),
                    Matrix(1, 5, 178), Vector(1, 179),
                    useConfidenceHead: false);

                AssertClose(expectedHeadsNoConfidence.regression, actual.regression, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).regression");
                AssertClose(expectedHeadsNoConfidence.range, actual.range, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).range");
                AssertClose(expectedHeadsNoConfidence.quality, actual.quality, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).quality");
                AssertClose(expectedHeadsNoConfidence.direction, actual.direction, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).direction");
                AssertClose(expectedHeadsNoConfidence.midDirection, actual.midDirection, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).midDirection");
                Assert(
                    expectedHeadsNoConfidence.confidence == null && actual.confidence == null,
                    backend.Name + ": ProjectMmtacOutputHeads(no confidence).confidence should be null");
                AssertClose(expectedHeadsNoConfidence.regressionLogits, actual.regressionLogits, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).regressionLogits");
                AssertClose(expectedHeadsNoConfidence.rangeLogits, actual.rangeLogits, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).rangeLogits");
                AssertClose(expectedHeadsNoConfidence.qualityLogits, actual.qualityLogits, backend.Tolerance, backend.Name + ": ProjectMmtacOutputHeads(no confidence).qualityLogits");
            }

            CompareVectorAll("SoftmaxVector", _cpu.SoftmaxVector(new[] { -1f, 0.5f, 2f, -0.25f }), m => m.SoftmaxVector(new[] { -1f, 0.5f, 2f, -0.25f }));
            CompareVectorAll("SoftmaxVector stable large values", _cpu.SoftmaxVector(new[] { 1000f, 999f, 998f }), m => m.SoftmaxVector(new[] { 1000f, 999f, 998f }), 5e-3f);
        }

        private void Test_ContentAwareCrossAttentionForward_AllBackendsMatch()
        {
            int seq = 3;
            int ctx = 4;
            int ed = 6;
            int heads = 2;
            float scale = 1.0f / MathF.Sqrt(ed / heads);

            var q = Matrix(seq, ed, 80);
            var k = Matrix(ctx, ed, 81);
            var v = Matrix(ctx, ed, 82);
            var decayBias = Tensor3(seq, ctx, heads, 83);

            var expected = _cpu.ContentAwareCrossAttentionForward(q, k, v, heads, scale, decayBias, out var expectedW, out var expectedScores);

            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.ContentAwareCrossAttentionForward(Clone(q), Clone(k), Clone(v), heads, scale, Clone(decayBias), out var actualW, out var actualScores);
                AssertClose(expected, actual, backend.Tolerance, backend.Name + ": ContentAwareCrossAttentionForward.output");
                AssertAttentionWeights(expectedW, actualW, backend.Tolerance, backend.Name + ": ContentAwareCrossAttentionForward.attentionWeights");
                AssertAttentionWeights(expectedScores, actualScores, backend.Tolerance, backend.Name + ": ContentAwareCrossAttentionForward.scoresPreSoftmax");
            }
        }

        private void Test_ContentAwareDecayForward_OptionalAllBackendsMatch()
        {
            if (!TryCreateContentAwareDecayNetwork(out var network, out string reason))
            {
                Console.WriteLine($"[SKIP] ContentAwareDecayForward parity skipped: {reason}");
                return;
            }

            string invalidReason = ValidateContentAwareDecayNetwork(network);
            if (invalidReason != null)
            {
                Console.WriteLine($"[SKIP] ContentAwareDecayForward parity skipped: {invalidReason}");
                return;
            }

            int queryLen = 3;
            int keyLen = 4;
            int contentDim = network.ContentDim;

            var q = Matrix(queryLen, contentDim, 84);
            var k = Matrix(keyLen, contentDim, 85);

            var timeDiffs = new float[,]
            {
                { 0f, 1f, 2f, 3f },
                { 0f, 0f, 1f, 2f },
                { 0f, 0f, 0f, 1f },
            };

            var keyTimes = new[] { -3f, -2f, -1f, 0f };

            var expected = _cpu.ContentAwareDecayForward(q, k, timeDiffs, keyTimes, network, isTraining: false, dropoutRng: null);

            foreach (var backend in NonReferenceBackends(8e-3f))
            {
                var actual = backend.Manager.ContentAwareDecayForward(Clone(q), Clone(k), Clone(timeDiffs), Clone(keyTimes), network, isTraining: false, dropoutRng: null);
                AssertClose(expected.decayBias, actual.decayBias, backend.Tolerance, backend.Name + ": ContentAwareDecayForward.decayBias");
            }
        }

        private void Test_ContentAwareCrossAttentionWithCache_OptionalAllBackendsMatch()
        {
            if (!TryCreateTacamtBlock(out var block, out string reason))
            {
                Console.WriteLine($"[SKIP] ContentAwareCrossAttentionWithCache parity skipped: {reason}");
                return;
            }

            int seq = 3;
            int ctx = 4;
            int ed = 8;
            int heads = 2;

            var q = Matrix(seq, ed, 86);
            var k = Matrix(ctx, ed, 87);
            var v = Matrix(ctx, ed, 88);

            var timeDiffs = new float[,]
            {
                { 0f, 1f, 2f, 3f },
                { 0f, 0f, 1f, 2f },
                { 0f, 0f, 0f, 1f },
            };

            var keyTimes = new[] { -3f, -2f, -1f, 0f };
            var queryEmb = Matrix(seq, ed, 89);
            var keyEmb = Matrix(ctx, ed, 90);

            var expectedBc = new BlockCache();
            var expected = _cpu.ContentAwareCrossAttentionWithCache(q, k, v, timeDiffs, keyTimes, queryEmb, keyEmb, block, expectedBc, ed, heads, enableDecayBias: false, isTraining: false, dropoutRng: null);

            foreach (var backend in NonReferenceBackends(8e-3f))
            {
                var bc = new BlockCache();
                var actual = backend.Manager.ContentAwareCrossAttentionWithCache(Clone(q), Clone(k), Clone(v), Clone(timeDiffs), Clone(keyTimes), Clone(queryEmb), Clone(keyEmb), block, bc, ed, heads, enableDecayBias: false, isTraining: false, dropoutRng: null);

                AssertClose(expected, actual, backend.Tolerance, backend.Name + ": ContentAwareCrossAttentionWithCache.output");

                if (expectedBc.CrossAttentionWeights != null && bc.CrossAttentionWeights != null)
                {
                    AssertAttentionWeights(expectedBc.CrossAttentionWeights, bc.CrossAttentionWeights, backend.Tolerance, backend.Name + ": ContentAwareCrossAttentionWithCache.cache.CrossAttentionWeights");
                }

                if (expectedBc.CrossScoresPreSoftmax != null && bc.CrossScoresPreSoftmax != null)
                {
                    AssertAttentionWeights(expectedBc.CrossScoresPreSoftmax, bc.CrossScoresPreSoftmax, backend.Tolerance, backend.Name + ": ContentAwareCrossAttentionWithCache.cache.CrossScoresPreSoftmax");
                }

                Assert(expectedBc.DecayCache == null && bc.DecayCache == null, backend.Name + ": ContentAwareCrossAttentionWithCache.DecayCache should be null when enableDecayBias is false");
            }
        }

        private void Test_BackpropTimeDecayedAttention_AllBackendsMatch()
        {
            int seq = 3;
            int ctx = 4;
            int ed = 6;
            int heads = 2;
            float scale = 1.0f / MathF.Sqrt(ed / heads);

            var q = Matrix(seq, ed, 91);
            var k = Matrix(ctx, ed, 92);
            var v = Matrix(ctx, ed, 93);
            var decay = Tensor3(seq, ctx, heads, 94);
            _ = _cpu.ContentAwareCrossAttentionForward(q, k, v, heads, scale, decay, out var attentionWeights, out _);
            var dOutput = Matrix(seq, ed, 95);
            var timeDiffs = Matrix(seq, ctx, 96);

            var expected = _cpu.BackpropTimeDecayedAttention(q, k, v, dOutput, attentionWeights, timeDiffs, ed, heads);

            foreach (var backend in NonReferenceBackends(5e-3f))
            {
                var actual = backend.Manager.BackpropTimeDecayedAttention(Clone(q), Clone(k), Clone(v), Clone(dOutput), Clone(attentionWeights), Clone(timeDiffs), ed, heads);
                AssertClose(expected.dQ, actual.dQ, backend.Tolerance, backend.Name + ": BackpropTimeDecayedAttention.dQ");
                AssertClose(expected.dK, actual.dK, backend.Tolerance, backend.Name + ": BackpropTimeDecayedAttention.dK");
                AssertClose(expected.dV, actual.dV, backend.Tolerance, backend.Name + ": BackpropTimeDecayedAttention.dV");
                AssertClose(expected.dDecayBias, actual.dDecayBias, backend.Tolerance, backend.Name + ": BackpropTimeDecayedAttention.dDecayBias");
            }
        }

        private void Test_TokenizerAcceleration_AllBackendsMatch()
        {
            var text = "Hello, world! Hello world 123.";
            var expectedTokens = _cpu.PreTokenize(text);
            foreach (var backend in NonReferenceBackends())
            {
                Assert(expectedTokens.SequenceEqual(backend.Manager.PreTokenize(text)), backend.Name + ": PreTokenize mismatch");
            }

            string[] corpus = { "aa aa ab", "aa!", "Hello HELLO" };
            var expectedFreq = _cpu.GetWordFrequencies(corpus, lowerCase: true);
            foreach (var backend in NonReferenceBackends())
            {
                AssertDictionaryEqual(expectedFreq, backend.Manager.GetWordFrequencies(corpus, lowerCase: true), backend.Name + ": GetWordFrequencies");
            }

            var expectedChars = _cpu.BuildCharacterVocabulary(expectedFreq);
            foreach (var backend in NonReferenceBackends())
            {
                Assert(expectedChars.SetEquals(backend.Manager.BuildCharacterVocabulary(expectedFreq)), backend.Name + ": BuildCharacterVocabulary mismatch");
            }

            var word = new List<string> { "a", "a", "b", "a", "a" };
            var expectedMerge = _cpu.ApplyMerge(word, "a", "a");
            foreach (var backend in NonReferenceBackends())
            {
                Assert(expectedMerge.SequenceEqual(backend.Manager.ApplyMerge(new List<string>(word), "a", "a")), backend.Name + ": ApplyMerge mismatch");
            }

            var mergePriority = new Dictionary<(string, string), int>
            {
                [("a", "a")] = 0,
                [("aa", "b")] = 1,
            };
            var vocab = new Dictionary<string, int>
            {
                ["a"] = 1,
                ["b"] = 2,
                ["aa"] = 3,
                ["aab"] = 4,
            };
            var expectedEncoded = _cpu.EncodeWord("aab", mergePriority, vocab, unkTokenId: 99);
            foreach (var backend in NonReferenceBackends())
            {
                Assert(expectedEncoded.SequenceEqual(backend.Manager.EncodeWord("aab", mergePriority, vocab, unkTokenId: 99)), backend.Name + ": EncodeWord mismatch");
            }

            var words = new Dictionary<List<string>, int>(new ListEqualityComparer<string>())
            {
                [new List<string> { "a", "a" }] = 3,
                [new List<string> { "a", "b" }] = 2,
            };

            var expectedPairs = _cpu.CountPairFrequencies(words);
            foreach (var backend in NonReferenceBackends())
            {
                AssertPairDictionaryEqual(expectedPairs, backend.Manager.CountPairFrequencies(CloneWords(words)), backend.Name + ": CountPairFrequencies");
            }

            var expectedBest = _cpu.SelectBestPair(expectedPairs, 1);
            foreach (var backend in NonReferenceBackends())
            {
                var actual = backend.Manager.SelectBestPair(new Dictionary<(string left, string right), int>(expectedPairs), 1);
                Assert(expectedBest.pair == actual.pair && expectedBest.frequency == actual.frequency, backend.Name + ": SelectBestPair mismatch");
            }

            var expectedVocabMerge = _cpu.ApplyMergeToVocabulary(words, "a", "a");
            foreach (var backend in NonReferenceBackends())
            {
                AssertWordDictionaryEqual(expectedVocabMerge, backend.Manager.ApplyMergeToVocabulary(CloneWords(words), "a", "a"), backend.Name + ": ApplyMergeToVocabulary");
            }

            var idToVocab = new Dictionary<int, string>
            {
                [0] = "<|pad|>",
                [1] = "hello",
                [2] = ",",
                [3] = "world",
            };
            string expectedDecoded = _cpu.DecodeTokens(new[] { 0, 1, 2, 3 }, idToVocab, "<unk>", skipSpecialTokens: true);
            foreach (var backend in NonReferenceBackends())
            {
                Assert(expectedDecoded == backend.Manager.DecodeTokens(new[] { 0, 1, 2, 3 }, idToVocab, "<unk>", skipSpecialTokens: true), backend.Name + ": DecodeTokens mismatch");
            }

            var expectedPad = _cpu.PadOrTruncate(new[] { 1, 5, 6, 2 }, 6, true, 0, 2);
            foreach (var backend in NonReferenceBackends())
            {
                Assert(expectedPad.SequenceEqual(backend.Manager.PadOrTruncate(new[] { 1, 5, 6, 2 }, 6, true, 0, 2)), backend.Name + ": PadOrTruncate mismatch");
            }
        }

        private void Test_RotaryEmbeddings_AllBackendsMatch()
        {
            var matrix = Matrix(4, 8, 97);
            var expected = Clone(matrix);
            _cpu.ApplyRotaryPositionEmbeddingInPlace(expected, numHeads: 2, baseTheta: 10000f, inverse: false);
            CompareMatrixAll("ApplyRotaryPositionEmbeddingInPlace", expected, m =>
            {
                var x = Clone(matrix);
                m.ApplyRotaryPositionEmbeddingInPlace(x, numHeads: 2, baseTheta: 10000f, inverse: false);
                return x;
            }, 5e-3f);

            var headExpected = Clone(matrix);
            _cpu.ApplyRotaryPositionEmbeddingHeadInPlace(headExpected, startCol: 2, headDim: 4, baseTheta: 10000f, inverse: false);
            CompareMatrixAll("ApplyRotaryPositionEmbeddingHeadInPlace", headExpected, m =>
            {
                var x = Clone(matrix);
                m.ApplyRotaryPositionEmbeddingHeadInPlace(x, startCol: 2, headDim: 4, baseTheta: 10000f, inverse: false);
                return x;
            }, 5e-3f);

            var roundTrip = Clone(matrix);
            _cpu.ApplyRotaryPositionEmbeddingInPlace(roundTrip, 2, 10000f, inverse: false);
            _cpu.ApplyRotaryPositionEmbeddingInPlace(roundTrip, 2, 10000f, inverse: true);
            AssertClose(matrix, roundTrip, 5e-4f, "CPU rotary round-trip");
        }

        private void Test_SigmoidAndDispose_AllBackendsMatch()
        {
            var expected = Matrix(4, 5, 98);
            _cpu.SigmoidInPlace(expected);
            CompareMatrixAll("SigmoidInPlace", expected, m =>
            {
                var x = Matrix(4, 5, 98);
                m.SigmoidInPlace(x);
                return x;
            });

            new AccelerationCPU().Dispose();
            new AccelerationMutliThreadCPU().Dispose();

            if (TryCreateGpu(out var gpu, out _, out string reason))
            {
                gpu.Dispose();
            }
            else
            {
                Console.WriteLine($"[SKIP] GPU Dispose smoke skipped: {reason}");
            }
        }

        private void Test_GpuThresholdSizedSharedPrimitive_OptionalMatchesCpu()
        {
            if (!TryGetGpu(out var gpu, out string name, out string reason))
            {
                Console.WriteLine($"[SKIP] GPU threshold-sized parity skipped: {reason}");
                return;
            }

            var a = Matrix(128, 128, 99);
            var b = Matrix(128, 64, 100);
            var expected = _cpu.MatrixMultiply(a, b);
            var actual = gpu.MatrixMultiply(Clone(a), Clone(b));
            AssertClose(expected, actual, GpuTolerance, name + ": threshold MatrixMultiply");
        }

        private void Test_TransformerCoreGpuKernelPaths_OptionalMatchCpu()
        {
            if (!TryGetGpu(out var gpu, out string name, out string reason))
            {
                Console.WriteLine($"[SKIP] GPU transformer-core kernel-path parity skipped: {reason}");
                return;
            }

            const int rows = 512;
            const int cols = 512;

            var scores = Matrix(rows, cols, 105);
            var mask = PeriodicMask(rows, cols);

            AssertClose(_cpu.Softmax(scores), gpu.Softmax(Clone(scores)), 8e-3f, name + ": true GPU Softmax no mask");
            AssertClose(_cpu.Softmax(scores, mask), gpu.Softmax(Clone(scores), Clone(mask)), 8e-3f, name + ": true GPU Softmax mask");

            var input = Matrix(rows, cols, 106);
            var gamma = PositiveVector(cols, 107);
            var beta = Vector(cols, 108);

            AssertClose(_cpu.LayerNorm(input, gamma, beta), gpu.LayerNorm(Clone(input), Clone(gamma), Clone(beta)), 8e-3f, name + ": true GPU LayerNorm");

            var expectedForward = _cpu.LayerNormForward(input, gamma, beta);
            var actualForward = gpu.LayerNormForward(Clone(input), Clone(gamma), Clone(beta));

            AssertClose(expectedForward.output, actualForward.output, 8e-3f, name + ": true GPU LayerNormForward.output");
            AssertClose(expectedForward.means, actualForward.means, 8e-3f, name + ": true GPU LayerNormForward.means");
            AssertClose(expectedForward.variances, actualForward.variances, 8e-3f, name + ": true GPU LayerNormForward.variances");
            AssertClose(expectedForward.normalized, actualForward.normalized, 8e-3f, name + ": true GPU LayerNormForward.normalized");

            var dOut = Matrix(rows, cols, 109);
            var expectedBackward = _cpu.LayerNormBackward(dOut, expectedForward.normalized, gamma, input, expectedForward.means, expectedForward.variances);
            var actualBackward = gpu.LayerNormBackward(Clone(dOut), Clone(expectedForward.normalized), Clone(gamma), Clone(input), Clone(expectedForward.means), Clone(expectedForward.variances));

            AssertClose(expectedBackward.dInput, actualBackward.dInput, 1.5e-2f, name + ": true GPU LayerNormBackward.dInput");
            AssertClose(expectedBackward.dGamma, actualBackward.dGamma, 2.5e-2f, name + ": true GPU LayerNormBackward.dGamma");
            AssertClose(expectedBackward.dBeta, actualBackward.dBeta, 2.5e-2f, name + ": true GPU LayerNormBackward.dBeta");

            const int seq = 128;
            const int embeddingDim = 64;
            const int mhaHeads = 4;

            var q = Matrix(seq, embeddingDim, 110);
            var k = Matrix(seq, embeddingDim, 111);
            var v = Matrix(seq, embeddingDim, 112);
            float scale = 1.0f / MathF.Sqrt(embeddingDim / mhaHeads);

            AssertClose(_cpu.MultiHeadAttentionForward(q, k, v, mhaHeads, scale), gpu.MultiHeadAttentionForward(Clone(q), Clone(k), Clone(v), mhaHeads, scale), 8e-3f, name + ": true GPU MultiHeadAttentionForward no mask");

            var causal = _cpu.CreateCausalMask(seq);

            AssertClose(_cpu.MultiHeadAttentionForward(q, k, v, mhaHeads, scale, causal), gpu.MultiHeadAttentionForward(Clone(q), Clone(k), Clone(v), mhaHeads, scale, Clone(causal)), 8e-3f, name + ": true GPU MultiHeadAttentionForward causal mask");

            AssertClose(_cpu.ScaledDotProductAttention(q, k, v, mhaHeads, mask: null, causal: false), gpu.ScaledDotProductAttention(Clone(q), Clone(k), Clone(v), mhaHeads, mask: null, causal: false), 8e-3f, name + ": true GPU ScaledDotProductAttention no mask");
            AssertClose(_cpu.ScaledDotProductAttention(q, k, v, mhaHeads, Clone(causal), causal: false), gpu.ScaledDotProductAttention(Clone(q), Clone(k), Clone(v), mhaHeads, Clone(causal), causal: false), 8e-3f, name + ": true GPU ScaledDotProductAttention explicit causal mask");
            AssertClose(_cpu.ScaledDotProductAttention(q, k, v, mhaHeads, mask: null, causal: true), gpu.ScaledDotProductAttention(Clone(q), Clone(k), Clone(v), mhaHeads, mask: null, causal: true), 8e-3f, name + ": true GPU ScaledDotProductAttention causal flag");

            const int qkvRows = 192;
            const int qkvInputDim = 128;
            const int qkvOutputDim = 128;

            var qkvInput = Matrix(qkvRows, qkvInputDim, 142);
            var qkvWQ = Matrix(qkvOutputDim, qkvInputDim, 143);
            var qkvWK = Matrix(qkvOutputDim, qkvInputDim, 144);
            var qkvWV = Matrix(qkvOutputDim, qkvInputDim, 145);
            var qkvBiasQ = Vector(qkvOutputDim, 146);
            var qkvBiasK = Vector(qkvOutputDim, 147);
            var qkvBiasV = Vector(qkvOutputDim, 148);

            var expectedQkv = _cpu.ProjectQKV(qkvInput, qkvWQ, qkvBiasQ, qkvWK, qkvBiasK, qkvWV, qkvBiasV);
            var actualQkv = gpu.ProjectQKV(Clone(qkvInput), Clone(qkvWQ), Clone(qkvBiasQ), Clone(qkvWK), Clone(qkvBiasK), Clone(qkvWV), Clone(qkvBiasV));

            AssertClose(expectedQkv.Q, actualQkv.Q, 8e-3f, name + ": true GPU ProjectQKV.Q");
            AssertClose(expectedQkv.K, actualQkv.K, 8e-3f, name + ": true GPU ProjectQKV.K");
            AssertClose(expectedQkv.V, actualQkv.V, 8e-3f, name + ": true GPU ProjectQKV.V");

            var qkvDQ = Matrix(qkvRows, qkvOutputDim, 149);
            var qkvDK = Matrix(qkvRows, qkvOutputDim, 150);
            var qkvDV = Matrix(qkvRows, qkvOutputDim, 151);
            var qkvWQGradInitial = Matrix(qkvOutputDim, qkvInputDim, 152);
            var qkvWKGradInitial = Matrix(qkvOutputDim, qkvInputDim, 153);
            var qkvWVGradInitial = Matrix(qkvOutputDim, qkvInputDim, 154);
            var qkvBiasQGradInitial = Vector(qkvOutputDim, 155);
            var qkvBiasKGradInitial = Vector(qkvOutputDim, 156);
            var qkvBiasVGradInitial = Vector(qkvOutputDim, 157);

            var expectedWQGrad = Clone(qkvWQGradInitial);
            var expectedWKGrad = Clone(qkvWKGradInitial);
            var expectedWVGrad = Clone(qkvWVGradInitial);
            var expectedBiasQGrad = Clone(qkvBiasQGradInitial);
            var expectedBiasKGrad = Clone(qkvBiasKGradInitial);
            var expectedBiasVGrad = Clone(qkvBiasVGradInitial);

            var expectedDInput = _cpu.BackpropQKV(qkvInput, qkvDQ, qkvDK, qkvDV, qkvWQ, qkvWK, qkvWV, expectedWQGrad, expectedBiasQGrad, expectedWKGrad, expectedBiasKGrad, expectedWVGrad, expectedBiasVGrad);

            var actualWQGrad = Clone(qkvWQGradInitial);
            var actualWKGrad = Clone(qkvWKGradInitial);
            var actualWVGrad = Clone(qkvWVGradInitial);
            var actualBiasQGrad = Clone(qkvBiasQGradInitial);
            var actualBiasKGrad = Clone(qkvBiasKGradInitial);
            var actualBiasVGrad = Clone(qkvBiasVGradInitial);

            var actualDInput = gpu.BackpropQKV(Clone(qkvInput), Clone(qkvDQ), Clone(qkvDK), Clone(qkvDV), Clone(qkvWQ), Clone(qkvWK), Clone(qkvWV), actualWQGrad, actualBiasQGrad, actualWKGrad, actualBiasKGrad, actualWVGrad, actualBiasVGrad);

            AssertClose(expectedDInput, actualDInput, 1.5e-2f, name + ": true GPU BackpropQKV.dInput");
            AssertClose(expectedWQGrad, actualWQGrad, 1.5e-2f, name + ": true GPU BackpropQKV.WQGrad");
            AssertClose(expectedWKGrad, actualWKGrad, 1.5e-2f, name + ": true GPU BackpropQKV.WKGrad");
            AssertClose(expectedWVGrad, actualWVGrad, 1.5e-2f, name + ": true GPU BackpropQKV.WVGrad");
            AssertClose(expectedBiasQGrad, actualBiasQGrad, 1.5e-2f, name + ": true GPU BackpropQKV.biasQGrad");
            AssertClose(expectedBiasKGrad, actualBiasKGrad, 1.5e-2f, name + ": true GPU BackpropQKV.biasKGrad");
            AssertClose(expectedBiasVGrad, actualBiasVGrad, 1.5e-2f, name + ": true GPU BackpropQKV.biasVGrad");
        }

        private void Test_GpuConcreteHelperKernels_OptionalMatchCpu()
        {
            if (!TryGetGpu(out var backend, out string name, out string reason))
            {
                Console.WriteLine($"[SKIP] GPU concrete helper kernel parity skipped: {reason}");
                return;
            }

            if (!(backend is AccelerationGPU gpu))
            {
                Console.WriteLine("[SKIP] GPU concrete helper kernel parity skipped: backend is not AccelerationGPU");
                return;
            }

            const int seqLen = 512;
            const int embeddingDim = 512;

            var projected = Matrix(seqLen, embeddingDim, 113);
            var bias = Vector(embeddingDim, 114);
            var positional = Matrix(seqLen, embeddingDim, 115);

            var expectedBiasPos = AddBiasAndPositionExpected(projected, bias, positional, seqLen, embeddingDim);

            AssertClose(expectedBiasPos, gpu.AddBiasAndPositionalEncoding(Clone(projected), Clone(bias), Clone(positional), seqLen, embeddingDim), 5e-3f, name + ": true GPU AddBiasAndPositionalEncoding");
            AssertClose(expectedBiasPos, gpu.EmbedWithBiasAndPositional(Clone(projected), Clone(bias), Clone(positional), seqLen, embeddingDim), 5e-3f, name + ": true GPU EmbedWithBiasAndPositional");

            var tokenEmbedding = Matrix(128, embeddingDim, 116);
            var tokenIds = new int[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                tokenIds[i] = i % tokenEmbedding.GetLength(0);
            }

            AssertClose(EmbedTokensWithPositionExpected(tokenEmbedding, tokenIds, positional, seqLen, embeddingDim), gpu.EmbedTokensWithPosition(Clone(tokenEmbedding), Clone(tokenIds), Clone(positional), seqLen, embeddingDim), 5e-3f, name + ": true GPU EmbedTokensWithPosition");

            var hidden = Matrix(1024, embeddingDim, 117);
            var offsets = new int[seqLen];
            var counts = new int[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                offsets[i] = i * 2;
                counts[i] = i % 5 == 0 ? 0 : 2;
            }

            AssertClose(MeanPoolRowsExpected(hidden, offsets, counts, seqLen, embeddingDim), gpu.MeanPoolRows(Clone(hidden), Clone(offsets), Clone(counts), seqLen, embeddingDim), 5e-3f, name + ": true GPU MeanPoolRows grouped");
        }

        private sealed class Backend
        {
            public Backend(string name, IAccelerationManager manager, float tolerance)
            {
                Name = name;
                Manager = manager;
                Tolerance = tolerance;
            }

            public string Name { get; }
            public IAccelerationManager Manager { get; }
            public float Tolerance { get; }
        }

        private sealed class ListEqualityComparer<T> : IEqualityComparer<List<T>>
        {
            public bool Equals(List<T> x, List<T> y)
            {
                if (ReferenceEquals(x, y)) return true;
                if (x == null || y == null || x.Count != y.Count) return false;

                var comparer = EqualityComparer<T>.Default;
                for (int i = 0; i < x.Count; i++)
                {
                    if (!comparer.Equals(x[i], y[i]))
                    {
                        return false;
                    }
                }

                return true;
            }

            public int GetHashCode(List<T> obj)
            {
                if (obj == null) return 0;

                unchecked
                {
                    int hash = 17;
                    var comparer = EqualityComparer<T>.Default;

                    foreach (var item in obj)
                    {
                        hash = hash * 31 + (item == null ? 0 : comparer.GetHashCode(item));
                    }

                    return hash;
                }
            }
        }

        private IEnumerable<Backend> NonReferenceBackends(float? customTolerance = null)
        {
            yield return new Backend("MultiThreadCPU", _multi, customTolerance ?? CpuTolerance);

            if (TryGetGpu(out var gpu, out string name, out _))
            {
                yield return new Backend(name, gpu, customTolerance ?? GpuTolerance);
            }
        }

        private bool TryGetGpu(out IAccelerationManager gpu, out string name, out string reason)
        {
            if (!_gpuInitAttempted)
            {
                _gpuInitAttempted = true;

                if (TryCreateGpu(out _gpu, out _gpuName, out _gpuSkipReason))
                {
                    _gpuSkipReason = null;
                }
            }

            gpu = _gpu;
            name = _gpuName;
            reason = _gpuSkipReason;
            return gpu != null;
        }

        private static bool TryCreateGpu(out IAccelerationManager gpu, out string name, out string reason)
        {
            var failures = new List<string>();

            foreach (var type in new[] { AccelerationType.CUDA, AccelerationType.GPU })
            {
                try
                {
                    gpu = new AccelerationGPU(type, 0);
                    name = type.ToString();
                    reason = null;
                    return true;
                }
                catch (Exception ex)
                {
                    failures.Add($"{type}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            gpu = null;
            name = null;
            reason = string.Join(" | ", failures);
            return false;
        }

        private void CompareMatrixAll(string label, float[,] expected, Func<IAccelerationManager, float[,]> actualFactory, float? tolerance = null)
        {
            foreach (var backend in NonReferenceBackends(tolerance))
            {
                AssertClose(expected, actualFactory(backend.Manager), backend.Tolerance, backend.Name + ": " + label);
            }
        }

        private void CompareVectorAll(string label, float[] expected, Func<IAccelerationManager, float[]> actualFactory, float? tolerance = null)
        {
            foreach (var backend in NonReferenceBackends(tolerance))
            {
                AssertClose(expected, actualFactory(backend.Manager), backend.Tolerance, backend.Name + ": " + label);
            }
        }

        private void CompareTensor3All(string label, float[,,] expected, Func<IAccelerationManager, float[,,]> actualFactory, float? tolerance = null)
        {
            foreach (var backend in NonReferenceBackends(tolerance))
            {
                AssertClose(expected, actualFactory(backend.Manager), backend.Tolerance, backend.Name + ": " + label);
            }
        }

        private void CompareScalarAll(string label, float expected, Func<IAccelerationManager, float> actualFactory, float? tolerance = null)
        {
            foreach (var backend in NonReferenceBackends(tolerance))
            {
                AssertClose(expected, actualFactory(backend.Manager), backend.Tolerance, backend.Name + ": " + label);
            }
        }

        private void CompareBoolMatrixAll(string label, bool[,] expected, Func<IAccelerationManager, bool[,]> actualFactory)
        {
            foreach (var backend in NonReferenceBackends())
            {
                AssertBoolMatrixEqual(expected, actualFactory(backend.Manager), backend.Name + ": " + label);
            }
        }

        private static float[] TinyForwardPass(float[] row)
        {
            var output = new float[row.Length];
            for (int i = 0; i < row.Length; i++)
            {
                output[i] = MathF.Tanh(row[i] * 0.5f + 0.1f);
            }
            return output;
        }

        private bool TryCreateContentAwareDecayNetwork(out ContentAwareDecayNetwork network, out string reason)
        {
            network = null;
            reason = null;

            Type t = typeof(ContentAwareDecayNetwork);
            var failures = new List<string>();

            foreach (var ctor in t.GetConstructors().OrderBy(c => c.GetParameters().Length))
            {
                if (!TryBuildConstructorArgs(ctor, out object[] args, out string argReason))
                {
                    failures.Add($"{ctor}: {argReason}");
                    continue;
                }

                try
                {
                    var candidate = (ContentAwareDecayNetwork)ctor.Invoke(args);
                    string invalidReason = ValidateContentAwareDecayNetwork(candidate);

                    if (invalidReason == null)
                    {
                        network = candidate;
                        return true;
                    }

                    failures.Add($"{ctor}: constructed invalid network: {invalidReason}");
                }
                catch (TargetInvocationException ex) when (ex.InnerException != null)
                {
                    failures.Add($"{ctor}: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
                }
                catch (Exception ex)
                {
                    failures.Add($"{ctor}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            reason = failures.Count == 0 ? "no public constructor found" : string.Join(" | ", failures.Take(5));
            return false;
        }

        private static string ValidateContentAwareDecayNetwork(ContentAwareDecayNetwork network)
        {
            if (network == null) return "network was null";
            if (network.ContentDim <= 0) return $"ContentDim must be positive, got {network.ContentDim}";
            if (network.NumHeads <= 0) return $"NumHeads must be positive, got {network.NumHeads}";
            if (network.ProjectionDim <= 0) return $"ProjectionDim must be positive, got {network.ProjectionDim}";
            if (network.HiddenDim <= 0) return $"HiddenDim must be positive, got {network.HiddenDim}";
            if (network.MLPInputDim <= 0) return $"MLPInputDim must be positive, got {network.MLPInputDim}";
            if (network.NumTimeBases <= 0) return $"NumTimeBases must be positive, got {network.NumTimeBases}";
            if (network.TimeRawDim <= 0) return $"TimeRawDim must be positive, got {network.TimeRawDim}";
            return null;
        }

        private bool TryBuildConstructorArgs(ConstructorInfo ctor, out object[] args, out string reason)
        {
            var parameters = ctor.GetParameters();
            args = new object[parameters.Length];
            reason = null;

            for (int i = 0; i < parameters.Length; i++)
            {
                var p = parameters[i];
                Type pt = p.ParameterType;
                string pn = p.Name ?? string.Empty;
                string lower = pn.ToLowerInvariant();

                if (pt == typeof(int))
                {
                    if (lower.Contains("head") && lower.Contains("dim")) args[i] = 4;
                    else if (lower.Contains("head")) args[i] = 2;
                    else if (lower.Contains("content") || lower.Contains("embed") || lower.Contains("embedding") || lower.Contains("model") || lower.Contains("dim") || lower.Contains("dimension")) args[i] = 8;
                    else if (lower.Contains("projection") || lower.Contains("proj")) args[i] = 4;
                    else if (lower.Contains("hidden")) args[i] = 16;
                    else if (lower.Contains("feed") || lower.Contains("ffn") || lower.Contains("mlp")) args[i] = 16;
                    else if (lower.Contains("basis") || lower.Contains("bases")) args[i] = 4;
                    else if (lower.Contains("raw")) args[i] = 4;
                    else if (lower.Contains("vocab")) args[i] = 32;
                    else if (lower.Contains("sequence") || lower.Contains("seq") || lower.Contains("context") || lower.Contains("memory")) args[i] = 8;
                    else if (lower.Contains("time") || lower.Contains("base")) args[i] = 4;
                    else args[i] = 2;
                    continue;
                }

                if (pt == typeof(float))
                {
                    if (lower.Contains("dropout")) args[i] = 0f;
                    else if (lower.Contains("normalization") || lower.Contains("norm") || lower.Contains("hour")) args[i] = 24f;
                    else if (p.HasDefaultValue && p.DefaultValue != DBNull.Value && p.DefaultValue != null) args[i] = p.DefaultValue;
                    else args[i] = 0f;
                    continue;
                }

                if (pt == typeof(double))
                {
                    if (lower.Contains("dropout")) args[i] = 0d;
                    else if (lower.Contains("normalization") || lower.Contains("norm") || lower.Contains("hour")) args[i] = 24d;
                    else if (p.HasDefaultValue && p.DefaultValue != DBNull.Value && p.DefaultValue != null) args[i] = p.DefaultValue;
                    else args[i] = 0d;
                    continue;
                }

                if (pt == typeof(bool))
                {
                    args[i] = p.HasDefaultValue && p.DefaultValue != DBNull.Value && p.DefaultValue != null ? p.DefaultValue : false;
                    continue;
                }

                if (pt == typeof(Random))
                {
                    args[i] = new Random(1234);
                    continue;
                }

                if (pt == typeof(ActivationType))
                {
                    args[i] = ActivationType.Relu;
                    continue;
                }

                if (pt == typeof(IAccelerationManager) || typeof(IAccelerationManager).IsAssignableFrom(pt))
                {
                    args[i] = new AccelerationCPU();
                    continue;
                }

                if (pt == typeof(AccelerationType))
                {
                    args[i] = AccelerationType.CPU;
                    continue;
                }

                if (p.HasDefaultValue && p.DefaultValue != DBNull.Value)
                {
                    args[i] = p.DefaultValue;
                    continue;
                }

                reason = $"unsupported constructor parameter {pt.FullName} {pn}";
                return false;
            }

            return true;
        }

        private bool TryCreateTacamtBlock(out TacamtBlock block, out string reason)
        {
            block = null;
            reason = null;

            Type t = typeof(TacamtBlock);
            var failures = new List<string>();

            foreach (var ctor in t.GetConstructors().OrderBy(c => c.GetParameters().Length))
            {
                if (!TryBuildConstructorArgs(ctor, out object[] args, out string argReason))
                {
                    failures.Add($"{ctor}: {argReason}");
                    continue;
                }

                try
                {
                    block = (TacamtBlock)ctor.Invoke(args);
                    if (block != null) return true;
                }
                catch (Exception ex)
                {
                    failures.Add($"{ctor}: {ex.GetType().Name}: {ex.Message}");
                }
            }

            reason = failures.Count == 0 ? "no public constructor found" : string.Join(" | ", failures.Take(3));
            return false;
        }

        private void AssertThrows<TException>(Action action, string label) where TException : Exception
        {
            try
            {
                action();
            }
            catch (TException)
            {
                return;
            }
            catch (Exception ex)
            {
                Assert(false, $"{label}: expected {typeof(TException).Name}, got {ex.GetType().Name}: {ex.Message}");
                return;
            }

            Assert(false, $"{label}: expected {typeof(TException).Name}, but no exception was thrown");
        }

        private void AssertClose(float expected, float actual, float tolerance, string label)
        {
            if (float.IsNaN(expected) && float.IsNaN(actual)) return;

            if (float.IsInfinity(expected) || float.IsInfinity(actual))
            {
                Assert(expected.Equals(actual), $"{label}: expected {expected}, got {actual}");
                return;
            }

            float scale = MathF.Max(1f, MathF.Abs(expected));
            float diff = MathF.Abs(expected - actual);
            Assert(diff <= tolerance * scale, $"{label}: expected {expected}, got {actual}, diff {diff}, tolerance {tolerance}");
        }

        private void AssertClose(float[] expected, float[] actual, float tolerance, string label)
        {
            Assert(expected != null && actual != null, label + ": one vector was null");
            Assert(expected.Length == actual.Length, $"{label}: length mismatch {expected.Length} vs {actual.Length}");

            for (int i = 0; i < expected.Length; i++)
            {
                AssertClose(expected[i], actual[i], tolerance, $"{label}[{i}]");
            }
        }

        private void AssertClose(float[,] expected, float[,] actual, float tolerance, string label)
        {
            Assert(expected != null && actual != null, label + ": one matrix was null");
            Assert(expected.GetLength(0) == actual.GetLength(0), $"{label}: row mismatch");
            Assert(expected.GetLength(1) == actual.GetLength(1), $"{label}: col mismatch");

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    AssertClose(expected[i, j], actual[i, j], tolerance, $"{label}[{i},{j}]");
                }
            }
        }

        private void AssertClose(float[,,] expected, float[,,] actual, float tolerance, string label)
        {
            Assert(expected != null && actual != null, label + ": one tensor was null");
            Assert(expected.GetLength(0) == actual.GetLength(0), $"{label}: dim0 mismatch");
            Assert(expected.GetLength(1) == actual.GetLength(1), $"{label}: dim1 mismatch");
            Assert(expected.GetLength(2) == actual.GetLength(2), $"{label}: dim2 mismatch");

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    for (int k = 0; k < expected.GetLength(2); k++)
                    {
                        AssertClose(expected[i, j, k], actual[i, j, k], tolerance, $"{label}[{i},{j},{k}]");
                    }
                }
            }
        }

        private void AssertAttentionWeights(float[][,] expected, float[][,] actual, float tolerance, string label)
        {
            Assert(expected != null && actual != null, label + ": one attention array was null");
            Assert(expected.Length == actual.Length, label + ": head count mismatch");

            for (int h = 0; h < expected.Length; h++)
            {
                AssertClose(expected[h], actual[h], tolerance, $"{label}.head{h}");
            }
        }

        private void AssertBoolMatrixEqual(bool[,] expected, bool[,] actual, string label)
        {
            Assert(expected.GetLength(0) == actual.GetLength(0), $"{label}: row mismatch");
            Assert(expected.GetLength(1) == actual.GetLength(1), $"{label}: col mismatch");

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    Assert(expected[i, j] == actual[i, j], $"{label}[{i},{j}]: expected {expected[i, j]}, got {actual[i, j]}");
                }
            }
        }

        private void AssertDictionaryEqual(Dictionary<string, int> expected, Dictionary<string, int> actual, string label)
        {
            Assert(expected.Count == actual.Count, $"{label}: count mismatch {expected.Count} vs {actual.Count}");
            foreach (var kv in expected)
            {
                Assert(actual.TryGetValue(kv.Key, out int value), $"{label}: missing key {kv.Key}");
                Assert(value == kv.Value, $"{label}: value mismatch for {kv.Key}: {kv.Value} vs {value}");
            }
        }

        private void AssertPairDictionaryEqual(Dictionary<(string left, string right), int> expected, Dictionary<(string left, string right), int> actual, string label)
        {
            Assert(expected.Count == actual.Count, $"{label}: count mismatch {expected.Count} vs {actual.Count}");
            foreach (var kv in expected)
            {
                Assert(actual.TryGetValue(kv.Key, out int value), $"{label}: missing key {kv.Key}");
                Assert(value == kv.Value, $"{label}: value mismatch for {kv.Key}: {kv.Value} vs {value}");
            }
        }

        private void AssertWordDictionaryEqual(Dictionary<List<string>, int> expected, Dictionary<List<string>, int> actual, string label)
        {
            Assert(expected.Count == actual.Count, $"{label}: count mismatch {expected.Count} vs {actual.Count}");
            foreach (var kv in expected)
            {
                bool found = actual.Any(a => a.Value == kv.Value && a.Key.SequenceEqual(kv.Key));
                Assert(found, $"{label}: missing merged word [{string.Join(" ", kv.Key)}] => {kv.Value}");
            }
        }

        private static (float[,] Q, float[,] K, float[,] V) ProjectQKVExpected(float[,] input, float[,] wq, float[] biasQ, float[,] wk, float[] biasK, float[,] wv, float[] biasV)
        {
            int rows = input.GetLength(0);
            int inputDim = input.GetLength(1);
            int outputDim = wq.GetLength(0);
            var q = new float[rows, outputDim];
            var k = new float[rows, outputDim];
            var v = new float[rows, outputDim];

            for (int i = 0; i < rows; i++)
            {
                for (int o = 0; o < outputDim; o++)
                {
                    float qSum = biasQ[o];
                    float kSum = biasK[o];
                    float vSum = biasV[o];

                    for (int d = 0; d < inputDim; d++)
                    {
                        float x = input[i, d];
                        qSum += wq[o, d] * x;
                        kSum += wk[o, d] * x;
                        vSum += wv[o, d] * x;
                    }

                    q[i, o] = qSum;
                    k[i, o] = kSum;
                    v[i, o] = vSum;
                }
            }

            return (q, k, v);
        }

        private static float[,] BackpropQKVExpected(float[,] input, float[,] dQ, float[,] dK, float[,] dV, float[,] wq, float[,] wk, float[,] wv, float[,] wqGrad, float[] biasQGrad, float[,] wkGrad, float[] biasKGrad, float[,] wvGrad, float[] biasVGrad)
        {
            int rows = input.GetLength(0);
            int inputDim = input.GetLength(1);
            int outputDim = dQ.GetLength(1);
            var dInput = new float[rows, inputDim];

            for (int i = 0; i < rows; i++)
            {
                for (int o = 0; o < outputDim; o++)
                {
                    float dq = dQ[i, o];
                    float dk = dK[i, o];
                    float dv = dV[i, o];

                    biasQGrad[o] += dq;
                    biasKGrad[o] += dk;
                    biasVGrad[o] += dv;

                    for (int d = 0; d < inputDim; d++)
                    {
                        float x = input[i, d];
                        wqGrad[o, d] += dq * x;
                        wkGrad[o, d] += dk * x;
                        wvGrad[o, d] += dv * x;
                        dInput[i, d] += dq * wq[o, d] + dk * wk[o, d] + dv * wv[o, d];
                    }
                }
            }

            return dInput;
        }

        private static float[,] Matrix(int rows, int cols, int seed)
        {
            var m = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = Math.Sin((i + 1) * 12.9898 + (j + 1) * 78.233 + seed * 0.137);
                    m[i, j] = (float)(x * 0.25);
                }
            }
            return m;
        }

        private static float[,] Zeros(int rows, int cols) => new float[rows, cols];

        private static float[] Vector(int len, int seed)
        {
            var v = new float[len];
            for (int i = 0; i < len; i++)
            {
                double x = Math.Sin((i + 1) * 19.19 + seed * 0.313);
                v[i] = (float)(x * 0.25);
            }
            return v;
        }

        private static float[] PositiveVector(int len, int seed)
        {
            var v = Vector(len, seed);
            for (int i = 0; i < len; i++)
            {
                v[i] = MathF.Abs(v[i]) + 0.5f;
            }
            return v;
        }

        private static float[,,] Tensor3(int d0, int d1, int d2, int seed)
        {
            var t = new float[d0, d1, d2];
            for (int i = 0; i < d0; i++)
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        double x = Math.Sin((i + 1) * 3.1 + (j + 1) * 5.7 + (k + 1) * 7.9 + seed * 0.101);
                        t[i, j, k] = (float)(x * 0.1);
                    }
                }
            }
            return t;
        }

        private static bool[,] PeriodicMask(int rows, int cols)
        {
            var mask = new bool[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                bool any = false;
                for (int j = 0; j < cols; j++)
                {
                    bool value = j == 0 || ((i * 31 + j * 17) % 7) != 0;
                    mask[i, j] = value;
                    any |= value;
                }
                if (!any) mask[i, 0] = true;
            }
            return mask;
        }

        private static float[,] AddBiasAndPositionExpected(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            var result = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    result[i, j] = projected[i, j] + bias[j] + positionalEncoding[i, j];
                }
            }
            return result;
        }

        private static float[,] EmbedTokensWithPositionExpected(float[,] tokenEmbedding, int[] tokenIds, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            var result = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = tokenIds[i];
                for (int j = 0; j < embeddingDim; j++)
                {
                    result[i, j] = tokenEmbedding[tokenId, j] + positionalEncoding[i, j];
                }
            }
            return result;
        }

        private static float[,] MeanPoolRowsExpected(float[,] hidden, int[] storyOffsets, int[] storyCounts, int numStories, int embeddingDim)
        {
            var result = new float[numStories, embeddingDim];
            for (int s = 0; s < numStories; s++)
            {
                int start = storyOffsets[s];
                int count = storyCounts[s];
                if (count <= 0) continue;

                for (int d = 0; d < embeddingDim; d++)
                {
                    float sum = 0.0f;
                    for (int i = start; i < start + count; i++)
                    {
                        sum += hidden[i, d];
                    }
                    result[s, d] = sum / count;
                }
            }
            return result;
        }

        private static float[,] Clone(float[,] input)
        {
            if (input == null) return null;
            var output = new float[input.GetLength(0), input.GetLength(1)];
            Array.Copy(input, output, input.Length);
            return output;
        }

        private static float[] Clone(float[] input)
        {
            if (input == null) return null;
            var output = new float[input.Length];
            Array.Copy(input, output, input.Length);
            return output;
        }

        private static int[] Clone(int[] input)
        {
            if (input == null) return null;
            var output = new int[input.Length];
            Array.Copy(input, output, input.Length);
            return output;
        }

        private static bool[,] Clone(bool[,] input)
        {
            if (input == null) return null;
            var output = new bool[input.GetLength(0), input.GetLength(1)];
            Array.Copy(input, output, input.Length);
            return output;
        }

        private static float[,,] Clone(float[,,] input)
        {
            if (input == null) return null;
            var output = new float[input.GetLength(0), input.GetLength(1), input.GetLength(2)];
            Array.Copy(input, output, input.Length);
            return output;
        }

        private static float[][,] Clone(float[][,] input)
        {
            if (input == null) return null;
            var output = new float[input.Length][,];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Clone(input[i]);
            }
            return output;
        }

        private static Dictionary<List<string>, int> CloneWords(Dictionary<List<string>, int> words)
        {
            var output = new Dictionary<List<string>, int>(new ListEqualityComparer<string>());
            foreach (var kv in words)
            {
                output[new List<string>(kv.Key)] = kv.Value;
            }
            return output;
        }
    }
}
