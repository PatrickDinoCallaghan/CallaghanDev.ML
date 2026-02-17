using CallaghanDev.ML.AccelerationManagers;
using CallaghanDev.ML.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.TestConsoleApp
{
    public class AccelerationConsistencyTests
    {
        private const float _tolerance_float = 1e-4f;
        private const float _tolerance_stricter = 1e-6f;

        private int _passed = 0;
        private int _failed = 0;
        private List<string> _failures = new List<string>();

        private readonly (Func<IAccelerationManager> factory, string name)[] _backends = new (Func<IAccelerationManager>, string)[]
        {
            (() => new AccelerationCPU(),              "CPU"),
            (() => new AccelerationMutliThreadCPU(),    "MultiThreadCPU"),
            (() => new AccelerationGPU(AccelerationType.GPU),  "GPU"),
            // (() => new AccelerationGPU(AccelerationType.CUDA), "CUDA"),
        };

        public void RunAllTests()
        {
            Console.WriteLine("=== Acceleration Backend Consistency Tests ===");
            _passed = 0;
            _failed = 0;
            _failures.Clear();

            var tests = new (Action test, string name)[]
            {
                // Basic linear algebra
                (Test_CalculateDotProduct,                    "CalculateDotProduct"),
                (Test_ActivateLayer_Relu,                    "ActivateLayer (ReLU)"),
                (Test_ActivateLayer_Sigmoid,                 "ActivateLayer (Sigmoid)"),
                (Test_ActivateLayer_Tanh,                    "ActivateLayer (Tanh)"),
                (Test_ActivateLayer_LeakyRelu,               "ActivateLayer (LeakyReLU)"),
                (Test_CalculateOutputGradients,               "CalculateOutputGradients"),
                (Test_CalculateHiddenGradients,               "CalculateHiddenGradients"),
                (Test_UpdateWeights,                          "UpdateWeights"),
                (Test_UpdateBias,                             "UpdateBias"),

                // Matrix operations
                (Test_MatrixMultiply,                         "MatrixMultiply"),
                (Test_MatrixMultiply_NonSquare,               "MatrixMultiply (non-square)"),
                (Test_MatrixMultiplyTranspose,                "MatrixMultiplyTranspose"),
                (Test_MatrixMultiplyTranspose_NonSquare,      "MatrixMultiplyTranspose (non-square)"),
                (Test_MatrixScale,                            "MatrixScale"),
                (Test_MatrixAdd,                              "MatrixAdd"),
                (Test_BatchDotProduct,                        "BatchDotProduct"),
                (Test_MatrixAddBias,                          "MatrixAddBias"),

                // Softmax
                (Test_Softmax_NoMask,                         "Softmax (no mask)"),
                (Test_Softmax_WithMask,                       "Softmax (with mask)"),

                // LayerNorm
                (Test_LayerNorm,                              "LayerNorm"),
                (Test_LayerNormForward,                       "LayerNormForward"),
                (Test_LayerNormBackward,                      "LayerNormBackward"),

                // Embedding / positional
                (Test_EmbedTokensWithPosition,                "EmbedTokensWithPosition"),
                (Test_AddBiasAndPositionalEncoding,           "AddBiasAndPositionalEncoding"),

                // Loss functions
                (Test_CrossEntropyLossAndGradient,            "CrossEntropyLossAndGradient"),
                (Test_MSELossAndGradient,                     "MSELossAndGradient"),

                // Backprop helpers
                (Test_BackpropOutputProjection,                "BackpropOutputProjection"),
                (Test_BackpropInputProjection,                 "BackpropInputProjection"),
                (Test_BackpropLinearProjection,                "BackpropLinearProjection"),
                (Test_AccumulateTokenEmbeddingGrad,            "AccumulateTokenEmbeddingGrad"),
                (Test_AccumulateVectorGradients,               "AccumulateVectorGradients"),

                // In-place operations
                (Test_MatrixScaleInPlace,                      "MatrixScaleInPlace"),
                (Test_VectorScaleInPlace,                      "VectorScaleInPlace"),
                (Test_MatrixUpdate,                            "MatrixUpdate"),
                (Test_VectorUpdate,                            "VectorUpdate"),
                (Test_ZeroMatrix,                              "ZeroMatrix"),
                (Test_ZeroVector,                              "ZeroVector"),
                (Test_MatrixAccumulate,                        "MatrixAccumulate"),
                (Test_SigmoidInPlace,                          "SigmoidInPlace"),

                // Norms
                (Test_MatrixSquaredNorm,                       "MatrixSquaredNorm"),
                (Test_VectorSquaredNorm,                       "VectorSquaredNorm"),

                // Row operations
                (Test_SliceRows,                               "SliceRows"),
                (Test_ExtractRow,                              "ExtractRow"),
                (Test_SetRow,                                  "SetRow"),
                (Test_CreateCausalMask,                        "CreateCausalMask"),

                // Multi-head attention
                (Test_MHA_Forward_SelfAttention,               "MHA Forward (self-attention)"),
                (Test_MHA_Forward_CrossAttention,              "MHA Forward (cross-attention)"),
                (Test_MHA_Forward_WithMask,                    "MHA Forward (with causal mask)"),
                (Test_MHA_Backward_NoMask,                     "MHA Backward (no mask)"),
                (Test_MHA_Backward_WithMask,                   "MHA Backward (with decoder mask)"),
                (Test_MHA_Backward_CrossAttention,             "MHA Backward (cross-attention)"),

                // Larger sizes to stress parallelism
                (Test_MatrixMultiply_Large,                    "MatrixMultiply (large 64x64)"),
                (Test_MHA_Forward_Large,                       "MHA Forward (large seqLen=32)"),
                (Test_Softmax_Large,                           "Softmax (large 32x64)"),
                (Test_BatchDotProduct_Large,                   "BatchDotProduct (large)"),
                (Test_CrossEntropyLoss_LargeVocab,             "CrossEntropyLoss (large vocab)"),
                (Test_LayerNormForward_Large,                  "LayerNormForward (large batch)"),
                (Test_BackpropOutputProjection_Large,          "BackpropOutputProjection (large)"),
            };

            for (int i = 0; i < tests.Length; i++)
            {
                Console.Write($"  [{i + 1,2}/{tests.Length}] {tests[i].name,-55} ");
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

            PrintSummary(tests.Length);
        }

        #region Helpers

        private void Assert(bool condition, string message)
        {
            if (!condition)
            {
                throw new Exception(message);
            }
        }

        private void AssertClose(float a, float b, string name, float tol = -1f)
        {
            if (tol < 0) tol = _tolerance_float;
            Assert(MathF.Abs(a - b) <= tol, $"{name}: {a} vs {b} (diff={MathF.Abs(a - b)}, tol={tol})");
        }

        private void AssertMatricesClose(float[,] a, float[,] b, string name, float tol = -1f)
        {
            if (tol < 0) tol = _tolerance_float;
            Assert(a.GetLength(0) == b.GetLength(0) && a.GetLength(1) == b.GetLength(1),
                $"{name}: shape mismatch [{a.GetLength(0)},{a.GetLength(1)}] vs [{b.GetLength(0)},{b.GetLength(1)}]");
            for (int i = 0; i < a.GetLength(0); i++)
                for (int j = 0; j < a.GetLength(1); j++)
                    Assert(MathF.Abs(a[i, j] - b[i, j]) <= tol,
                        $"{name}[{i},{j}]: {a[i, j]} vs {b[i, j]} (diff={MathF.Abs(a[i, j] - b[i, j])})");
        }

        private void AssertVectorsClose(float[] a, float[] b, string name, float tol = -1f)
        {
            if (tol < 0) tol = _tolerance_float;
            Assert(a.Length == b.Length, $"{name}: length mismatch {a.Length} vs {b.Length}");
            for (int i = 0; i < a.Length; i++)
                Assert(MathF.Abs(a[i] - b[i]) <= tol,
                    $"{name}[{i}]: {a[i]} vs {b[i]} (diff={MathF.Abs(a[i] - b[i])})");
        }

        private float[,] RandMatrix(Random rng, int rows, int cols, float scale = 1f)
        {
            var m = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = ((float)rng.NextDouble() - 0.5f) * 2f * scale;
            return m;
        }

        private float[] RandVector(Random rng, int len, float scale = 1f)
        {
            var v = new float[len];
            for (int i = 0; i < len; i++)
                v[i] = ((float)rng.NextDouble() - 0.5f) * 2f * scale;
            return v;
        }

        private bool[,] MakeCausalMask(int seqLen)
        {
            var mask = new bool[seqLen, seqLen];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j <= i; j++)
                    mask[i, j] = true;
            return mask;
        }

        private void CompareAllBackends<T>(string testName, Func<IAccelerationManager, T> run, Action<T, T, string> compare)
        {
            var results = new (T result, string name)[_backends.Length];
            for (int i = 0; i < _backends.Length; i++)
            {
                var mgr = _backends[i].factory();
                try
                {
                    results[i] = (run(mgr), _backends[i].name);
                }
                finally
                {
                    (mgr as IDisposable)?.Dispose();
                }
            }

            for (int i = 1; i < results.Length; i++)
            {
                compare(results[0].result, results[i].result, $"{testName}: {results[0].name} vs {results[i].name}");
            }
        }
        private void PrintSummary(int total)
        {
            Console.WriteLine($"\n{"",3}{new string('─', 58)}");
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
            Console.WriteLine($" / {total} total\n");
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

        #endregion

        // =====================================================================
        // Basic linear algebra
        // =====================================================================

        public void Test_CalculateDotProduct()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 16, 12);
            var vector = RandVector(rng, 12);

            CompareAllBackends("CalculateDotProduct",
                mgr => mgr.CalculateDotProduct(matrix, vector),
                (a, b, name) => AssertVectorsClose(a, b, name));
        }

        public void Test_ActivateLayer_Relu()
        {
            var rng = new Random(42);
            var dot = RandVector(rng, 32);
            var bias = RandVector(rng, 32);

            CompareAllBackends("ActivateLayer_Relu",
                mgr => mgr.ActivateLayer(dot, bias, ActivationType.Relu),
                (a, b, name) =>
                {
                    AssertVectorsClose(a.activation, b.activation, name + ".activation");
                    AssertVectorsClose(a.derivative, b.derivative, name + ".derivative");
                });
        }

        public void Test_ActivateLayer_Sigmoid()
        {
            var rng = new Random(42);
            var dot = RandVector(rng, 32);
            var bias = RandVector(rng, 32);

            CompareAllBackends("ActivateLayer_Sigmoid",
                mgr => mgr.ActivateLayer(dot, bias, ActivationType.Sigmoid),
                (a, b, name) =>
                {
                    AssertVectorsClose(a.activation, b.activation, name + ".activation");
                    AssertVectorsClose(a.derivative, b.derivative, name + ".derivative");
                });
        }

        public void Test_ActivateLayer_Tanh()
        {
            var rng = new Random(42);
            var dot = RandVector(rng, 32);
            var bias = RandVector(rng, 32);

            CompareAllBackends("ActivateLayer_Tanh",
                mgr => mgr.ActivateLayer(dot, bias, ActivationType.Tanh),
                (a, b, name) =>
                {
                    AssertVectorsClose(a.activation, b.activation, name + ".activation");
                    AssertVectorsClose(a.derivative, b.derivative, name + ".derivative");
                });
        }

        public void Test_ActivateLayer_LeakyRelu()
        {
            var rng = new Random(42);
            var dot = RandVector(rng, 32);
            var bias = RandVector(rng, 32);

            CompareAllBackends("ActivateLayer_LeakyRelu",
                mgr => mgr.ActivateLayer(dot, bias, ActivationType.Leakyrelu),
                (a, b, name) =>
                {
                    AssertVectorsClose(a.activation, b.activation, name + ".activation");
                    AssertVectorsClose(a.derivative, b.derivative, name + ".derivative");
                });
        }

        public void Test_CalculateOutputGradients()
        {
            var rng = new Random(42);
            var cost = RandVector(rng, 20);
            var derivative = RandVector(rng, 20);

            CompareAllBackends("CalculateOutputGradients",
                mgr => mgr.CalculateOutputGradients(cost, derivative),
                (a, b, name) => AssertVectorsClose(a, b, name));
        }

        public void Test_CalculateHiddenGradients()
        {
            var rng = new Random(42);
            var weights = RandMatrix(rng, 16, 12);
            var nextDeltas = RandVector(rng, 16);
            var derivative = RandVector(rng, 12);

            CompareAllBackends("CalculateHiddenGradients",
                mgr => mgr.CalculateHiddenGradients(weights, nextDeltas, derivative),
                (a, b, name) => AssertVectorsClose(a, b, name));
        }

        public void Test_UpdateWeights()
        {
            var rng = new Random(42);
            var weights = RandMatrix(rng, 10, 8);
            var deltas = RandVector(rng, 10);
            var prevActivations = RandVector(rng, 8);
            float lr = 0.01f;
            float lambda = 0.001f;

            CompareAllBackends("UpdateWeights",
                mgr => mgr.UpdateWeights(weights, deltas, prevActivations, lr, lambda),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_UpdateBias()
        {
            var rng = new Random(42);
            var bias = RandVector(rng, 16);
            var deltas = RandVector(rng, 16);
            float lr = 0.01f;

            CompareAllBackends("UpdateBias",
                mgr => mgr.UpdateBias(bias, deltas, lr),
                (a, b, name) => AssertVectorsClose(a, b, name));
        }

        // =====================================================================
        // Matrix operations
        // =====================================================================

        public void Test_MatrixMultiply()
        {
            var rng = new Random(42);
            var A = RandMatrix(rng, 8, 12);
            var B = RandMatrix(rng, 12, 6);

            CompareAllBackends("MatrixMultiply",
                mgr => mgr.MatrixMultiply(A, B),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MatrixMultiply_NonSquare()
        {
            var rng = new Random(42);
            var A = RandMatrix(rng, 3, 17);
            var B = RandMatrix(rng, 17, 5);

            CompareAllBackends("MatrixMultiply_NonSquare",
                mgr => mgr.MatrixMultiply(A, B),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MatrixMultiplyTranspose()
        {
            var rng = new Random(42);
            var A = RandMatrix(rng, 8, 10);
            var B = RandMatrix(rng, 6, 10); // B^T: 10x6, so A*B^T = 8x6

            CompareAllBackends("MatrixMultiplyTranspose",
                mgr => mgr.MatrixMultiplyTranspose(A, B),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MatrixMultiplyTranspose_NonSquare()
        {
            var rng = new Random(42);
            var A = RandMatrix(rng, 5, 13);
            var B = RandMatrix(rng, 9, 13);

            CompareAllBackends("MatrixMultiplyTranspose_NonSquare",
                mgr => mgr.MatrixMultiplyTranspose(A, B),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MatrixScale()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 10, 8);
            float scalar = 0.73f;

            CompareAllBackends("MatrixScale",
                mgr => mgr.MatrixScale(matrix, scalar),
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        public void Test_MatrixAdd()
        {
            var rng = new Random(42);
            var A = RandMatrix(rng, 10, 8);
            var B = RandMatrix(rng, 10, 8);

            CompareAllBackends("MatrixAdd",
                mgr => mgr.MatrixAdd(A, B),
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        public void Test_BatchDotProduct()
        {
            var rng = new Random(42);
            var weights = RandMatrix(rng, 16, 8);   // [outputDim, inputDim]
            var input = RandMatrix(rng, 6, 8);       // [seqLen, inputDim]

            CompareAllBackends("BatchDotProduct",
                mgr => mgr.BatchDotProduct(weights, input),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MatrixAddBias()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 8, 12);
            var bias = RandVector(rng, 12);

            CompareAllBackends("MatrixAddBias",
                mgr => mgr.MatrixAddBias(matrix, bias),
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        // =====================================================================
        // Softmax
        // =====================================================================

        public void Test_Softmax_NoMask()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 8, 10);

            CompareAllBackends("Softmax_NoMask",
                mgr => mgr.Softmax(matrix, null),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_Softmax_WithMask()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 6, 6);
            var mask = MakeCausalMask(6);

            CompareAllBackends("Softmax_WithMask",
                mgr => mgr.Softmax(matrix, mask),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        // =====================================================================
        // LayerNorm
        // =====================================================================

        public void Test_LayerNorm()
        {
            var rng = new Random(42);
            var input = RandMatrix(rng, 6, 16);
            var gamma = RandVector(rng, 16);
            var beta = RandVector(rng, 16);

            CompareAllBackends("LayerNorm",
                mgr => mgr.LayerNorm(input, gamma, beta),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_LayerNormForward()
        {
            var rng = new Random(42);
            var input = RandMatrix(rng, 6, 16);
            var gamma = RandVector(rng, 16);
            var beta = RandVector(rng, 16);

            CompareAllBackends("LayerNormForward",
                mgr => mgr.LayerNormForward(input, gamma, beta),
                (a, b, name) =>
                {
                    AssertMatricesClose(a.output, b.output, name + ".output");
                    AssertVectorsClose(a.means, b.means, name + ".means");
                    AssertVectorsClose(a.variances, b.variances, name + ".variances");
                    AssertMatricesClose(a.normalized, b.normalized, name + ".normalized");
                });
        }

        public void Test_LayerNormBackward()
        {
            var rng = new Random(42);
            int batch = 4, features = 8;
            var input = RandMatrix(rng, batch, features);
            var gamma = RandVector(rng, features);
            var beta = RandVector(rng, features);

            // Compute forward with CPU to get shared inputs for backward
            var cpu = new AccelerationCPU();
            var (_, means, variances, normalized) = cpu.LayerNormForward(input, gamma, beta);
            var dOut = RandMatrix(rng, batch, features);

            CompareAllBackends("LayerNormBackward",
                mgr => mgr.LayerNormBackward(dOut, normalized, gamma, input, means, variances),
                (a, b, name) =>
                {
                    AssertMatricesClose(a.dInput, b.dInput, name + ".dInput");
                    AssertVectorsClose(a.dGamma, b.dGamma, name + ".dGamma");
                    AssertVectorsClose(a.dBeta, b.dBeta, name + ".dBeta");
                });
        }

        // =====================================================================
        // Embedding / positional
        // =====================================================================

        public void Test_EmbedTokensWithPosition()
        {
            var rng = new Random(42);
            int vocabSize = 20, embDim = 8, seqLen = 6;
            var tokenEmbedding = RandMatrix(rng, vocabSize, embDim);
            var tokenIds = new int[] { 3, 7, 1, 15, 0, 12 };
            var positionalEncoding = RandMatrix(rng, seqLen, embDim);

            CompareAllBackends("EmbedTokensWithPosition",
                mgr => mgr.EmbedTokensWithPosition(tokenEmbedding, tokenIds, positionalEncoding, seqLen, embDim),
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        public void Test_AddBiasAndPositionalEncoding()
        {
            var rng = new Random(42);
            int seqLen = 6, embDim = 8;
            var projected = RandMatrix(rng, seqLen, embDim);
            var bias = RandVector(rng, embDim);
            var positionalEncoding = RandMatrix(rng, seqLen, embDim);

            CompareAllBackends("AddBiasAndPositionalEncoding",
                mgr => mgr.AddBiasAndPositionalEncoding(projected, bias, positionalEncoding, seqLen, embDim),
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        // =====================================================================
        // Loss functions
        // =====================================================================

        public void Test_CrossEntropyLossAndGradient()
        {
            var rng = new Random(42);
            int seqLen = 6, vocabSize = 10;
            var logits = RandMatrix(rng, seqLen, vocabSize);
            var targets = new int[] { 3, 7, 1, 5, 0, 9 };

            CompareAllBackends("CrossEntropyLossAndGradient",
                mgr => mgr.CrossEntropyLossAndGradient(logits, targets, seqLen),
                (a, b, name) =>
                {
                    AssertClose(a.loss, b.loss, name + ".loss");
                    AssertMatricesClose(a.dLogits, b.dLogits, name + ".dLogits");
                });
        }

        public void Test_MSELossAndGradient()
        {
            var rng = new Random(42);
            int seqLen = 6, outputDim = 3;
            var predictions = RandMatrix(rng, seqLen, outputDim);
            var targets = RandMatrix(rng, seqLen, outputDim);

            CompareAllBackends("MSELossAndGradient",
                mgr => mgr.MSELossAndGradient(predictions, targets, seqLen),
                (a, b, name) =>
                {
                    AssertClose(a.loss, b.loss, name + ".loss");
                    AssertMatricesClose(a.dOutput, b.dOutput, name + ".dOutput");
                });
        }

        // =====================================================================
        // Backprop helpers
        // =====================================================================

        public void Test_BackpropOutputProjection()
        {
            var rng = new Random(42);
            int seqLen = 4, outputDim = 8, embDim = 6;
            var dLogits = RandMatrix(rng, seqLen, outputDim);
            var input = RandMatrix(rng, seqLen, embDim);
            var weights = RandMatrix(rng, outputDim, embDim);

            CompareAllBackends("BackpropOutputProjection",
                mgr =>
                {
                    var wg = new float[outputDim, embDim];
                    var bg = new float[outputDim];
                    var dX = mgr.BackpropOutputProjection(dLogits, input, weights, wg, bg, seqLen, outputDim, embDim);
                    return (dX, wg, bg);
                },
                (a, b, name) =>
                {
                    AssertMatricesClose(a.dX, b.dX, name + ".dX");
                    AssertMatricesClose(a.wg, b.wg, name + ".weightGrad");
                    AssertVectorsClose(a.bg, b.bg, name + ".biasGrad");
                });
        }

        public void Test_BackpropInputProjection()
        {
            var rng = new Random(42);
            int seqLen = 4, embDim = 8, inputFeatureDim = 3;
            var dX = RandMatrix(rng, seqLen, embDim);
            var continuousInput = RandMatrix(rng, seqLen, inputFeatureDim);

            CompareAllBackends("BackpropInputProjection",
                mgr =>
                {
                    var wg = new float[embDim, inputFeatureDim];
                    var bg = new float[embDim];
                    mgr.BackpropInputProjection(dX, continuousInput, wg, bg, seqLen, embDim, inputFeatureDim);
                    return (wg, bg);
                },
                (a, b, name) =>
                {
                    AssertMatricesClose(a.wg, b.wg, name + ".weightGrad");
                    AssertVectorsClose(a.bg, b.bg, name + ".biasGrad");
                });
        }

        public void Test_BackpropLinearProjection()
        {
            var rng = new Random(42);
            int seqLen = 4, embDim = 8;
            var input = RandMatrix(rng, seqLen, embDim);
            var dOutput = RandMatrix(rng, seqLen, embDim);
            var weights = RandMatrix(rng, embDim, embDim);

            CompareAllBackends("BackpropLinearProjection",
                mgr =>
                {
                    var wg = new float[embDim, embDim];
                    var bg = new float[embDim];
                    var dInput = new float[seqLen, embDim];
                    mgr.BackpropLinearProjection(input, dOutput, weights, wg, bg, dInput);
                    return (wg, bg, dInput);
                },
                (a, b, name) =>
                {
                    AssertMatricesClose(a.wg, b.wg, name + ".weightGrad");
                    AssertVectorsClose(a.bg, b.bg, name + ".biasGrad");
                    AssertMatricesClose(a.dInput, b.dInput, name + ".dInput");
                });
        }

        public void Test_AccumulateTokenEmbeddingGrad()
        {
            var rng = new Random(42);
            int vocabSize = 10, embDim = 8, seqLen = 6;
            var dX = RandMatrix(rng, seqLen, embDim);
            var tokenIds = new int[] { 3, 7, 1, 3, 0, 7 }; // includes duplicates

            CompareAllBackends("AccumulateTokenEmbeddingGrad",
                mgr =>
                {
                    var grad = new float[vocabSize, embDim];
                    mgr.AccumulateTokenEmbeddingGrad(grad, dX, tokenIds, seqLen, embDim);
                    return grad;
                },
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_AccumulateVectorGradients()
        {
            var rng = new Random(42);
            var source = RandVector(rng, 16);

            CompareAllBackends("AccumulateVectorGradients",
                mgr =>
                {
                    var target = RandVector(new Random(99), 16); // same initial for all
                    mgr.AccumulateVectorGradients(target, source);
                    return target;
                },
                (a, b, name) => AssertVectorsClose(a, b, name));
        }

        // =====================================================================
        // In-place operations
        // =====================================================================

        public void Test_MatrixScaleInPlace()
        {
            var rng = new Random(42);
            float scalar = 0.5f;

            CompareAllBackends("MatrixScaleInPlace",
                mgr =>
                {
                    var m = RandMatrix(new Random(42), 8, 6);
                    mgr.MatrixScaleInPlace(m, scalar);
                    return m;
                },
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        public void Test_VectorScaleInPlace()
        {
            float scalar = 2.5f;

            CompareAllBackends("VectorScaleInPlace",
                mgr =>
                {
                    var v = RandVector(new Random(42), 16);
                    mgr.VectorScaleInPlace(v, scalar);
                    return v;
                },
                (a, b, name) => AssertVectorsClose(a, b, name, _tolerance_stricter));
        }

        public void Test_MatrixUpdate()
        {
            var rng = new Random(42);
            var gradients = RandMatrix(rng, 8, 6);
            float lr = 0.01f;

            CompareAllBackends("MatrixUpdate",
                mgr =>
                {
                    var w = RandMatrix(new Random(42), 8, 6);
                    mgr.MatrixUpdate(w, gradients, lr);
                    return w;
                },
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        public void Test_VectorUpdate()
        {
            var rng = new Random(42);
            var gradients = RandVector(rng, 16);
            float lr = 0.01f;

            CompareAllBackends("VectorUpdate",
                mgr =>
                {
                    var w = RandVector(new Random(42), 16);
                    mgr.VectorUpdate(w, gradients, lr);
                    return w;
                },
                (a, b, name) => AssertVectorsClose(a, b, name, _tolerance_stricter));
        }

        public void Test_ZeroMatrix()
        {
            CompareAllBackends("ZeroMatrix",
                mgr =>
                {
                    var m = RandMatrix(new Random(42), 8, 6);
                    mgr.ZeroMatrix(m);
                    return m;
                },
                (a, b, name) => AssertMatricesClose(a, b, name, 0f));
        }

        public void Test_ZeroVector()
        {
            CompareAllBackends("ZeroVector",
                mgr =>
                {
                    var v = RandVector(new Random(42), 16);
                    mgr.ZeroVector(v);
                    return v;
                },
                (a, b, name) => AssertVectorsClose(a, b, name, 0f));
        }

        public void Test_MatrixAccumulate()
        {
            var rng = new Random(42);
            var source = RandMatrix(rng, 8, 6);

            CompareAllBackends("MatrixAccumulate",
                mgr =>
                {
                    var target = RandMatrix(new Random(99), 8, 6);
                    mgr.MatrixAccumulate(target, source);
                    return target;
                },
                (a, b, name) => AssertMatricesClose(a, b, name, _tolerance_stricter));
        }

        public void Test_SigmoidInPlace()
        {
            CompareAllBackends("SigmoidInPlace",
                mgr =>
                {
                    var m = RandMatrix(new Random(42), 8, 6, scale: 3f);
                    mgr.SigmoidInPlace(m);
                    return m;
                },
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        // =====================================================================
        // Norms
        // =====================================================================

        public void Test_MatrixSquaredNorm()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 10, 8);

            CompareAllBackends("MatrixSquaredNorm",
                mgr => mgr.MatrixSquaredNorm(matrix),
                (a, b, name) => AssertClose(a, b, name));
        }

        public void Test_VectorSquaredNorm()
        {
            var rng = new Random(42);
            var vector = RandVector(rng, 32);

            CompareAllBackends("VectorSquaredNorm",
                mgr => mgr.VectorSquaredNorm(vector),
                (a, b, name) => AssertClose(a, b, name));
        }

        // =====================================================================
        // Row operations
        // =====================================================================

        public void Test_SliceRows()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 10, 8);

            CompareAllBackends("SliceRows",
                mgr => mgr.SliceRows(matrix, 2, 7),
                (a, b, name) => AssertMatricesClose(a, b, name, 0f));
        }

        public void Test_ExtractRow()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 10, 8);

            CompareAllBackends("ExtractRow",
                mgr => mgr.ExtractRow(matrix, 4, 8),
                (a, b, name) => AssertVectorsClose(a, b, name, 0f));
        }

        public void Test_SetRow()
        {
            var rng = new Random(42);
            var values = RandVector(rng, 8);

            CompareAllBackends("SetRow",
                mgr =>
                {
                    var m = RandMatrix(new Random(99), 10, 8);
                    mgr.SetRow(m, 3, values, 8);
                    return m;
                },
                (a, b, name) => AssertMatricesClose(a, b, name, 0f));
        }

        public void Test_CreateCausalMask()
        {
            CompareAllBackends("CreateCausalMask",
                mgr => mgr.CreateCausalMask(8),
                (a, b, name) =>
                {
                    Assert(a.GetLength(0) == b.GetLength(0) && a.GetLength(1) == b.GetLength(1),
                        $"{name}: shape mismatch");
                    for (int i = 0; i < a.GetLength(0); i++)
                        for (int j = 0; j < a.GetLength(1); j++)
                            Assert(a[i, j] == b[i, j], $"{name}[{i},{j}]: {a[i, j]} vs {b[i, j]}");
                });
        }

        // =====================================================================
        // Multi-head attention
        // =====================================================================

        public void Test_MHA_Forward_SelfAttention()
        {
            var rng = new Random(42);
            int seqLen = 6, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = RandMatrix(rng, seqLen, embDim);
            var K = RandMatrix(rng, seqLen, embDim);
            var V = RandMatrix(rng, seqLen, embDim);

            CompareAllBackends("MHA_Forward_Self",
                mgr => mgr.MultiHeadAttentionForward(Q, K, V, numHeads, scale, null),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MHA_Forward_CrossAttention()
        {
            var rng = new Random(42);
            int seqLenQ = 5, seqLenK = 8, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = RandMatrix(rng, seqLenQ, embDim);
            var K = RandMatrix(rng, seqLenK, embDim);
            var V = RandMatrix(rng, seqLenK, embDim);

            CompareAllBackends("MHA_Forward_Cross",
                mgr => mgr.MultiHeadAttentionForward(Q, K, V, numHeads, scale, null),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MHA_Forward_WithMask()
        {
            var rng = new Random(42);
            int seqLen = 6, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = RandMatrix(rng, seqLen, embDim);
            var K = RandMatrix(rng, seqLen, embDim);
            var V = RandMatrix(rng, seqLen, embDim);
            var mask = MakeCausalMask(seqLen);

            CompareAllBackends("MHA_Forward_Masked",
                mgr => mgr.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_MHA_Backward_NoMask()
        {
            var rng = new Random(42);
            int seqLen = 6, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = RandMatrix(rng, seqLen, embDim);
            var K = RandMatrix(rng, seqLen, embDim);
            var V = RandMatrix(rng, seqLen, embDim);
            var dConcat = RandMatrix(rng, seqLen, embDim);

            CompareAllBackends("MHA_Backward_NoMask",
                mgr => mgr.MultiHeadAttentionBackward(Q, K, V, dConcat, numHeads, scale, false),
                (a, b, name) =>
                {
                    AssertMatricesClose(a.dQ, b.dQ, name + ".dQ");
                    AssertMatricesClose(a.dK, b.dK, name + ".dK");
                    AssertMatricesClose(a.dV, b.dV, name + ".dV");
                });
        }

        public void Test_MHA_Backward_WithMask()
        {
            var rng = new Random(42);
            int seqLen = 6, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = RandMatrix(rng, seqLen, embDim);
            var K = RandMatrix(rng, seqLen, embDim);
            var V = RandMatrix(rng, seqLen, embDim);
            var dConcat = RandMatrix(rng, seqLen, embDim);

            CompareAllBackends("MHA_Backward_Masked",
                mgr => mgr.MultiHeadAttentionBackward(Q, K, V, dConcat, numHeads, scale, true),
                (a, b, name) =>
                {
                    AssertMatricesClose(a.dQ, b.dQ, name + ".dQ");
                    AssertMatricesClose(a.dK, b.dK, name + ".dK");
                    AssertMatricesClose(a.dV, b.dV, name + ".dV");
                });
        }

        public void Test_MHA_Backward_CrossAttention()
        {
            var rng = new Random(42);
            int seqLenQ = 5, seqLenK = 8, embDim = 8, numHeads = 2;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = RandMatrix(rng, seqLenQ, embDim);
            var K = RandMatrix(rng, seqLenK, embDim);
            var V = RandMatrix(rng, seqLenK, embDim);
            var dConcat = RandMatrix(rng, seqLenQ, embDim);

            CompareAllBackends("MHA_Backward_Cross",
                mgr => mgr.MultiHeadAttentionBackward(Q, K, V, dConcat, numHeads, scale, false),
                (a, b, name) =>
                {
                    AssertMatricesClose(a.dQ, b.dQ, name + ".dQ");
                    AssertMatricesClose(a.dK, b.dK, name + ".dK");
                    AssertMatricesClose(a.dV, b.dV, name + ".dV");
                });
        }

        // =====================================================================
        // Larger sizes to stress parallel paths
        // =====================================================================

        public void Test_MatrixMultiply_Large()
        {
            var rng = new Random(42);
            var A = RandMatrix(rng, 64, 64);
            var B = RandMatrix(rng, 64, 64);

            CompareAllBackends("MatrixMultiply_Large",
                mgr => mgr.MatrixMultiply(A, B),
                (a, b, name) => AssertMatricesClose(a, b, name, 1e-3f));
        }

        public void Test_MHA_Forward_Large()
        {
            var rng = new Random(42);
            int seqLen = 32, embDim = 16, numHeads = 4;
            float scale = 1.0f / MathF.Sqrt(embDim / numHeads);
            var Q = RandMatrix(rng, seqLen, embDim);
            var K = RandMatrix(rng, seqLen, embDim);
            var V = RandMatrix(rng, seqLen, embDim);
            var mask = MakeCausalMask(seqLen);

            CompareAllBackends("MHA_Forward_Large",
                mgr => mgr.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask),
                (a, b, name) => AssertMatricesClose(a, b, name, 1e-4f));
        }

        public void Test_Softmax_Large()
        {
            var rng = new Random(42);
            var matrix = RandMatrix(rng, 32, 64);

            CompareAllBackends("Softmax_Large",
                mgr => mgr.Softmax(matrix, null),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_BatchDotProduct_Large()
        {
            var rng = new Random(42);
            var weights = RandMatrix(rng, 32, 16);
            var input = RandMatrix(rng, 24, 16);

            CompareAllBackends("BatchDotProduct_Large",
                mgr => mgr.BatchDotProduct(weights, input),
                (a, b, name) => AssertMatricesClose(a, b, name));
        }

        public void Test_CrossEntropyLoss_LargeVocab()
        {
            var rng = new Random(42);
            int seqLen = 16, vocabSize = 100;
            var logits = RandMatrix(rng, seqLen, vocabSize);
            var targets = new int[seqLen];
            for (int i = 0; i < seqLen; i++) targets[i] = rng.Next(vocabSize);

            CompareAllBackends("CrossEntropyLoss_LargeVocab",
                mgr => mgr.CrossEntropyLossAndGradient(logits, targets, seqLen),
                (a, b, name) =>
                {
                    AssertClose(a.loss, b.loss, name + ".loss");
                    AssertMatricesClose(a.dLogits, b.dLogits, name + ".dLogits");
                });
        }

        public void Test_LayerNormForward_Large()
        {
            var rng = new Random(42);
            int batch = 16, features = 32;
            var input = RandMatrix(rng, batch, features);
            var gamma = RandVector(rng, features);
            var beta = RandVector(rng, features);

            CompareAllBackends("LayerNormForward_Large",
                mgr => mgr.LayerNormForward(input, gamma, beta),
                (a, b, name) =>
                {
                    AssertMatricesClose(a.output, b.output, name + ".output");
                    AssertVectorsClose(a.means, b.means, name + ".means");
                    AssertVectorsClose(a.variances, b.variances, name + ".variances");
                    AssertMatricesClose(a.normalized, b.normalized, name + ".normalized");
                });
        }

        public void Test_BackpropOutputProjection_Large()
        {
            var rng = new Random(42);
            int seqLen = 16, outputDim = 32, embDim = 16;
            var dLogits = RandMatrix(rng, seqLen, outputDim);
            var input = RandMatrix(rng, seqLen, embDim);
            var weights = RandMatrix(rng, outputDim, embDim);

            CompareAllBackends("BackpropOutputProjection_Large",
                mgr =>
                {
                    var wg = new float[outputDim, embDim];
                    var bg = new float[outputDim];
                    var dX = mgr.BackpropOutputProjection(dLogits, input, weights, wg, bg, seqLen, outputDim, embDim);
                    return (dX, wg, bg);
                },
                (a, b, name) =>
                {
                    AssertMatricesClose(a.dX, b.dX, name + ".dX", 1e-3f);
                    AssertMatricesClose(a.wg, b.wg, name + ".weightGrad", 1e-3f);
                    AssertVectorsClose(a.bg, b.bg, name + ".biasGrad", 1e-3f);
                });
        }
    }
}
