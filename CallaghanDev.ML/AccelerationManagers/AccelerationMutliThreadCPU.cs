using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.TACAMT;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationMutliThreadCPU : IAccelerationManager
    {
        private readonly ParallelOptions _parallelOptions;

        // Minimum number of "work units" (e.g. rows * cols) before we bother
        // spinning up the thread-pool.  Below this threshold every method falls
        // back to a plain sequential loop, which avoids the ~2-5 µs overhead of
        // Parallel.For on tiny arrays.
        private const int PARALLEL_THRESHOLD = 512;

        private const bool AlwaysParallel = false; // Just for testing purposes. Forgot to turn it off.

        private AccelerationCPU _singleThreadCPU = null;

        private readonly object _singleThreadCPULock = new object();
        public AccelerationMutliThreadCPU()
        {
            _parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };
        }

        private bool ShouldParallelize(int workUnits)
        {
            if (workUnits < PARALLEL_THRESHOLD && !AlwaysParallel)
            {
                if (_singleThreadCPU == null)
                {
                    lock (_singleThreadCPULock)
                    {
                        // Double-check pattern to avoid multiple instantiations
                        if (_singleThreadCPU == null)
                        {
                            _singleThreadCPU = new AccelerationCPU();
                        }
                    }
                }
            }
            return workUnits >= PARALLEL_THRESHOLD || AlwaysParallel;
        }


        #region Shared Tensor primitives

        public float[,] MatrixMultiply(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int colsB = B.GetLength(1);


            if (!ShouldParallelize(rowsA * colsB))
            {
                return _singleThreadCPU.MatrixMultiply(A, B);
            }

            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);

            if (colsA != rowsB)
            {
                throw new ArgumentException($"Matrix dimensions don't match: [{rowsA}x{colsA}] * [{rowsB}x{colsB}]");
            }

            var C = new float[rowsA, colsB];
            const int BLOCK = 32;

            Parallel.For(0, (rowsA + BLOCK - 1) / BLOCK, _parallelOptions, ii =>
            {
                int iStart = ii * BLOCK;
                int iEnd = Math.Min(iStart + BLOCK, rowsA);

                for (int jj = 0; jj < colsB; jj += BLOCK)
                {
                    int jEnd = Math.Min(jj + BLOCK, colsB);
                    for (int kk = 0; kk < colsA; kk += BLOCK)
                    {
                        int kEnd = Math.Min(kk + BLOCK, colsA);
                        for (int i = iStart; i < iEnd; i++)
                        {
                            for (int j = jj; j < jEnd; j++)
                            {
                                float sum = C[i, j];
                                for (int k = kk; k < kEnd; k++)
                                {
                                    sum += A[i, k] * B[k, j];
                                }
                                C[i, j] = sum;
                            }
                        }
                    }
                }
            });
            return C;
        }

        public float[,] MatrixMultiplyTranspose(float[,] A, float[,] B)
        {
            int rowsA = A.GetLength(0);
            int rowsB = B.GetLength(0);

            if (!ShouldParallelize(rowsA * rowsB))
            {
                return _singleThreadCPU.MatrixMultiplyTranspose(A, B);
            }

            int colsB = B.GetLength(1);
            int colsA = A.GetLength(1);

            if (colsA != colsB)
            {
                throw new ArgumentException($"Matrix dimensions don't match for A*B^T");
            }

            var C = new float[rowsA, rowsB];


            Parallel.For(0, rowsA, _parallelOptions, i =>
            {
                for (int j = 0; j < rowsB; j++)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < colsA; k++)
                    {
                        sum += A[i, k] * B[j, k];
                    }
                    C[i, j] = sum;
                }
            });
            return C;
        }

        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.MatrixScale(matrix, scalar);
            }

            var result = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            });

            return result;
        }

        public float[,] MatrixAdd(float[,] A, float[,] B)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.MatrixAdd(A, B);
            }
            var result = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = A[i, j] + B[i, j];
                }
            });

            return result;
        }

        public float[,] MatrixAddBias(float[,] matrix, float[] bias)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.MatrixAddBias(matrix, bias);
            }

            var result = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] + bias[j];
                }
            });
            return result;
        }

        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix)
        {
            return BatchDotProduct(weights, inputMatrix, 0, inputMatrix.GetLength(0));
        }

        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix, int rowStart, int rowCount)
        {
            if (rowStart < 0 || rowCount < 0)
            {
                throw new ArgumentOutOfRangeException();
            }

            if (rowStart + rowCount > inputMatrix.GetLength(0))
            {
                throw new ArgumentException("Invalid row slice.");
            }

            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);

            if (inputMatrix.GetLength(1) != inputDim)
            {
                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");
            }

            if (!ShouldParallelize(rowCount * outputDim))
            {
                return _singleThreadCPU.BatchDotProduct(weights, inputMatrix, rowStart, rowCount);
            }

            var result = new float[rowCount, outputDim];

            Parallel.For(0, rowCount, _parallelOptions, i =>
            {
                int srcRow = rowStart + i;

                for (int j = 0; j < outputDim; j++)
                {
                    float sum = 0.0f;

                    for (int k = 0; k < inputDim; k++)
                    {
                        sum += weights[j, k] * inputMatrix[srcRow, k];
                    }

                    result[i, j] = sum;
                }
            });

            return result;
        }

        public float[] ExtractRow(float[,] matrix, int rowIndex, int cols)
        {
            var result = new float[cols];

            Buffer.BlockCopy(matrix, rowIndex * cols * sizeof(float), result, 0, cols * sizeof(float));

            return result;
        }

        public float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            int cols = matrix.GetLength(1);
            int numRows = endRow - startRow;

            if (!ShouldParallelize(numRows * cols))
            {
                return _singleThreadCPU.SliceRows(matrix, startRow, endRow);
            }

            var result = new float[numRows, cols];

            Parallel.For(0, numRows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[startRow + i, j];
                }
            });
            return result;
        }

        public void SetRow(float[,] matrix, int rowIndex, float[] values, int cols)
        {
            Buffer.BlockCopy(values, 0, matrix, rowIndex * cols * sizeof(float), cols * sizeof(float));
        }

        public void ZeroMatrix(float[,] matrix)
        {
            Array.Clear(matrix, 0, matrix.Length);
        }

        public void ZeroVector(float[] vector)
        {
            Array.Clear(vector, 0, vector.Length);
        }


        public unsafe void MatrixAccumulate(float[,] target, float[,] source)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                _singleThreadCPU.MatrixAccumulate(target, source);

                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                MatrixAccumulateRow(target, source, cols, i);
            });
        }

        public void MatrixAddInPlace(float[,] target, float[,] addend)
        {
            MatrixAccumulate(target, addend);
        }
        public void VectorAccumulate(float[] target, float[] source)
        {
            AccumulateVectorGradients(target, source);
        }

        public void AccumulateVectorGradients(float[] targetGrad, float[] sourceGrad)
        {
            int len = targetGrad.Length;

            if (!ShouldParallelize(len))
            {
                _singleThreadCPU.AccumulateVectorGradients(targetGrad, sourceGrad);
                return;
            }


            // Small vector=> SIMD without thread overhead
            int vecSize = Vector<float>.Count;
            int vecEnd = len - (len % vecSize);

            int j = 0;
            while (j < vecEnd)
            {
                var t = new Vector<float>(targetGrad, j);
                var s = new Vector<float>(sourceGrad, j);
                (t + s).CopyTo(targetGrad, j);
                j += vecSize;
            }

            while (j < len)
            {
                targetGrad[j] += sourceGrad[j];
                j++;
            }
        }


        #endregion


        #region Neural network
        public float[] CalculateDotProduct(float[,] matrix, float[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (vector.Length != cols)
            {
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");
            }

            var result = new float[rows];

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.CalculateDotProduct(matrix, vector);
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++)
                {
                    sum += matrix[i, j] * vector[j];
                }
                result[i] = sum;
            });

            return result;
        }

        public (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType activationType)
        {
            int n = dot.Length;

            if (!ShouldParallelize(n))
            {
                return _singleThreadCPU.ActivateLayer(dot, bias, activationType);
            }
            var activation = new float[n];
            var derivative = new float[n];

            var func = Functions.GetActivationFunction(activationType);
            var deriv = Functions.GetActivationDerivative(activationType);


            Parallel.For(0, n, _parallelOptions, i =>
            {
                float z = dot[i] + bias[i];
                activation[i] = func(z);
                derivative[i] = deriv(z);
            });

            return (activation, derivative);
        }

        public float[] CalculateOutputGradients(float[] cost, float[] derivative)
        {
            int n = cost.Length;

            if (!ShouldParallelize(n))
            {
                return _singleThreadCPU.CalculateOutputGradients(cost, derivative);
            }

            var grad = new float[n];


            Parallel.For(0, n, _parallelOptions, i =>
                grad[i] = -cost[i] * derivative[i]
            );

            return grad;
        }

        public float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            var pre = new float[cols];

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.CalculateHiddenGradients(weights, nextDeltas, derivative);
            }
            else
            {
                Parallel.For(0, cols, _parallelOptions, j =>
                {
                    float sum = 0.0f;

                    for (int i = 0; i < rows; i++)
                    {
                        sum += weights[i, j] * nextDeltas[i];
                    }
                    pre[j] = sum;
                });

                var delta = new float[cols];
                if (!ShouldParallelize(delta.Length))
                {
                    for (int i = 0; i < cols; i++)
                    {
                        delta[i] = pre[i] * derivative[i];
                    }
                }
                else
                {
                    Parallel.For(0, cols, _parallelOptions, i =>
                    {
                        delta[i] = pre[i] * derivative[i];
                    });
                }
                return delta;
            }
        }

        public float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.UpdateWeights(weights, deltas, prevActivations, learningRate, lambda);
            }
            else
            {
                var updated = new float[rows, cols];

                Parallel.For(0, rows, _parallelOptions, i =>
                {
                    for (int j = 0; j < cols; j++)
                    {
                        float gradStep = deltas[i] * prevActivations[j];
                        float regTerm = lambda * weights[i, j];
                        updated[i, j] = weights[i, j] - learningRate * (gradStep + regTerm);
                    }
                });

                return updated;
            }

        }

        public float[] UpdateBias(float[] bias, float[] deltas, float learningRate)
        {
            int n = bias.Length;

            if (!ShouldParallelize(n))
            {
                return _singleThreadCPU.UpdateBias(bias, deltas, learningRate);
            }
            else
            {
                var updated = new float[n];

                Parallel.For(0, n, _parallelOptions, i =>
                    updated[i] = bias[i] - learningRate * deltas[i]
                );

                return updated;
            }
        }
        #endregion

        #region Transformer core

        #region SoftMax

        private readonly ConcurrentDictionary<int, bool> _softmaxStrategyCache = new();

        private readonly ConcurrentDictionary<int, long> _parellelTime = new();

        private readonly ConcurrentDictionary<int, long> _sequentialTime = new();

        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.Softmax(matrix, mask);
            }

            var result = new float[rows, cols];

            Parallel.For(0, rows, _parallelOptions, i =>
                SoftmaxRow(matrix, mask, result, cols, i)
            );

            return result;
        }
        private void SoftmaxRow(float[,] matrix, bool[,] mask, float[,] result, int cols, int i)
        {
            if (cols == 0)
            {
                return;
            }
            var ShouldUseParallelSoftmaxRowFound = _softmaxStrategyCache.TryGetValue(cols, out bool ShouldUseParallelSoftmaxRow);

            if (ShouldUseParallelSoftmaxRowFound)
            {
                if (AlwaysParallel || ShouldUseParallelSoftmaxRow)
                {
                    SoftmaxRowParallel(matrix, mask, result, cols, i);
                    return;
                }
                else
                {
                    SoftmaxRowSequential(matrix, mask, result, cols, i);
                    return;
                }
            }
            else
            {
                if (!_sequentialTime.TryGetValue(cols, out long sticks))
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    SoftmaxRowSequential(matrix, mask, result, cols, i);
                    sw.Stop();
                    _sequentialTime[cols] = sw.ElapsedTicks;
                    return;
                }

                if (!_parellelTime.TryGetValue(cols, out long pticks))
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    SoftmaxRowParallel(matrix, mask, result, cols, i);
                    sw.Stop();
                    _parellelTime[cols] = sw.ElapsedTicks;
                }

                _softmaxStrategyCache[cols] = sticks > pticks;

            }
        }
        private void SoftmaxRowParallel(float[,] matrix, bool[,] mask, float[,] result, int cols, int i)
        {
            object lockObj = new object();

            float globalMax = float.NegativeInfinity;
            Parallel.For(0, cols, _parallelOptions, () => float.NegativeInfinity,
                (j, state, localMax) =>
                {
                    if (mask == null || mask[i, j])
                    {
                        float val = matrix[i, j];
                        if (val > localMax) localMax = val;
                    }
                    return localMax;
                },
                localMax => { lock (lockObj) { if (localMax > globalMax) globalMax = localMax; } }
            );

            // All masked out - zero the row and bail
            if (float.IsNegativeInfinity(globalMax))
            {
                Parallel.For(0, cols, _parallelOptions, j => result[i, j] = 0.0f);
                return;
            }

            float globalSum = 0.0f;
            Parallel.For(0, cols, _parallelOptions, () => 0.0f,
                (j, state, localSum) =>
                {
                    if (mask != null && !mask[i, j])
                        result[i, j] = 0.0f;
                    else
                    {
                        float e = MathF.Exp(matrix[i, j] - globalMax);
                        result[i, j] = e;
                        localSum += e;
                    }
                    return localSum;
                },
                localSum => { lock (lockObj) { globalSum += localSum; } }
            );

            float invSum = 1.0f / globalSum;

            Parallel.For(0, cols, _parallelOptions, j =>
                result[i, j] *= invSum
            );
        }
        private void SoftmaxRowSequential(float[,] matrix, bool[,] mask, float[,] result, int cols, int i)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < cols; j++)
            {
                if (mask == null || mask[i, j])
                {
                    max = Math.Max(max, matrix[i, j]);
                }
            }

            // All masked out - zero the row and bail
            if (float.IsNegativeInfinity(max))
            {
                for (int j = 0; j < cols; j++)
                    result[i, j] = 0.0f;
                return;
            }

            float sum = 0.0f;
            for (int j = 0; j < cols; j++)
            {
                if (mask != null && !mask[i, j])
                {
                    result[i, j] = 0.0f;
                }
                else
                {
                    result[i, j] = MathF.Exp(matrix[i, j] - max);
                    sum += result[i, j];
                }
            }

            float invSum = 1.0f / sum;
            for (int j = 0; j < cols; j++)
            {
                result[i, j] *= invSum;
            }
        }

        #endregion

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            if (!ShouldParallelize(batchSize * features))
            {
                return _singleThreadCPU.LayerNorm(input, gamma, beta, epsilon);
            }
            var result = new float[batchSize, features];

            Parallel.For(0, batchSize, _parallelOptions, i =>
                LayerNormRow(input, gamma, beta, epsilon, features, result, i)
            );
            return result;
        }

        //Making the SoftmaxRow method able to run in parallel was a big strech. But attempting to do something similar for layerNormRow is just mental.
        // even if we assume very large embedding dims say really big, like 16,384 or even 32,768. The math still wont favour within-row parallelism for LayerNorm.
        // - 32,768 features sequential: ~10-15µs(add, subtract, multiply - no transcendental functions)
        // - Three Parallel.For overheads: ~10-15µs
        // Thats break even at best. And at this point using cpu accelleration is just always the wrong choice. Use gpu or something.
        // TLDR, Leave this
        private static void LayerNormRow(float[,] input, float[] gamma, float[] beta, float epsilon, int features, float[,] result, int i)
        {
            float mean = 0.0f;

            for (int j = 0; j < features; j++)
            {
                mean += input[i, j];
            }

            mean = mean / features;

            float variance = 0.0f;

            for (int j = 0; j < features; j++)
            {
                float diff = input[i, j] - mean;
                variance += diff * diff;
            }

            variance = variance / features;

            float stdDev = MathF.Sqrt(variance + epsilon);

            for (int j = 0; j < features; j++)
            {
                result[i, j] = gamma[j] * (input[i, j] - mean) / stdDev + beta[j];
            }
        }

        public (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            if (!ShouldParallelize(batchSize * features))
            {
                return _singleThreadCPU.LayerNormForward(input, gamma, beta, epsilon);
            }
            var means = new float[batchSize];
            var variances = new float[batchSize];
            var normalized = new float[batchSize, features];
            var output = new float[batchSize, features];

            void ProcessRow(int i)
            {
                float mean = 0.0f;

                for (int j = 0; j < features; j++)
                {
                    mean += input[i, j];
                }

                mean = mean / features;

                means[i] = mean;

                float variance = 0.0f;

                for (int j = 0; j < features; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance = variance / features;
                variances[i] = variance;

                float stdDev = MathF.Sqrt(variance + epsilon);

                for (int j = 0; j < features; j++)
                {
                    normalized[i, j] = (input[i, j] - mean) / stdDev;
                    output[i, j] = gamma[j] * normalized[i, j] + beta[j];
                }
            }

            Parallel.For(0, batchSize, _parallelOptions, ProcessRow);


            return (output, means, variances, normalized);
        }

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);

            var dInput = new float[batchSize, features];
            var dGamma = new float[features];
            var dBeta = new float[features];

            var localDGammas = new float[batchSize][];
            var localDBetas = new float[batchSize][];

            void ProcessRow(int i)
            {
                float invStd = 1.0f / MathF.Sqrt(variance[i] + epsilon);

                var ldGamma = new float[features];
                var ldBeta = new float[features];

                for (int j = 0; j < features; j++)
                {
                    ldGamma[j] = dOut[i, j] * normalized[i, j];
                    ldBeta[j] = dOut[i, j];
                }

                localDGammas[i] = ldGamma;
                localDBetas[i] = ldBeta;

                var dNorm = new float[features];
                for (int j = 0; j < features; j++)
                {
                    dNorm[j] = dOut[i, j] * gamma[j];
                }

                float dVar = 0;
                float invStdCubed = invStd * invStd * invStd;

                for (int j = 0; j < features; j++)
                {
                    float xMinusMean = input[i, j] - mean[i];
                    dVar += dNorm[j] * xMinusMean * (-0.5f) * invStdCubed;
                }

                float dMean = 0;
                for (int j = 0; j < features; j++)
                {
                    dMean += dNorm[j] * (-invStd);
                }

                float invN = 1.0f / features;
                for (int j = 0; j < features; j++)
                {
                    float xMinusMean = input[i, j] - mean[i];
                    dInput[i, j] = dNorm[j] * invStd + dVar * 2.0f * xMinusMean * invN + dMean * invN;
                }
            }

            if (!ShouldParallelize(batchSize * features))
            {
                for (int i = 0; i < batchSize; i++)
                {
                    ProcessRow(i);
                }
            }
            else
            {
                Parallel.For(0, batchSize, _parallelOptions, ProcessRow);
            }

            // Reduce
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    dGamma[j] += localDGammas[i][j];
                    dBeta[j] += localDBetas[i][j];
                }
            }

            return (dInput, dGamma, dBeta);
        }


        public bool[,] CreateCausalMask(int seqLen)
        {
            var mask = new bool[seqLen, seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    mask[i, j] = true;
                }
            }
            return mask;
        }

        #region MultiHeadAttentionForward

        // This is the best i could attempt. Please please please let me never have to try this again. Jesus.
        public unsafe float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }

            var concatenated = new float[seqLenQ, embeddingDim];

            // Always parallelise across heads - each head is fully independent
            Parallel.For(0, numHeads, _parallelOptions, head =>
                MHAForwardHeadFast(Q, K, V, concatenated, head, headDim, seqLenQ, seqLenK, scale, mask)
            );

            return concatenated;
        }

        private unsafe void MHAForwardHeadFast(float[,] Q, float[,] K, float[,] V, float[,] concatenated, int head, int headDim, int seqLenQ, int seqLenK, float scale, bool[,] mask)
        {
            int offset = head * headDim;
            int vecSize = Vector<float>.Count;
            int headVecEnd = headDim - (headDim % vecSize);

            // Single allocation for this head - reused every row
            float[] scores = new float[seqLenK];

            fixed (float* pQ = Q, pK = K, pV = V, pOut = concatenated, pScores = scores)
            {
                int strideQ = Q.GetLength(1);
                int strideK = K.GetLength(1);
                int strideV = V.GetLength(1);
                int strideOut = concatenated.GetLength(1);

                for (int i = 0; i < seqLenQ; i++)
                {
                    float* qi = pQ + i * strideQ + offset;
                    float* oi = pOut + i * strideOut + offset;

                    float max = float.NegativeInfinity;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (mask != null && !mask[i, j]) { pScores[j] = float.NegativeInfinity; continue; }
                        float* kj = pK + j * strideK + offset;
                        float dot = 0;
                        int k = 0;
                        for (; k < headVecEnd; k += vecSize)
                            dot += Vector.Dot(
                                new Vector<float>(new ReadOnlySpan<float>(qi + k, vecSize)),
                                new Vector<float>(new ReadOnlySpan<float>(kj + k, vecSize)));
                        for (; k < headDim; k++) dot += qi[k] * kj[k];
                        pScores[j] = dot * scale;
                        if (pScores[j] > max) max = pScores[j];
                    }

                    float sum = 0;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (mask != null && !mask[i, j]) { pScores[j] = 0; continue; }
                        pScores[j] = MathF.Exp(pScores[j] - max);
                        sum += pScores[j];
                    }
                    float invSum = 1.0f / sum;
                    for (int j = 0; j < seqLenK; j++) pScores[j] *= invSum;

                    for (int k = 0; k < headDim; k++) oi[k] = 0;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        float sj = pScores[j];
                        if (sj == 0) continue;
                        float* vj = pV + j * strideV + offset;
                        var sjVec = new Vector<float>(sj);
                        int k = 0;
                        for (; k < headVecEnd; k += vecSize)
                        {
                            var cur = new Vector<float>(new ReadOnlySpan<float>(oi + k, vecSize));
                            var vv = new Vector<float>(new ReadOnlySpan<float>(vj + k, vecSize));
                            (cur + sjVec * vv).CopyTo(new Span<float>(oi + k, vecSize));
                        }
                        for (; k < headDim; k++) oi[k] += sj * vj[k];
                    }
                }
            }
        }

        #endregion

        #region MultiHeadAttentionBackward

        public unsafe (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }

            var dQ_full = new float[seqLenQ, embeddingDim];
            var dK_full = new float[seqLenK, embeddingDim];
            var dV_full = new float[seqLenK, embeddingDim];

            Parallel.For(0, numHeads, _parallelOptions, head =>
                MHABackwardHeadFast(Q, K, V, dConcatenated, dQ_full, dK_full, dV_full,
                    head, headDim, seqLenQ, seqLenK, scale, useDecoderMask)
            );

            return (dQ_full, dK_full, dV_full);
        }

        private unsafe void MHABackwardHeadFast(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, float[,] dQ_full, float[,] dK_full, float[,] dV_full, int head, int headDim, int seqLenQ, int seqLenK, float scale, bool useDecoderMask)
        {
            int offset = head * headDim;
            int vecSize = Vector<float>.Count;
            int headVecEnd = headDim - (headDim % vecSize);

            // Single allocation per head - reused every row
            float[] attn = new float[seqLenK];
            float[] dAttn = new float[seqLenK];

            fixed (float* pQ = Q, pK = K, pV = V, pDC = dConcatenated, pDQ = dQ_full, pDK = dK_full, pDV = dV_full, pAttn = attn, pDAttn = dAttn)
            {
                int strideQ = Q.GetLength(1);
                int strideK = K.GetLength(1);
                int strideV = V.GetLength(1);
                int strideDC = dConcatenated.GetLength(1);
                int strideDQ = dQ_full.GetLength(1);
                int strideDK = dK_full.GetLength(1);
                int strideDV = dV_full.GetLength(1);

                for (int i = 0; i < seqLenQ; i++)
                {
                    float* qi = pQ + i * strideQ + offset;
                    float* dci = pDC + i * strideDC + offset;
                    float* dqi = pDQ + i * strideDQ + offset;

                    // Recompute attention
                    float max = float.NegativeInfinity;

                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            pAttn[j] = 0;
                            continue;
                        }

                        float* kj = pK + j * strideK + offset;
                        float dot = 0;
                        int k = 0;

                        while (k < headVecEnd)
                        {
                            dot += Vector.Dot(new Vector<float>(new ReadOnlySpan<float>(qi + k, vecSize)), new Vector<float>(new ReadOnlySpan<float>(kj + k, vecSize)));
                            k += vecSize;
                        }

                        while (k < headDim)
                        {
                            dot += qi[k] * kj[k];
                            k++;
                        }
                        pAttn[j] = dot * scale;

                        if (pAttn[j] > max)
                        {
                            max = pAttn[j];
                        }
                    }

                    float expSum = 0;

                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            continue;
                        }
                        pAttn[j] = MathF.Exp(pAttn[j] - max);
                        expSum += pAttn[j];
                    }
                    float invExp = 1.0f / (expSum + 1e-10f);
                    for (int j = 0; j < seqLenK; j++)
                    {
                        pAttn[j] *= invExp;
                    }

                    for (int j = 0; j < seqLenK; j++)
                    {
                        float* vj = pV + j * strideV + offset;
                        float dot = 0;
                        int k = 0;
                        for (; k < headVecEnd; k += vecSize)
                        {
                            dot += Vector.Dot(new Vector<float>(new ReadOnlySpan<float>(dci + k, vecSize)), new Vector<float>(new ReadOnlySpan<float>(vj + k, vecSize)));
                        }
                        for (; k < headDim; k++) dot += dci[k] * vj[k];
                        pDAttn[j] = dot;
                    }

                    // -- Softmax backward --
                    float sDot = 0;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        sDot += pAttn[j] * pDAttn[j];
                    }

                    // -- dQ, dK, dV --
                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            continue;
                        }
                        float dScore = pAttn[j] * (pDAttn[j] - sDot) * scale;
                        float aij = pAttn[j];
                        float* kj = pK + j * strideK + offset;
                        float* dkj = pDK + j * strideDK + offset;
                        float* dvj = pDV + j * strideDV + offset;

                        var dsVec = new Vector<float>(dScore);
                        var aVec = new Vector<float>(aij);

                        int k = 0;
                        for (; k < headVecEnd; k += vecSize)
                        {
                            var kvec = new Vector<float>(new ReadOnlySpan<float>(kj + k, vecSize));
                            var dqv = new Vector<float>(new ReadOnlySpan<float>(dqi + k, vecSize));
                            (dqv + dsVec * kvec).CopyTo(new Span<float>(dqi + k, vecSize));

                            var qvec = new Vector<float>(new ReadOnlySpan<float>(qi + k, vecSize));
                            var dkv = new Vector<float>(new ReadOnlySpan<float>(dkj + k, vecSize));
                            (dkv + dsVec * qvec).CopyTo(new Span<float>(dkj + k, vecSize));

                            var dcvec = new Vector<float>(new ReadOnlySpan<float>(dci + k, vecSize));
                            var dvv = new Vector<float>(new ReadOnlySpan<float>(dvj + k, vecSize));
                            (dvv + aVec * dcvec).CopyTo(new Span<float>(dvj + k, vecSize));
                        }
                        for (; k < headDim; k++)
                        {
                            dqi[k] += dScore * kj[k];
                            dkj[k] += dScore * qi[k];
                            dvj[k] += aij * dci[k];
                        }
                    }
                }
            }
        }


        #endregion
        public float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn)
        {
            int inputDim = input.GetLength(1);

            if (!ShouldParallelize(seqLen * inputDim))
            {
                return _singleThreadCPU.FFNForwardBatch(input, seqLen, outputDim, forwardPassFn);
            }

            var result = new float[seqLen, outputDim];

            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                // Each thread gets its own input row buffer (no sharing)
                var row = new float[inputDim];
                for (int j = 0; j < inputDim; j++)
                    row[j] = input[i, j];

                var outRow = forwardPassFn(row);

                for (int j = 0; j < outputDim; j++)
                    result[i, j] = outRow[j];
            });

            return result;
        }

        #endregion



        #region BackpropLinearProjection

        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            int seqLen = input.GetLength(0);
            int embeddingDim = input.GetLength(1);

            if (!ShouldParallelize(seqLen * embeddingDim))
            {
                _singleThreadCPU.BackpropLinearProjection(input, dOutput, weights, weightGrad, biasGrad, dInput);
                return;
            }

            var localWGrads = new float[seqLen][,];
            var localBGrads = new float[seqLen][];

            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                var lwg = new float[embeddingDim, embeddingDim];
                var lbg = new float[embeddingDim];

                for (int j = 0; j < embeddingDim; j++)
                {
                    float dOutVal = dOutput[i, j];
                    for (int k = 0; k < embeddingDim; k++)
                    {
                        lwg[k, j] += input[i, k] * dOutVal;
                    }
                    lbg[j] += dOutVal;
                }

                for (int k = 0; k < embeddingDim; k++)
                {
                    float sum = 0;
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        sum += dOutput[i, j] * weights[k, j];
                    }
                    dInput[i, k] += sum;
                }

                localWGrads[i] = lwg;
                localBGrads[i] = lbg;
            });

            for (int s = 0; s < seqLen; s++)
            {
                var lwg = localWGrads[s];
                var lbg = localBGrads[s];
                for (int k = 0; k < embeddingDim; k++)
                {
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        weightGrad[k, j] += lwg[k, j];
                    }
                }
                for (int j = 0; j < embeddingDim; j++)
                { 
                    biasGrad[j] += lbg[j];
                }
            }
        }

        //not worth it
        private static void BackpropLinearProjectionSeq(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput, int seqLen, int embeddingDim)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    float dOutVal = dOutput[i, j];
                    for (int k = 0; k < embeddingDim; k++)
                    {
                        weightGrad[k, j] += input[i, k] * dOutVal;
                    }
                    biasGrad[j] += dOutVal;
                }
                for (int k = 0; k < embeddingDim; k++)
                {
                    float sum = 0;
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        sum += dOutput[i, j] * weights[k, j];
                    }
                    dInput[i, k] += sum;
                }
            }
        }

        #endregion



        public float MatrixSquaredNorm(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                return _singleThreadCPU.MatrixSquaredNorm(matrix);
            }

            float sum = 0;
            object lockObj = new object();

            Parallel.For(0, rows, _parallelOptions, () => 0.0f, (i, state, localSum) =>
                {
                    for (int j = 0; j < cols; j++)
                    {
                        localSum += matrix[i, j] * matrix[i, j];
                    }
                    return localSum;
                },
                localSum => { lock (lockObj) { sum += localSum; } }
            );

            return sum;
        }

        public float VectorSquaredNorm(float[] vector)
        {
            if (!ShouldParallelize(vector.Length))
            {
                return _singleThreadCPU.VectorSquaredNorm(vector);
            }

            float sum = 0;
            object lockObj = new object();

            Parallel.For(0, vector.Length, _parallelOptions, 
                () => 0.0f, (i, state, localSum) => localSum + vector[i] * vector[i],
                localSum => { lock (lockObj) { sum += localSum; } }
            );

            return sum;
        }

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                _singleThreadCPU.MatrixScaleInPlace(matrix, scale);
                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] *= scale;
                }
            });
        }

        public void VectorScaleInPlace(float[] vector, float scale)
        {
            if (!ShouldParallelize(vector.Length))
            {
                _singleThreadCPU.VectorScaleInPlace(vector, scale);
                return;
            }

            Parallel.For(0, vector.Length, _parallelOptions, i =>
                vector[i] *= scale
            );
        }

        public void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                _singleThreadCPU.MatrixUpdate(weights, gradients, learningRate);
                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    weights[i, j] -= learningRate * gradients[i, j];
                }
            });
        }

        public void VectorUpdate(float[] weights, float[] gradients, float learningRate)
        {
            if (!ShouldParallelize(weights.Length))
            {
                _singleThreadCPU.VectorUpdate(weights, gradients, learningRate);
                return;
            }

            Parallel.For(0, weights.Length, _parallelOptions, i =>
                weights[i] -= learningRate * gradients[i]
            );
        }

        public (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen)
        {
            int outputDim = logits.GetLength(1);
            var dLogits = new float[logits.GetLength(0), outputDim];
            float invLen = 1.0f / effectiveLen;


            if (!ShouldParallelize(effectiveLen * outputDim))
            {
                return _singleThreadCPU.CrossEntropyLossAndGradient(logits, targets, effectiveLen);
            }
            var localLosses = new float[effectiveLen];

            void ProcessRow(int i)
            {
                float max = float.NegativeInfinity;

                for (int j = 0; j < outputDim; j++)
                {
                    max = Math.Max(max, logits[i, j]);
                }

                float sum = 0;
                var probs = new float[outputDim];

                for (int j = 0; j < outputDim; j++)
                {
                    probs[j] = MathF.Exp(logits[i, j] - max);
                    sum += probs[j];
                }

                for (int j = 0; j < outputDim; j++)
                {
                    probs[j] = probs[j] / sum;
                }

                int targetToken = targets[i];
                localLosses[i] = -MathF.Log(probs[targetToken] + 1e-10f);

                for (int j = 0; j < outputDim; j++)
                {
                    dLogits[i, j] = probs[j] * invLen;

                    if (j == targetToken)
                    {
                        dLogits[i, j] -= invLen;
                    }
                }
            }

            Parallel.For(0, effectiveLen, _parallelOptions, ProcessRow);

            float loss = 0;

            for (int i = 0; i < effectiveLen; i++)
            {
                loss += localLosses[i];
            }

            loss = loss / effectiveLen;

            return (loss, dLogits);
        }

        public (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen)
        {
            int outputDim = predictions.GetLength(1);
            var dOutput = new float[predictions.GetLength(0), outputDim];

            float invLen = 1.0f / (effectiveLen * outputDim);

            if (!ShouldParallelize(effectiveLen * outputDim))
            {
                return _singleThreadCPU.MSELossAndGradient(predictions, targets, effectiveLen);
            }

            var localLosses = new float[effectiveLen];

            void ProcessRow(int i)
            {
                float rowLoss = 0;

                for (int j = 0; j < outputDim; j++)
                {
                    float diff = predictions[i, j] - targets[i, j];
                    rowLoss += diff * diff;
                    dOutput[i, j] = 2.0f * diff * invLen;
                }

                localLosses[i] = rowLoss;
            }

            Parallel.For(0, effectiveLen, _parallelOptions, ProcessRow);

            float loss = 0;

            for (int i = 0; i < effectiveLen; i++)
            {
                loss += localLosses[i];
            }

            loss = loss/ (effectiveLen * outputDim);

            return (loss, dOutput);
        }

        public float[,] BackpropOutputProjection(float[,] dLogits, float[,] input, float[,] weights, float[,] weightGrad, float[] biasGrad, int seqLen, int outputDim, int embeddingDim)
        {
            var dX = new float[seqLen, embeddingDim];

            if (!ShouldParallelize(seqLen * outputDim))
            {
                return _singleThreadCPU.BackpropOutputProjection(dLogits, input, weights, weightGrad, biasGrad, seqLen, outputDim, embeddingDim);
            }

            // Thread-local accumulation for shared weightGrad/biasGrad
            var localWGrads = new float[seqLen][,];
            var localBGrads = new float[seqLen][];

            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                var lwg = new float[outputDim, embeddingDim];
                var lbg = new float[outputDim];

                for (int v = 0; v < outputDim; v++)
                {
                    float dVal = dLogits[i, v];

                    for (int e = 0; e < embeddingDim; e++)
                    {
                        lwg[v, e] += input[i, e] * dVal;
                    }

                    lbg[v] += dVal;
                }

                for (int e = 0; e < embeddingDim; e++)
                {
                    float grad = 0;

                    for (int v = 0; v < outputDim; v++)
                    {
                        grad += dLogits[i, v] * weights[v, e];
                    }

                    dX[i, e] = grad;
                }

                localWGrads[i] = lwg;
                localBGrads[i] = lbg;
            });

            for (int s = 0; s < seqLen; s++)
            {
                var lwg = localWGrads[s];
                var lbg = localBGrads[s];

                for (int v = 0; v < outputDim; v++)
                {
                    for (int e = 0; e < embeddingDim; e++)
                    {
                        weightGrad[v, e] += lwg[v, e];
                    }

                    biasGrad[v] += lbg[v];
                }
            }

            return dX;
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            if (!ShouldParallelize(seqLen * embeddingDim))
            {
                _singleThreadCPU.BackpropInputProjection(dX, continuousInput, weightGrad, biasGrad, seqLen, embeddingDim, inputFeatureDim);
                return;
            }

            var localWGrads = new float[seqLen][,];
            var localBGrads = new float[seqLen][];

            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                var lwg = new float[embeddingDim, inputFeatureDim];
                var lbg = new float[embeddingDim];

                for (int e = 0; e < embeddingDim; e++)
                {
                    float dVal = dX[i, e];
                    for (int f = 0; f < inputFeatureDim; f++)
                    {
                        lwg[e, f] += dVal * continuousInput[i, f];
                    }
                    
                    lbg[e] += dVal;
                }

                localWGrads[i] = lwg;
                localBGrads[i] = lbg;
            });

            for (int s = 0; s < seqLen; s++)
            {
                var lwg = localWGrads[s];
                var lbg = localBGrads[s];

                for (int e = 0; e < embeddingDim; e++)
                {
                    for (int f = 0; f < inputFeatureDim; f++)
                    {
                        weightGrad[e, f] += lwg[e, f];
                    }

                    biasGrad[e] += lbg[e];
                }
            }
        }

        public void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim)
        {
            if (!ShouldParallelize(seqLen * embeddingDim))
            {
                _singleThreadCPU.AccumulateTokenEmbeddingGrad(embeddingGrad, dX, tokenIds, seqLen, embeddingDim);
                return;
            }


            Parallel.For(0, embeddingDim, _parallelOptions, j =>
            {
                int i = 0;

                while (i < seqLen)
                {
                    int tokenId = tokenIds[i];
                    embeddingGrad[tokenId, j] += dX[i, j];
                    i++;
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void MatrixAccumulateRow(float[,] target, float[,] source, int cols, int i)
        {
            int vecSize = Vector<float>.Count;
            int vecEnd = cols - (cols % vecSize);

            // Pin inside the non-lambda method - this is legal.
            fixed (float* pTgt = target, pSrc = source)
            {
                float* t = pTgt + i * cols;
                float* s = pSrc + i * cols;

                int j = 0;
                while (j < vecEnd)
                {
                    var tv = new Vector<float>(new ReadOnlySpan<float>(t + j, vecSize));
                    var sv = new Vector<float>(new ReadOnlySpan<float>(s + j, vecSize));
                    (tv + sv).CopyTo(new Span<float>(t + j, vecSize));
                    j += vecSize;
                }

                while (j < cols)
                {
                    t[j] += s[j];
                    j++;
                }
            }
        }

        public void SigmoidInPlace(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldParallelize(rows * cols))
            {
                _singleThreadCPU.SigmoidInPlace(matrix);
                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = StableSigmoid(matrix[i, j]);
                }
            });
        }

        private static float StableSigmoid(float x)
        {
            if (x >= 0)
            {
                float ex = MathF.Exp(-x);
                return 1.0f / (1.0f + ex);
            }
            else
            {
                float ex = MathF.Exp(x);
                return ex / (1.0f + ex);
            }
        }


        #region ContentAwareCrossAttentionForward

        public float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            int psl = Q.GetLength(0);
            int tsl = K.GetLength(0);
            int embDim = Q.GetLength(1);
            int headDim = embDim / numHeads;

            var output = new float[psl, embDim];
            var aw = new float[numHeads][,];
            var sp = new float[numHeads][,];

            // Allocate per-head arrays up-front (avoids allocation inside parallel body)
            for (int h = 0; h < numHeads; h++)
            {
                aw[h] = new float[psl, tsl];
                sp[h] = new float[psl, tsl];
            }

            bool parallelHeads = ShouldParallelize(numHeads * psl * tsl);

            if (parallelHeads)
            {
                Parallel.For(0, numHeads, _parallelOptions, h => ContentAwareCrossAttentionHead(Q, K, V, output, aw[h], sp[h], h, headDim, psl, tsl, scale, decayBias));
            }
            else
            {
                for (int h = 0; h < numHeads; h++)
                {
                    ContentAwareCrossAttentionHead(Q, K, V, output, aw[h], sp[h], h, headDim, psl, tsl, scale, decayBias);
                }
            }

            attentionWeights = aw;
            scoresPreSoftmax = sp;
            return output;
        }

        private static unsafe void ContentAwareCrossAttentionHead(float[,] Q, float[,] K, float[,] V,float[,] output, float[,] weights, float[,] scores, int h, int headDim, int psl, int tsl, float scale, float[,,] decayBias)
        {
            int si = h * headDim;
            int embDim = Q.GetLength(1);
            int vecSize = Vector<float>.Count;
            int vecEnd = headDim - (headDim % vecSize);

            fixed (float* pQ = Q, pK = K, pV = V, pOut = output,
                          pW = weights, pS = scores)
            {
                int strideQ = embDim;
                int strideK = K.GetLength(1);
                int strideV = V.GetLength(1);
                int strideOut = embDim;
                int strideW = tsl;
                int strideScr = tsl;

                for (int p = 0; p < psl; p++)
                {
                    float* qi = pQ + p * strideQ + si;
                    float* oi = pOut + p * strideOut + si;
                    float* wi = pW + p * strideW;
                    float* sci = pS + p * strideScr;

                    float mx = float.MinValue;

                    // -- Compute raw scores --
                    for (int s = 0; s < tsl; s++)
                    {
                        float* kj = pK + s * strideK + si;
                        float dot = 0f;
                        int k = 0;

                        for (; k < vecEnd; k += vecSize)
                            dot += Vector.Dot(
                                new Vector<float>(new ReadOnlySpan<float>(qi + k, vecSize)),
                                new Vector<float>(new ReadOnlySpan<float>(kj + k, vecSize)));
                        for (; k < headDim; k++) dot += qi[k] * kj[k];

                        float sc = dot * scale;
                        if (decayBias != null) sc += decayBias[p, s, h];
                        sci[s] = sc;
                        if (sc > mx) mx = sc;
                    }

                    // -- Softmax --
                    float se = 0f;
                    for (int s = 0; s < tsl; s++)
                    {
                        float e = MathF.Exp(sci[s] - mx);
                        wi[s] = e;
                        se += e;
                    }
                    float invSe = se > 0f ? 1f / se : 0f;
                    for (int s = 0; s < tsl; s++) wi[s] *= invSe;

                    // -- Weighted sum of V --
                    // Zero the head-slice first
                    for (int d = 0; d < headDim; d++) oi[d] = 0f;

                    for (int s = 0; s < tsl; s++)
                    {
                        float wis = wi[s];
                        if (wis == 0f) continue;

                        float* vj = pV + s * strideV + si;
                        var wVec = new Vector<float>(wis);
                        int k = 0;

                        for (; k < vecEnd; k += vecSize)
                        {
                            var ov = new Vector<float>(new ReadOnlySpan<float>(oi + k, vecSize));
                            var vv = new Vector<float>(new ReadOnlySpan<float>(vj + k, vecSize));
                            (ov + wVec * vv).CopyTo(new Span<float>(oi + k, vecSize));
                        }
                        for (; k < headDim; k++) oi[k] += wis * vj[k];
                    }
                }
            }
        }

        public void ApplyContextTypeEmbedding(float[,] contextHidden, float[,] typeEmbedding, int[] typeIndices)
        {
            int n = contextHidden.GetLength(0);
            int embDim = contextHidden.GetLength(1);

            if (!ShouldParallelize(n * embDim))
            {
                _singleThreadCPU.ApplyContextTypeEmbedding(contextHidden, typeEmbedding, typeIndices);
                return;
            }

            Parallel.For(0, n, _parallelOptions, i =>
            {
                int t = typeIndices[i];
                int vecSize = Vector<float>.Count;
                int vecEnd = embDim - (embDim % vecSize);
                int d = 0;

                // SIMD path for the inner dimension
                for (; d < vecEnd; d += vecSize)
                {
                    // contextHidden does not have a direct Span constructor (2-D array),
                    // so we fall back to scalar for safety. SIMD would require unsafe/pinning.
                    for (int k = d; k < d + vecSize; k++)
                        contextHidden[i, k] += typeEmbedding[t, k];
                }
                for (; d < embDim; d++)
                    contextHidden[i, d] += typeEmbedding[t, d];
            });
        }

        public float[,] ComputeTimeDiffMatrix( int priceSeqLen, float[] keyArrivalTimes)
        {
            int numKeys = keyArrivalTimes.Length;

            if (!ShouldParallelize(priceSeqLen * numKeys))
            {
                return _singleThreadCPU.ComputeTimeDiffMatrix(priceSeqLen, keyArrivalTimes);
            }

            var td = new float[priceSeqLen, numKeys];

            Parallel.For(0, priceSeqLen, _parallelOptions, p =>
            {
                float fp = p;
                for (int s = 0; s < numKeys; s++)
                    td[p, s] = MathF.Abs(fp - keyArrivalTimes[s]);
            });

            return td;
        }
    

        public float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos,  float[,] contextHidden,  int totalCtx, float scale)
        {
            int embDim = priceHidden.GetLength(1);

            if (!ShouldParallelize(totalCtx * embDim))
            {
                return _singleThreadCPU.ComputeMemoryAttentionScores(priceHidden, lastPos, contextHidden, totalCtx, scale);
            }

            var scores = new float[totalCtx];

            Parallel.For(0, totalCtx, _parallelOptions, s =>
            {
                float dot = 0f;
                int vecSize = Vector<float>.Count;
                int vecEnd = embDim - (embDim % vecSize);
                int d = 0;

                // Scalar fallback (2-D array → no direct span).
                // For hot paths this should be moved to unsafe code; for now
                // the parallelism across context entries is the primary win.
                for (; d < embDim; d++)
                    dot += priceHidden[lastPos, d] * contextHidden[s, d];

                scores[s] = dot * scale;
            });

            return scores;
        }

        public float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim)
        {
            int embDim = hidden.GetLength(1);

            if (!ShouldParallelize(seqLen * outputDim))
            {
                return _singleThreadCPU.ProjectOutputBatch(hidden, outputProjection, outputBias, seqLen, outputDim);
            }

            var pred = new float[seqLen, outputDim];

            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float sum = outputBias[j];
                    for (int k = 0; k < embDim; k++)
                        sum += outputProjection[j, k] * hidden[i, k];
                    pred[i, j] = sum;
                }
            });

            return pred;
        }


        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(
           float[,] queryEmbeddings,
           float[,] keyEmbeddings,
           float[,] timeDiffs,
           float[] keyTimesFromRef,
           ContentAwareDecayNetwork network,
           bool isTraining = false,
           Random dropoutRng = null)
        {
            int queryLen = timeDiffs.GetLength(0);
            int keyLen = timeDiffs.GetLength(1);
            int numHeads = network.NumHeads;
            int projDim = network.ProjectionDim;
            int contentDim = network.ContentDim;
            int hiddenDim = network.HiddenDim;
            int mlpInputDim = network.MLPInputDim;
            int numBases = network.NumTimeBases;
            int rawDim = network.TimeRawDim;

            var cache = new ContentAwareDecayCache
            {
                QueryEmbeddings = queryEmbeddings,
                KeyEmbeddings = keyEmbeddings,
                TimeDiffs = timeDiffs,
                KeyTimesFromRef = keyTimesFromRef,
                QueryProj = new float[numHeads, queryLen, projDim],
                KeyProj = new float[numHeads, keyLen, projDim],
                TimeRawFeatures = new float[numHeads, keyLen, rawDim],
                TimeEncoding = new float[numHeads, keyLen, projDim],
                MemAttnQInput = new float[numHeads, keyLen, projDim],
                MemAttnKInput = new float[numHeads, keyLen, projDim],
                MemAttnWeights = new float[numHeads, keyLen, keyLen],
                MemAttnOutput = new float[numHeads, keyLen, projDim],
                RefinedKey = new float[numHeads, keyLen, projDim],
                MLPInput = new float[queryLen, keyLen, numHeads, mlpInputDim],
                MLPHiddenPreAct = new float[queryLen, keyLen, numHeads, hiddenDim],
                MLPHidden = new float[queryLen, keyLen, numHeads, hiddenDim],
                GateLogits = new float[queryLen, keyLen, numHeads],
                Gates = new float[queryLen, keyLen, numHeads],
                MemAttnDropoutMask = null,
                MLPDropoutMask = null
            };

            bool useMemAttnDrop = isTraining && network.MemoryAttentionDropout > 0 && dropoutRng != null;
            bool useMLPDrop = isTraining && network.MLPDropout > 0 && dropoutRng != null;

            if (useMemAttnDrop)
                cache.MemAttnDropoutMask = new float[numHeads, keyLen, keyLen];
            if (useMLPDrop)
                cache.MLPDropoutMask = new float[queryLen, keyLen, numHeads, hiddenDim];

            var decayBias = new float[queryLen, keyLen, numHeads];

            int totalWork = numHeads * queryLen * keyLen;

            if (!ShouldParallelize(totalWork))
            {
                return _singleThreadCPU.ContentAwareDecayForward(
                    queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef,
                    network, isTraining, dropoutRng);
            }

            // FIX: derive one deterministic per-head seed from dropoutRng sequentially,
            // before the parallel loop, so each head's draw sequence matches single-
            // threaded CPU regardless of thread scheduling order.
            Random[] perHeadRngs = null;
            if ((useMemAttnDrop || useMLPDrop) && dropoutRng != null)
            {
                perHeadRngs = new Random[numHeads];
                for (int h = 0; h < numHeads; h++)
                    perHeadRngs[h] = new Random(dropoutRng.Next());
            }

            Parallel.For(0, numHeads, _parallelOptions, h =>
            {
                ProcessDecayHead(
                    h, queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef,
                    network, cache, decayBias,
                    isTraining, useMemAttnDrop, useMLPDrop,
                    perHeadRngs?[h],
                    queryLen, keyLen, projDim, contentDim,
                    hiddenDim, mlpInputDim, numBases, rawDim);
            });

            return (decayBias, cache);
        }
        private void ProcessDecayHead(int h, float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network, ContentAwareDecayCache cache, float[,,] decayBias, bool isTraining, bool useMemAttnDrop, bool useMLPDrop, Random dropoutRng, int queryLen, int keyLen, int projDim, int contentDim, int hiddenDim, int mlpInputDim, int numBases, int rawDim)
        {
            // 1. Project queries
            for (int q = 0; q < queryLen; q++)
            {
                for (int p = 0; p < projDim; p++)
                {
                    float val = network.QueryProjectionBias[h, p];
                    for (int d = 0; d < contentDim; d++)
                        val += network.QueryProjection[h, p, d] * queryEmbeddings[q, d];
                    cache.QueryProj[h, q, p] = val;
                }
            }

            // 2. Project keys
            for (int s = 0; s < keyLen; s++)
            {
                for (int p = 0; p < projDim; p++)
                {
                    float val = network.KeyProjectionBias[h, p];
                    for (int d = 0; d < contentDim; d++)
                        val += network.KeyProjection[h, p, d] * keyEmbeddings[s, d];
                    cache.KeyProj[h, s, p] = val;
                }
            }

            // 3. Multi-scale sinusoidal time encoding
            for (int s = 0; s < keyLen; s++)
            {
                float t = keyTimesFromRef != null ? keyTimesFromRef[s] : 0f;
                for (int b = 0; b < numBases; b++)
                {
                    float freq = MathF.Exp(network.TimeLogFreq[h, b]);
                    float angle = freq * t;
                    cache.TimeRawFeatures[h, s, b * 2] = MathF.Sin(angle);
                    cache.TimeRawFeatures[h, s, b * 2 + 1] = MathF.Cos(angle);
                }
                for (int p = 0; p < projDim; p++)
                {
                    float val = network.TimeProjBias[h, p];
                    for (int r = 0; r < rawDim; r++)
                        val += network.TimeProj[h, p, r] * cache.TimeRawFeatures[h, s, r];
                    cache.TimeEncoding[h, s, p] = val;
                }
            }

            // 4. Memory interaction attention
            float memScale = 1.0f / MathF.Sqrt(projDim);

            for (int s = 0; s < keyLen; s++)
            {
                for (int p = 0; p < projDim; p++)
                {
                    cache.MemAttnQInput[h, s, p] = cache.KeyProj[h, s, p] + cache.TimeEncoding[h, s, p];
                    cache.MemAttnKInput[h, s, p] = cache.KeyProj[h, s, p] + cache.TimeEncoding[h, s, p];
                }
            }

            for (int i = 0; i < keyLen; i++)
            {
                float maxScore = float.MinValue;
                var scores = new float[keyLen];

                for (int j = 0; j < keyLen; j++)
                {
                    float dot = 0;
                    for (int p = 0; p < projDim; p++)
                        dot += cache.MemAttnQInput[h, i, p] * cache.MemAttnKInput[h, j, p];
                    scores[j] = dot * memScale;
                    if (scores[j] > maxScore) maxScore = scores[j];
                }

                float sumExp = 0;
                for (int j = 0; j < keyLen; j++)
                {
                    cache.MemAttnWeights[h, i, j] = MathF.Exp(scores[j] - maxScore);
                    sumExp += cache.MemAttnWeights[h, i, j];
                }
                if (sumExp > 0)
                    for (int j = 0; j < keyLen; j++)
                        cache.MemAttnWeights[h, i, j] /= sumExp;

                // FIX: no lock - dropoutRng is now a per-head-local instance
                if (useMemAttnDrop)
                {
                    float keepProb = 1.0f - network.MemoryAttentionDropout;
                    float dscale = 1.0f / keepProb;
                    for (int j = 0; j < keyLen; j++)
                    {
                        float mask = dropoutRng.NextSingle() < keepProb ? dscale : 0f;
                        cache.MemAttnDropoutMask[h, i, j] = mask;
                        cache.MemAttnWeights[h, i, j] *= mask;
                    }
                }

                for (int p = 0; p < projDim; p++)
                {
                    float val = 0;
                    for (int j = 0; j < keyLen; j++)
                        val += cache.MemAttnWeights[h, i, j] * cache.KeyProj[h, j, p];
                    cache.MemAttnOutput[h, i, p] = val;
                }
            }

            // 5. Output projection + residual
            for (int s = 0; s < keyLen; s++)
            {
                for (int p = 0; p < projDim; p++)
                {
                    float val = network.MemAttnOutputB[h, p];
                    for (int q = 0; q < projDim; q++)
                        val += network.MemAttnOutputW[h, p, q] * cache.MemAttnOutput[h, s, q];
                    cache.RefinedKey[h, s, p] = val + cache.KeyProj[h, s, p];
                }
            }

            // 6. Gating MLP
            for (int qi = 0; qi < queryLen; qi++)
            {
                for (int si = 0; si < keyLen; si++)
                {
                    float td = timeDiffs[qi, si];
                    float logTd = MathF.Log(1f + td);

                    int idx = 0;
                    for (int p = 0; p < projDim; p++) cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p];
                    for (int p = 0; p < projDim; p++) cache.MLPInput[qi, si, h, idx++] = cache.RefinedKey[h, si, p];
                    for (int p = 0; p < projDim; p++) cache.MLPInput[qi, si, h, idx++] = cache.QueryProj[h, qi, p] * cache.RefinedKey[h, si, p];
                    cache.MLPInput[qi, si, h, idx++] = td;
                    cache.MLPInput[qi, si, h, idx++] = logTd;

                    for (int j = 0; j < hiddenDim; j++)
                    {
                        float val = network.B1[h, j];
                        for (int k = 0; k < mlpInputDim; k++)
                            val += network.W1[h, j, k] * cache.MLPInput[qi, si, h, k];
                        cache.MLPHiddenPreAct[qi, si, h, j] = val;
                        float activated = val > 0f ? val : 0.01f * val;
                        // FIX: no lock - dropoutRng is now a per-head-local instance
                        if (useMLPDrop)
                        {
                            float keepProb = 1.0f - network.MLPDropout;
                            float mask = dropoutRng.NextSingle() < keepProb ? (1.0f / keepProb) : 0f;
                            cache.MLPDropoutMask[qi, si, h, j] = mask;
                            activated *= mask;
                        }
                        cache.MLPHidden[qi, si, h, j] = activated;
                    }

                    float logit = network.B2[h];
                    for (int j = 0; j < hiddenDim; j++)
                        logit += network.W2[h, j] * cache.MLPHidden[qi, si, h, j];
                    cache.GateLogits[qi, si, h] = logit;

                    float gate = StableSigmoid(logit);
                    cache.Gates[qi, si, h] = gate;

                    float baseRate = MathF.Exp(network.LogBaseDecayRate[h]);
                    float effectiveTd = 1f + td;
                    decayBias[qi, si, h] = -(baseRate * gate) * effectiveTd;
                }
            }
        }
        #endregion
        public float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, int PriceEmbeddingDim, int PriceNumHeads, bool isTraining = false,  Random dropoutRng = null)
        {
            int psl = Q.GetLength(0);
            int tsl = K.GetLength(0);
            int ed = PriceEmbeddingDim;
            int nh = PriceNumHeads;
            int hd = ed / nh;
            float scale = 1.0f / MathF.Sqrt(hd);

            float[,,] decayBias = null;
            if (timeDiffs != null)
            {
                var (bias, decayCache) = ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, block.DecayNetwork, isTraining, dropoutRng);
                decayBias = bias;
                bc.DecayCache = decayCache;
            }


            float[][,] attentionWeights;
            float[][,] scoresPreSoftmax;

            var output = ContentAwareCrossAttentionForward(Q, K, V, nh, scale, decayBias, out attentionWeights, out scoresPreSoftmax);


            bc.CrossAttentionWeights = attentionWeights;
            bc.CrossScoresPreSoftmax = scoresPreSoftmax;

            return output;
        }
   
        public void Matrix3DScaleInPlace(float[,,] matrix, float scale)
        {
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);
            int total = d0 * d1 * d2;

            if (!ShouldParallelize(total))
            {
                _singleThreadCPU.Matrix3DScaleInPlace(matrix, scale);
                return;
            }

            Parallel.For(0, d0, _parallelOptions, i =>
            {
                for (int j = 0; j < d1; j++)
                    for (int k = 0; k < d2; k++)
                        matrix[i, j, k] *= scale;
            });
        }
        public float MatrixSquaredNorm3D(float[,,] matrix)
        {
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);
            int total = d0 * d1 * d2;

            if (!ShouldParallelize(total))
                return _singleThreadCPU.MatrixSquaredNorm3D(matrix);

            float sum = 0;
            object lockObj = new object();

            Parallel.For(0, d0, _parallelOptions,
                () => 0.0f,
                (i, state, localSum) =>
                {
                    for (int j = 0; j < d1; j++)
                        for (int k = 0; k < d2; k++)
                            localSum += matrix[i, j, k] * matrix[i, j, k];
                    return localSum;
                },
                localSum => { lock (lockObj) { sum += localSum; } }
            );

            return sum;
        }


        public void Dispose()
        { 

        }

    }
}
