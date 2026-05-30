using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers;
using CallaghanDev.ML.Transformers.TACAMT;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AccelerationManagers
{
    public class AccelerationMutliThreadCPU : IAccelerationManager
    {
        private readonly ParallelOptions _parallelOptions;
        private AccelerationCPU _singleThreadCPU;
        private readonly object _singleThreadCPULock = new object();

        private const bool AlwaysParallel = false;


        private const int FixedMaxDegreeOfParallelism = 0; // 0 = logical CPU count.
        private const int MinParallelThreshold = 512;
        private const int MaxParallelThreshold = 1_048_576;
        private const int FallbackParallelThreshold = 8192;
        private const int CalibrationWarmupWorkUnits = 4096;
        private const int RequiredConsecutiveParallelWins = 2;
        private const long ParallelWinNumerator = 90L;
        private const long ParallelWinDenominator = 100L;

        private static readonly int MAX_DEGREE_OF_PARALLELISM = GetMaxDegreeOfParallelism();
        private static readonly int PARALLEL_THRESHOLD = GetParallelThreshold();

        public AccelerationMutliThreadCPU()
        {
            _parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = MAX_DEGREE_OF_PARALLELISM
            };
        }

        private AccelerationCPU SingleThreadCPU
        {
            get
            {
                if (_singleThreadCPU != null)
                {
                    return _singleThreadCPU;
                }

                lock (_singleThreadCPULock)
                {
                    if (_singleThreadCPU == null)
                    {
                        _singleThreadCPU = new AccelerationCPU();
                    }

                    return _singleThreadCPU;
                }
            }
        }

        private static int GetMaxDegreeOfParallelism()
        {
            int processorCount = Math.Max(1, Environment.ProcessorCount);
            if (FixedMaxDegreeOfParallelism > 0)
            {
                return Math.Max(1, Math.Min(processorCount, FixedMaxDegreeOfParallelism));
            }
            return processorCount;
        }

        private static int GetParallelThreshold()
        {
            int dop = MAX_DEGREE_OF_PARALLELISM;
            if (dop <= 1)
            {
                return int.MaxValue;
            }

            try
            {
                var options = new ParallelOptions { MaxDegreeOfParallelism = dop };
                float warmup = 0f;
                warmup += RunSequentialThresholdProbe(CalibrationWarmupWorkUnits);
                warmup += RunParallelThresholdProbe(CalibrationWarmupWorkUnits, options, new float[dop]);
                GC.KeepAlive(warmup);

                int[] candidates =
                {
                    512, 1024, 2048, 4096, 8192, 16384,
                    32768, 65536, 131072, 262144, 524288, 1048576
                };

                int firstWinningCandidate = -1;
                int consecutiveWins = 0;
                var partials = new float[dop];

                foreach (int workUnits in candidates)
                {
                    int repetitions = workUnits <= 32768 ? 5 : 3;
                    long sequentialTicks = MeasureBestTicks(() => RunSequentialThresholdProbe(workUnits), repetitions);
                    long parallelTicks = MeasureBestTicks(() => RunParallelThresholdProbe(workUnits, options, partials), repetitions);

                    if (sequentialTicks <= 0 || parallelTicks <= 0)
                    {
                        continue;
                    }

                    bool parallelClearlyWins = parallelTicks * ParallelWinDenominator <= sequentialTicks * ParallelWinNumerator;
                    if (parallelClearlyWins)
                    {
                        if (firstWinningCandidate < 0)
                        {
                            firstWinningCandidate = workUnits;
                        }

                        consecutiveWins++;
                        if (consecutiveWins >= RequiredConsecutiveParallelWins)
                        {
                            long conservative = firstWinningCandidate + firstWinningCandidate / 2L;
                            int rounded = RoundUpPowerOfTwo((int)Math.Min(MaxParallelThreshold, Math.Max(MinParallelThreshold, conservative)));
                            return ClampInt(rounded, MinParallelThreshold, MaxParallelThreshold);
                        }
                    }
                    else
                    {
                        firstWinningCandidate = -1;
                        consecutiveWins = 0;
                    }
                }

                return MaxParallelThreshold;
            }
            catch
            {
                return FallbackParallelThreshold;
            }
        }

        private bool ShouldParallelize(long workUnits)
        {
            if (MAX_DEGREE_OF_PARALLELISM <= 1)
            {
                return false;
            }

            if (AlwaysParallel)
            {
                return true;
            }

            return workUnits >= PARALLEL_THRESHOLD;
        }

        private static long MeasureBestTicks(Func<float> action, int repetitions)
        {
            long best = long.MaxValue;
            float sink = 0f;
            for (int i = 0; i < repetitions; i++)
            {
                long start = Stopwatch.GetTimestamp();
                sink += action();
                long elapsed = Stopwatch.GetTimestamp() - start;
                if (elapsed > 0 && elapsed < best)
                {
                    best = elapsed;
                }
            }
            GC.KeepAlive(sink);
            return best == long.MaxValue ? 0 : best;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float RunSequentialThresholdProbe(int workUnits)
        {
            float acc = 1.0f;
            for (int i = 0; i < workUnits; i++)
            {
                acc += ((i & 1023) + 1) * 0.000001f;
                acc *= 0.99999994f;
            }
            return acc;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float RunParallelThresholdProbe(int workUnits, ParallelOptions options, float[] partials)
        {
            int workers = partials.Length;
            Array.Clear(partials, 0, workers);
            Parallel.For(0, workers, options, worker =>
            {
                int start = (int)((long)workUnits * worker / workers);
                int end = (int)((long)workUnits * (worker + 1) / workers);
                float local = 1.0f;
                for (int i = start; i < end; i++)
                {
                    local += ((i & 1023) + 1) * 0.000001f;
                    local *= 0.99999994f;
                }
                partials[worker] = local;
            });

            float sum = 0f;
            for (int i = 0; i < workers; i++)
            {
                sum += partials[i];
            }
            return sum;
        }

        private static int RoundUpPowerOfTwo(int value)
        {
            if (value <= 1) return 1;
            value--;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;
            if (value < 0) return 1 << 30;
            return value + 1;
        }

        private static int ClampInt(int value, int min, int max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        #region Safe parallel shared tensor primitives

        public float[,] MatrixMultiply(float[,] A, float[,] B)
        {
            if (A == null) throw new ArgumentNullException(nameof(A));
            if (B == null) throw new ArgumentNullException(nameof(B));

            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != rowsB)
            {
                throw new ArgumentException($"Matrix dimensions don't match: [{rowsA}x{colsA}] * [{rowsB}x{colsB}]");
            }

            long workUnits = (long)rowsA * colsA * colsB;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.MatrixMultiply(A, B);
            }

            var C = new float[rowsA, colsB];
            const int BLOCK = 32;
            int rowBlocks = (rowsA + BLOCK - 1) / BLOCK;
            int colBlocks = (colsB + BLOCK - 1) / BLOCK;
            long tileCountLong = (long)rowBlocks * colBlocks;

            if (tileCountLong > int.MaxValue)
            {
                return SingleThreadCPU.MatrixMultiply(A, B);
            }

            int tileCount = (int)tileCountLong;

            Parallel.For(0, tileCount, _parallelOptions, tile =>
            {
                int rb = tile / colBlocks;
                int cb = tile - rb * colBlocks;
                int ii = rb * BLOCK;
                int jj = cb * BLOCK;
                int iMax = Math.Min(ii + BLOCK, rowsA);
                int jMax = Math.Min(jj + BLOCK, colsB);

                for (int kk = 0; kk < colsA; kk += BLOCK)
                {
                    int kMax = Math.Min(kk + BLOCK, colsA);
                    for (int i = ii; i < iMax; i++)
                    {
                        for (int j = jj; j < jMax; j++)
                        {
                            float sum = C[i, j];
                            for (int k = kk; k < kMax; k++)
                            {
                                sum += A[i, k] * B[k, j];
                            }
                            C[i, j] = sum;
                        }
                    }
                }
            });

            return C;
        }

        public float[,] MatrixMultiplyTranspose(float[,] A, float[,] B)
        {
            if (A == null) throw new ArgumentNullException(nameof(A));
            if (B == null) throw new ArgumentNullException(nameof(B));

            int rowsA = A.GetLength(0);
            int colsA = A.GetLength(1);
            int rowsB = B.GetLength(0);
            int colsB = B.GetLength(1);

            if (colsA != colsB)
            {
                throw new ArgumentException($"Matrix dimensions don't match for A*B^T");
            }

            long workUnits = (long)rowsA * rowsB * colsA;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.MatrixMultiplyTranspose(A, B);
            }

            var C = new float[rowsA, rowsB];
            long cellCountLong = (long)rowsA * rowsB;

            if (cellCountLong > int.MaxValue)
            {
                return SingleThreadCPU.MatrixMultiplyTranspose(A, B);
            }

            int cellCount = (int)cellCountLong;
            Parallel.For(0, cellCount, _parallelOptions, flat =>
            {
                int i = flat / rowsB;
                int j = flat - i * rowsB;
                float sum = 0.0f;
                for (int k = 0; k < colsA; k++)
                {
                    sum += A[i, k] * B[j, k];
                }
                C[i, j] = sum;
            });

            return C;
        }

        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];
            long workUnits = (long)rows * cols;

            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.MatrixScale(matrix, scalar);
            }

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
            if (A == null) throw new ArgumentNullException(nameof(A));
            if (B == null) throw new ArgumentNullException(nameof(B));
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);
            if (B.GetLength(0) != rows || B.GetLength(1) != cols)
            {
                throw new ArgumentException("Matrix dimensions must match.");
            }

            if (!ShouldParallelize((long)rows * cols))
            {
                return SingleThreadCPU.MatrixAdd(A, B);
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
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            if (bias == null) throw new ArgumentNullException(nameof(bias));
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (bias.Length != cols)
            {
                throw new ArgumentException("Bias length must match matrix column count.", nameof(bias));
            }

            if (!ShouldParallelize((long)rows * cols))
            {
                return SingleThreadCPU.MatrixAddBias(matrix, bias);
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
            if (inputMatrix == null) throw new ArgumentNullException(nameof(inputMatrix));
            return BatchDotProduct(weights, inputMatrix, 0, inputMatrix.GetLength(0));
        }

        public float[,] BatchDotProduct(float[,] weights, float[,] inputMatrix, int rowStart, int rowCount)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (inputMatrix == null) throw new ArgumentNullException(nameof(inputMatrix));
            if (rowStart < 0 || rowCount < 0) throw new ArgumentOutOfRangeException();
            if (rowStart + rowCount > inputMatrix.GetLength(0)) throw new ArgumentException("Invalid row slice.");

            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);
            if (inputMatrix.GetLength(1) != inputDim)
            {
                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");
            }

            long workUnits = (long)rowCount * outputDim * inputDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.BatchDotProduct(weights, inputMatrix, rowStart, rowCount);
            }

            var result = new float[rowCount, outputDim];
            long cellCountLong = (long)rowCount * outputDim;
            if (cellCountLong > int.MaxValue)
            {
                return SingleThreadCPU.BatchDotProduct(weights, inputMatrix, rowStart, rowCount);
            }

            int cellCount = (int)cellCountLong;
            Parallel.For(0, cellCount, _parallelOptions, flat =>
            {
                int i = flat / outputDim;
                int j = flat - i * outputDim;
                int srcRow = rowStart + i;
                float sum = 0.0f;
                for (int k = 0; k < inputDim; k++)
                {
                    sum += weights[j, k] * inputMatrix[srcRow, k];
                }
                result[i, j] = sum;
            });

            return result;
        }

        public float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            if (startRow < 0 || endRow > matrix.GetLength(0) || startRow > endRow)
            {
                throw new ArgumentOutOfRangeException();
            }

            int cols = matrix.GetLength(1);
            int numRows = endRow - startRow;
            var result = new float[numRows, cols];
            int bytesPerRow = cols * sizeof(float);

            if (!ShouldParallelize((long)numRows * cols))
            {
                return SingleThreadCPU.SliceRows(matrix, startRow, endRow);
            }

            Parallel.For(0, numRows, _parallelOptions, i =>
            {
                Buffer.BlockCopy(matrix, (startRow + i) * bytesPerRow, result, i * bytesPerRow, bytesPerRow);
            });

            return result;
        }

        public void MatrixAddInPlace(float[,] target, float[,] addend)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (addend == null) throw new ArgumentNullException(nameof(addend));
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);
            if (addend.GetLength(0) != rows || addend.GetLength(1) != cols)
            {
                throw new ArgumentException("Matrix dimensions must match.", nameof(addend));
            }

            if (!ShouldParallelize((long)rows * cols))
            {
                SingleThreadCPU.MatrixAddInPlace(target, addend);
                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    target[i, j] += addend[i, j];
                }
            });
        }

        public void VectorAccumulate(float[] target, float[] source)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (source == null) throw new ArgumentNullException(nameof(source));
            if (source.Length != target.Length)
            {
                throw new ArgumentException("Vector lengths must match.", nameof(source));
            }

            if (!ShouldParallelize(target.Length))
            {
                SingleThreadCPU.VectorAccumulate(target, source);
                return;
            }

            Parallel.For(0, target.Length, _parallelOptions, i => target[i] += source[i]);
        }

        #endregion

        #region Safe parallel neural-network helpers

        public float[] CalculateDotProduct(float[,] matrix, float[] vector)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            if (vector == null) throw new ArgumentNullException(nameof(vector));
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (vector.Length != cols)
            {
                throw new ArgumentException($"Expected vector of length {cols}, got {vector.Length}");
            }

            if (!ShouldParallelize((long)rows * cols))
            {
                return SingleThreadCPU.CalculateDotProduct(matrix, vector);
            }

            var result = new float[rows];
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

        public float[] CalculateOutputGradients(float[] cost, float[] derivative)
        {
            if (cost == null) throw new ArgumentNullException(nameof(cost));
            if (derivative == null) throw new ArgumentNullException(nameof(derivative));
            if (derivative.Length != cost.Length) throw new ArgumentException("Vector lengths must match.", nameof(derivative));

            if (!ShouldParallelize(cost.Length))
            {
                return SingleThreadCPU.CalculateOutputGradients(cost, derivative);
            }

            var grad = new float[cost.Length];
            Parallel.For(0, cost.Length, _parallelOptions, i => grad[i] = -cost[i] * derivative[i]);
            return grad;
        }

        public float[] CalculateHiddenGradients(float[,] weights, float[] nextDeltas, float[] derivative)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (nextDeltas == null) throw new ArgumentNullException(nameof(nextDeltas));
            if (derivative == null) throw new ArgumentNullException(nameof(derivative));

            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            if (nextDeltas.Length != rows) throw new ArgumentException("nextDeltas length must match weight row count.", nameof(nextDeltas));
            if (derivative.Length != cols) throw new ArgumentException("derivative length must match weight column count.", nameof(derivative));

            if (!ShouldParallelize((long)rows * cols))
            {
                return SingleThreadCPU.CalculateHiddenGradients(weights, nextDeltas, derivative);
            }

            var delta = new float[cols];
            Parallel.For(0, cols, _parallelOptions, j =>
            {
                float sum = 0.0f;
                for (int i = 0; i < rows; i++)
                {
                    sum += weights[i, j] * nextDeltas[i];
                }
                delta[j] = sum * derivative[j];
            });
            return delta;
        }

        public float[,] UpdateWeights(float[,] weights, float[] deltas, float[] prevActivations, float learningRate, float lambda)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (deltas == null) throw new ArgumentNullException(nameof(deltas));
            if (prevActivations == null) throw new ArgumentNullException(nameof(prevActivations));
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            if (deltas.Length != rows) throw new ArgumentException("deltas length must match row count.", nameof(deltas));
            if (prevActivations.Length != cols) throw new ArgumentException("prevActivations length must match column count.", nameof(prevActivations));

            if (!ShouldParallelize((long)rows * cols))
            {
                return SingleThreadCPU.UpdateWeights(weights, deltas, prevActivations, learningRate, lambda);
            }

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

        public float[] UpdateBias(float[] bias, float[] deltas, float learningRate)
        {
            if (bias == null) throw new ArgumentNullException(nameof(bias));
            if (deltas == null) throw new ArgumentNullException(nameof(deltas));
            if (deltas.Length != bias.Length) throw new ArgumentException("deltas length must match bias length.", nameof(deltas));

            if (!ShouldParallelize(bias.Length))
            {
                return SingleThreadCPU.UpdateBias(bias, deltas, learningRate);
            }

            var updated = new float[bias.Length];
            Parallel.For(0, bias.Length, _parallelOptions, i => updated[i] = bias[i] - learningRate * deltas[i]);
            return updated;
        }

        #endregion

        #region Safe parallel projection/backprop kernels

        public bool[,] CreateCausalMask(int seqLen)
        {
            if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
            if (!ShouldParallelize((long)seqLen * seqLen))
            {
                return SingleThreadCPU.CreateCausalMask(seqLen);
            }

            var mask = new bool[seqLen, seqLen];
            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                for (int j = 0; j <= i; j++)
                {
                    mask[i, j] = true;
                }
            });
            return mask;
        }

        public (float[,] Q, float[,] K, float[,] V) ProjectQKV(float[,] input, float[,] WQ, float[] biasQ, float[,] WK, float[] biasK, float[,] WV, float[] biasV)
        {
            ValidateProjectQKVInputs(input, WQ, biasQ, WK, biasK, WV, biasV, out int rows, out int inputDim, out int outputDim);
            long workUnits = 3L * rows * outputDim * inputDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ProjectQKV(input, WQ, biasQ, WK, biasK, WV, biasV);
            }

            var Q = new float[rows, outputDim];
            var K = new float[rows, outputDim];
            var V = new float[rows, outputDim];
            long cellsLong = (long)rows * outputDim;
            if (cellsLong > int.MaxValue)
            {
                return SingleThreadCPU.ProjectQKV(input, WQ, biasQ, WK, biasK, WV, biasV);
            }

            int cells = (int)cellsLong;
            Parallel.For(0, cells, _parallelOptions, flat =>
            {
                int i = flat / outputDim;
                int o = flat - i * outputDim;
                float qSum = biasQ[o];
                float kSum = biasK[o];
                float vSum = biasV[o];
                for (int d = 0; d < inputDim; d++)
                {
                    float x = input[i, d];
                    qSum += WQ[o, d] * x;
                    kSum += WK[o, d] * x;
                    vSum += WV[o, d] * x;
                }
                Q[i, o] = qSum;
                K[i, o] = kSum;
                V[i, o] = vSum;
            });
            return (Q, K, V);
        }

        public float[,] BackpropQKV(float[,] input, float[,] dQ, float[,] dK, float[,] dV, float[,] WQ, float[,] WK, float[,] WV, float[,] WQGrad, float[] biasQGrad, float[,] WKGrad, float[] biasKGrad, float[,] WVGrad, float[] biasVGrad)
        {
            ValidateBackpropQKVInputs(input, dQ, dK, dV, WQ, WK, WV, WQGrad, biasQGrad, WKGrad, biasKGrad, WVGrad, biasVGrad, out int rows, out int inputDim, out int outputDim);
            long workUnits = 6L * rows * inputDim * outputDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.BackpropQKV(input, dQ, dK, dV, WQ, WK, WV, WQGrad, biasQGrad, WKGrad, biasKGrad, WVGrad, biasVGrad);
            }

            var dInput = new float[rows, inputDim];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int d = 0; d < inputDim; d++)
                {
                    float sum = 0f;
                    for (int o = 0; o < outputDim; o++)
                    {
                        sum += dQ[i, o] * WQ[o, d]
                             + dK[i, o] * WK[o, d]
                             + dV[i, o] * WV[o, d];
                    }
                    dInput[i, d] = sum;
                }
            });

            Parallel.For(0, outputDim, _parallelOptions, o =>
            {
                float bq = biasQGrad[o];
                float bk = biasKGrad[o];
                float bv = biasVGrad[o];
                for (int i = 0; i < rows; i++)
                {
                    bq += dQ[i, o];
                    bk += dK[i, o];
                    bv += dV[i, o];
                }
                biasQGrad[o] = bq;
                biasKGrad[o] = bk;
                biasVGrad[o] = bv;

                for (int d = 0; d < inputDim; d++)
                {
                    float wq = WQGrad[o, d];
                    float wk = WKGrad[o, d];
                    float wv = WVGrad[o, d];
                    for (int i = 0; i < rows; i++)
                    {
                        float x = input[i, d];
                        wq += dQ[i, o] * x;
                        wk += dK[i, o] * x;
                        wv += dV[i, o] * x;
                    }
                    WQGrad[o, d] = wq;
                    WKGrad[o, d] = wk;
                    WVGrad[o, d] = wv;
                }
            });

            return dInput;
        }

        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (dOutput == null) throw new ArgumentNullException(nameof(dOutput));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (weightGrad == null) throw new ArgumentNullException(nameof(weightGrad));
            if (biasGrad == null) throw new ArgumentNullException(nameof(biasGrad));
            if (dInput == null) throw new ArgumentNullException(nameof(dInput));

            int seqLen = input.GetLength(0);
            int inDim = input.GetLength(1);
            int outDim = dOutput.GetLength(1);

            if (dOutput.GetLength(0) != seqLen) throw new ArgumentException("dOutput row count must match input row count.", nameof(dOutput));
            if (weights.GetLength(0) != outDim || weights.GetLength(1) != inDim) throw new ArgumentException("weights shape mismatch.", nameof(weights));
            if (weightGrad.GetLength(0) != outDim || weightGrad.GetLength(1) != inDim) throw new ArgumentException("weightGrad shape mismatch.", nameof(weightGrad));
            if (biasGrad.Length != outDim) throw new ArgumentException("biasGrad length mismatch.", nameof(biasGrad));
            if (dInput.GetLength(0) != seqLen || dInput.GetLength(1) != inDim) throw new ArgumentException("dInput shape mismatch.", nameof(dInput));

            long workUnits = 2L * seqLen * inDim * outDim;
            if (!ShouldParallelize(workUnits))
            {
                SingleThreadCPU.BackpropLinearProjection(input, dOutput, weights, weightGrad, biasGrad, dInput);
                return;
            }

            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                for (int k = 0; k < inDim; k++)
                {
                    float sum = 0f;
                    for (int j = 0; j < outDim; j++)
                    {
                        sum += dOutput[i, j] * weights[j, k];
                    }
                    dInput[i, k] += sum;
                }
            });

            Parallel.For(0, outDim, _parallelOptions, j =>
            {
                float b = biasGrad[j];
                for (int i = 0; i < seqLen; i++)
                {
                    b += dOutput[i, j];
                }
                biasGrad[j] = b;

                for (int k = 0; k < inDim; k++)
                {
                    float wg = weightGrad[j, k];
                    for (int i = 0; i < seqLen; i++)
                    {
                        wg += dOutput[i, j] * input[i, k];
                    }
                    weightGrad[j, k] = wg;
                }
            });
        }

        public float[,] BackpropOutputProjection(float[,] dLogits, float[,] input, float[,] weights, float[,] weightGrad, float[] biasGrad, int seqLen, int outputDim, int embeddingDim)
        {
            if (dLogits == null) throw new ArgumentNullException(nameof(dLogits));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (weightGrad == null) throw new ArgumentNullException(nameof(weightGrad));
            if (biasGrad == null) throw new ArgumentNullException(nameof(biasGrad));

            if (dLogits.GetLength(0) < seqLen || dLogits.GetLength(1) < outputDim) throw new ArgumentException("dLogits shape mismatch.", nameof(dLogits));
            if (input.GetLength(0) < seqLen || input.GetLength(1) < embeddingDim) throw new ArgumentException("input shape mismatch.", nameof(input));
            if (weights.GetLength(0) < outputDim || weights.GetLength(1) < embeddingDim) throw new ArgumentException("weights shape mismatch.", nameof(weights));
            if (weightGrad.GetLength(0) < outputDim || weightGrad.GetLength(1) < embeddingDim) throw new ArgumentException("weightGrad shape mismatch.", nameof(weightGrad));
            if (biasGrad.Length < outputDim) throw new ArgumentException("biasGrad length mismatch.", nameof(biasGrad));

            long workUnits = 2L * seqLen * outputDim * embeddingDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.BackpropOutputProjection(dLogits, input, weights, weightGrad, biasGrad, seqLen, outputDim, embeddingDim);
            }

            var dX = new float[seqLen, embeddingDim];

            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                for (int e = 0; e < embeddingDim; e++)
                {
                    float grad = 0f;
                    for (int v = 0; v < outputDim; v++)
                    {
                        grad += dLogits[i, v] * weights[v, e];
                    }
                    dX[i, e] = grad;
                }
            });

            Parallel.For(0, outputDim, _parallelOptions, v =>
            {
                float b = biasGrad[v];
                for (int i = 0; i < seqLen; i++)
                {
                    b += dLogits[i, v];
                }
                biasGrad[v] = b;

                for (int e = 0; e < embeddingDim; e++)
                {
                    float wg = weightGrad[v, e];
                    for (int i = 0; i < seqLen; i++)
                    {
                        wg += input[i, e] * dLogits[i, v];
                    }
                    weightGrad[v, e] = wg;
                }
            });

            return dX;
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            BackpropInputProjection(dX, continuousInput, 0, weightGrad, biasGrad, seqLen, embeddingDim, inputFeatureDim);
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, int inputRowStart, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            if (dX == null) throw new ArgumentNullException(nameof(dX));
            if (continuousInput == null) throw new ArgumentNullException(nameof(continuousInput));
            if (weightGrad == null) throw new ArgumentNullException(nameof(weightGrad));
            if (biasGrad == null) throw new ArgumentNullException(nameof(biasGrad));

            if (inputRowStart < 0 || seqLen < 0 || inputRowStart + seqLen > continuousInput.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(inputRowStart), $"Invalid input row slice: start={inputRowStart}, count={seqLen}, rows={continuousInput.GetLength(0)}.");
            }
            if (dX.GetLength(0) < seqLen || dX.GetLength(1) < embeddingDim) throw new ArgumentException("dX shape mismatch.", nameof(dX));
            if (continuousInput.GetLength(1) < inputFeatureDim) throw new ArgumentException("continuousInput shape mismatch.", nameof(continuousInput));
            if (weightGrad.GetLength(0) < embeddingDim || weightGrad.GetLength(1) < inputFeatureDim) throw new ArgumentException("weightGrad shape mismatch.", nameof(weightGrad));
            if (biasGrad.Length < embeddingDim) throw new ArgumentException("biasGrad length mismatch.", nameof(biasGrad));

            long workUnits = (long)seqLen * embeddingDim * inputFeatureDim;
            if (!ShouldParallelize(workUnits))
            {
                SingleThreadCPU.BackpropInputProjection(dX, continuousInput, inputRowStart, weightGrad, biasGrad, seqLen, embeddingDim, inputFeatureDim);
                return;
            }

            Parallel.For(0, embeddingDim, _parallelOptions, e =>
            {
                float b = biasGrad[e];
                for (int i = 0; i < seqLen; i++)
                {
                    b += dX[i, e];
                }
                biasGrad[e] = b;

                for (int f = 0; f < inputFeatureDim; f++)
                {
                    float wg = weightGrad[e, f];
                    for (int i = 0; i < seqLen; i++)
                    {
                        wg += dX[i, e] * continuousInput[inputRowStart + i, f];
                    }
                    weightGrad[e, f] = wg;
                }
            });
        }

        public (float[,] K, float[,] V) ProjectKV(float[,] input, float[,] WK, float[] biasK, float[,] WV, float[] biasV)
        {
            ValidateProjectKVInputs(input, WK, biasK, WV, biasV, out int rows, out int inputDim, out int outputDim);
            long workUnits = 2L * rows * outputDim * inputDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ProjectKV(input, WK, biasK, WV, biasV);
            }

            var K = new float[rows, outputDim];
            var V = new float[rows, outputDim];
            long cellsLong = (long)rows * outputDim;
            if (cellsLong > int.MaxValue)
            {
                return SingleThreadCPU.ProjectKV(input, WK, biasK, WV, biasV);
            }

            int cells = (int)cellsLong;
            Parallel.For(0, cells, _parallelOptions, flat =>
            {
                int i = flat / outputDim;
                int o = flat - i * outputDim;
                float kSum = biasK[o];
                float vSum = biasV[o];
                for (int d = 0; d < inputDim; d++)
                {
                    float x = input[i, d];
                    kSum += WK[o, d] * x;
                    vSum += WV[o, d] * x;
                }
                K[i, o] = kSum;
                V[i, o] = vSum;
            });
            return (K, V);
        }

        public float[,] BackpropKV(float[,] input, float[,] dK, float[,] dV, float[,] WK, float[,] WV, float[,] WKGrad, float[] biasKGrad, float[,] WVGrad, float[] biasVGrad)
        {
            ValidateBackpropKVInputs(input, dK, dV, WK, WV, WKGrad, biasKGrad, WVGrad, biasVGrad, out int rows, out int inputDim, out int outputDim);
            long workUnits = 4L * rows * inputDim * outputDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.BackpropKV(input, dK, dV, WK, WV, WKGrad, biasKGrad, WVGrad, biasVGrad);
            }

            var dInput = new float[rows, inputDim];

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int d = 0; d < inputDim; d++)
                {
                    float sum = 0f;
                    for (int o = 0; o < outputDim; o++)
                    {
                        sum += dK[i, o] * WK[o, d]
                             + dV[i, o] * WV[o, d];
                    }
                    dInput[i, d] = sum;
                }
            });

            Parallel.For(0, outputDim, _parallelOptions, o =>
            {
                float bk = biasKGrad[o];
                float bv = biasVGrad[o];
                for (int i = 0; i < rows; i++)
                {
                    bk += dK[i, o];
                    bv += dV[i, o];
                }
                biasKGrad[o] = bk;
                biasVGrad[o] = bv;

                for (int d = 0; d < inputDim; d++)
                {
                    float wk = WKGrad[o, d];
                    float wv = WVGrad[o, d];
                    for (int i = 0; i < rows; i++)
                    {
                        float x = input[i, d];
                        wk += dK[i, o] * x;
                        wv += dV[i, o] * x;
                    }
                    WKGrad[o, d] = wk;
                    WVGrad[o, d] = wv;
                }
            });

            return dInput;
        }

        #endregion

        #region Safe parallel in-place updates

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (!ShouldParallelize((long)rows * cols))
            {
                SingleThreadCPU.MatrixScaleInPlace(matrix, scale);
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
            if (vector == null) throw new ArgumentNullException(nameof(vector));
            if (!ShouldParallelize(vector.Length))
            {
                SingleThreadCPU.VectorScaleInPlace(vector, scale);
                return;
            }

            Parallel.For(0, vector.Length, _parallelOptions, i => vector[i] *= scale);
        }

        public void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (gradients == null) throw new ArgumentNullException(nameof(gradients));
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);
            if (gradients.GetLength(0) != rows || gradients.GetLength(1) != cols)
            {
                throw new ArgumentException("Matrix dimensions must match.", nameof(gradients));
            }

            if (!ShouldParallelize((long)rows * cols))
            {
                SingleThreadCPU.MatrixUpdate(weights, gradients, learningRate);
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
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (gradients == null) throw new ArgumentNullException(nameof(gradients));
            if (weights.Length != gradients.Length) throw new ArgumentException("weights and gradients must have the same length.", nameof(gradients));

            if (!ShouldParallelize(weights.Length))
            {
                SingleThreadCPU.VectorUpdate(weights, gradients, learningRate);
                return;
            }

            Parallel.For(0, weights.Length, _parallelOptions, i => weights[i] -= learningRate * gradients[i]);
        }

        public void VectorUpdateClamped(float[] weights, float[] gradients, float learningRate, float minValue, float maxValue)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (gradients == null) throw new ArgumentNullException(nameof(gradients));
            if (weights.Length != gradients.Length) throw new ArgumentException("weights and gradients must have the same length.");

            if (!ShouldParallelize(weights.Length))
            {
                SingleThreadCPU.VectorUpdateClamped(weights, gradients, learningRate, minValue, maxValue);
                return;
            }

            Parallel.For(0, weights.Length, _parallelOptions, i =>
            {
                weights[i] = Math.Clamp(weights[i] - learningRate * gradients[i], minValue, maxValue);
            });
        }

        public void Matrix3DScaleInPlace(float[,,] matrix, float scale)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);
            if (!ShouldParallelize((long)d0 * d1 * d2))
            {
                SingleThreadCPU.Matrix3DScaleInPlace(matrix, scale);
                return;
            }

            Parallel.For(0, d0, _parallelOptions, i =>
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        matrix[i, j, k] *= scale;
                    }
                }
            });
        }

        public void Matrix3DAddInPlace(float[,,] target, float[,,] addend)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (addend == null) throw new ArgumentNullException(nameof(addend));
            int d0 = target.GetLength(0);
            int d1 = target.GetLength(1);
            int d2 = target.GetLength(2);
            if (addend.GetLength(0) != d0 || addend.GetLength(1) != d1 || addend.GetLength(2) != d2)
            {
                throw new ArgumentException("3D tensor shape mismatch.", nameof(addend));
            }

            if (!ShouldParallelize((long)d0 * d1 * d2))
            {
                SingleThreadCPU.Matrix3DAddInPlace(target, addend);
                return;
            }

            Parallel.For(0, d0, _parallelOptions, i =>
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        target[i, j, k] += addend[i, j, k];
                    }
                }
            });
        }

        public void Matrix3DUpdate(float[,,] weights, float[,,] gradients, float learningRate)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (gradients == null) throw new ArgumentNullException(nameof(gradients));
            int d0 = weights.GetLength(0);
            int d1 = weights.GetLength(1);
            int d2 = weights.GetLength(2);
            if (gradients.GetLength(0) != d0 || gradients.GetLength(1) != d1 || gradients.GetLength(2) != d2)
            {
                throw new ArgumentException("3D tensor shape mismatch.", nameof(gradients));
            }

            if (!ShouldParallelize((long)d0 * d1 * d2))
            {
                SingleThreadCPU.Matrix3DUpdate(weights, gradients, learningRate);
                return;
            }

            Parallel.For(0, d0, _parallelOptions, i =>
            {
                for (int j = 0; j < d1; j++)
                {
                    for (int k = 0; k < d2; k++)
                    {
                        weights[i, j, k] -= learningRate * gradients[i, j, k];
                    }
                }
            });
        }

        public void SigmoidInPlace(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            if (!ShouldParallelize((long)rows * cols))
            {
                SingleThreadCPU.SigmoidInPlace(matrix);
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

        public void ZeroMatrixColumns(float[,] matrix, int columnCount)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            if (columnCount <= 0) return;
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            int count = Math.Min(columnCount, cols);
            if (!ShouldParallelize((long)rows * count))
            {
                SingleThreadCPU.ZeroMatrixColumns(matrix, columnCount);
                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < count; j++)
                {
                    matrix[i, j] = 0f;
                }
            });
        }

        #endregion

        #region CPU-reference delegated methods

        public float[] ExtractRow(float[,] matrix, int rowIndex, int cols)
        {
            return SingleThreadCPU.ExtractRow(matrix, rowIndex, cols);
        }

        public void SetRow(float[,] matrix, int rowIndex, float[] values, int cols)
        {
            SingleThreadCPU.SetRow(matrix, rowIndex, values, cols);
        }

        public void ZeroMatrix(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            long workUnits = (long)rows * cols;

            if (!ShouldParallelize(workUnits))
            {
                SingleThreadCPU.ZeroMatrix(matrix);
                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = 0f;
                }
            });
        }

        public void ZeroVector(float[] vector)
        {
            if (vector == null) throw new ArgumentNullException(nameof(vector));

            if (!ShouldParallelize(vector.Length))
            {
                SingleThreadCPU.ZeroVector(vector);
                return;
            }

            Parallel.For(0, vector.Length, _parallelOptions, i => vector[i] = 0f);
        }
        public (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType activationType)
        {
            if (dot == null) throw new ArgumentNullException(nameof(dot));
            if (bias == null) throw new ArgumentNullException(nameof(bias));
            if (bias.Length != dot.Length)
            {
                throw new ArgumentException("Bias length must match dot product length");
            }

            if (!ShouldParallelize(dot.Length))
            {
                return SingleThreadCPU.ActivateLayer(dot, bias, activationType);
            }

            int n = dot.Length;
            var activation = new float[n];
            var derivative = new float[n];

            var activationFunction = CallaghanDev.ML.Functions.GetActivationFunction(activationType);
            var activationDerivative = CallaghanDev.ML.Functions.GetActivationDerivative(activationType);

            Parallel.For(0, n, _parallelOptions, i =>
            {
                float z = dot[i] + bias[i];
                activation[i] = activationFunction(z);
                derivative[i] = activationDerivative(z);
            });

            return (activation, derivative);
        }
        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (mask != null && (mask.GetLength(0) != rows || mask.GetLength(1) != cols))
            {
                throw new ArgumentException($"Mask shape must be [{rows},{cols}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));
            }

            if (!ShouldParallelize((long)rows * cols))
            {
                return SingleThreadCPU.Softmax(matrix, mask);
            }

            var result = new float[rows, cols];

            if (mask == null)
            {
                Parallel.For(0, rows, _parallelOptions, i =>
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < cols; j++)
                    {
                        float value = matrix[i, j];
                        if (float.IsNaN(value))
                        {
                            throw new InvalidOperationException($"Softmax input contains NaN at [{i},{j}].");
                        }
                        if (value > max)
                        {
                            max = value;
                        }
                    }

                    if (float.IsNegativeInfinity(max))
                    {
                        return;
                    }

                    float sum = 0f;
                    for (int j = 0; j < cols; j++)
                    {
                        float exp = MathF.Exp(matrix[i, j] - max);
                        result[i, j] = exp;
                        sum += exp;
                    }

                    if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
                    {
                        return;
                    }

                    float invSum = 1f / sum;
                    for (int j = 0; j < cols; j++)
                    {
                        result[i, j] *= invSum;
                    }
                });
            }
            else
            {
                Parallel.For(0, rows, _parallelOptions, i =>
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < cols; j++)
                    {
                        if (!mask[i, j])
                        {
                            continue;
                        }

                        float value = matrix[i, j];
                        if (float.IsNaN(value))
                        {
                            throw new InvalidOperationException($"Softmax input contains NaN at [{i},{j}].");
                        }
                        if (value > max)
                        {
                            max = value;
                        }
                    }

                    if (float.IsNegativeInfinity(max))
                    {
                        return;
                    }

                    float sum = 0f;
                    for (int j = 0; j < cols; j++)
                    {
                        if (!mask[i, j])
                        {
                            result[i, j] = 0f;
                            continue;
                        }

                        float exp = MathF.Exp(matrix[i, j] - max);
                        result[i, j] = exp;
                        sum += exp;
                    }

                    if (sum <= 0f || float.IsNaN(sum) || float.IsInfinity(sum))
                    {
                        return;
                    }

                    float invSum = 1f / sum;
                    for (int j = 0; j < cols; j++)
                    {
                        result[i, j] *= invSum;
                    }
                });
            }

            return result;
        }

        [Obsolete]
        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            return SingleThreadCPU.MultiHeadAttentionBackward_Obsolete(Q, K, V, dConcatenated, numHeads, scale, useDecoderMask);
        }

        [Obsolete]
        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            return SingleThreadCPU.MultiHeadAttentionBackward_Obsolete(Q, K, V, dConcatenated, numHeads, scale, mask);
        }


        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            if (!ShouldParallelize(EstimateAttentionBackwardWork(Q, K, numHeads)))
            {
                return SingleThreadCPU.MultiHeadAttentionBackward(Q, K, V, dConcatenated, numHeads, scale, useDecoderMask);
            }

            return MultiHeadAttentionBackwardCoreParallel(Q, K, V, dConcatenated, numHeads, scale, mask: null, useDecoderMask: useDecoderMask);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            if (!ShouldParallelize(EstimateAttentionBackwardWork(Q, K, numHeads)))
            {
                return SingleThreadCPU.MultiHeadAttentionBackward(Q, K, V, dConcatenated, numHeads, scale, mask);
            }

            return MultiHeadAttentionBackwardCoreParallel(Q, K, V, dConcatenated, numHeads, scale, mask, useDecoderMask: false);
        }

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (gamma == null) throw new ArgumentNullException(nameof(gamma));
            if (beta == null) throw new ArgumentNullException(nameof(beta));

            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            if (!ShouldParallelize((long)batchSize * features))
            {
                return SingleThreadCPU.LayerNorm(input, gamma, beta, epsilon);
            }

            var result = new float[batchSize, features];

            Parallel.For(0, batchSize, _parallelOptions, i =>
            {
                float mean = 0f;
                for (int j = 0; j < features; j++)
                {
                    mean += input[i, j];
                }
                mean /= features;

                float variance = 0f;
                for (int j = 0; j < features; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance /= features;

                float invStd = 1f / MathF.Sqrt(variance + epsilon);
                for (int j = 0; j < features; j++)
                {
                    result[i, j] = gamma[j] * ((input[i, j] - mean) * invStd) + beta[j];
                }
            });

            return result;
        }

        public (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (gamma == null) throw new ArgumentNullException(nameof(gamma));
            if (beta == null) throw new ArgumentNullException(nameof(beta));

            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            if (!ShouldParallelize((long)batchSize * features))
            {
                return SingleThreadCPU.LayerNormForward(input, gamma, beta, epsilon);
            }

            var means = new float[batchSize];
            var variances = new float[batchSize];
            var normalized = new float[batchSize, features];
            var output = new float[batchSize, features];

            Parallel.For(0, batchSize, _parallelOptions, i =>
            {
                float mean = 0f;
                for (int j = 0; j < features; j++)
                {
                    mean += input[i, j];
                }
                mean /= features;
                means[i] = mean;

                float variance = 0f;
                for (int j = 0; j < features; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance /= features;
                variances[i] = variance;

                float invStd = 1f / MathF.Sqrt(variance + epsilon);
                for (int j = 0; j < features; j++)
                {
                    float n = (input[i, j] - mean) * invStd;
                    normalized[i, j] = n;
                    output[i, j] = gamma[j] * n + beta[j];
                }
            });

            return (output, means, variances, normalized);
        }

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            if (dOut == null) throw new ArgumentNullException(nameof(dOut));
            if (normalized == null) throw new ArgumentNullException(nameof(normalized));
            if (gamma == null) throw new ArgumentNullException(nameof(gamma));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (mean == null) throw new ArgumentNullException(nameof(mean));
            if (variance == null) throw new ArgumentNullException(nameof(variance));

            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);

            if (!ShouldParallelize((long)batchSize * features * 3L))
            {
                return SingleThreadCPU.LayerNormBackward(dOut, normalized, gamma, input, mean, variance, epsilon);
            }

            var dInput = new float[batchSize, features];
            var dGamma = new float[features];
            var dBeta = new float[features];

            Parallel.For(0, batchSize, _parallelOptions, i =>
            {
                float invStd = 1f / MathF.Sqrt(variance[i] + epsilon);
                float invStdCubed = invStd * invStd * invStd;
                float dVar = 0f;
                float dMean = 0f;
                float meanI = mean[i];

                for (int j = 0; j < features; j++)
                {
                    float dOutVal = dOut[i, j];
                    float dNorm = dOutVal * gamma[j];
                    float xMinusMean = input[i, j] - meanI;
                    dVar += dNorm * xMinusMean * (-0.5f) * invStdCubed;
                    dMean += dNorm * (-invStd);
                }

                float invN = 1f / features;
                float dVarScale = dVar * 2f * invN;
                float dMeanScale = dMean * invN;

                for (int j = 0; j < features; j++)
                {
                    float dNorm = dOut[i, j] * gamma[j];
                    float xMinusMean = input[i, j] - meanI;
                    dInput[i, j] = dNorm * invStd + dVarScale * xMinusMean + dMeanScale;
                }
            });

            Parallel.For(0, features, _parallelOptions, j =>
            {
                float gammaSum = 0f;
                float betaSum = 0f;
                for (int i = 0; i < batchSize; i++)
                {
                    float dOutVal = dOut[i, j];
                    gammaSum += dOutVal * normalized[i, j];
                    betaSum += dOutVal;
                }
                dGamma[j] = gammaSum;
                dBeta[j] = betaSum;
            });

            return (dInput, dGamma, dBeta);
        }

        [Obsolete]
        public float[,] MultiHeadAttentionForward_Obsolete(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            if (Q == null || K == null || V == null)
            {
                return SingleThreadCPU.MultiHeadAttentionForward_Obsolete(Q, K, V, numHeads, scale, mask);
            }

            long workUnits = (long)Q.GetLength(0) * K.GetLength(0) * Q.GetLength(1);
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.MultiHeadAttentionForward_Obsolete(Q, K, V, numHeads, scale, mask);
            }

            return MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask);
        }

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);

            if (K.GetLength(1) != embeddingDim || V.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("Q, K and V must have the same embedding dimension.");
            }
            if (V.GetLength(0) != seqLenK)
            {
                throw new ArgumentException("K and V must have the same sequence length.");
            }
            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads.");
            }
            if (mask != null && (mask.GetLength(0) != seqLenQ || mask.GetLength(1) != seqLenK))
            {
                throw new ArgumentException("Mask shape must be [seqLenQ, seqLenK].");
            }

            int headDim = embeddingDim / numHeads;
            long workUnits = (long)numHeads * seqLenQ * seqLenK * Math.Max(1, headDim);
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask);
            }

            var output = new float[seqLenQ, embeddingDim];
            if (seqLenQ == 0 || seqLenK == 0 || headDim == 0)
            {
                return output;
            }

            Parallel.For(0, numHeads, _parallelOptions, head =>
            {
                int offset = head * headDim;
                var scores = new float[seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    float maxScore = float.NegativeInfinity;

                    for (int key = 0; key < seqLenK; key++)
                    {
                        if (mask != null && !mask[i, key])
                        {
                            scores[key] = float.NegativeInfinity;
                            continue;
                        }

                        float dot = DotHeadRows(Q, i, K, key, offset, headDim);
                        float score = dot * scale;
                        scores[key] = score;
                        if (score > maxScore)
                        {
                            maxScore = score;
                        }
                    }

                    if (float.IsNegativeInfinity(maxScore))
                    {
                        continue;
                    }

                    float sumExp = 0f;
                    for (int key = 0; key < seqLenK; key++)
                    {
                        float score = scores[key];
                        if (float.IsNegativeInfinity(score))
                        {
                            scores[key] = 0f;
                            continue;
                        }

                        float exp = MathF.Exp(score - maxScore);
                        scores[key] = exp;
                        sumExp += exp;
                    }

                    float invSumExp = sumExp > 0f ? 1f / sumExp : 0f;
                    for (int d = 0; d < headDim; d++)
                    {
                        float value = 0f;
                        for (int key = 0; key < seqLenK; key++)
                        {
                            float weight = scores[key] * invSumExp;
                            if (weight != 0f)
                            {
                                value += weight * V[key, offset + d];
                            }
                        }
                        output[i, offset + d] = value;
                    }
                }
            });

            return output;
        }

        public float[,] ScaledDotProductAttention(float[,] q, float[,] k, float[,] v, int numHeads, bool[,] mask = null, bool causal = false)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

            int queryLen = q.GetLength(0);
            int keyLen = k.GetLength(0);
            int valueLen = v.GetLength(0);
            int embeddingDim = q.GetLength(1);

            if (queryLen <= 0) throw new ArgumentException("Q must contain at least one row.", nameof(q));
            if (keyLen <= 0) throw new ArgumentException("K must contain at least one row.", nameof(k));
            if (embeddingDim <= 0) throw new ArgumentException("Q must contain at least one column.", nameof(q));
            if (k.GetLength(1) != embeddingDim) throw new ArgumentException("K width must match Q width.", nameof(k));
            if (v.GetLength(1) != embeddingDim) throw new ArgumentException("V width must match Q width.", nameof(v));
            if (valueLen != keyLen) throw new ArgumentException("V row count must match K row count.", nameof(v));
            if (embeddingDim % numHeads != 0) throw new ArgumentException("Embedding dimension must be divisible by numHeads.", nameof(numHeads));
            if (mask != null && (mask.GetLength(0) != queryLen || mask.GetLength(1) != keyLen))
            {
                throw new ArgumentException($"Mask shape must be [{queryLen},{keyLen}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));
            }
            if (causal && queryLen != keyLen)
            {
                throw new ArgumentException("The simple causal path assumes queryLen == keyLen. Use an explicit mask for cross-attention or cached decoding.");
            }

            int headDim = embeddingDim / numHeads;
            float scale = 1f / MathF.Sqrt(headDim);
            long workUnits = (long)numHeads * queryLen * keyLen * Math.Max(1, headDim);
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ScaledDotProductAttention(q, k, v, numHeads, mask, causal);
            }

            var output = new float[queryLen, embeddingDim];

            Parallel.For(0, numHeads, _parallelOptions, head =>
            {
                int offset = head * headDim;
                var scores = new float[keyLen];

                for (int qi = 0; qi < queryLen; qi++)
                {
                    int usableKeyLen = causal ? Math.Min(qi + 1, keyLen) : keyLen;
                    float maxScore = float.NegativeInfinity;

                    for (int ki = 0; ki < usableKeyLen; ki++)
                    {
                        if (mask != null && !mask[qi, ki])
                        {
                            scores[ki] = float.NegativeInfinity;
                            continue;
                        }

                        float score = DotHeadRows(q, qi, k, ki, offset, headDim) * scale;
                        scores[ki] = score;
                        if (score > maxScore)
                        {
                            maxScore = score;
                        }
                    }

                    if (float.IsNegativeInfinity(maxScore))
                    {
                        continue;
                    }

                    float sumExp = 0f;
                    for (int ki = 0; ki < usableKeyLen; ki++)
                    {
                        float score = scores[ki];
                        if (float.IsNegativeInfinity(score))
                        {
                            scores[ki] = 0f;
                            continue;
                        }

                        float exp = MathF.Exp(score - maxScore);
                        scores[ki] = exp;
                        sumExp += exp;
                    }

                    if (sumExp <= 0f || float.IsNaN(sumExp) || float.IsInfinity(sumExp))
                    {
                        continue;
                    }

                    float invSumExp = 1f / sumExp;
                    for (int d = 0; d < headDim; d++)
                    {
                        float value = 0f;
                        for (int ki = 0; ki < usableKeyLen; ki++)
                        {
                            float weight = scores[ki] * invSumExp;
                            if (weight != 0f)
                            {
                                value += weight * v[ki, offset + d];
                            }
                        }
                        output[qi, offset + d] = value;
                    }
                }
            });

            return output;
        }

        public void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim)
        {
            if (embeddingGrad == null) throw new ArgumentNullException(nameof(embeddingGrad));
            if (dX == null) throw new ArgumentNullException(nameof(dX));
            if (tokenIds == null) throw new ArgumentNullException(nameof(tokenIds));

            long workUnits = (long)seqLen * embeddingDim;
            if (!ShouldParallelize(workUnits))
            {
                SingleThreadCPU.AccumulateTokenEmbeddingGrad(embeddingGrad, dX, tokenIds, seqLen, embeddingDim);
                return;
            }

            int vocabSize = embeddingGrad.GetLength(0);
            var positionsByToken = new Dictionary<int, List<int>>();
            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = tokenIds[i];
                if ((uint)tokenId >= (uint)vocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token out of range: {tokenId}");
                }

                if (!positionsByToken.TryGetValue(tokenId, out var positions))
                {
                    positions = new List<int>();
                    positionsByToken[tokenId] = positions;
                }
                positions.Add(i);
            }

            var groups = new List<KeyValuePair<int, List<int>>>(positionsByToken);
            if (!ShouldParallelize((long)groups.Count * embeddingDim))
            {
                SingleThreadCPU.AccumulateTokenEmbeddingGrad(embeddingGrad, dX, tokenIds, seqLen, embeddingDim);
                return;
            }

            Parallel.For(0, groups.Count, _parallelOptions, g =>
            {
                int tokenId = groups[g].Key;
                List<int> positions = groups[g].Value;

                for (int p = 0; p < positions.Count; p++)
                {
                    int srcRow = positions[p];
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        embeddingGrad[tokenId, j] += dX[srcRow, j];
                    }
                }
            });
        }

        public (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen)
        {
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            int outputDim = logits.GetLength(1);
            long workUnits = (long)effectiveLen * outputDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.CrossEntropyLossAndGradient(logits, targets, effectiveLen);
            }

            var dLogits = new float[logits.GetLength(0), outputDim];
            var rowLoss = new float[effectiveLen];
            float invLen = 1f / effectiveLen;

            Parallel.For(0, effectiveLen, _parallelOptions, i =>
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < outputDim; j++)
                {
                    max = Math.Max(max, logits[i, j]);
                }

                float sum = 0f;
                for (int j = 0; j < outputDim; j++)
                {
                    float p = MathF.Exp(logits[i, j] - max);
                    dLogits[i, j] = p;
                    sum += p;
                }

                float invSum = 1f / sum;
                int targetToken = targets[i];
                rowLoss[i] = -MathF.Log((dLogits[i, targetToken] * invSum) + 1e-10f);

                for (int j = 0; j < outputDim; j++)
                {
                    float grad = (dLogits[i, j] * invSum) * invLen;
                    if (j == targetToken)
                    {
                        grad -= invLen;
                    }
                    dLogits[i, j] = grad;
                }
            });

            float loss = 0f;
            for (int i = 0; i < effectiveLen; i++)
            {
                loss += rowLoss[i];
            }

            loss /= effectiveLen;
            return (loss, dLogits);
        }

        public (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen)
        {
            if (predictions == null) throw new ArgumentNullException(nameof(predictions));
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            int outputDim = predictions.GetLength(1);
            if (targets.GetLength(0) < effectiveLen || targets.GetLength(1) != outputDim)
            {
                throw new ArgumentException("Target shape must match predictions for the effective length.", nameof(targets));
            }

            long lengthLong = (long)effectiveLen * outputDim;
            if (!ShouldParallelize(lengthLong) || lengthLong > int.MaxValue)
            {
                return SingleThreadCPU.MSELossAndGradient(predictions, targets, effectiveLen);
            }

            int length = (int)lengthLong;
            var dOutput = new float[predictions.GetLength(0), outputDim];
            var losses = new float[length];
            float invLen = 1f / length;

            Parallel.For(0, length, _parallelOptions, index =>
            {
                int row = index / outputDim;
                int col = index - row * outputDim;
                float diff = predictions[row, col] - targets[row, col];
                losses[index] = diff * diff;
                dOutput[row, col] = 2f * diff * invLen;
            });

            float loss = 0f;
            for (int i = 0; i < length; i++)
            {
                loss += losses[i];
            }

            loss /= length;
            return (loss, dOutput);
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            long workUnits = (long)rows * cols;

            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.MatrixSquaredNorm(matrix);
            }

            var partials = new float[Math.Min(MAX_DEGREE_OF_PARALLELISM, Math.Max(1, rows))];
            Parallel.For(0, partials.Length, _parallelOptions, worker =>
            {
                int start = (int)((long)rows * worker / partials.Length);
                int end = (int)((long)rows * (worker + 1) / partials.Length);
                float local = 0f;

                for (int i = start; i < end; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        float v = matrix[i, j];
                        local += v * v;
                    }
                }

                partials[worker] = local;
            });

            float sum = 0f;
            for (int i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }
            return sum;
        }

        public float VectorSquaredNorm(float[] vector)
        {
            if (vector == null) throw new ArgumentNullException(nameof(vector));

            if (!ShouldParallelize(vector.Length))
            {
                return SingleThreadCPU.VectorSquaredNorm(vector);
            }

            var partials = new float[Math.Min(MAX_DEGREE_OF_PARALLELISM, Math.Max(1, vector.Length))];
            Parallel.For(0, partials.Length, _parallelOptions, worker =>
            {
                int start = (int)((long)vector.Length * worker / partials.Length);
                int end = (int)((long)vector.Length * (worker + 1) / partials.Length);
                float local = 0f;
                for (int i = start; i < end; i++)
                {
                    float v = vector[i];
                    local += v * v;
                }
                partials[worker] = local;
            });

            float sum = 0f;
            for (int i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }
            return sum;
        }

        public void ApplyContextTypeEmbedding(float[,] contextHidden, float[,] typeEmbedding, int[] typeIndices)
        {
            if (contextHidden == null) throw new ArgumentNullException(nameof(contextHidden));
            if (typeEmbedding == null) throw new ArgumentNullException(nameof(typeEmbedding));
            if (typeIndices == null) throw new ArgumentNullException(nameof(typeIndices));

            int rows = contextHidden.GetLength(0);
            int embDim = contextHidden.GetLength(1);
            if (typeIndices.Length != rows)
            {
                throw new ArgumentException("typeIndices length must match contextHidden row count.", nameof(typeIndices));
            }
            if (typeEmbedding.GetLength(1) != embDim)
            {
                throw new ArgumentException("typeEmbedding dimension must match contextHidden embedding dimension.", nameof(typeEmbedding));
            }

            int typeCount = typeEmbedding.GetLength(0);
            for (int i = 0; i < rows; i++)
            {
                int type = typeIndices[i];
                if ((uint)type >= (uint)typeCount)
                {
                    throw new ArgumentOutOfRangeException(nameof(typeIndices), $"Invalid context type index {type} at row {i}.");
                }
            }

            if (!ShouldParallelize((long)rows * embDim))
            {
                SingleThreadCPU.ApplyContextTypeEmbedding(contextHidden, typeEmbedding, typeIndices);
                return;
            }

            Parallel.For(0, rows, _parallelOptions, i =>
            {
                int type = typeIndices[i];
                for (int d = 0; d < embDim; d++)
                {
                    contextHidden[i, d] += typeEmbedding[type, d];
                }
            });
        }

        public float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes)
        {
            if (priceSeqLen <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(priceSeqLen), "priceSeqLen must be positive.");
            }
            if (keyArrivalTimes == null) throw new ArgumentNullException(nameof(keyArrivalTimes));

            int numKeys = keyArrivalTimes.Length;
            long workUnits = (long)priceSeqLen * numKeys;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ComputeTimeDiffMatrix(priceSeqLen, keyArrivalTimes);
            }

            var td = new float[priceSeqLen, numKeys];
            Parallel.For(0, priceSeqLen, _parallelOptions, p =>
            {
                for (int s = 0; s < numKeys; s++)
                {
                    td[p, s] = p - keyArrivalTimes[s];
                }
            });

            return td;
        }

        public float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos, float[,] contextHidden, int totalCtx, float scale)
        {
            if (priceHidden == null) throw new ArgumentNullException(nameof(priceHidden));
            if (contextHidden == null) throw new ArgumentNullException(nameof(contextHidden));

            int embDim = priceHidden.GetLength(1);
            long workUnits = (long)totalCtx * embDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ComputeMemoryAttentionScores(priceHidden, lastPos, contextHidden, totalCtx, scale);
            }

            var scores = new float[totalCtx];
            Parallel.For(0, totalCtx, _parallelOptions, s =>
            {
                float dot = 0f;
                for (int d = 0; d < embDim; d++)
                {
                    dot += priceHidden[lastPos, d] * contextHidden[s, d];
                }
                scores[s] = dot * scale;
            });

            return scores;
        }

        public float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim)
        {
            if (hidden == null) throw new ArgumentNullException(nameof(hidden));
            if (outputProjection == null) throw new ArgumentNullException(nameof(outputProjection));
            if (outputBias == null) throw new ArgumentNullException(nameof(outputBias));

            int embDim = hidden.GetLength(1);
            long workUnits = (long)seqLen * outputDim * embDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ProjectOutputBatch(hidden, outputProjection, outputBias, seqLen, outputDim);
            }

            var pred = new float[seqLen, outputDim];
            Parallel.For(0, seqLen, _parallelOptions, i =>
            {
                for (int j = 0; j < outputDim; j++)
                {
                    float sum = outputBias[j];
                    for (int d = 0; d < embDim; d++)
                    {
                        sum += outputProjection[j, d] * hidden[i, d];
                    }
                    pred[i, j] = sum;
                }
            });

            return pred;
        }

        public float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

            int queryLen = Q.GetLength(0);
            int keyLen = K.GetLength(0);
            int embDim = Q.GetLength(1);
            int headDim = embDim / numHeads;

            long workUnits = (long)numHeads * queryLen * keyLen * Math.Max(1, headDim);
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ContentAwareCrossAttentionForward(Q, K, V, numHeads, scale, decayBias, out attentionWeights, out scoresPreSoftmax);
            }

            return ContentAwareCrossAttentionForwardCoreParallel(Q, K, V, numHeads, scale, decayBias, timeDiffs: null, out attentionWeights, out scoresPreSoftmax);
        }

        public float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, int PriceEmbeddingDim, int PriceNumHeads, bool enableDecayBias = true, bool isTraining = false, Random dropoutRng = null)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (block == null) throw new ArgumentNullException(nameof(block));
            if (bc == null) throw new ArgumentNullException(nameof(bc));

            int queryLen = Q.GetLength(0);
            int keyLen = K.GetLength(0);
            int ed = PriceEmbeddingDim;
            int nh = PriceNumHeads;
            int hd = ed / nh;
            long workUnits = (long)nh * queryLen * keyLen * Math.Max(1, hd);

            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ContentAwareCrossAttentionWithCache(Q, K, V, timeDiffs, keyTimesFromRef, queryEmbeddings, keyEmbeddings, block, bc, PriceEmbeddingDim, PriceNumHeads, enableDecayBias, isTraining, dropoutRng);
            }

            float[,,] decayBias = null;
            if (enableDecayBias && timeDiffs != null)
            {
                var (bias, decayCache) = ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, block.DecayNetwork, isTraining, dropoutRng);
                decayBias = bias;
                bc.DecayCache = decayCache;
            }
            else
            {
                bc.DecayCache = null;
            }

            float scale = 1f / MathF.Sqrt(hd);
            var output = ContentAwareCrossAttentionForwardCoreParallel(Q, K, V, nh, scale, decayBias, timeDiffs, out var attentionWeights, out var scoresPreSoftmax);
            bc.CrossAttentionWeights = attentionWeights;
            bc.CrossScoresPreSoftmax = scoresPreSoftmax;
            return output;
        }

        public float MatrixSquaredNorm3D(float[,,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);
            long workUnits = (long)d0 * d1 * d2;

            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.MatrixSquaredNorm3D(matrix);
            }

            var partials = new float[Math.Min(MAX_DEGREE_OF_PARALLELISM, Math.Max(1, d0))];
            Parallel.For(0, partials.Length, _parallelOptions, worker =>
            {
                int start = (int)((long)d0 * worker / partials.Length);
                int end = (int)((long)d0 * (worker + 1) / partials.Length);
                float local = 0f;

                for (int i = start; i < end; i++)
                {
                    for (int j = 0; j < d1; j++)
                    {
                        for (int k = 0; k < d2; k++)
                        {
                            float v = matrix[i, j, k];
                            local += v * v;
                        }
                    }
                }

                partials[worker] = local;
            });

            float sum = 0f;
            for (int i = 0; i < partials.Length; i++)
            {
                sum += partials[i];
            }
            return sum;
        }

        public float[] ProjectGlobalFeatures(float[] globalFeatures, float[,] projection, float[] bias)
        {
            if (globalFeatures == null) throw new ArgumentNullException(nameof(globalFeatures));
            if (projection == null) throw new ArgumentNullException(nameof(projection));
            if (bias == null) throw new ArgumentNullException(nameof(bias));

            int ed = projection.GetLength(0);
            int gd = projection.GetLength(1);
            if (globalFeatures.Length != gd)
            {
                throw new ArgumentException($"Expected global feature length {gd}, got {globalFeatures.Length}.");
            }
            if (bias.Length != ed)
            {
                throw new ArgumentException($"Expected bias length {ed}, got {bias.Length}.");
            }

            if (!ShouldParallelize((long)ed * gd))
            {
                return SingleThreadCPU.ProjectGlobalFeatures(globalFeatures, projection, bias);
            }

            var output = new float[ed];
            Parallel.For(0, ed, _parallelOptions, d =>
            {
                float sum = bias[d];
                for (int g = 0; g < gd; g++)
                {
                    sum += projection[d, g] * globalFeatures[g];
                }
                output[d] = sum;
            });

            return output;
        }

        public float[,] EmbedTokenIds(int[] tokenIds, float[,] embedding, int embeddingDim)
        {
            if (tokenIds == null || tokenIds.Length == 0)
            {
                return new float[0, embeddingDim];
            }
            if (embedding == null) throw new ArgumentNullException(nameof(embedding));

            int vocabSize = embedding.GetLength(0);
            if (embedding.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("Embedding dimension mismatch.", nameof(embedding));
            }

            long workUnits = (long)tokenIds.Length * embeddingDim;
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.EmbedTokenIds(tokenIds, embedding, embeddingDim);
            }

            for (int i = 0; i < tokenIds.Length; i++)
            {
                int tokenId = tokenIds[i];
                if ((uint)tokenId >= (uint)vocabSize)
                {
                    throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token out of range: {tokenId}");
                }
            }

            var output = new float[tokenIds.Length, embeddingDim];
            int bytesPerRow = embeddingDim * sizeof(float);
            Parallel.For(0, tokenIds.Length, _parallelOptions, i =>
            {
                Buffer.BlockCopy(embedding, tokenIds[i] * bytesPerRow, output, i * bytesPerRow, bytesPerRow);
            });

            return output;
        }

        public float[] MeanPoolRows(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var output = new float[cols];

            if (rows == 0)
            {
                return output;
            }

            if (!ShouldParallelize((long)rows * cols))
            {
                return SingleThreadCPU.MeanPoolRows(matrix);
            }

            float inv = 1f / rows;
            Parallel.For(0, cols, _parallelOptions, d =>
            {
                float sum = 0f;
                for (int r = 0; r < rows; r++)
                {
                    sum += matrix[r, d];
                }
                output[d] = sum * inv;
            });

            return output;
        }

        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews) BuildMmtacContext(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] contextTypeEmbedding)
        {
            if (contextTypeEmbedding == null) throw new ArgumentNullException(nameof(contextTypeEmbedding));

            int ed = contextTypeEmbedding.GetLength(1);
            int numGlobal = globalToken != null ? 1 : 0;
            int numNews = newsHidden != null ? newsHidden.GetLength(0) : 0;
            int total = numGlobal + numNews;

            if (total == 0)
            {
                return (null, null, 0, 0);
            }

            if (!ShouldParallelize((long)total * ed))
            {
                return SingleThreadCPU.BuildMmtacContext(newsHidden, newsTimes, globalToken, contextTypeEmbedding);
            }

            if (globalToken != null && globalToken.Length != ed) throw new ArgumentException("globalToken length must match embedding dimension.", nameof(globalToken));
            if (newsHidden != null && newsHidden.GetLength(1) != ed) throw new ArgumentException("newsHidden embedding dimension mismatch.", nameof(newsHidden));
            if (newsTimes != null && newsTimes.Length != numNews) throw new ArgumentException("newsTimes length must match newsHidden row count.", nameof(newsTimes));

            var contextHidden = new float[total, ed];
            var contextTimes = new float[total];

            if (globalToken != null)
            {
                for (int d = 0; d < ed; d++)
                {
                    contextHidden[0, d] = globalToken[d] + contextTypeEmbedding[2, d];
                }
                contextTimes[0] = 0f;
            }

            Parallel.For(0, numNews, _parallelOptions, i =>
            {
                int row = numGlobal + i;
                for (int d = 0; d < ed; d++)
                {
                    contextHidden[row, d] = newsHidden[i, d] + contextTypeEmbedding[0, d];
                }
                contextTimes[row] = newsTimes != null ? newsTimes[i] : 0f;
            });

            return (contextHidden, contextTimes, numGlobal, numNews);
        }

        public (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] regressionLogits, float[] rangeLogits, float[] qualityLogits) ProjectMmtacOutputHeads(float[,] hidden, float[,] regressionProjection, float[] regressionBias, float[,] rangeProjection, float[] rangeBias, float[,] qualityProjection, float[] qualityBias, float[,] directionProjection, float[] directionBias, float[,] midDirectionProjection, float[] midDirectionBias, float[,] confidenceProjection, float[] confidenceBias, bool useConfidenceHead)
        {
            if (hidden == null) throw new ArgumentNullException(nameof(hidden));

            int sl = hidden.GetLength(0);
            int ed = hidden.GetLength(1);
            const int rDim = 3;

            long workUnits = (long)sl * ed * (useConfidenceHead ? 6 : 5);
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.ProjectMmtacOutputHeads(hidden, regressionProjection, regressionBias, rangeProjection, rangeBias, qualityProjection, qualityBias, directionProjection, directionBias, midDirectionProjection, midDirectionBias, confidenceProjection, confidenceBias, useConfidenceHead);
            }

            var rawRegression = ProjectOutputBatch(hidden, regressionProjection, regressionBias, sl, rDim);
            var regression = new float[sl, rDim];
            var range = new float[sl, 1];
            var rangeLogits = new float[sl];
            var quality = new float[sl, 1];
            var qualityLogits = new float[sl];
            var direction = new float[sl, 1];
            var midDirection = new float[sl, 1];
            float[,] confidence = useConfidenceHead ? new float[sl, 1] : null;

            Parallel.For(0, sl, _parallelOptions, t =>
            {
                float upBase = StableSoftplus(rawRegression[t, 0]);
                float downBase = StableSoftplus(rawRegression[t, 1]);
                float rangeLogit = rangeBias[0] + DotProjectionRow(rangeProjection, 0, hidden, t, ed);
                rangeLogits[t] = rangeLogit;

                float rangeValue = StableSoftplus(rangeLogit);
                float den = upBase + downBase;
                float upShare = den > 1e-6f ? upBase / den : 0.5f;
                float downShare = 1f - upShare;
                float close = rawRegression[t, 2];

                regression[t, 0] = close + rangeValue * upShare;
                regression[t, 1] = close - rangeValue * downShare;
                regression[t, 2] = close;
                range[t, 0] = rangeValue;

                float qualityLogit = qualityBias[0] + DotProjectionRow(qualityProjection, 0, hidden, t, ed);
                qualityLogits[t] = qualityLogit;
                quality[t, 0] = StableSigmoid(qualityLogit);

                float directionLogit = directionBias[0] + DotProjectionRow(directionProjection, 0, hidden, t, ed);
                direction[t, 0] = StableSigmoid(directionLogit);

                float midDirectionLogit = midDirectionBias[0] + DotProjectionRow(midDirectionProjection, 0, hidden, t, ed);
                midDirection[t, 0] = StableSigmoid(midDirectionLogit);

                if (useConfidenceHead)
                {
                    float confidenceLogit = confidenceBias[0] + DotProjectionRow(confidenceProjection, 0, hidden, t, ed);
                    confidence[t, 0] = StableSigmoid(confidenceLogit);
                }
            });

            return (regression, range, quality, direction, midDirection, confidence, rawRegression, rangeLogits, qualityLogits);
        }

        public float[] SoftmaxVector(float[] scores)
        {
            if (scores == null) throw new ArgumentNullException(nameof(scores));

            var output = new float[scores.Length];
            if (scores.Length == 0)
            {
                return output;
            }

            if (!ShouldParallelize(scores.Length))
            {
                return SingleThreadCPU.SoftmaxVector(scores);
            }

            int workers = Math.Min(MAX_DEGREE_OF_PARALLELISM, scores.Length);
            var partialMax = new float[workers];
            Parallel.For(0, workers, _parallelOptions, worker =>
            {
                int start = (int)((long)scores.Length * worker / workers);
                int end = (int)((long)scores.Length * (worker + 1) / workers);
                float localMax = float.NegativeInfinity;
                for (int i = start; i < end; i++)
                {
                    if (scores[i] > localMax)
                    {
                        localMax = scores[i];
                    }
                }
                partialMax[worker] = localMax;
            });

            float max = float.NegativeInfinity;
            for (int i = 0; i < workers; i++)
            {
                if (partialMax[i] > max)
                {
                    max = partialMax[i];
                }
            }

            var partialSum = new float[workers];
            Parallel.For(0, workers, _parallelOptions, worker =>
            {
                int start = (int)((long)scores.Length * worker / workers);
                int end = (int)((long)scores.Length * (worker + 1) / workers);
                float localSum = 0f;
                for (int i = start; i < end; i++)
                {
                    float value = MathF.Exp(scores[i] - max);
                    output[i] = value;
                    localSum += value;
                }
                partialSum[worker] = localSum;
            });

            float sum = 0f;
            for (int i = 0; i < workers; i++)
            {
                sum += partialSum[i];
            }

            if (sum <= 0f)
            {
                return output;
            }

            float inv = 1f / sum;
            Parallel.For(0, output.Length, _parallelOptions, i => output[i] *= inv);
            return output;
        }

        public (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttention(float[,] q, float[,] k, float[,] v, float[,] dOutput, float[][,] attentionWeights, float[,] timeDiffs, int embeddingDim, int numHeads)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (dOutput == null) throw new ArgumentNullException(nameof(dOutput));
            if (attentionWeights == null) throw new ArgumentNullException(nameof(attentionWeights));

            int queryLen = q.GetLength(0);
            int keyLen = k.GetLength(0);
            int headDim = embeddingDim / numHeads;
            long workUnits = (long)numHeads * queryLen * keyLen * Math.Max(1, headDim);

            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.BackpropTimeDecayedAttention(q, k, v, dOutput, attentionWeights, timeDiffs, embeddingDim, numHeads);
            }

            float scale = 1f / MathF.Sqrt(headDim);
            var dQ = new float[queryLen, embeddingDim];
            var dK = new float[keyLen, embeddingDim];
            var dV = new float[keyLen, embeddingDim];
            float[,,] dDecayBias = timeDiffs != null ? new float[queryLen, keyLen, numHeads] : null;

            Parallel.For(0, numHeads, _parallelOptions, h =>
            {
                int offset = h * headDim;
                var weights = attentionWeights[h];
                var dWeights = new float[keyLen];

                for (int p = 0; p < queryLen; p++)
                {
                    for (int s = 0; s < keyLen; s++)
                    {
                        float dot = 0f;
                        float weight = weights[p, s];
                        for (int d = 0; d < headDim; d++)
                        {
                            float dout = dOutput[p, offset + d];
                            dV[s, offset + d] += weight * dout;
                            dot += dout * v[s, offset + d];
                        }
                        dWeights[s] = dot;
                    }

                    float dotWeightGrad = 0f;
                    for (int s = 0; s < keyLen; s++)
                    {
                        dotWeightGrad += weights[p, s] * dWeights[s];
                    }

                    for (int s = 0; s < keyLen; s++)
                    {
                        float dScore = weights[p, s] * (dWeights[s] - dotWeightGrad);
                        float dScaled = dScore * scale;

                        for (int d = 0; d < headDim; d++)
                        {
                            dQ[p, offset + d] += dScaled * k[s, offset + d];
                            dK[s, offset + d] += dScaled * q[p, offset + d];
                        }

                        if (dDecayBias != null)
                        {
                            dDecayBias[p, s, h] += dScore;
                        }
                    }
                }
            });

            return (dQ, dK, dV, dDecayBias);
        }

        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews, int numPrice) BuildMmtacContextWithPrice(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] priceContextHidden, float[] priceContextTimes, float[,] contextTypeEmbedding)
        {
            if (contextTypeEmbedding == null) throw new ArgumentNullException(nameof(contextTypeEmbedding));

            int ed = contextTypeEmbedding.GetLength(1);
            int numGlobal = globalToken != null ? 1 : 0;
            int numNews = newsHidden != null ? newsHidden.GetLength(0) : 0;
            int numPrice = priceContextHidden != null ? priceContextHidden.GetLength(0) : 0;
            int total = numGlobal + numNews + numPrice;

            if (total == 0)
            {
                return (null, null, 0, 0, 0);
            }

            if (!ShouldParallelize((long)total * ed))
            {
                return SingleThreadCPU.BuildMmtacContextWithPrice(newsHidden, newsTimes, globalToken, priceContextHidden, priceContextTimes, contextTypeEmbedding);
            }

            if (globalToken != null && globalToken.Length != ed) throw new ArgumentException("globalToken length must match embedding dimension.", nameof(globalToken));
            if (newsHidden != null && newsHidden.GetLength(1) != ed) throw new ArgumentException("newsHidden embedding dimension mismatch.", nameof(newsHidden));
            if (newsTimes != null && newsTimes.Length != numNews) throw new ArgumentException("newsTimes length must match newsHidden row count.", nameof(newsTimes));
            if (priceContextHidden != null && priceContextHidden.GetLength(1) != ed) throw new ArgumentException("priceContextHidden embedding dimension mismatch.", nameof(priceContextHidden));
            if (priceContextTimes != null && priceContextTimes.Length != numPrice) throw new ArgumentException("priceContextTimes length must match priceContextHidden row count.", nameof(priceContextTimes));

            var contextHidden = new float[total, ed];
            var contextTimes = new float[total];

            if (globalToken != null)
            {
                for (int d = 0; d < ed; d++)
                {
                    contextHidden[0, d] = globalToken[d] + contextTypeEmbedding[2, d];
                }
                contextTimes[0] = 0f;
            }

            Parallel.For(0, numNews, _parallelOptions, i =>
            {
                int row = numGlobal + i;
                for (int d = 0; d < ed; d++)
                {
                    contextHidden[row, d] = newsHidden[i, d] + contextTypeEmbedding[0, d];
                }
                contextTimes[row] = newsTimes != null ? newsTimes[i] : 0f;
            });

            int priceOffset = numGlobal + numNews;
            Parallel.For(0, numPrice, _parallelOptions, i =>
            {
                int row = priceOffset + i;
                for (int d = 0; d < ed; d++)
                {
                    contextHidden[row, d] = priceContextHidden[i, d] + contextTypeEmbedding[1, d];
                }
                contextTimes[row] = priceContextTimes != null ? priceContextTimes[i] : 0f;
            });

            return (contextHidden, contextTimes, numGlobal, numNews, numPrice);
        }


        public (float loss, float[,] dHidden) BackpropMmtacOutputHeads(float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] targetRegression, float[,] targetRange, float[,] targetQuality, float[,] targetDirection, float[,] targetMidDirection, float[] previousClose, float[] confidenceTargets, float[,] hidden, float[,] regressionLogits, float[] rangeLogits, float[,] regressionProjection, float[,] rangeProjection, float[,] qualityProjection, float[,] directionProjection, float[,] midDirectionProjection, float[,] confidenceProjection, float[,] regressionProjectionGrad, float[] regressionBiasGrad, float[,] rangeProjectionGrad, float[] rangeBiasGrad, float[,] qualityProjectionGrad, float[] qualityBiasGrad, float[,] directionProjectionGrad, float[] directionBiasGrad, float[,] midDirectionProjectionGrad, float[] midDirectionBiasGrad, float[,] confidenceProjectionGrad, float[] confidenceBiasGrad, float rangeLossWeight, float qualityLossWeight, float directionLossWeight, float midDirectionLossWeight, float closeDirectionConsistencyWeight, float closeDirectionConsistencyMargin, float confidenceLossWeight, bool useConfidenceHead)
        {
            if (regression == null) throw new ArgumentNullException(nameof(regression));
            if (targetRegression == null) throw new ArgumentNullException(nameof(targetRegression));
            if (hidden == null) throw new ArgumentNullException(nameof(hidden));
            if (regressionLogits == null) throw new ArgumentNullException(nameof(regressionLogits));
            if (rangeLogits == null) throw new ArgumentNullException(nameof(rangeLogits));

            int sl = regression.GetLength(0);
            int rDim = regression.GetLength(1);
            int ed = hidden.GetLength(1);

            if (rDim < 3)
            {
                throw new ArgumentException("Regression output must contain high, low and close columns.", nameof(regression));
            }
            if (hidden.GetLength(0) != sl)
            {
                throw new ArgumentException("hidden row count must match output row count.", nameof(hidden));
            }
            if (confidenceTargets != null && confidenceTargets.Length < sl)
            {
                throw new ArgumentException("confidenceTargets length must be at least the sequence length.", nameof(confidenceTargets));
            }

            long workUnits = (long)Math.Max(1, sl) * Math.Max(1, ed) * (useConfidenceHead ? 7L : 6L);
            if (!ShouldParallelize(workUnits))
            {
                return SingleThreadCPU.BackpropMmtacOutputHeads(regression, range, quality, direction, midDirection, confidence, targetRegression, targetRange, targetQuality, targetDirection, targetMidDirection, previousClose, confidenceTargets, hidden, regressionLogits, rangeLogits, regressionProjection, rangeProjection, qualityProjection, directionProjection, midDirectionProjection, confidenceProjection, regressionProjectionGrad, regressionBiasGrad, rangeProjectionGrad, rangeBiasGrad, qualityProjectionGrad, qualityBiasGrad, directionProjectionGrad, directionBiasGrad, midDirectionProjectionGrad, midDirectionBiasGrad, confidenceProjectionGrad, confidenceBiasGrad, rangeLossWeight, qualityLossWeight, directionLossWeight, midDirectionLossWeight, closeDirectionConsistencyWeight, closeDirectionConsistencyMargin, confidenceLossWeight, useConfidenceHead);
            }

            float invRegCount = 1f / (sl * rDim);
            var mseLossByRow = new float[sl];
            var dHigh = new float[sl];
            var dLow = new float[sl];
            var dClose = new float[sl];
            var dRangeOutput = new float[sl];

            Parallel.For(0, sl, _parallelOptions, t =>
            {
                float diffHigh = regression[t, 0] - targetRegression[t, 0];
                float diffLow = regression[t, 1] - targetRegression[t, 1];
                float diffClose = regression[t, 2] - targetRegression[t, 2];

                mseLossByRow[t] = diffHigh * diffHigh + diffLow * diffLow + diffClose * diffClose;
                dHigh[t] = 2f * diffHigh * invRegCount;
                dLow[t] = 2f * diffLow * invRegCount;
                dClose[t] = 2f * diffClose * invRegCount;
            });

            float mseLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                mseLoss += mseLossByRow[t];
            }
            mseLoss *= invRegCount;

            float closeDirectionLoss = 0f;
            if (closeDirectionConsistencyWeight > 0f && previousClose != null)
            {
                var closeDirectionLossByRow = new float[sl];
                var dCloseConsistency = new float[sl];

                Parallel.For(0, sl, _parallelOptions, t =>
                {
                    float sign = targetDirection[t, 0] >= 0.5f ? 1f : -1f;
                    float z = sign * (regression[t, 2] - previousClose[t] - sign * closeDirectionConsistencyMargin);

                    if (z > 20f)
                    {
                        closeDirectionLossByRow[t] = MathF.Exp(-z);
                    }
                    else if (z < -20f)
                    {
                        closeDirectionLossByRow[t] = -z;
                    }
                    else
                    {
                        closeDirectionLossByRow[t] = MathF.Log(1f + MathF.Exp(-z));
                    }

                    float sigmoidNegZ;
                    if (z >= 0f)
                    {
                        float ez = MathF.Exp(-z);
                        sigmoidNegZ = ez / (1f + ez);
                    }
                    else
                    {
                        float ez = MathF.Exp(z);
                        sigmoidNegZ = 1f / (1f + ez);
                    }

                    dCloseConsistency[t] = -sign * sigmoidNegZ * closeDirectionConsistencyWeight / sl;
                });

                for (int t = 0; t < sl; t++)
                {
                    closeDirectionLoss += closeDirectionLossByRow[t];
                    dClose[t] += dCloseConsistency[t];
                }
                closeDirectionLoss /= sl;
            }

            var rangeLossByRow = new float[sl];
            Parallel.For(0, sl, _parallelOptions, t =>
            {
                float diff = range[t, 0] - targetRange[t, 0];
                rangeLossByRow[t] = diff * diff;
                dRangeOutput[t] = 2f * diff / sl * rangeLossWeight;
            });

            float rangeLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                rangeLoss += rangeLossByRow[t];
            }
            rangeLoss /= sl;

            var dRawRegression = new float[sl, rDim];
            var dRangeLogit = new float[sl];

            Parallel.For(0, sl, _parallelOptions, t =>
            {
                float upLogit = regressionLogits[t, 0];
                float downLogit = regressionLogits[t, 1];
                float rangeLogit = rangeLogits[t];

                float upBase = StableSoftplus(upLogit);
                float downBase = StableSoftplus(downLogit);
                float den = upBase + downBase;
                float upShare = den > 1e-6f ? upBase / den : 0.5f;
                float downShare = 1f - upShare;
                float rangeValue = StableSoftplus(rangeLogit);

                float gHigh = dHigh[t];
                float gLow = dLow[t];
                float gClose = dClose[t];
                float gRange = dRangeOutput[t];

                float dCloseRaw = gHigh + gLow + gClose;
                float dRangeValue = gHigh * upShare - gLow * downShare + gRange;
                float dShare = rangeValue * (gHigh + gLow);

                float dUpBase = 0f;
                float dDownBase = 0f;
                if (den > 1e-6f)
                {
                    float invDenSq = 1f / (den * den);
                    dUpBase = dShare * downBase * invDenSq;
                    dDownBase = -dShare * upBase * invDenSq;
                }

                dRawRegression[t, 0] = dUpBase * StableSigmoid(upLogit);
                dRawRegression[t, 1] = dDownBase * StableSigmoid(downLogit);
                dRawRegression[t, 2] = dCloseRaw;
                dRangeLogit[t] = dRangeValue * StableSigmoid(rangeLogit);
            });

            var dQualityLogit = new float[sl];
            var qualityLossByRow = new float[sl];
            Parallel.For(0, sl, _parallelOptions, t =>
            {
                float p = quality[t, 0];
                float diff = p - targetQuality[t, 0];
                qualityLossByRow[t] = diff * diff;

                float dOutput = 2f * diff / sl * qualityLossWeight;
                dQualityLogit[t] = dOutput * p * (1f - p);
            });

            float qualityLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                qualityLoss += qualityLossByRow[t];
            }
            qualityLoss /= sl;

            var dDirectionLogit = new float[sl];
            var directionLossByRow = new float[sl];
            Parallel.For(0, sl, _parallelOptions, t =>
            {
                float p = direction[t, 0];
                float y = targetDirection[t, 0];
                float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                directionLossByRow[t] = -(y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc));
                dDirectionLogit[t] = (p - y) * directionLossWeight / sl;
            });

            float directionLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                directionLoss += directionLossByRow[t];
            }
            directionLoss /= sl;

            var dMidDirectionLogit = new float[sl];
            var midDirectionLossByRow = new float[sl];
            Parallel.For(0, sl, _parallelOptions, t =>
            {
                float p = midDirection[t, 0];
                float y = targetMidDirection[t, 0];
                float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                midDirectionLossByRow[t] = -(y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc));
                dMidDirectionLogit[t] = (p - y) * midDirectionLossWeight / sl;
            });

            float midDirectionLoss = 0f;
            for (int t = 0; t < sl; t++)
            {
                midDirectionLoss += midDirectionLossByRow[t];
            }
            midDirectionLoss /= sl;

            float confidenceLoss = 0f;
            float effectiveConfidenceWeight = useConfidenceHead ? MathF.Max(0f, confidenceLossWeight) : 0f;
            float[] dConfidenceLogit = null;

            if (useConfidenceHead && confidence != null && confidenceProjection != null && confidenceProjectionGrad != null && confidenceBiasGrad != null && effectiveConfidenceWeight > 0f)
            {
                dConfidenceLogit = new float[sl];
                var confidenceLossByRow = new float[sl];

                Parallel.For(0, sl, _parallelOptions, t =>
                {
                    float p = confidence[t, 0];
                    float y;

                    if (confidenceTargets != null)
                    {
                        y = confidenceTargets[t];
                    }
                    else
                    {
                        float sq = 0f;
                        for (int j = 0; j < rDim; j++)
                        {
                            float diff = regression[t, j] - targetRegression[t, j];
                            sq += diff * diff;
                        }
                        y = MathF.Exp(-5f * MathF.Sqrt(sq / rDim));
                    }

                    float pc = Math.Clamp(p, 1e-7f, 1f - 1e-7f);
                    confidenceLossByRow[t] = -(y * MathF.Log(pc) + (1f - y) * MathF.Log(1f - pc));
                    dConfidenceLogit[t] = (p - y) * effectiveConfidenceWeight / sl;
                });

                for (int t = 0; t < sl; t++)
                {
                    confidenceLoss += confidenceLossByRow[t];
                }
                confidenceLoss /= sl;
            }

            for (int t = 0; t < sl; t++)
            {
                rangeBiasGrad[0] += dRangeLogit[t];
                qualityBiasGrad[0] += dQualityLogit[t];
                directionBiasGrad[0] += dDirectionLogit[t];
                midDirectionBiasGrad[0] += dMidDirectionLogit[t];

                if (dConfidenceLogit != null)
                {
                    confidenceBiasGrad[0] += dConfidenceLogit[t];
                }

                for (int v = 0; v < rDim; v++)
                {
                    regressionBiasGrad[v] += dRawRegression[t, v];
                }
            }

            Parallel.For(0, rDim * ed, _parallelOptions, flat =>
            {
                int v = flat / ed;
                int k = flat - v * ed;
                for (int t = 0; t < sl; t++)
                {
                    regressionProjectionGrad[v, k] += dRawRegression[t, v] * hidden[t, k];
                }
            });

            Parallel.For(0, ed, _parallelOptions, k =>
            {
                for (int t = 0; t < sl; t++)
                {
                    rangeProjectionGrad[0, k] += dRangeLogit[t] * hidden[t, k];
                    qualityProjectionGrad[0, k] += dQualityLogit[t] * hidden[t, k];
                    directionProjectionGrad[0, k] += dDirectionLogit[t] * hidden[t, k];
                    midDirectionProjectionGrad[0, k] += dMidDirectionLogit[t] * hidden[t, k];

                    if (dConfidenceLogit != null)
                    {
                        confidenceProjectionGrad[0, k] += dConfidenceLogit[t] * hidden[t, k];
                    }
                }
            });

            var dHidden = new float[sl, ed];
            Parallel.For(0, sl, _parallelOptions, t =>
            {
                for (int k = 0; k < ed; k++)
                {
                    float grad = dRangeLogit[t] * rangeProjection[0, k];

                    for (int v = 0; v < rDim; v++)
                    {
                        grad += dRawRegression[t, v] * regressionProjection[v, k];
                    }

                    grad += dQualityLogit[t] * qualityProjection[0, k];
                    grad += dDirectionLogit[t] * directionProjection[0, k];
                    grad += dMidDirectionLogit[t] * midDirectionProjection[0, k];

                    if (dConfidenceLogit != null)
                    {
                        grad += dConfidenceLogit[t] * confidenceProjection[0, k];
                    }

                    dHidden[t, k] = grad;
                }
            });

            float loss = mseLoss
                       + closeDirectionConsistencyWeight * closeDirectionLoss
                       + rangeLossWeight * rangeLoss
                       + qualityLossWeight * qualityLoss
                       + directionLossWeight * directionLoss
                       + midDirectionLossWeight * midDirectionLoss
                       + effectiveConfidenceWeight * confidenceLoss;

            return (loss, dHidden);
        }

        public void AccumulateMmtacContextGradients(float[,] dContextA, float[,] dContextB, float[,] contextTypeEmbeddingGrad, float[,] dLiveNewsHidden, float[] dGlobalHidden, int numGlobal, int numStoredNews, int numNews, int numLiveNews, int numPriceContext, int totalContext, int priceOffset)
        {
            if (contextTypeEmbeddingGrad == null) throw new ArgumentNullException(nameof(contextTypeEmbeddingGrad));
            if (dContextA == null && dContextB == null)
            {
                return;
            }

            int ed = contextTypeEmbeddingGrad.GetLength(1);
            if (!ShouldParallelize((long)Math.Max(1, totalContext) * ed))
            {
                SingleThreadCPU.AccumulateMmtacContextGradients(dContextA, dContextB, contextTypeEmbeddingGrad, dLiveNewsHidden, dGlobalHidden, numGlobal, numStoredNews, numNews, numLiveNews, numPriceContext, totalContext, priceOffset);
                return;
            }

            Parallel.For(0, ed, _parallelOptions, j =>
            {
                for (int gi = 0; gi < numGlobal && gi < totalContext; gi++)
                {
                    float g = GetMmtacContextGrad(dContextA, dContextB, gi, j);
                    contextTypeEmbeddingGrad[2, j] += g;
                    if (dGlobalHidden != null)
                    {
                        dGlobalHidden[j] += g;
                    }
                }

                for (int i = 0; i < numNews; i++)
                {
                    int ctxIdx = numGlobal + i;
                    if (ctxIdx >= totalContext)
                    {
                        break;
                    }

                    bool isLive = i >= numStoredNews;
                    int liveIdx = i - numStoredNews;
                    float g = GetMmtacContextGrad(dContextA, dContextB, ctxIdx, j);
                    contextTypeEmbeddingGrad[0, j] += g;

                    if (isLive && dLiveNewsHidden != null && liveIdx >= 0 && liveIdx < numLiveNews)
                    {
                        dLiveNewsHidden[liveIdx, j] += g;
                    }
                }

                for (int i = 0; i < numPriceContext; i++)
                {
                    int ctxIdx = priceOffset + i;
                    if (ctxIdx >= totalContext)
                    {
                        break;
                    }

                    contextTypeEmbeddingGrad[1, j] += GetMmtacContextGrad(dContextA, dContextB, ctxIdx, j);
                }
            });
        }

        public void AccumulateGlobalProjectionGradients(float[] dGlobalHidden, float[] globalFeatures, float[,] projectionGrad, float[] biasGrad)
        {
            if (dGlobalHidden == null || globalFeatures == null || projectionGrad == null || biasGrad == null)
            {
                return;
            }

            int ed = dGlobalHidden.Length;
            int gd = globalFeatures.Length;

            if (!ShouldParallelize((long)ed * gd))
            {
                SingleThreadCPU.AccumulateGlobalProjectionGradients(dGlobalHidden, globalFeatures, projectionGrad, biasGrad);
                return;
            }

            Parallel.For(0, ed, _parallelOptions, d =>
            {
                float gToken = dGlobalHidden[d];
                biasGrad[d] += gToken;
                for (int g = 0; g < gd; g++)
                {
                    projectionGrad[d, g] += gToken * globalFeatures[g];
                }
            });
        }

        public float[,] ExpandMeanPoolGradient(float[,] pooledGradient, int rowIndex, int rowCount, int embeddingDim)
        {
            if (pooledGradient == null) throw new ArgumentNullException(nameof(pooledGradient));
            if (rowCount <= 0)
            {
                return new float[0, embeddingDim];
            }
            if (rowIndex < 0 || rowIndex >= pooledGradient.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex));
            }

            if (!ShouldParallelize((long)rowCount * embeddingDim))
            {
                return SingleThreadCPU.ExpandMeanPoolGradient(pooledGradient, rowIndex, rowCount, embeddingDim);
            }

            var result = new float[rowCount, embeddingDim];
            float inv = 1f / rowCount;
            int pooledCols = pooledGradient.GetLength(1);

            Parallel.For(0, rowCount, _parallelOptions, t =>
            {
                for (int d = 0; d < embeddingDim; d++)
                {
                    result[t, d] = pooledGradient[rowIndex, d] * inv;
                }
            });

            return result;
        }

        public void ApplyRotaryPositionEmbeddingInPlace(float[,] matrix, int numHeads, float baseTheta, bool inverse)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
            if (baseTheta <= 0f || float.IsNaN(baseTheta) || float.IsInfinity(baseTheta)) throw new ArgumentOutOfRangeException(nameof(baseTheta));

            int seqLen = matrix.GetLength(0);
            int embeddingDim = matrix.GetLength(1);
            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads.");
            }

            int headDim = embeddingDim / numHeads;
            if ((headDim & 1) != 0)
            {
                throw new ArgumentException("RoPE requires an even per-head dimension.");
            }

            long workUnits = (long)seqLen * embeddingDim;
            if (!ShouldParallelize(workUnits))
            {
                SingleThreadCPU.ApplyRotaryPositionEmbeddingInPlace(matrix, numHeads, baseTheta, inverse);
                return;
            }

            ApplyRotaryPositionEmbeddingAllHeadsParallelInPlace(matrix, numHeads, headDim, baseTheta, inverse);
        }

        public void ApplyRotaryPositionEmbeddingHeadInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));
            if (startCol < 0) throw new ArgumentOutOfRangeException(nameof(startCol));
            if (headDim < 0) throw new ArgumentOutOfRangeException(nameof(headDim));
            if ((headDim & 1) != 0) throw new ArgumentException("RoPE requires an even per-head dimension.", nameof(headDim));
            if (baseTheta <= 0f || float.IsNaN(baseTheta) || float.IsInfinity(baseTheta)) throw new ArgumentOutOfRangeException(nameof(baseTheta));

            int seqLen = matrix.GetLength(0);
            int embeddingDim = matrix.GetLength(1);
            if (startCol + headDim > embeddingDim)
            {
                throw new ArgumentException("Head range exceeds matrix embedding dimension.");
            }

            long workUnits = (long)seqLen * headDim;
            if (!ShouldParallelize(workUnits))
            {
                SingleThreadCPU.ApplyRotaryPositionEmbeddingHeadInPlace(matrix, startCol, headDim, baseTheta, inverse);
                return;
            }

            ApplyRotaryPositionEmbeddingSingleHeadParallelInPlace(matrix, startCol, headDim, baseTheta, inverse);
        }

        public float[,] ContentAwareCrossAttentionWithCache(
            float[,] Q,
            float[,] K,
            float[,] V,
            float[,] timeDiffs,
            float[] keyTimesFromRef,
            float[,] queryEmbeddings,
            float[,] keyEmbeddings,
            TacamtBlock block,
            BlockCache bc,
            int PriceEmbeddingDim,
            int PriceNumHeads,
            bool isTraining = false,
            Random dropoutRng = null)
        {
            return ContentAwareCrossAttentionWithCache(
                Q,
                K,
                V,
                timeDiffs,
                keyTimesFromRef,
                queryEmbeddings,
                keyEmbeddings,
                block,
                bc,
                PriceEmbeddingDim,
                PriceNumHeads,
                enableDecayBias: true,
                isTraining: isTraining,
                dropoutRng: dropoutRng);
        }

        public void ApplyRotaryPositionEmbeddingInPlace(float[,] matrix, int numHeads)
        {
            ApplyRotaryPositionEmbeddingInPlace(matrix, numHeads, 10000f, inverse: false);
        }

        public void ApplyRotaryPositionEmbeddingBackwardInPlace(float[,] matrix, int numHeads)
        {
            ApplyRotaryPositionEmbeddingInPlace(matrix, numHeads, 10000f, inverse: true);
        }

        #endregion

        #region Helpers required by the replacements

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float DotHeadRows(float[,] a, int aRow, float[,] b, int bRow, int offset, int length)
        {
            float sum = 0f;
            for (int d = 0; d < length; d++)
            {
                sum += a[aRow, offset + d] * b[bRow, offset + d];
            }
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float DotProjectionRow(float[,] projection, int projectionRow, float[,] input, int inputRow, int length)
        {
            float sum = 0f;
            for (int d = 0; d < length; d++)
            {
                sum += projection[projectionRow, d] * input[inputRow, d];
            }
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float StableSoftplus(float x)
        {
            if (x > 20f)
            {
                return x;
            }
            if (x < -20f)
            {
                return MathF.Exp(x);
            }
            return MathF.Log(1f + MathF.Exp(x));
        }

        private static long EstimateAttentionBackwardWork(float[,] Q, float[,] K, int numHeads)
        {
            if (Q == null || K == null || numHeads <= 0)
            {
                return long.MaxValue;
            }
            return (long)Q.GetLength(0) * K.GetLength(0) * Q.GetLength(1);
        }

        private (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackwardCoreParallel(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask, bool useDecoderMask)
        {
            if (Q == null) throw new ArgumentNullException(nameof(Q));
            if (K == null) throw new ArgumentNullException(nameof(K));
            if (V == null) throw new ArgumentNullException(nameof(V));
            if (dConcatenated == null) throw new ArgumentNullException(nameof(dConcatenated));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));

            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);

            if (K.GetLength(1) != embeddingDim || V.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("Q, K and V must have the same embedding dimension.");
            }
            if (V.GetLength(0) != seqLenK)
            {
                throw new ArgumentException("K and V must have the same sequence length.");
            }
            if (dConcatenated.GetLength(0) != seqLenQ || dConcatenated.GetLength(1) != embeddingDim)
            {
                throw new ArgumentException("dConcatenated shape must match Q shape.", nameof(dConcatenated));
            }
            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads.", nameof(numHeads));
            }
            if (mask != null && (mask.GetLength(0) != seqLenQ || mask.GetLength(1) != seqLenK))
            {
                throw new ArgumentException($"Mask shape must be [{seqLenQ},{seqLenK}], got [{mask.GetLength(0)},{mask.GetLength(1)}].", nameof(mask));
            }

            int headDim = embeddingDim / numHeads;
            var dQFull = new float[seqLenQ, embeddingDim];
            var dKFull = new float[seqLenK, embeddingDim];
            var dVFull = new float[seqLenK, embeddingDim];

            Parallel.For(0, numHeads, _parallelOptions, head =>
            {
                int offset = head * headDim;
                var weights = new float[seqLenK];
                var dAttn = new float[seqLenK];
                var activeIndices = new int[seqLenK];
                var dqRow = new float[headDim];

                for (int i = 0; i < seqLenQ; i++)
                {
                    Array.Clear(dqRow, 0, headDim);
                    float maxScore = float.NegativeInfinity;
                    int activeCount = 0;

                    if (mask != null)
                    {
                        for (int j = 0; j < seqLenK; j++)
                        {
                            if (!mask[i, j])
                            {
                                continue;
                            }

                            float score = DotHeadRows(Q, i, K, j, offset, headDim) * scale;
                            weights[j] = score;
                            activeIndices[activeCount++] = j;
                            if (score > maxScore)
                            {
                                maxScore = score;
                            }
                        }
                    }
                    else
                    {
                        activeCount = useDecoderMask ? Math.Min(i + 1, seqLenK) : seqLenK;
                        for (int j = 0; j < activeCount; j++)
                        {
                            float score = DotHeadRows(Q, i, K, j, offset, headDim) * scale;
                            weights[j] = score;
                            if (score > maxScore)
                            {
                                maxScore = score;
                            }
                        }
                    }

                    if (activeCount == 0 || float.IsNegativeInfinity(maxScore))
                    {
                        continue;
                    }

                    float sumExp = 0f;
                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            float w = MathF.Exp(weights[j] - maxScore);
                            weights[j] = w;
                            sumExp += w;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                        {
                            float w = MathF.Exp(weights[j] - maxScore);
                            weights[j] = w;
                            sumExp += w;
                        }
                    }

                    if (sumExp <= 0f)
                    {
                        continue;
                    }

                    float invSumExp = 1f / sumExp;
                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            weights[j] *= invSumExp;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                        {
                            weights[j] *= invSumExp;
                        }
                    }

                    float rowDot = 0f;
                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            float w = weights[j];
                            float dAttnJ = 0f;
                            for (int d = 0; d < headDim; d++)
                            {
                                float dout = dConcatenated[i, offset + d];
                                dVFull[j, offset + d] += w * dout;
                                dAttnJ += dout * V[j, offset + d];
                            }
                            dAttn[j] = dAttnJ;
                            rowDot += w * dAttnJ;
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                        {
                            float w = weights[j];
                            float dAttnJ = 0f;
                            for (int d = 0; d < headDim; d++)
                            {
                                float dout = dConcatenated[i, offset + d];
                                dVFull[j, offset + d] += w * dout;
                                dAttnJ += dout * V[j, offset + d];
                            }
                            dAttn[j] = dAttnJ;
                            rowDot += w * dAttnJ;
                        }
                    }

                    if (mask != null)
                    {
                        for (int n = 0; n < activeCount; n++)
                        {
                            int j = activeIndices[n];
                            float dDot = weights[j] * (dAttn[j] - rowDot) * scale;
                            if (dDot == 0f)
                            {
                                continue;
                            }

                            for (int d = 0; d < headDim; d++)
                            {
                                dqRow[d] += dDot * K[j, offset + d];
                                dKFull[j, offset + d] += dDot * Q[i, offset + d];
                            }
                        }
                    }
                    else
                    {
                        for (int j = 0; j < activeCount; j++)
                        {
                            float dDot = weights[j] * (dAttn[j] - rowDot) * scale;
                            if (dDot == 0f)
                            {
                                continue;
                            }

                            for (int d = 0; d < headDim; d++)
                            {
                                dqRow[d] += dDot * K[j, offset + d];
                                dKFull[j, offset + d] += dDot * Q[i, offset + d];
                            }
                        }
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        dQFull[i, offset + d] += dqRow[d];
                    }
                }
            });

            return (dQFull, dKFull, dVFull);
        }

        private float[,] ContentAwareCrossAttentionForwardCoreParallel(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, float[,] timeDiffs, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            int queryLen = Q.GetLength(0);
            int keyLen = K.GetLength(0);
            int embDim = Q.GetLength(1);
            int headDim = embDim / numHeads;

            var output = new float[queryLen, embDim];
            attentionWeights = new float[numHeads][,];
            scoresPreSoftmax = new float[numHeads][,];

            var localWeights = attentionWeights;
            var localScores = scoresPreSoftmax;

            Parallel.For(0, numHeads, _parallelOptions, h =>
            {
                int offset = h * headDim;
                var scores = new float[queryLen, keyLen];
                var weights = new float[queryLen, keyLen];

                for (int q = 0; q < queryLen; q++)
                {
                    float max = float.NegativeInfinity;

                    for (int s = 0; s < keyLen; s++)
                    {
                        if (timeDiffs != null && timeDiffs[q, s] < 0f)
                        {
                            scores[q, s] = float.NegativeInfinity;
                            continue;
                        }

                        float score = DotHeadRows(Q, q, K, s, offset, headDim) * scale;
                        if (decayBias != null)
                        {
                            score += decayBias[q, s, h];
                        }

                        scores[q, s] = score;
                        if (!float.IsNegativeInfinity(score) && score > max)
                        {
                            max = score;
                        }
                    }

                    if (float.IsNegativeInfinity(max))
                    {
                        continue;
                    }

                    float sum = 0f;
                    for (int s = 0; s < keyLen; s++)
                    {
                        float score = scores[q, s];
                        if (float.IsNegativeInfinity(score))
                        {
                            weights[q, s] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(score - max);
                        weights[q, s] = w;
                        sum += w;
                    }

                    if (sum > 0f)
                    {
                        float inv = 1f / sum;
                        for (int s = 0; s < keyLen; s++)
                        {
                            weights[q, s] *= inv;
                        }
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        float value = 0f;
                        for (int s = 0; s < keyLen; s++)
                        {
                            float w = weights[q, s];
                            if (w != 0f)
                            {
                                value += w * V[s, offset + d];
                            }
                        }
                        output[q, offset + d] = value;
                    }
                }

                localWeights[h] = weights;
                localScores[h] = scores;
            });

            return output;
        }

        private static float GetMmtacContextGrad(float[,] dContextA, float[,] dContextB, int row, int col)
        {
            float g = 0f;
            if (dContextA != null && row < dContextA.GetLength(0))
            {
                g += dContextA[row, col];
            }
            if (dContextB != null && row < dContextB.GetLength(0))
            {
                g += dContextB[row, col];
            }
            return g;
        }

        private void ApplyRotaryPositionEmbeddingAllHeadsParallelInPlace(float[,] matrix, int numHeads, int headDim, float baseTheta, bool inverse)
        {
            int seqLen = matrix.GetLength(0);
            int pairCount = headDim / 2;
            if (seqLen == 0 || pairCount == 0)
            {
                return;
            }

            var denom = BuildRotaryDenominators(pairCount, headDim, baseTheta);
            long iterationsLong = (long)numHeads * seqLen;
            if (iterationsLong > int.MaxValue)
            {
                SingleThreadCPU.ApplyRotaryPositionEmbeddingInPlace(matrix, numHeads, baseTheta, inverse);
                return;
            }

            int iterations = (int)iterationsLong;
            Parallel.For(0, iterations, _parallelOptions, flat =>
            {
                int head = flat / seqLen;
                int pos = flat - head * seqLen;
                int startCol = head * headDim;
                ApplyRotaryPositionEmbeddingPositionInPlace(matrix, pos, startCol, headDim, denom, inverse);
            });
        }

        private void ApplyRotaryPositionEmbeddingSingleHeadParallelInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            int seqLen = matrix.GetLength(0);
            int pairCount = headDim / 2;
            if (seqLen == 0 || pairCount == 0)
            {
                return;
            }

            var denom = BuildRotaryDenominators(pairCount, headDim, baseTheta);
            Parallel.For(0, seqLen, _parallelOptions, pos =>
            {
                ApplyRotaryPositionEmbeddingPositionInPlace(matrix, pos, startCol, headDim, denom, inverse);
            });
        }

        private static float[] BuildRotaryDenominators(int pairCount, int headDim, float baseTheta)
        {
            var denom = new float[pairCount];
            for (int pair = 0; pair < pairCount; pair++)
            {
                denom[pair] = MathF.Pow(baseTheta, (2f * pair) / headDim);
            }
            return denom;
        }

        private static void ApplyRotaryPositionEmbeddingPositionInPlace(float[,] matrix, int pos, int startCol, int headDim, float[] denom, bool inverse)
        {
            int pairCount = headDim / 2;
            for (int pair = 0; pair < pairCount; pair++)
            {
                int evenCol = startCol + (pair * 2);
                int oddCol = evenCol + 1;

                float theta = pos / denom[pair];
                float cos = MathF.Cos(theta);
                float sin = MathF.Sin(theta);
                if (inverse)
                {
                    sin = -sin;
                }

                float x0 = matrix[pos, evenCol];
                float x1 = matrix[pos, oddCol];

                matrix[pos, evenCol] = (x0 * cos) - (x1 * sin);
                matrix[pos, oddCol] = (x0 * sin) + (x1 * cos);
            }
        }

    
        public float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            if (forwardPassFn == null)
            {
                throw new ArgumentNullException(nameof(forwardPassFn));
            }

            if (seqLen < 0 || seqLen > input.GetLength(0))
            {
                throw new ArgumentOutOfRangeException(nameof(seqLen));
            }

            if (forwardPassFn.Target is CallaghanDev.ML.NeuralNetwork network)
            {
                float[,] batchInput = seqLen == input.GetLength(0)
                    ? input
                    : SliceRows(input, 0, seqLen);

                return network.ForwardPassOnlyBatch(batchInput);
            }

            // Unknown delegates may mutate captured state, so keep the conservative
            // single-threaded fallback for non-NeuralNetwork call sites. MMTAC and
            // transformer FFNs use the batched branch above.
            return SingleThreadCPU.FFNForwardBatch(input, seqLen, outputDim, forwardPassFn);
        }

  
        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network, bool isTraining = false, Random dropoutRng = null)
        {
            // Training needs the full ContentAwareDecayCache for Backward(...), so keep
            // the exact reference implementation there. Validation/inference only needs
            // the decay bias; use a cache-free parallel path that avoids allocating the
            // large backprop tensors.
            if (isTraining)
            {
                return SingleThreadCPU.ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, network, isTraining, dropoutRng);
            }

            return (ContentAwareDecayForwardInferenceOnly(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, network), null);
        }

        private float[,,] ContentAwareDecayForwardInferenceOnly(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network)
        {
            if (queryEmbeddings == null) throw new ArgumentNullException(nameof(queryEmbeddings));
            if (keyEmbeddings == null) throw new ArgumentNullException(nameof(keyEmbeddings));
            if (timeDiffs == null) throw new ArgumentNullException(nameof(timeDiffs));
            if (network == null) throw new ArgumentNullException(nameof(network));

            int queryLen = timeDiffs.GetLength(0);
            int keyLen = timeDiffs.GetLength(1);
            int numHeads = network.NumHeads;
            int projDim = network.ProjectionDim;
            int contentDim = network.ContentDim;
            int hiddenDim = network.HiddenDim;
            int mlpInputDim = network.MLPInputDim;
            int numBases = network.NumTimeBases;
            int rawDim = network.TimeRawDim;

            if (queryEmbeddings.GetLength(0) != queryLen || queryEmbeddings.GetLength(1) != contentDim)
            {
                throw new ArgumentException("queryEmbeddings shape does not match the decay network.", nameof(queryEmbeddings));
            }

            if (keyEmbeddings.GetLength(0) != keyLen || keyEmbeddings.GetLength(1) != contentDim)
            {
                throw new ArgumentException("keyEmbeddings shape does not match the decay network.", nameof(keyEmbeddings));
            }

            if (keyTimesFromRef != null && keyTimesFromRef.Length != keyLen)
            {
                throw new ArgumentException("keyTimesFromRef length must match key length.", nameof(keyTimesFromRef));
            }

            var decayBias = new float[queryLen, keyLen, numHeads];
            float timeNorm = MathF.Max(network.TimeNormalizationHours, 1e-4f);

            Parallel.For(0, numHeads, _parallelOptions, h =>
            {
                var queryProj = new float[queryLen, projDim];
                var keyProj = new float[keyLen, projDim];
                var timeEncoding = new float[keyLen, projDim];
                var refinedKey = new float[keyLen, projDim];
                var memOutput = new float[keyLen, projDim];
                var scores = new float[keyLen];
                var mlpInput = new float[mlpInputDim];

                for (int q = 0; q < queryLen; q++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.QueryProjectionBias[h, p];
                        for (int d = 0; d < contentDim; d++)
                        {
                            val += network.QueryProjection[h, p, d] * queryEmbeddings[q, d];
                        }
                        queryProj[q, p] = val;
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.KeyProjectionBias[h, p];
                        for (int d = 0; d < contentDim; d++)
                        {
                            val += network.KeyProjection[h, p, d] * keyEmbeddings[s, d];
                        }
                        keyProj[s, p] = val;
                    }

                    float t = keyTimesFromRef != null ? keyTimesFromRef[s] : 0f;

                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.TimeProjBias[h, p];
                        for (int b = 0; b < numBases; b++)
                        {
                            float freq = MathF.Exp(network.TimeLogFreq[h, b]);
                            float angle = freq * t;
                            float sin = MathF.Sin(angle);
                            float cos = MathF.Cos(angle);
                            int raw = b * 2;

                            if (raw < rawDim)
                            {
                                val += network.TimeProj[h, p, raw] * sin;
                            }

                            if (raw + 1 < rawDim)
                            {
                                val += network.TimeProj[h, p, raw + 1] * cos;
                            }
                        }
                        timeEncoding[s, p] = val;
                    }
                }

                float memScale = 1.0f / MathF.Sqrt(projDim);

                for (int i = 0; i < keyLen; i++)
                {
                    float maxScore = float.NegativeInfinity;

                    for (int j = 0; j < keyLen; j++)
                    {
                        bool valid = keyTimesFromRef == null || keyTimesFromRef[j] <= keyTimesFromRef[i];

                        if (!valid)
                        {
                            scores[j] = float.NegativeInfinity;
                            continue;
                        }

                        float dot = 0f;
                        for (int p = 0; p < projDim; p++)
                        {
                            float qi = keyProj[i, p] + timeEncoding[i, p];
                            float kj = keyProj[j, p] + timeEncoding[j, p];
                            dot += qi * kj;
                        }

                        float score = dot * memScale;
                        scores[j] = score;
                        if (score > maxScore)
                        {
                            maxScore = score;
                        }
                    }

                    if (float.IsNegativeInfinity(maxScore))
                    {
                        continue;
                    }

                    float sumExp = 0f;
                    for (int j = 0; j < keyLen; j++)
                    {
                        if (float.IsNegativeInfinity(scores[j]))
                        {
                            scores[j] = 0f;
                            continue;
                        }

                        float w = MathF.Exp(scores[j] - maxScore);
                        scores[j] = w;
                        sumExp += w;
                    }

                    if (sumExp <= 0f)
                    {
                        continue;
                    }

                    float inv = 1f / sumExp;
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = 0f;
                        for (int j = 0; j < keyLen; j++)
                        {
                            val += (scores[j] * inv) * keyProj[j, p];
                        }
                        memOutput[i, p] = val;
                    }
                }

                for (int s = 0; s < keyLen; s++)
                {
                    for (int p = 0; p < projDim; p++)
                    {
                        float val = network.MemAttnOutputB[h, p];
                        for (int q = 0; q < projDim; q++)
                        {
                            val += network.MemAttnOutputW[h, p, q] * memOutput[s, q];
                        }
                        refinedKey[s, p] = val + keyProj[s, p];
                    }
                }

                float baseRate = MathF.Exp(network.LogBaseDecayRate[h]);

                for (int qi = 0; qi < queryLen; qi++)
                {
                    for (int si = 0; si < keyLen; si++)
                    {
                        float td = timeDiffs[qi, si];

                        if (td < 0f)
                        {
                            decayBias[qi, si, h] = float.NegativeInfinity;
                            continue;
                        }

                        float normTd = td > 0f ? td / timeNorm : 0f;
                        float logTd = MathF.Log(1f + normTd);

                        int inputIndex = 0;
                        for (int p = 0; p < projDim; p++)
                        {
                            mlpInput[inputIndex++] = queryProj[qi, p];
                        }
                        for (int p = 0; p < projDim; p++)
                        {
                            mlpInput[inputIndex++] = refinedKey[si, p];
                        }
                        for (int p = 0; p < projDim; p++)
                        {
                            mlpInput[inputIndex++] = queryProj[qi, p] * refinedKey[si, p];
                        }
                        mlpInput[inputIndex++] = normTd;
                        mlpInput[inputIndex++] = logTd;

                        float logit = network.B2[h];

                        for (int j = 0; j < hiddenDim; j++)
                        {
                            float hidden = network.B1[h, j];
                            for (int k = 0; k < mlpInputDim; k++)
                            {
                                hidden += network.W1[h, j, k] * mlpInput[k];
                            }

                            float activated = hidden > 0f ? hidden : 0.01f * hidden;
                            logit += network.W2[h, j] * activated;
                        }

                        float gate = network.ClampGate(StableSigmoid(logit));
                        decayBias[qi, si, h] = -(baseRate * (1f - gate)) * normTd;
                    }
                }
            });

            return decayBias;
        }

        public string[] PreTokenize(string text)
        {
            return SingleThreadCPU.PreTokenize(text);
        }

        public Dictionary<string, int> GetWordFrequencies(string[] texts, bool lowerCase)
        {
            return SingleThreadCPU.GetWordFrequencies(texts, lowerCase);
        }

        public HashSet<string> BuildCharacterVocabulary(Dictionary<string, int> wordFreqs)
        {
            return SingleThreadCPU.BuildCharacterVocabulary(wordFreqs);
        }

        public List<string> ApplyMerge(List<string> word, string left, string right)
        {
            return SingleThreadCPU.ApplyMerge(word, left, right);
        }

        public List<int> EncodeWord(string word, Dictionary<(string, string), int> mergePriority, Dictionary<string, int> vocabToId, int unkTokenId)
        {
            return SingleThreadCPU.EncodeWord(word, mergePriority, vocabToId, unkTokenId);
        }

        public Dictionary<(string left, string right), int> CountPairFrequencies(Dictionary<List<string>, int> words)
        {
            return SingleThreadCPU.CountPairFrequencies(words);
        }

        public ((string left, string right) pair, int frequency) SelectBestPair(Dictionary<(string left, string right), int> pairCounts, int minFrequency)
        {
            return SingleThreadCPU.SelectBestPair(pairCounts, minFrequency);
        }

        public Dictionary<List<string>, int> ApplyMergeToVocabulary(Dictionary<List<string>, int> words, string left, string right)
        {
            return SingleThreadCPU.ApplyMergeToVocabulary(words, left, right);
        }

        public string DecodeTokens(int[] tokenIds, Dictionary<int, string> idToVocab, string unkToken, bool skipSpecialTokens)
        {
            return SingleThreadCPU.DecodeTokens(tokenIds, idToVocab, unkToken, skipSpecialTokens);
        }

        public int[] PadOrTruncate(int[] tokenIds, int maxLength, bool addSpecialTokens, int padTokenId, int endTokenId)
        {
            return SingleThreadCPU.PadOrTruncate(tokenIds, maxLength, addSpecialTokens, padTokenId, endTokenId);
        }

        #endregion


        #region Validation/helpers for safe parallel kernels

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
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

        private static void ValidateProjectQKVInputs(
            float[,] input,
            float[,] WQ,
            float[] biasQ,
            float[,] WK,
            float[] biasK,
            float[,] WV,
            float[] biasV,
            out int rows,
            out int inputDim,
            out int outputDim)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (WQ == null) throw new ArgumentNullException(nameof(WQ));
            if (WK == null) throw new ArgumentNullException(nameof(WK));
            if (WV == null) throw new ArgumentNullException(nameof(WV));
            if (biasQ == null) throw new ArgumentNullException(nameof(biasQ));
            if (biasK == null) throw new ArgumentNullException(nameof(biasK));
            if (biasV == null) throw new ArgumentNullException(nameof(biasV));

            rows = input.GetLength(0);
            inputDim = input.GetLength(1);
            int qDim = WQ.GetLength(0);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);

            if (WQ.GetLength(1) != inputDim) throw new ArgumentException("WQ input dimension does not match input width.", nameof(WQ));
            if (WK.GetLength(1) != inputDim) throw new ArgumentException("WK input dimension does not match input width.", nameof(WK));
            if (WV.GetLength(1) != inputDim) throw new ArgumentException("WV input dimension does not match input width.", nameof(WV));
            if (biasQ.Length != qDim) throw new ArgumentException("biasQ length does not match WQ output dimension.", nameof(biasQ));
            if (biasK.Length != kDim) throw new ArgumentException("biasK length does not match WK output dimension.", nameof(biasK));
            if (biasV.Length != vDim) throw new ArgumentException("biasV length does not match WV output dimension.", nameof(biasV));
            if (qDim != kDim || qDim != vDim) throw new ArgumentException("Q, K and V output dimensions must match.");
            outputDim = qDim;
        }

        private static void ValidateProjectKVInputs(
            float[,] input,
            float[,] WK,
            float[] biasK,
            float[,] WV,
            float[] biasV,
            out int rows,
            out int inputDim,
            out int outputDim)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (WK == null) throw new ArgumentNullException(nameof(WK));
            if (WV == null) throw new ArgumentNullException(nameof(WV));
            if (biasK == null) throw new ArgumentNullException(nameof(biasK));
            if (biasV == null) throw new ArgumentNullException(nameof(biasV));

            rows = input.GetLength(0);
            inputDim = input.GetLength(1);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);

            if (kDim != vDim) throw new ArgumentException("K and V output dimensions must match.");
            if (WK.GetLength(1) != inputDim) throw new ArgumentException("WK input dimension does not match input width.", nameof(WK));
            if (WV.GetLength(1) != inputDim) throw new ArgumentException("WV input dimension does not match input width.", nameof(WV));
            if (biasK.Length != kDim) throw new ArgumentException("biasK length does not match WK output dimension.", nameof(biasK));
            if (biasV.Length != vDim) throw new ArgumentException("biasV length does not match WV output dimension.", nameof(biasV));
            outputDim = kDim;
        }

        private static void ValidateBackpropQKVInputs(
            float[,] input,
            float[,] dQ,
            float[,] dK,
            float[,] dV,
            float[,] WQ,
            float[,] WK,
            float[,] WV,
            float[,] WQGrad,
            float[] biasQGrad,
            float[,] WKGrad,
            float[] biasKGrad,
            float[,] WVGrad,
            float[] biasVGrad,
            out int rows,
            out int inputDim,
            out int outputDim)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (dQ == null) throw new ArgumentNullException(nameof(dQ));
            if (dK == null) throw new ArgumentNullException(nameof(dK));
            if (dV == null) throw new ArgumentNullException(nameof(dV));
            if (WQ == null) throw new ArgumentNullException(nameof(WQ));
            if (WK == null) throw new ArgumentNullException(nameof(WK));
            if (WV == null) throw new ArgumentNullException(nameof(WV));
            if (WQGrad == null) throw new ArgumentNullException(nameof(WQGrad));
            if (WKGrad == null) throw new ArgumentNullException(nameof(WKGrad));
            if (WVGrad == null) throw new ArgumentNullException(nameof(WVGrad));
            if (biasQGrad == null) throw new ArgumentNullException(nameof(biasQGrad));
            if (biasKGrad == null) throw new ArgumentNullException(nameof(biasKGrad));
            if (biasVGrad == null) throw new ArgumentNullException(nameof(biasVGrad));

            rows = input.GetLength(0);
            inputDim = input.GetLength(1);
            int qDim = WQ.GetLength(0);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);
            if (qDim != kDim || qDim != vDim) throw new ArgumentException("Q, K and V dimensions must match.");
            outputDim = qDim;
            if (WQ.GetLength(1) != inputDim || WK.GetLength(1) != inputDim || WV.GetLength(1) != inputDim) throw new ArgumentException("Q/K/V weight input dimensions must match input width.");
            if (dQ.GetLength(0) != rows || dK.GetLength(0) != rows || dV.GetLength(0) != rows) throw new ArgumentException("dQ, dK and dV row counts must match input row count.");
            if (dQ.GetLength(1) != outputDim || dK.GetLength(1) != outputDim || dV.GetLength(1) != outputDim) throw new ArgumentException("dQ, dK and dV widths must match Q/K/V output dimension.");
            if (WQGrad.GetLength(0) != outputDim || WQGrad.GetLength(1) != inputDim) throw new ArgumentException("WQGrad shape mismatch.", nameof(WQGrad));
            if (WKGrad.GetLength(0) != outputDim || WKGrad.GetLength(1) != inputDim) throw new ArgumentException("WKGrad shape mismatch.", nameof(WKGrad));
            if (WVGrad.GetLength(0) != outputDim || WVGrad.GetLength(1) != inputDim) throw new ArgumentException("WVGrad shape mismatch.", nameof(WVGrad));
            if (biasQGrad.Length != outputDim || biasKGrad.Length != outputDim || biasVGrad.Length != outputDim) throw new ArgumentException("Q/K/V bias gradient lengths must match output dimension.");
        }

        private static void ValidateBackpropKVInputs(
            float[,] input,
            float[,] dK,
            float[,] dV,
            float[,] WK,
            float[,] WV,
            float[,] WKGrad,
            float[] biasKGrad,
            float[,] WVGrad,
            float[] biasVGrad,
            out int rows,
            out int inputDim,
            out int outputDim)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (dK == null) throw new ArgumentNullException(nameof(dK));
            if (dV == null) throw new ArgumentNullException(nameof(dV));
            if (WK == null) throw new ArgumentNullException(nameof(WK));
            if (WV == null) throw new ArgumentNullException(nameof(WV));
            if (WKGrad == null) throw new ArgumentNullException(nameof(WKGrad));
            if (WVGrad == null) throw new ArgumentNullException(nameof(WVGrad));
            if (biasKGrad == null) throw new ArgumentNullException(nameof(biasKGrad));
            if (biasVGrad == null) throw new ArgumentNullException(nameof(biasVGrad));

            rows = input.GetLength(0);
            inputDim = input.GetLength(1);
            int kDim = WK.GetLength(0);
            int vDim = WV.GetLength(0);
            if (kDim != vDim) throw new ArgumentException("K and V dimensions must match.");
            outputDim = kDim;
            if (WK.GetLength(1) != inputDim || WV.GetLength(1) != inputDim) throw new ArgumentException("K/V weight input dimensions must match input width.");
            if (dK.GetLength(0) != rows || dV.GetLength(0) != rows) throw new ArgumentException("dK and dV row counts must match input row count.");
            if (dK.GetLength(1) != outputDim || dV.GetLength(1) != outputDim) throw new ArgumentException("dK and dV widths must match K/V output dimension.");
            if (WKGrad.GetLength(0) != outputDim || WKGrad.GetLength(1) != inputDim) throw new ArgumentException("WKGrad shape mismatch.", nameof(WKGrad));
            if (WVGrad.GetLength(0) != outputDim || WVGrad.GetLength(1) != inputDim) throw new ArgumentException("WVGrad shape mismatch.", nameof(WVGrad));
            if (biasKGrad.Length != outputDim || biasVGrad.Length != outputDim) throw new ArgumentException("K/V bias gradient lengths must match output dimension.");
        }

        #endregion

        public void Dispose()
        {
            SingleThreadCPU.Dispose();
        }
    }
}
