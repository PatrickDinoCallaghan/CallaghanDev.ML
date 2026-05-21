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

        private const bool AlwaysParallel = true;


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
            SingleThreadCPU.ZeroMatrix(matrix);
        }

        public void ZeroVector(float[] vector)
        {
            SingleThreadCPU.ZeroVector(vector);
        }

        public (float[] activation, float[] derivative) ActivateLayer(float[] dot, float[] bias, ActivationType activationType)
        {
            return SingleThreadCPU.ActivateLayer(dot, bias, activationType);
        }

        public float[,] Softmax(float[,] matrix, bool[,] mask = null)
        {
            return SingleThreadCPU.Softmax(matrix, mask);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            return SingleThreadCPU.MultiHeadAttentionBackward_Obsolete(Q, K, V, dConcatenated, numHeads, scale, useDecoderMask);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward_Obsolete(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            return SingleThreadCPU.MultiHeadAttentionBackward_Obsolete(Q, K, V, dConcatenated, numHeads, scale, mask);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            return SingleThreadCPU.MultiHeadAttentionBackward(Q, K, V, dConcatenated, numHeads, scale, useDecoderMask);
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            return SingleThreadCPU.MultiHeadAttentionBackward(Q, K, V, dConcatenated, numHeads, scale, mask);
        }

        public float[,] LayerNorm(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            return SingleThreadCPU.LayerNorm(input, gamma, beta, epsilon);
        }

        public (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            return SingleThreadCPU.LayerNormForward(input, gamma, beta, epsilon);
        }

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            return SingleThreadCPU.LayerNormBackward(dOut, normalized, gamma, input, mean, variance, epsilon);
        }

        public float[,] MultiHeadAttentionForward_Obsolete(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            return SingleThreadCPU.MultiHeadAttentionForward_Obsolete(Q, K, V, numHeads, scale, mask);
        }

        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            return SingleThreadCPU.MultiHeadAttentionForward(Q, K, V, numHeads, scale, mask);
        }

        public float[,] ScaledDotProductAttention(float[,] q, float[,] k, float[,] v, int numHeads, bool[,] mask = null, bool causal = false)
        {
            return SingleThreadCPU.ScaledDotProductAttention(q, k, v, numHeads, mask, causal);
        }

        public float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn)
        {
            return SingleThreadCPU.FFNForwardBatch(input, seqLen, outputDim, forwardPassFn);
        }

        public void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim)
        {
            SingleThreadCPU.AccumulateTokenEmbeddingGrad(embeddingGrad, dX, tokenIds, seqLen, embeddingDim);
        }

        public (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen)
        {
            return SingleThreadCPU.CrossEntropyLossAndGradient(logits, targets, effectiveLen);
        }

        public (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen)
        {
            return SingleThreadCPU.MSELossAndGradient(predictions, targets, effectiveLen);
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            return SingleThreadCPU.MatrixSquaredNorm(matrix);
        }

        public float VectorSquaredNorm(float[] vector)
        {
            return SingleThreadCPU.VectorSquaredNorm(vector);
        }

        public void ApplyContextTypeEmbedding(float[,] contextHidden, float[,] typeEmbedding, int[] typeIndices)
        {
            SingleThreadCPU.ApplyContextTypeEmbedding(contextHidden, typeEmbedding, typeIndices);
        }

        public float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes)
        {
            return SingleThreadCPU.ComputeTimeDiffMatrix(priceSeqLen, keyArrivalTimes);
        }

        public float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos, float[,] contextHidden, int totalCtx, float scale)
        {
            return SingleThreadCPU.ComputeMemoryAttentionScores(priceHidden, lastPos, contextHidden, totalCtx, scale);
        }

        public float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim)
        {
            return SingleThreadCPU.ProjectOutputBatch(hidden, outputProjection, outputBias, seqLen, outputDim);
        }

        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network, bool isTraining = false, Random dropoutRng = null)
        {
            return SingleThreadCPU.ContentAwareDecayForward(queryEmbeddings, keyEmbeddings, timeDiffs, keyTimesFromRef, network, isTraining, dropoutRng);
        }

        public float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            return SingleThreadCPU.ContentAwareCrossAttentionForward(Q, K, V, numHeads, scale, decayBias, out attentionWeights, out scoresPreSoftmax);
        }

        public float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, int PriceEmbeddingDim, int PriceNumHeads, bool enableDecayBias = true, bool isTraining = false, Random dropoutRng = null)
        {
            return SingleThreadCPU.ContentAwareCrossAttentionWithCache(Q, K, V, timeDiffs, keyTimesFromRef, queryEmbeddings, keyEmbeddings, block, bc, PriceEmbeddingDim, PriceNumHeads, enableDecayBias, isTraining, dropoutRng);
        }

        public float MatrixSquaredNorm3D(float[,,] matrix)
        {
            return SingleThreadCPU.MatrixSquaredNorm3D(matrix);
        }

        public float[] ProjectGlobalFeatures(float[] globalFeatures, float[,] projection, float[] bias)
        {
            return SingleThreadCPU.ProjectGlobalFeatures(globalFeatures, projection, bias);
        }

        public float[,] EmbedTokenIds(int[] tokenIds, float[,] embedding, int embeddingDim)
        {
            return SingleThreadCPU.EmbedTokenIds(tokenIds, embedding, embeddingDim);
        }

        public float[] MeanPoolRows(float[,] matrix)
        {
            return SingleThreadCPU.MeanPoolRows(matrix);
        }

        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews) BuildMmtacContext(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] contextTypeEmbedding)
        {
            return SingleThreadCPU.BuildMmtacContext(newsHidden, newsTimes, globalToken, contextTypeEmbedding);
        }

        public (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] regressionLogits, float[] rangeLogits, float[] qualityLogits) ProjectMmtacOutputHeads(float[,] hidden, float[,] regressionProjection, float[] regressionBias, float[,] rangeProjection, float[] rangeBias, float[,] qualityProjection, float[] qualityBias, float[,] directionProjection, float[] directionBias, float[,] midDirectionProjection, float[] midDirectionBias, float[,] confidenceProjection, float[] confidenceBias, bool useConfidenceHead)
        {
            return SingleThreadCPU.ProjectMmtacOutputHeads(hidden, regressionProjection, regressionBias, rangeProjection, rangeBias, qualityProjection, qualityBias, directionProjection, directionBias, midDirectionProjection, midDirectionBias, confidenceProjection, confidenceBias, useConfidenceHead);
        }

        public float[] SoftmaxVector(float[] scores)
        {
            return SingleThreadCPU.SoftmaxVector(scores);
        }

        public (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttention(float[,] q, float[,] k, float[,] v, float[,] dOutput, float[][,] attentionWeights, float[,] timeDiffs, int embeddingDim, int numHeads)
        {
            return SingleThreadCPU.BackpropTimeDecayedAttention(q, k, v, dOutput, attentionWeights, timeDiffs, embeddingDim, numHeads);
        }

        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews, int numPrice) BuildMmtacContextWithPrice(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] priceContextHidden, float[] priceContextTimes, float[,] contextTypeEmbedding)
        {
            return SingleThreadCPU.BuildMmtacContextWithPrice(newsHidden, newsTimes, globalToken, priceContextHidden, priceContextTimes, contextTypeEmbedding);
        }

        public (float loss, float[,] dHidden) BackpropMmtacOutputHeads(float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] targetRegression, float[,] targetRange, float[,] targetQuality, float[,] targetDirection, float[,] targetMidDirection, float[] previousClose, float[] confidenceTargets, float[,] hidden, float[,] regressionLogits, float[] rangeLogits, float[,] regressionProjection, float[,] rangeProjection, float[,] qualityProjection, float[,] directionProjection, float[,] midDirectionProjection, float[,] confidenceProjection, float[,] regressionProjectionGrad, float[] regressionBiasGrad, float[,] rangeProjectionGrad, float[] rangeBiasGrad, float[,] qualityProjectionGrad, float[] qualityBiasGrad, float[,] directionProjectionGrad, float[] directionBiasGrad, float[,] midDirectionProjectionGrad, float[] midDirectionBiasGrad, float[,] confidenceProjectionGrad, float[] confidenceBiasGrad, float rangeLossWeight, float qualityLossWeight, float directionLossWeight, float midDirectionLossWeight, float closeDirectionConsistencyWeight, float closeDirectionConsistencyMargin, float confidenceLossWeight, bool useConfidenceHead)
        {
            return SingleThreadCPU.BackpropMmtacOutputHeads(regression, range, quality, direction, midDirection, confidence, targetRegression, targetRange, targetQuality, targetDirection, targetMidDirection, previousClose, confidenceTargets, hidden, regressionLogits, rangeLogits, regressionProjection, rangeProjection, qualityProjection, directionProjection, midDirectionProjection, confidenceProjection, regressionProjectionGrad, regressionBiasGrad, rangeProjectionGrad, rangeBiasGrad, qualityProjectionGrad, qualityBiasGrad, directionProjectionGrad, directionBiasGrad, midDirectionProjectionGrad, midDirectionBiasGrad, confidenceProjectionGrad, confidenceBiasGrad, rangeLossWeight, qualityLossWeight, directionLossWeight, midDirectionLossWeight, closeDirectionConsistencyWeight, closeDirectionConsistencyMargin, confidenceLossWeight, useConfidenceHead);
        }

        public void AccumulateMmtacContextGradients(float[,] dContextA, float[,] dContextB, float[,] contextTypeEmbeddingGrad, float[,] dLiveNewsHidden, float[] dGlobalHidden, int numGlobal, int numStoredNews, int numNews, int numLiveNews, int numPriceContext, int totalContext, int priceOffset)
        {
            SingleThreadCPU.AccumulateMmtacContextGradients(dContextA, dContextB, contextTypeEmbeddingGrad, dLiveNewsHidden, dGlobalHidden, numGlobal, numStoredNews, numNews, numLiveNews, numPriceContext, totalContext, priceOffset);
        }

        public void AccumulateGlobalProjectionGradients(float[] dGlobalHidden, float[] globalFeatures, float[,] projectionGrad, float[] biasGrad)
        {
            SingleThreadCPU.AccumulateGlobalProjectionGradients(dGlobalHidden, globalFeatures, projectionGrad, biasGrad);
        }

        public float[,] ExpandMeanPoolGradient(float[,] pooledGradient, int rowIndex, int rowCount, int embeddingDim)
        {
            return SingleThreadCPU.ExpandMeanPoolGradient(pooledGradient, rowIndex, rowCount, embeddingDim);
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

        public void ApplyRotaryPositionEmbeddingInPlace(float[,] matrix, int numHeads, float baseTheta, bool inverse)
        {
            SingleThreadCPU.ApplyRotaryPositionEmbeddingInPlace(matrix, numHeads, baseTheta, inverse);
        }

        public void ApplyRotaryPositionEmbeddingHeadInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            SingleThreadCPU.ApplyRotaryPositionEmbeddingHeadInPlace(matrix, startCol, headDim, baseTheta, inverse);
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
            return SingleThreadCPU.ContentAwareCrossAttentionWithCache(
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
            SingleThreadCPU.ApplyRotaryPositionEmbeddingInPlace(matrix, numHeads, 10000f, inverse: false);
        }

        public void ApplyRotaryPositionEmbeddingBackwardInPlace(float[,] matrix, int numHeads)
        {
            SingleThreadCPU.ApplyRotaryPositionEmbeddingInPlace(matrix, numHeads, 10000f, inverse: true);
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
