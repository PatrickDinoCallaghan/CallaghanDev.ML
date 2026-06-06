using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private readonly Dictionary<(int r1, int c1, int c2), (MemoryBuffer2D<float, Stride2D.DenseX> a, MemoryBuffer2D<float, Stride2D.DenseX> b, MemoryBuffer2D<float, Stride2D.DenseX> c)> _matMulCache = new();
        private readonly Dictionary<(int r1, int c1, int r2), (MemoryBuffer2D<float, Stride2D.DenseX> a, MemoryBuffer2D<float, Stride2D.DenseX> b, MemoryBuffer2D<float, Stride2D.DenseX> c)> _matMulTransposeCache = new();
        private readonly Dictionary<(int rows, int cols), (MemoryBuffer2D<float, Stride2D.DenseX> a, MemoryBuffer2D<float, Stride2D.DenseX> b, MemoryBuffer2D<float, Stride2D.DenseX> c)> _matrixAddCache = new();
        private readonly Dictionary<(int rows, int cols), (MemoryBuffer2D<float, Stride2D.DenseX> input, MemoryBuffer1D<float, Stride1D.Dense> bias, MemoryBuffer2D<float, Stride2D.DenseX> output)> _matrixAddBiasCache = new();
        private readonly Dictionary<(int rows, int cols, int rowCount), (MemoryBuffer2D<float, Stride2D.DenseX> input, MemoryBuffer2D<float, Stride2D.DenseX> output)> _sliceRowsCache = new();
        private readonly Dictionary<(int outputDim, int inputDim, int inputRows, int rowCount), (MemoryBuffer2D<float, Stride2D.DenseX> w, MemoryBuffer2D<float, Stride2D.DenseX> inp, MemoryBuffer2D<float, Stride2D.DenseX> res)> _batchDotCache = new();
        private readonly Dictionary<(int outputDim, int inputDim, int inputRows, int rowCount), (MemoryBuffer2D<float, Stride2D.DenseX> w, MemoryBuffer2D<float, Stride2D.DenseX> inp, MemoryBuffer1D<float, Stride1D.Dense> bias, MemoryBuffer2D<float, Stride2D.DenseX> res)> _batchDotAddBiasCache = new();
        private readonly Dictionary<(int rows, int cols), (MemoryBuffer2D<float, Stride2D.DenseX> input, MemoryBuffer2D<float, Stride2D.DenseX> subLayer, MemoryBuffer1D<float, Stride1D.Dense> gamma, MemoryBuffer1D<float, Stride1D.Dense> beta, MemoryBuffer2D<float, Stride2D.DenseX> output)> _residualLayerNormCache = new();
        private readonly Dictionary<(int rows, int cols), (MemoryBuffer2D<float, Stride2D.DenseX> input, MemoryBuffer2D<float, Stride2D.DenseX> subLayer, MemoryBuffer1D<float, Stride1D.Dense> gamma, MemoryBuffer1D<float, Stride1D.Dense> beta, MemoryBuffer2D<float, Stride2D.DenseX> output, MemoryBuffer1D<float, Stride1D.Dense> means, MemoryBuffer1D<float, Stride1D.Dense> variances, MemoryBuffer2D<float, Stride2D.DenseX> normalized, MemoryBuffer2D<float, Stride2D.DenseX> residual)> _residualLayerNormForwardCache = new();

        private void InitSharedTensorKernels()
        {
            //Matrix Multiply
            _matMulKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatMulKernel);
            //Matrix Multiply Transpose
            _matMulTransposeKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatMulTransposeKernel);
            //Matrix Scale
            _matScaleKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, float>(MatScaleKernel);
            //Matrix Add
            _matAddKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatAddKernel);
            //Matrix Add Bias
            _matAddBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>>(MatAddBiasKernel);
            //BatchDotProduct
            _batchDotKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, ArrayView2D<float, Stride2D.DenseX>>(BatchDotKernel);
            _batchDotAddBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int, ArrayView2D<float, Stride2D.DenseX>>(BatchDotAddBiasKernel);

            //SliceRows, ExtractRow and SetRow
            _sliceRowsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int>(SliceRowsKernel);
            _extractRowKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int>(ExtractRowKernel);
            _setRowKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int>(SetRowKernel);

            //ZeroMatrix / ZeroVector
            _zeroMatrixKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(ZeroMatrixKernel);
            _zeroMatrixColumnsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(ZeroMatrixColumnsKernel);
            _zeroVectorKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(ZeroVectorKernel);

            //MatrixAddInPlace
            _matAccumulateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(MatAccumulateKernel);


            _vecAccumulateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(VecAccumulateKernel);
        }

        #region MatrixMultiply

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matMulKernel;
        public float[,] MatrixMultiply(float[,] A, float[,] B)
        {
            int r1 = A.GetLength(0);
            int c1 = A.GetLength(1);
            int r2 = B.GetLength(0);
            int c2 = B.GetLength(1);

            if (c1 != r2)
            {
                throw new ArgumentException($"Matrix dimensions don't match");
            }

            if (!ShouldUseGpu((long)r1 * c2 * c1, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.MatrixMultiply(A, B);
            }

            var key = (r1, c1, c2);
            if (!_matMulCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r1, c1)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r2, c2)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r1, c2))
                );
                _matMulCache[key] = bufs;
            }

            bufs.a.CopyFromCPU(A);
            bufs.b.CopyFromCPU(B);
            _matMulKernel(new Index2D(r1, c2), bufs.a.View, bufs.b.View, bufs.c.View);

            var result = new float[r1, c2];
            bufs.c.CopyToCPU(result);
            return result;
        }
        private static void MatMulKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> C)
        {
            int row = idx.X, col = idx.Y;
            float sum = 0.0f;
            int K = (int)A.Extent.Y;
            for (int k = 0; k < K; k++)
            {
                sum += A[row, k] * B[k, col];
            }
            C[row, col] = sum;
        }

        #endregion

        #region MatrixMultiplyTranspose

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matMulTransposeKernel;
        public float[,] MatrixMultiplyTranspose(float[,] A, float[,] B)
        {
            int r1 = A.GetLength(0);
            int c1 = A.GetLength(1);
            int r2 = B.GetLength(0);
            int c2 = B.GetLength(1);

            if (c1 != c2)
            {
                throw new ArgumentException($"Matrix dimensions don't match for A*B^T");
            }

            if (!ShouldUseGpu((long)r1 * r2 * c1, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.MatrixMultiplyTranspose(A, B);
            }

            var key = (r1, c1, r2);
            if (!_matMulTransposeCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r1, c1)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r2, c2)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(r1, r2))
                );
                _matMulTransposeCache[key] = bufs;
            }

            bufs.a.CopyFromCPU(A);
            bufs.b.CopyFromCPU(B);
            _matMulTransposeKernel(new Index2D(r1, r2), bufs.a.View, bufs.b.View, bufs.c.View);

            var result = new float[r1, r2];
            bufs.c.CopyToCPU(result);
            return result;
        }


        private static void MatMulTransposeKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> C)
        {
            int row = idx.X;
            int col = idx.Y;
            float sum = 0.0f;
            int kCount = (int)A.Extent.Y;

            for (int k = 0; k < kCount; k++)
            {
                sum += A[row, k] * B[col, k];
            }

            C[row, col] = sum;
        }

        #endregion

        #region MatrixScale

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float> _matScaleKernel;
        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.MatrixScale(matrix, scalar);
            }

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                bufIn.CopyFromCPU(matrix);
                _matScaleKernel(new Index2D(rows, cols), bufIn.View, scalar);

                var result = new float[rows, cols];
                bufIn.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufIn.Dispose();
            }
        }
        private static void MatScaleKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> mat, float scalar)
        {
            mat[idx] *= scalar;
        }

        #endregion

        #region MatrixAdd
        private static void MatAddKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> C)
        {
            C[idx] = A[idx] + B[idx];
        }

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matAddKernel;
        public float[,] MatrixAdd(float[,] A, float[,] B)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.MatrixAdd(A, B);
            }

            var key = (rows, cols);
            if (!_matrixAddCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols))
                );
                _matrixAddCache[key] = bufs;
            }

            bufs.a.CopyFromCPU(A);
            bufs.b.CopyFromCPU(B);
            _matAddKernel(new Index2D(rows, cols), bufs.a.View, bufs.b.View, bufs.c.View);

            var result = new float[rows, cols];
            bufs.c.CopyToCPU(result);
            return result;
        }
        #endregion

        #region MatrixAddBias

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _matAddBiasKernel;
        public float[,] MatrixAddBias(float[,] matrix, float[] bias)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.MatrixAddBias(matrix, bias);
            }

            var key = (rows, cols);
            if (!_matrixAddBiasCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate1D<float>(cols),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols))
                );
                _matrixAddBiasCache[key] = bufs;
            }

            bufs.input.CopyFromCPU(matrix);
            bufs.bias.CopyFromCPU(bias);
            _matAddBiasKernel(new Index2D(rows, cols), bufs.input.View, bufs.bias.View, bufs.output.View);

            var result = new float[rows, cols];
            bufs.output.CopyToCPU(result);
            return result;
        }
        private static void MatAddBiasKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> bias, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[idx] = matrix[idx] + bias[idx.Y];
        }

        #endregion

        #region BatchDotProductAddBias

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int, ArrayView2D<float, Stride2D.DenseX>> _batchDotAddBiasKernel;

        public float[,] BatchDotProductAddBias(float[,] weights, float[,] inputMatrix, float[] bias)
        {
            if (inputMatrix == null) throw new ArgumentNullException(nameof(inputMatrix));
            return BatchDotProductAddBias(weights, inputMatrix, 0, inputMatrix.GetLength(0), bias);
        }

        public float[,] BatchDotProductAddBias(float[,] weights, float[,] inputMatrix, int rowStart, int rowCount, float[] bias)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (inputMatrix == null) throw new ArgumentNullException(nameof(inputMatrix));
            if (bias == null) throw new ArgumentNullException(nameof(bias));
            if (rowStart < 0 || rowCount < 0) throw new ArgumentOutOfRangeException();

            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);
            if (inputMatrix.GetLength(1) != inputDim)
            {
                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");
            }
            if (rowStart + rowCount > inputMatrix.GetLength(0))
            {
                throw new ArgumentException("Invalid row slice.");
            }
            if (bias.Length != outputDim)
            {
                throw new ArgumentException("Bias length must match output dimension.", nameof(bias));
            }

            if (!ShouldUseGpu((long)rowCount * outputDim * inputDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.BatchDotProductAddBias(weights, inputMatrix, rowStart, rowCount, bias);
            }

            int inputRows = inputMatrix.GetLength(0);
            var key = (outputDim, inputDim, inputRows, rowCount);

            if (!_batchDotAddBiasCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(inputRows, inputDim)),
                    _accelerator.Allocate1D<float>(outputDim),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rowCount, outputDim))
                );

                _batchDotAddBiasCache[key] = bufs;
            }

            var residentWeights = GetResidentMatrixReadOnly(weights);
            var residentBias = GetResidentVectorReadOnly(bias);
            bufs.inp.CopyFromCPU(inputMatrix);

            _batchDotAddBiasKernel(
                new Index2D(rowCount, outputDim),
                residentWeights.View,
                bufs.inp.View,
                residentBias.View,
                rowStart,
                bufs.res.View);

            var result = new float[rowCount, outputDim];
            bufs.res.CopyToCPU(result);

            return result;
        }

        private static void BatchDotAddBiasKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> bias, int rowStart, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int seq = idx.X;
            int outDim = idx.Y;
            int inputDim = (int)weights.Extent.Y;
            int srcRow = rowStart + seq;
            float sum = 0.0f;

            for (int k = 0; k < inputDim; k++)
            {
                sum += weights[outDim, k] * input[srcRow, k];
            }

            result[seq, outDim] = sum + bias[outDim];
        }

        #endregion

        #region BatchDotProduct

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, ArrayView2D<float, Stride2D.DenseX>> _batchDotKernel;

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

            int outputDim = weights.GetLength(0);
            int inputDim = weights.GetLength(1);

            if (inputMatrix.GetLength(1) != inputDim)
            {
                throw new ArgumentException($"Expected input columns {inputDim}, got {inputMatrix.GetLength(1)}");
            }

            if (rowStart + rowCount > inputMatrix.GetLength(0))
            {
                throw new ArgumentException("Invalid row slice.");
            }

            if (!ShouldUseGpu((long)rowCount * outputDim * inputDim, GPU_MATMUL_OP_THRESHOLD))
            {
                return _mutliThreadCPU.BatchDotProduct(weights, inputMatrix, rowStart, rowCount);
            }

            int inputRows = inputMatrix.GetLength(0);
            var key = (outputDim, inputDim, inputRows, rowCount);

            if (!_batchDotCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(inputRows, inputDim)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rowCount, outputDim))
                );

                _batchDotCache[key] = bufs;
            }

            var residentWeights = GetResidentMatrixReadOnly(weights);
            bufs.inp.CopyFromCPU(inputMatrix);

            _batchDotKernel(
                new Index2D(rowCount, outputDim),
                residentWeights.View,
                bufs.inp.View,
                rowStart,
                bufs.res.View);

            var result = new float[rowCount, outputDim];
            bufs.res.CopyToCPU(result);

            return result;
        }

        private static void BatchDotKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> input, int rowStart, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int seq = idx.X;
            int outDim = idx.Y;
            int inputDim = (int)weights.Extent.Y;
            int srcRow = rowStart + seq;
            float sum = 0.0f;

            for (int k = 0; k < inputDim; k++)
            {
                sum += weights[outDim, k] * input[srcRow, k];
            }

            result[seq, outDim] = sum;
        }

        #endregion

        #region SliceRows

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int> _sliceRowsKernel;
        public float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (startRow < 0 || endRow < startRow || endRow > rows)
            {
                throw new ArgumentOutOfRangeException(nameof(startRow), "Invalid row range.");
            }

            int rowCount = endRow - startRow;

            if (rowCount == 0)
            {
                return new float[0, cols];
            }

            if (!ShouldUseGpu((long)rowCount * cols))
            {
                return _mutliThreadCPU.SliceRows(matrix, startRow, endRow);
            }

            var key = (rows, cols, rowCount);
            if (!_sliceRowsCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rowCount, cols))
                );
                _sliceRowsCache[key] = bufs;
            }

            bufs.input.CopyFromCPU(matrix);
            _sliceRowsKernel(new Index2D(rowCount, cols), bufs.input.View, bufs.output.View, startRow);

            var result = new float[rowCount, cols];
            bufs.output.CopyToCPU(result);
            return result;
        }

        private static void SliceRowsKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView2D<float, Stride2D.DenseX> result, int startRow)
        {
            result[idx] = matrix[startRow + idx.X, idx.Y];
        }

        #endregion

        #region ExtractRow

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int> _extractRowKernel;
        public float[] ExtractRow(float[,] matrix, int rowIndex, int cols)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            int rows = matrix.GetLength(0);
            int actualCols = matrix.GetLength(1);

            if (rowIndex < 0 || rowIndex >= rows)
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex));
            }

            if (cols < 0 || cols > actualCols)
            {
                throw new ArgumentOutOfRangeException(nameof(cols));
            }

            if (cols == 0)
            {
                return Array.Empty<float>();
            }

            if (!ShouldUseGpu(cols))
            {
                return _mutliThreadCPU.ExtractRow(matrix, rowIndex, cols);
            }

            var bufMatrix = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, actualCols));
            var bufResult = _accelerator.Allocate1D<float>(cols);

            try
            {
                bufMatrix.CopyFromCPU(matrix);
                _extractRowKernel(new Index1D(cols), bufMatrix.View, bufResult.View, rowIndex);

                var result = new float[cols];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufMatrix.Dispose();
                bufResult.Dispose();
            }
        }

        private static void ExtractRowKernel(Index1D col, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> result, int rowIndex)
        {
            result[col] = matrix[rowIndex, col];
        }

        #endregion

        #region SetRow

        private Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, int> _setRowKernel;
        public void SetRow(float[,] matrix, int rowIndex, float[] values, int cols)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (values == null)
            {
                throw new ArgumentNullException(nameof(values));
            }

            int rows = matrix.GetLength(0);
            int actualCols = matrix.GetLength(1);

            if (rowIndex < 0 || rowIndex >= rows)
            {
                throw new ArgumentOutOfRangeException(nameof(rowIndex));
            }

            if (cols < 0 || cols > actualCols || cols > values.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(cols));
            }

            if (cols == 0)
            {
                return;
            }

            if (!ShouldUseGpu(cols))
            {
                _mutliThreadCPU.SetRow(matrix, rowIndex, values, cols);
                return;
            }

            var bufMatrix = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, actualCols));
            var bufValues = _accelerator.Allocate1D<float>(cols);

            try
            {
                var valueSlice = values;
                if (values.Length != cols)
                {
                    valueSlice = new float[cols];
                    Array.Copy(values, valueSlice, cols);
                }

                bufMatrix.CopyFromCPU(matrix);
                bufValues.CopyFromCPU(valueSlice);
                _setRowKernel(new Index1D(cols), bufMatrix.View, bufValues.View, rowIndex);
                bufMatrix.CopyToCPU(matrix);
            }
            finally
            {
                bufMatrix.Dispose();
                bufValues.Dispose();
            }
        }

        private static void SetRowKernel(Index1D col, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> values, int rowIndex)
        {
            matrix[rowIndex, col] = values[col];
        }

        #endregion

        #region ZeroMatrix

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> _zeroMatrixKernel;
        public void ZeroMatrix(float[,] matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                _mutliThreadCPU.ZeroMatrix(matrix);
                return;
            }

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                _zeroMatrixKernel(new Index2D(rows, cols), buf.View);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        private static void ZeroMatrixKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> mat)
        {
            mat[idx] = 0.0f;
        }

        #endregion

        #region ZeroVector

        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>> _zeroVectorKernel;
        public void ZeroVector(float[] vector)
        {
            if (vector == null)
            {
                throw new ArgumentNullException(nameof(vector));
            }

            int n = vector.Length;

            if (!ShouldUseGpu(n))
            {
                _mutliThreadCPU.ZeroVector(vector);
                return;
            }

            var buf = _accelerator.Allocate1D<float>(n);

            try
            {
                _zeroVectorKernel(new Index1D(n), buf.View);
                buf.CopyToCPU(vector);
            }
            finally
            {
                buf.Dispose();
            }
        }

        private static void ZeroVectorKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> vector)
        {
            vector[i] = 0.0f;
        }

        #endregion

        #region MatrixAddInPlace

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matAccumulateKernel;
        public void MatrixAddInPlace(float[,] target, float[,] source)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);

            if (!ShouldUseGpu((long)rows * cols))
            {
                _mutliThreadCPU.MatrixAddInPlace(target, source);
                return;
            }

            var bufTgt = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufSrc = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                bufTgt.CopyFromCPU(target);
                bufSrc.CopyFromCPU(source);
                _matAccumulateKernel(new Index2D(rows, cols), bufTgt.View, bufSrc.View);
                bufTgt.CopyToCPU(target);
            }
            finally
            {
                bufTgt.Dispose();
                bufSrc.Dispose();
            }
        }
        private static void MatAccumulateKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> target, ArrayView2D<float, Stride2D.DenseX> source)
        {
            target[idx] += source[idx];
        }
        #endregion

        #region VectorAccumulate

        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _vecAccumulateKernel;
        public void VectorAccumulate(float[] target, float[] source)
        {
            int n = target.Length;

            if (!ShouldUseGpu(n))
            {
                _mutliThreadCPU.VectorAccumulate(target, source);
                return;
            }

            var bufTgt = _accelerator.Allocate1D<float>(n);
            var bufSrc = _accelerator.Allocate1D<float>(n);

            try
            {
                bufTgt.CopyFromCPU(target);
                bufSrc.CopyFromCPU(source);

                _vecAccumulateKernel(new Index1D(n), bufTgt.View, bufSrc.View);

                bufTgt.CopyToCPU(target);
            }
            finally
            {
                bufTgt.Dispose(); bufSrc.Dispose();
            }
        }

        private static void VecAccumulateKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> target, ArrayView1D<float, Stride1D.Dense> source)
        {
            target[i] += source[i];
        }

        #endregion


        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> _zeroMatrixColumnsKernel;
        private static void ZeroMatrixColumnsKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> mat)
        {
            mat[idx] = 0.0f;
        }
        public void ZeroMatrixColumns(float[,] matrix, int columnCount)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (columnCount <= 0)
            {
                return;
            }

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            int count = Math.Min(columnCount, cols);

            if (!ShouldUseGpu((long)rows * count))
            {
                _mutliThreadCPU.ZeroMatrixColumns(matrix, count);
                return;
            }

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                buf.CopyFromCPU(matrix);
                _zeroMatrixColumnsKernel(new Index2D(rows, count), buf.View);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }
        public float[,] ResidualLayerNorm(float[,] input, float[,] subLayer, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            ValidateResidualLayerNormInputs(input, subLayer, gamma, beta, out int rows, out int cols);

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.ResidualLayerNorm(input, subLayer, gamma, beta, epsilon);
            }

            var key = (rows, cols);
            if (!_residualLayerNormCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate1D<float>(cols),
                    _accelerator.Allocate1D<float>(cols),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols))
                );
                _residualLayerNormCache[key] = bufs;
            }

            bufs.input.CopyFromCPU(input);
            bufs.subLayer.CopyFromCPU(subLayer);
            bufs.gamma.CopyFromCPU(gamma);
            bufs.beta.CopyFromCPU(beta);

            _residualLayerNormKernel(new Index1D(rows), bufs.input.View, bufs.subLayer.View, bufs.gamma.View, bufs.beta.View, bufs.output.View, epsilon);

            var output = new float[rows, cols];
            bufs.output.CopyToCPU(output);
            return output;
        }

        public (float[,] output, float[] means, float[] variances, float[,] normalized, float[,] residual) ResidualLayerNormForward(float[,] input, float[,] subLayer, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            ValidateResidualLayerNormInputs(input, subLayer, gamma, beta, out int rows, out int cols);

            if (!ShouldUseGpu((long)rows * cols))
            {
                return _mutliThreadCPU.ResidualLayerNormForward(input, subLayer, gamma, beta, epsilon);
            }

            var key = (rows, cols);
            if (!_residualLayerNormForwardCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate1D<float>(cols),
                    _accelerator.Allocate1D<float>(cols),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate1D<float>(rows),
                    _accelerator.Allocate1D<float>(rows),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols))
                );
                _residualLayerNormForwardCache[key] = bufs;
            }

            bufs.input.CopyFromCPU(input);
            bufs.subLayer.CopyFromCPU(subLayer);
            bufs.gamma.CopyFromCPU(gamma);
            bufs.beta.CopyFromCPU(beta);

            _residualLayerNormForwardKernel(new Index1D(rows), bufs.input.View, bufs.subLayer.View, bufs.gamma.View, bufs.beta.View, bufs.output.View, bufs.means.View, bufs.variances.View, bufs.normalized.View, bufs.residual.View, epsilon);

            var output = new float[rows, cols];
            var means = new float[rows];
            var variances = new float[rows];
            var normalized = new float[rows, cols];
            var residual = new float[rows, cols];

            bufs.output.CopyToCPU(output);
            bufs.means.CopyToCPU(means);
            bufs.variances.CopyToCPU(variances);
            bufs.normalized.CopyToCPU(normalized);
            bufs.residual.CopyToCPU(residual);

            return (output, means, variances, normalized, residual);
        }

        private static void ValidateResidualLayerNormInputs(float[,] input, float[,] subLayer, float[] gamma, float[] beta, out int rows, out int cols)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (subLayer == null) throw new ArgumentNullException(nameof(subLayer));
            if (gamma == null) throw new ArgumentNullException(nameof(gamma));
            if (beta == null) throw new ArgumentNullException(nameof(beta));

            rows = input.GetLength(0);
            cols = input.GetLength(1);

            if (subLayer.GetLength(0) != rows || subLayer.GetLength(1) != cols)
            {
                throw new ArgumentException("Residual input dimensions must match.", nameof(subLayer));
            }
            if (gamma.Length != cols || beta.Length != cols)
            {
                throw new ArgumentException("Layer norm gamma/beta must match the feature dimension.");
            }
        }

        private void DisposeSharedTensorBuffers()
        {
            foreach (var v in _matMulCache.Values)
            {
                v.a.Dispose();
                v.b.Dispose();
                v.c.Dispose();
            }

            foreach (var v in _matMulTransposeCache.Values)
            {
                v.a.Dispose();
                v.b.Dispose();
                v.c.Dispose();
            }

            foreach (var v in _matrixAddCache.Values)
            {
                v.a.Dispose();
                v.b.Dispose();
                v.c.Dispose();
            }

            foreach (var v in _matrixAddBiasCache.Values)
            {
                v.input.Dispose();
                v.bias.Dispose();
                v.output.Dispose();
            }

            foreach (var v in _sliceRowsCache.Values)
            {
                v.input.Dispose();
                v.output.Dispose();
            }

            foreach (var v in _batchDotCache.Values)
            {
                v.w.Dispose();
                v.inp.Dispose();
                v.res.Dispose();
            }

            foreach (var v in _batchDotAddBiasCache.Values)
            {
                v.w.Dispose();
                v.inp.Dispose();
                v.bias.Dispose();
                v.res.Dispose();
            }

            foreach (var v in _residualLayerNormCache.Values)
            {
                v.input.Dispose();
                v.subLayer.Dispose();
                v.gamma.Dispose();
                v.beta.Dispose();
                v.output.Dispose();
            }

            foreach (var v in _residualLayerNormForwardCache.Values)
            {
                v.input.Dispose();
                v.subLayer.Dispose();
                v.gamma.Dispose();
                v.beta.Dispose();
                v.output.Dispose();
                v.means.Dispose();
                v.variances.Dispose();
                v.normalized.Dispose();
                v.residual.Dispose();
            }

            _matMulCache.Clear();
            _matMulTransposeCache.Clear();
            _matrixAddCache.Clear();
            _matrixAddBiasCache.Clear();
            _sliceRowsCache.Clear();
            _batchDotCache.Clear();
            _batchDotAddBiasCache.Clear();
            _residualLayerNormCache.Clear();
            _residualLayerNormForwardCache.Clear();
        }

    }
}
