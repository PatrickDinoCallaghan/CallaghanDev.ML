using CallaghanDev.ML.Enums;
using ILGPU;
using ILGPU.Runtime;
using MathNet.Symbolics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
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
            _batchDotKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(BatchDotKernel);

            //SliceRows, ExtractRow and SetRow not GPU accellerated

            //ZeroMatrix
            _zeroMatrixKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(ZeroMatrixKernel);

            //ZeroVector not GPU accellerated

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

            var bufA = _accelerator.Allocate2DDenseX<float>(new Index2D(r1, c1));
            var bufB = _accelerator.Allocate2DDenseX<float>(new Index2D(r2, c2));
            var bufC = _accelerator.Allocate2DDenseX<float>(new Index2D(r1, r2));

            try
            {
                bufA.CopyFromCPU(A);
                bufB.CopyFromCPU(B);
                _matMulTransposeKernel(new Index2D(r1, r2), bufA.View, bufB.View, bufC.View);

                var result = new float[r1, r2];
                bufC.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufA.Dispose(); bufB.Dispose(); bufC.Dispose();
            }
        }

        #endregion

        #region MatrixScale

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float> _matScaleKernel;
        public float[,] MatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

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

            var bufA = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufB = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufC = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                bufA.CopyFromCPU(A);
                bufB.CopyFromCPU(B);
                _matAddKernel(new Index2D(rows, cols), bufA.View, bufB.View, bufC.View);

                var result = new float[rows, cols];
                bufC.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufA.Dispose(); bufB.Dispose(); bufC.Dispose();
            }
        }
        #endregion

        #region MatrixAddBias

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _matAddBiasKernel;
        public float[,] MatrixAddBias(float[,] matrix, float[] bias)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufBias = _accelerator.Allocate1D<float>(cols);
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                bufIn.CopyFromCPU(matrix);
                bufBias.CopyFromCPU(bias);
                _matAddBiasKernel(new Index2D(rows, cols), bufIn.View, bufBias.View, bufOut.View);

                var result = new float[rows, cols];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufIn.Dispose(); bufBias.Dispose(); bufOut.Dispose();
            }
        }
        private static void MatAddBiasKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> bias, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[idx] = matrix[idx] + bias[idx.Y];
        }

        #endregion

        #region BatchDotProduct

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _batchDotKernel;

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

            var key = (outputDim, inputDim, rowCount);

            if (!_batchDotCache.TryGetValue(key, out var bufs))
            {
                bufs = (
                    _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, inputDim)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rowCount, inputDim)),
                    _accelerator.Allocate2DDenseX<float>(new Index2D(rowCount, outputDim))
                );

                _batchDotCache[key] = bufs;
            }

            bufs.w.CopyFromCPU(weights);

            // Create temporary CPU slice (required)
            var slice = new float[rowCount, inputDim];

            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    slice[i, j] = inputMatrix[rowStart + i, j];
                }
            }

            bufs.inp.CopyFromCPU(slice);

            _batchDotKernel(
                new Index2D(rowCount, outputDim),
                bufs.w.View,
                bufs.inp.View,
                bufs.res.View);

            var result = new float[rowCount, outputDim];
            bufs.res.CopyToCPU(result);

            return result;
        }

        private static void BatchDotKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int seq = idx.X;
            int outDim = idx.Y;
            int inputDim = (int)weights.Extent.Y;
            float sum = 0.0f;


            for (int k = 0; k < inputDim; k++)
            {
                sum += weights[outDim, k] * input[seq, k];
            }
            result[seq, outDim] = sum;
        }

        #endregion

        #region SliceRows
        //SliceRows is just a memory copy. Moving it to ILGPU usually costs meaning the transfer overhead will usually beat and GPU benifit, so we will just use multi thread slicing
        public float[,] SliceRows(float[,] matrix, int startRow, int endRow)
        {
            return _mutliThreadCPU.SliceRows(matrix, startRow, endRow);
        }
        #endregion

        #region ExtractRow
        public float[] ExtractRow(float[,] matrix, int rowIndex, int cols)
        {
            return _mutliThreadCPU.ExtractRow(matrix, rowIndex, cols);
        }
        #endregion

        #region SetRow
        public void SetRow(float[,] matrix, int rowIndex, float[] values, int cols)
        {
            _mutliThreadCPU.SetRow(matrix, rowIndex, values, cols);
        }
        #endregion

        #region ZeroMatrix

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> _zeroMatrixKernel;
        public void ZeroMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                // No need to copy data to GPU afterall we are just going to zeroing it
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

        public void ZeroVector(float[] vector)
        {
            _mutliThreadCPU.ZeroVector(vector);
        }

        #endregion

        #region MatrixAddInPlace

        private Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _matAccumulateKernel;
        public void MatrixAddInPlace(float[,] target, float[,] source)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);

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

    }
}
