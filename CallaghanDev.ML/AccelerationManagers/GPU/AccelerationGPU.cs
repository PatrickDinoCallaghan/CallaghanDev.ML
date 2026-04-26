using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.TACAMT;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using static Microsoft.FSharp.Core.ByRefKinds;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly AccelerationMutliThreadCPU _mutliThreadCPU;
       
     
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, float> _matScaleInPlaceKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float> _vecScaleInPlaceKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float> _matUpdateKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _vecUpdateKernel;
       
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _layerNormForwardKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float> _layerNormBackwardKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>> _matSquaredNormKernel;

        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _embedTokensKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _addBiasPosKernel;
      
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> _sigmoidInPlaceKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>> _zeroVectorKernel;
        private readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> _vecDotSumKernel;
        private readonly Action<Index1D, ArrayView2D<int, Stride2D.DenseX>> _causalMaskKernel;


        private readonly Dictionary<int, (MemoryBuffer1D<float, Stride1D.Dense> cost, MemoryBuffer1D<float, Stride1D.Dense> der, MemoryBuffer1D<float, Stride1D.Dense> grad)> _outGradCache = new();
        private readonly Dictionary<int, (MemoryBuffer1D<float, Stride1D.Dense> pre, MemoryBuffer1D<float, Stride1D.Dense> der, MemoryBuffer1D<float, Stride1D.Dense> delta)> _hidGradCache = new();
       
        private readonly Dictionary<(int r1, int c1, int c2), (MemoryBuffer2D<float, Stride2D.DenseX> a, MemoryBuffer2D<float, Stride2D.DenseX> b, MemoryBuffer2D<float, Stride2D.DenseX> c)> _matMulCache = new();
        private readonly Dictionary<(int outputDim, int inputDim, int rowCount), (MemoryBuffer2D<float, Stride2D.DenseX> w, MemoryBuffer2D<float, Stride2D.DenseX> inp, MemoryBuffer2D<float, Stride2D.DenseX> res)> _batchDotCache = new();
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _applyContextTypeEmbeddingKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _computeTimeDiffMatrixKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _meanPoolRowsKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _embedWithBiasAndPositionalKernel;
        private readonly Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> _computeMemoryAttentionScoresKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>> _projectOutputBatchKernel;

        // Add these to the AccelerationGPU class fields
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> _decayProjectQueriesKernel;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>> _decayProjectKeysKernel;
        private readonly Action<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>> _decayTimeEncodingRawKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>> _decayTimeEncodingProjKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _decayMemAttnQKInputKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float> _decayMemAttnScoresKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _decayMemAttnSoftmaxKernel;
        private readonly Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float, float, int> _decayMemAttnDropoutKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _decayMemAttnWeightedSumKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _decayMemAttnOutputProjKernel;
        private readonly Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, float> _mat3DScaleInPlaceKernel;

        #region Content aware decay forward only
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView3D<float, Stride3D.DenseXY>> _decayFinalBiasKernel;
        private readonly Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>> _decayMLPInputKernel;
        private readonly Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _decayMLPHiddenKernel;
        private readonly Action<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float, float, int> _decayMLPDropoutKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _decayMLPOutputKernel;
        #endregion

        #region Content aware cross attention forward
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, int, int> _extractHeadQKVKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float> _contentAwareScoresKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _addDecayBiasKernel;
        private readonly Action<Index1D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _contentAwareSoftmaxKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>> _contentAwareWeightedSumKernel;
        private readonly Action<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, int, int> _assembleHeadOutputKernel;
        #endregion


        public AccelerationGPU(AccelerationType accelerationType, int deviceIndex = 0)
        {
            Context context = Context.Create(builder =>
            {
                builder.EnableAlgorithms();
                builder.AllAccelerators();
            });
            if (accelerationType == AccelerationType.GPU)
            {
                _accelerator = context.CreateCLAccelerator(deviceIndex);
            }
            else if (accelerationType == AccelerationType.CUDA)
            {
                _accelerator = context.CreateCudaAccelerator(deviceIndex);
            }

            _mutliThreadCPU = new AccelerationMutliThreadCPU();

            InitSharedTensorKernels();
            InitNeuralNetworkKernels();
            InitTransformerCoreKernels();

            _matScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, float>(MatScaleInPlaceKernel);
            _vecScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, float>(VecScaleInPlaceKernel);
            _matUpdateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, float>(MatUpdateKernel);
            _vecUpdateKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(VecUpdateKernel);
            _layerNormForwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormForwardKernel);
            _layerNormBackwardKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>(LayerNormBackwardKernel);

            _embedTokensKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(EmbedTokensKernel);
            _addBiasPosKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(AddBiasPosKernel);
            
            _sigmoidInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(SigmoidInPlaceKernel);
            _zeroVectorKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(ZeroVectorKernel);
            _causalMaskKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<int, Stride2D.DenseX>>(CausalMaskKernel);

            _applyContextTypeEmbeddingKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>>(ApplyContextTypeEmbeddingKernel);
            _computeTimeDiffMatrixKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(ComputeTimeDiffMatrixKernel);
            _meanPoolRowsKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>>(MeanPoolRowsKernel);
            _embedWithBiasAndPositionalKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(EmbedWithBiasAndPositionalKernel);
            _computeMemoryAttentionScoresKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(ComputeMemoryAttentionScoresKernel);
            _projectOutputBatchKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>>(ProjectOutputBatchKernel);

            _mat3DScaleInPlaceKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, float>(Mat3DScaleInPlaceKernel);

            #region Content aware decay forward only

            _decayProjectQueriesKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>>(DecayProjectQueriesKernel);
            _decayProjectKeysKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>>(DecayProjectKeysKernel);
            _decayTimeEncodingRawKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>>(DecayTimeEncodingRawKernel);
            _decayTimeEncodingProjKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>>(DecayTimeEncodingProjKernel);
            _decayMemAttnQKInputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMemAttnQKInputKernel);
            _decayMemAttnScoresKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float>(DecayMemAttnScoresKernel);
            _decayMemAttnSoftmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMemAttnSoftmaxKernel);
            _decayMemAttnDropoutKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float, float, int>(DecayMemAttnDropoutKernel);
            _decayMemAttnWeightedSumKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMemAttnWeightedSumKernel);
            //_decayMemAttnOutputProjKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMemAttnOutputProjKernel);
            _decayMLPInputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMLPInputKernel);
            _decayMLPHiddenKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMLPHiddenKernel);
            _decayMLPDropoutKernel = _accelerator.LoadAutoGroupedStreamKernel<Index3D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float, float, int>(DecayMLPDropoutKernel);
            _decayMLPOutputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMLPOutputKernel);
            _decayFinalBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView3D<float, Stride3D.DenseXY>>(DecayFinalBiasKernel);
            _decayMemAttnOutputProjKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(DecayMemAttnOutputProjKernel);

            #endregion

            #region Content aware cross attention forward
            _extractHeadQKVKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView3D<float, Stride3D.DenseXY>, int, int>(ExtractHeadQKVKernel);
            _contentAwareScoresKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, float>(ContentAwareScoresKernel);
           
            
           
            _addDecayBiasKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(AddDecayBiasKernel);
            
            
            _contentAwareSoftmaxKernel = _accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(ContentAwareSoftmaxKernel);
            
            
            _contentAwareWeightedSumKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>, ArrayView3D<float, Stride3D.DenseXY>>(ContentAwareWeightedSumKernel);
            _assembleHeadOutputKernel = _accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView3D<float, Stride3D.DenseXY>, ArrayView2D<float, Stride2D.DenseX>, int, int>(AssembleHeadOutputKernel);
            #endregion
        }

      
 
        private static void MatScaleInPlaceKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> mat, float scale)
        {
            mat[idx] *= scale;
        }

        private static void VecScaleInPlaceKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> vec, float scale)
        {
            vec[i] *= scale;
        }

        private static void MatUpdateKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView2D<float, Stride2D.DenseX> gradients, float learningRate)
        {
            weights[idx] -= learningRate * gradients[idx];
        }

        private static void VecUpdateKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> weights, ArrayView1D<float, Stride1D.Dense> gradients, float learningRate)
        {
            weights[i] -= learningRate * gradients[i];
        }

    
        private static void LayerNormForwardKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView1D<float, Stride1D.Dense> beta, ArrayView2D<float, Stride2D.DenseX> output, ArrayView1D<float, Stride1D.Dense> means, ArrayView1D<float, Stride1D.Dense> variances, float epsilon)
        {
            int features = (int)input.Extent.Y;

            float mean = 0.0f;


            for (int j = 0; j < features; j++)
            {
                mean += input[batch, j];
            }
            mean = mean / features;
            means[batch] = mean;

            float variance = 0.0f;

            for (int j = 0; j < features; j++)
            {
                float diff = input[batch, j] - mean;
                variance += diff * diff;
            }

            variance = variance/ features;
            variances[batch] = variance;

            float stdDev = XMath.Sqrt(variance + epsilon);

            for (int j = 0; j < features; j++)
            {
                output[batch, j] = gamma[j] * (input[batch, j] - mean) / stdDev + beta[j];
            }
        }

        private static void LayerNormBackwardKernel(Index1D batch, ArrayView2D<float, Stride2D.DenseX> dOut, ArrayView2D<float, Stride2D.DenseX> normalized, ArrayView1D<float, Stride1D.Dense> gamma, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> mean, ArrayView1D<float, Stride1D.Dense> variance, ArrayView2D<float, Stride2D.DenseX> dInput, ArrayView1D<float, Stride1D.Dense> dGamma,  ArrayView1D<float, Stride1D.Dense> dBeta, float epsilon)
        {
            int features = (int)input.Extent.Y;
            float invStd = 1.0f / XMath.Sqrt(variance[batch] + epsilon);

            for (int j = 0; j < features; j++)
            {
                Atomic.Add(ref dGamma[j], dOut[batch, j] * normalized[batch, j]);
                Atomic.Add(ref dBeta[j], dOut[batch, j]);
            }

            float dVar = 0.0f;
            float dMean = 0.0f;
            float invStdCubed = invStd * invStd * invStd;

            for (int j = 0; j < features; j++)
            {
                float dNorm = dOut[batch, j] * gamma[j];
                float xMinusMean = input[batch, j] - mean[batch];
                dVar += dNorm * xMinusMean * (-0.5f) * invStdCubed;
                dMean += dNorm * (-invStd);
            }

            float invN = 1.0f / features;

            for (int j = 0; j < features; j++)
            {
                float dNorm = dOut[batch, j] * gamma[j];
                float xMinusMean = input[batch, j] - mean[batch];
                dInput[batch, j] = dNorm * invStd + dVar * 2.0f * xMinusMean * invN + dMean * invN;
            }
        }
        private static void Mat3DScaleInPlaceKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> mat, float scale)
        {
            mat[idx] *= scale;
        }

        #region ContentAwareDecayForward Kernels
        private static void DecayProjectQueriesKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> queryEmbeddings, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView2D<float, Stride2D.DenseX> bias, ArrayView2D<float, Stride2D.DenseX> output)
        {
            int h = idx.X;
            int q = idx.Y;
            int projDim = (int)bias.Extent.Y;
            int contentDim = (int)queryEmbeddings.Extent.Y;

            for (int p = 0; p < projDim; p++)
            {
                float val = bias[h, p];
                for (int d = 0; d < contentDim; d++)
                    val += weights[h, p, d] * queryEmbeddings[q, d];
                output[h * projDim + p, q] = val;
            }
        }

        private static void DecayProjectKeysKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> keyEmbeddings, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView2D<float, Stride2D.DenseX> bias, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int h = idx.X;
            int s = idx.Y;
            int projDim = (int)bias.Extent.Y;
            int contentDim = (int)keyEmbeddings.Extent.Y;

            for (int p = 0; p < projDim; p++)
            {
                float val = bias[h, p];
                for (int d = 0; d < contentDim; d++)
                    val += weights[h, p, d] * keyEmbeddings[s, d];
                output[h, s, p] = val;
            }
        }

        private static void DecayTimeEncodingRawKernel(Index2D idx, ArrayView1D<float, Stride1D.Dense> keyTimesFromRef, ArrayView2D<float, Stride2D.DenseX> timeLogFreq, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int h = idx.X;
            int s = idx.Y;
            int numBases = (int)timeLogFreq.Extent.Y;
            float t = keyTimesFromRef[s];

            for (int b = 0; b < numBases; b++)
            {
                float freq = XMath.Exp(timeLogFreq[h, b]);
                float angle = freq * t;
                output[h, s, b * 2] = XMath.Sin(angle);
                output[h, s, b * 2 + 1] = XMath.Cos(angle);
            }
        }

        private static void DecayTimeEncodingProjKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> timeRawFeatures, ArrayView3D<float, Stride3D.DenseXY> timeProj, ArrayView2D<float, Stride2D.DenseX> timeProjBias, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int h = idx.X;
            int s = idx.Y;
            int projDim = (int)output.Extent.Z;
            int rawDim = (int)timeRawFeatures.Extent.Z;

            for (int p = 0; p < projDim; p++)
            {
                float val = timeProjBias[h, p];
                for (int r = 0; r < rawDim; r++)
                    val += timeProj[h, p, r] * timeRawFeatures[h, s, r];
                output[h, s, p] = val;
            }
        }

        private static void DecayMemAttnQKInputKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> keyProj, ArrayView3D<float, Stride3D.DenseXY> timeEncoding, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int h = idx.X;
            int s = idx.Y;
            int projDim = (int)output.Extent.Z;

            for (int p = 0; p < projDim; p++)
                output[h, s, p] = keyProj[h, s, p] + timeEncoding[h, s, p];
        }

        private static void DecayMemAttnScoresKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> qInput, ArrayView3D<float, Stride3D.DenseXY> kInput, ArrayView3D<float, Stride3D.DenseXY> scores, float scale)
        {
            int h = idx.X;
            int i = idx.Y;
            int keyLen = (int)kInput.Extent.Y;
            int projDim = (int)qInput.Extent.Z;

            for (int j = 0; j < keyLen; j++)
            {
                float dot = 0f;
                for (int p = 0; p < projDim; p++)
                    dot += qInput[h, i, p] * kInput[h, j, p];
                scores[h, i, j] = dot * scale;
            }
        }

        private static void DecayMemAttnSoftmaxKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> scores, ArrayView3D<float, Stride3D.DenseXY> weights)
        {
            int h = idx.X;
            int i = idx.Y;
            int keyLen = (int)scores.Extent.Z;

            float maxScore = float.MinValue;
            for (int j = 0; j < keyLen; j++)
                if (scores[h, i, j] > maxScore)
                    maxScore = scores[h, i, j];

            float sumExp = 0f;
            for (int j = 0; j < keyLen; j++)
            {
                float e = XMath.Exp(scores[h, i, j] - maxScore);
                weights[h, i, j] = e;
                sumExp += e;
            }

            if (sumExp > 0f)
                for (int j = 0; j < keyLen; j++)
                    weights[h, i, j] /= sumExp;
        }

        private static void DecayMemAttnDropoutKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> dropoutMask, ArrayView3D<float, Stride3D.DenseXY> output, float keepProb, float scale, int seed)
        {
            int h = idx.X;
            int i = idx.Y;
            int j = idx.Z;

            int state = seed + h * 10007 + i * 997 + j * 101;
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            float random = (float)state / 0x7fffffff;

            float mask = random < keepProb ? scale : 0f;
            dropoutMask[h, i, j] = mask;
            output[h, i, j] = weights[h, i, j] * mask;
        }

        private static void DecayMemAttnWeightedSumKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> keyProj, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int h = idx.X;
            int i = idx.Y;
            int keyLen = (int)keyProj.Extent.Y;
            int projDim = (int)keyProj.Extent.Z;

            for (int p = 0; p < projDim; p++)
            {
                float val = 0f;
                for (int j = 0; j < keyLen; j++)
                    val += weights[h, i, j] * keyProj[h, j, p];
                output[h, i, p] = val;
            }
        }

        private static void DecayMemAttnOutputProjKernel(
            Index2D idx,
            ArrayView3D<float, Stride3D.DenseXY> memAttnOutput,
            ArrayView3D<float, Stride3D.DenseXY> outputW,
            ArrayView2D<float, Stride2D.DenseX> outputB,
            ArrayView3D<float, Stride3D.DenseXY> keyProj,
            ArrayView3D<float, Stride3D.DenseXY> refinedKey)
        {
            int h = idx.X;
            int s = idx.Y;
            int projDim = (int)refinedKey.Extent.Z;

            for (int p = 0; p < projDim; p++)
            {
                float val = outputB[h, p];
                for (int q = 0; q < projDim; q++)
                    val += outputW[h, p, q] * memAttnOutput[h, s, q];
                refinedKey[h, s, p] = val + keyProj[h, s, p];
            }
        }

        // Flatten 4D [queryLen, keyLen, numHeads, mlpInputDim] -> 3D [queryLen*keyLen, numHeads, mlpInputDim]
        private static void DecayMLPInputKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> queryProj, ArrayView3D<float, Stride3D.DenseXY> refinedKey, ArrayView2D<float, Stride2D.DenseX> timeDiffs, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int flatIdx = idx.X;  // flattened qi*keyLen + si
            int h = idx.Y;
            int mlpInputDim = (int)output.Extent.Z;
            int projDim = (int)queryProj.Extent.Z;

            // Recover qi, si from flatIdx
            int keyLen = (int)refinedKey.Extent.Y;
            int qi = flatIdx / keyLen;
            int si = flatIdx % keyLen;

            float td = timeDiffs[qi, si];
            float logTd = XMath.Log(1f + td);

            int idx_out = 0;
            for (int p = 0; p < projDim; p++)
                output[flatIdx, h, idx_out++] = queryProj[h, qi, p];
            for (int p = 0; p < projDim; p++)
                output[flatIdx, h, idx_out++] = refinedKey[h, si, p];
            for (int p = 0; p < projDim; p++)
                output[flatIdx, h, idx_out++] = queryProj[h, qi, p] * refinedKey[h, si, p];
            output[flatIdx, h, idx_out++] = td;
            output[flatIdx, h, idx_out] = logTd;
        }

        private static void DecayMLPHiddenKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> mlpInput, ArrayView3D<float, Stride3D.DenseXY> W1, ArrayView2D<float, Stride2D.DenseX> B1, ArrayView3D<float, Stride3D.DenseXY> preAct, ArrayView3D<float, Stride3D.DenseXY> hidden)
        {
            int flatIdx = idx.X;
            int h = idx.Y;
            int j = idx.Z;  // hidden dim index
            int mlpInputDim = (int)mlpInput.Extent.Z;

            float val = B1[h, j];
            for (int k = 0; k < mlpInputDim; k++)
                val += W1[h, j, k] * mlpInput[flatIdx, h, k];

            preAct[flatIdx, h, j] = val;
            hidden[flatIdx, h, j] = val > 0f ? val : 0f;
        }

        private static void DecayMLPDropoutKernel(Index3D idx, ArrayView3D<float, Stride3D.DenseXY> hidden, ArrayView3D<float, Stride3D.DenseXY> dropoutMask, float keepProb, float scale, int seed)
        {
            int flatIdx = idx.X;
            int h = idx.Y;
            int j = idx.Z;

            int state = seed + flatIdx * 10007 + h * 997 + j * 101;
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            float random = (float)state / 0x7fffffff;

            float mask = random < keepProb ? scale : 0f;
            dropoutMask[flatIdx, h, j] = mask;
            hidden[flatIdx, h, j] *= mask;
        }

        private static void DecayMLPOutputKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> mlpHidden, ArrayView2D<float, Stride2D.DenseX> W2, ArrayView1D<float, Stride1D.Dense> B2, ArrayView3D<float, Stride3D.DenseXY> gateLogits, ArrayView3D<float, Stride3D.DenseXY> gates)
        {
            int qi = idx.X;
            int si = idx.Y;
            int numHeads = (int)gateLogits.Extent.Z;
            int hiddenDim = (int)mlpHidden.Extent.Z;
            int keyLen = (int)gateLogits.Extent.Y;
            int flatIdx = qi * keyLen + si;

            for (int h = 0; h < numHeads; h++)
            {
                float logit = B2[h];
                for (int j = 0; j < hiddenDim; j++)
                    logit += W2[h, j] * mlpHidden[flatIdx, h, j];

                gateLogits[qi, si, h] = logit;

                float gate;
                if (logit >= 0f)
                {
                    float ex = XMath.Exp(-logit);
                    gate = 1f / (1f + ex);
                }
                else
                {
                    float ex = XMath.Exp(logit);
                    gate = ex / (1f + ex);
                }
                gates[qi, si, h] = gate;
            }
        }

        private static void DecayFinalBiasKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> gates, ArrayView2D<float, Stride2D.DenseX> timeDiffs, ArrayView1D<float, Stride1D.Dense> logBaseDecayRate, ArrayView3D<float, Stride3D.DenseXY> decayBias)
        {
            int qi = idx.X;
            int si = idx.Y;
            int numHeads = (int)gates.Extent.Z;
            float td = timeDiffs[qi, si];

            for (int h = 0; h < numHeads; h++)
            {
                float baseRate = XMath.Exp(logBaseDecayRate[h]);
                decayBias[qi, si, h] = -(baseRate * gates[qi, si, h]) * td;
            }
        }

        #endregion

        #region GPU Kernels

        private static void MatMulTransposeKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> C)
        {
            int row = idx.X, col = idx.Y;
            float sum = 0.0f;
            int K = (int)A.Extent.Y;
            for (int k = 0; k < K; k++)
                sum += A[row, k] * B[col, k];  // B[col,k] not B[k,col]
            C[row, col] = sum;
        }

  
       
        private static void EmbedTokensKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> tokenEmbedding, ArrayView1D<int, Stride1D.Dense> tokenIds, ArrayView2D<float, Stride2D.DenseX> positionalEncoding, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int i = idx.X;
            int j = idx.Y;
            int tokenId = tokenIds[i];
            result[idx] = tokenEmbedding[tokenId, j] + positionalEncoding[i, j];
        }

        private static void AddBiasPosKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> projected, ArrayView1D<float, Stride1D.Dense> bias, ArrayView2D<float, Stride2D.DenseX> positionalEncoding, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int j = idx.Y;
            result[idx] = projected[idx] + bias[j] + positionalEncoding[idx];
        }

    
        private static void SigmoidInPlaceKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> matrix)
        {
            float x = matrix[idx];

            if (x >= 0)
            {
                float ex = XMath.Exp(-x);
                matrix[idx] = 1.0f / (1.0f + ex);
            }
            else
            {
                float ex = XMath.Exp(x);
                matrix[idx] = ex / (1.0f + ex);
            }
        }

        private static void ZeroVectorKernel(Index1D i, ArrayView1D<float, Stride1D.Dense> vec)
        {
            vec[i] = 0.0f;
        }


        private static void CausalMaskKernel(Index1D i, ArrayView2D<int, Stride2D.DenseX> mask)
        {
            int cols = (int)mask.Extent.Y;
            for (int j = 0; j < cols; j++)
            {
                mask[i, j] = j <= i ? 1 : 0;
            }
        }

        #endregion

        #region ContentAwareCrossAttention GPU Kernels

        private static void ExtractHeadQKVKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> input, ArrayView3D<float, Stride3D.DenseXY> output, int headDim, int startIdx)
        {
            int i = idx.X;  // sequence position
            int j = idx.Y;  // dimension within head
            int h = 0;      // head index (kernel called per head)

            output[h, i, j] = input[i, startIdx + j];
        }

        private static void ContentAwareScoresKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> Q_head, ArrayView3D<float, Stride3D.DenseXY> K_head, ArrayView3D<float, Stride3D.DenseXY> scores, float scale)
        {
            int i = idx.X;  // query position
            int j = idx.Y;  // key position
            int h = 0;      // head index
            int headDim = (int)Q_head.Extent.Z;

            float dot = 0f;
            for (int d = 0; d < headDim; d++)
                dot += Q_head[h, i, d] * K_head[h, j, d];

            scores[h, i, j] = dot * scale;
        }

        private static void AddDecayBiasKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> scores, ArrayView3D<float, Stride3D.DenseXY> decayBias, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int i = idx.X;
            int j = idx.Y;
            int h = 0;  // head index

            output[h, i, j] = scores[h, i, j] + decayBias[h, i, j];
        }

        /*private static void ContentAwareSoftmaxKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> scores, ArrayView3D<float, Stride3D.DenseXY> weights)
        {
            int h = 0;      // head index
            int i = idx.X;  // query position
            int seqLenK = (int)scores.Extent.Z;

            // Find max
            float maxScore = float.MinValue;
            for (int j = 0; j < seqLenK; j++)
                if (scores[h, i, j] > maxScore)
                    maxScore = scores[h, i, j];

            // Exp and sum
            float sumExp = 0f;
            for (int j = 0; j < seqLenK; j++)
            {
                float e = XMath.Exp(scores[h, i, j] - maxScore);
                weights[h, i, j] = e;
                sumExp += e;
            }

            // Normalize
            if (sumExp > 0f)
                for (int j = 0; j < seqLenK; j++)
                    weights[h, i, j] /= sumExp;
        }*/
        static void ContentAwareSoftmaxKernel(Index1D q, ArrayView3D<float, Stride3D.DenseXY> scores, ArrayView3D<float, Stride3D.DenseXY> weights)
        {
            int h = 0; // because your buffers are [1, seqLenQ, seqLenK]
            int seqLenK = (int)scores.Extent.Z;

            float maxVal = float.NegativeInfinity;
            for (int k = 0; k < seqLenK; k++)
            {
                float v = scores[h, q, k];
                if (v > maxVal)
                {
                    maxVal = v;

                }
            }
                float sum = 0.0f;
            for (int k = 0; k < seqLenK; k++)
            {
                float e = XMath.Exp(scores[h, q, k] - maxVal);
                weights[h, q, k] = e;
                sum += e;
            }

            float inv = 1.0f / XMath.Max(sum, 1e-20f);

            for (int k = 0; k < seqLenK; k++)
            {
                weights[h, q, k] *= inv;
            }
        }
        private static void ContentAwareWeightedSumKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> weights, ArrayView3D<float, Stride3D.DenseXY> V_head, ArrayView3D<float, Stride3D.DenseXY> output)
        {
            int h = 0;      // head index
            int i = idx.X;  // query position
            int d = idx.Y;  // dimension
            int seqLenK = (int)weights.Extent.Z;

            float sum = 0f;
            for (int j = 0; j < seqLenK; j++)
            {
                sum += weights[h, i, j] * V_head[h, j, d];
            }

            output[h, i, d] = sum;
        }

        private static void AssembleHeadOutputKernel(Index2D idx, ArrayView3D<float, Stride3D.DenseXY> headOutput, ArrayView2D<float, Stride2D.DenseX> concatenated, int headDim, int startIdx)
        {
            int i = idx.X;  // sequence position
            int j = idx.Y;  // dimension within head
            int h = 0;      // head index

            concatenated[i, startIdx + j] = headOutput[h, i, j];
        }

        #endregion


        public float[,] MultiHeadAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale, bool[,] mask = null)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }

            int headDim = embeddingDim / numHeads;

            var concatenated = new float[seqLenQ, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = new float[seqLenQ, headDim];

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                    }
                }

                var K_head = new float[seqLenK, headDim];
                var V_head = new float[seqLenK, headDim];


                for (int i = 0; i < seqLenK; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                    }
                }
                
                var scores = MatrixMultiplyTranspose(Q_head, K_head);

                var scaledScores = MatrixScale(scores, scale);

                var attnWeights = Softmax(scaledScores, mask);

                var headOutput = MatrixMultiply(attnWeights, V_head);


                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        concatenated[i, startIdx + j] = headOutput[i, j];
                    }
                }
            }

            return concatenated;
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool useDecoderMask = false)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);


            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
                throw new ArgumentException("Embedding dim must be divisible by numHeads");

            var dQ_full = new float[seqLenQ, embeddingDim];
            var dK_full = new float[seqLenK, embeddingDim];
            var dV_full = new float[seqLenK, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                var Q_head = new float[seqLenQ, headDim];
                var dHeadOutput = new float[seqLenQ, headDim];


                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        Q_head[i, j] = Q[i, startIdx + j];
                        dHeadOutput[i, j] = dConcatenated[i, startIdx + j];
                    }
                }

                var K_head = new float[seqLenK, headDim];
                var V_head = new float[seqLenK, headDim];

                for (int i = 0; i < seqLenK; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        K_head[i, j] = K[i, startIdx + j];
                        V_head[i, j] = V[i, startIdx + j];
                    }
                }
                var scores = MatrixMultiplyTranspose(Q_head, K_head);
                var scaledScores = MatrixScale(scores, scale);

                var attnWeights = new float[seqLenQ, seqLenK];


                for (int i = 0; i < seqLenQ; i++)
                {
                    float max = float.NegativeInfinity;

                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            continue;
                        }
                        max = Math.Max(max, scaledScores[i, j]);
                    }

                    float expSum = 0;

                    for (int j = 0; j < seqLenK; j++)
                    {
                        if (useDecoderMask && j > i)
                        {
                            attnWeights[i, j] = 0;
                            continue;
                        }
                        attnWeights[i, j] = MathF.Exp(scaledScores[i, j] - max);
                        expSum += attnWeights[i, j];
                    }
                    for (int j = 0; j < seqLenK; j++)
                    {
                        attnWeights[i, j] = attnWeights[i, j]/(expSum + 1e-10f);
                    }
                }

                var dAttnWeights = MatrixMultiplyTranspose(dHeadOutput, V_head);

                var attnWeightsT = new float[seqLenK, seqLenQ];
                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        attnWeightsT[j, i] = attnWeights[i, j];
                    }
                }

                var dV_head = MatrixMultiply(attnWeightsT, dHeadOutput);

                var dScaledScores = new float[seqLenQ, seqLenK];

                for (int i = 0; i < seqLenQ; i++)
                {
                    float dot = 0;
                    for (int j = 0; j < seqLenK; j++)
                    {
                        dot += attnWeights[i, j] * dAttnWeights[i, j];
                    }
                    for (int j = 0; j < seqLenK; j++)
                    {
                        dScaledScores[i, j] = attnWeights[i, j] * (dAttnWeights[i, j] - dot);
                        if (useDecoderMask && j > i)
                        {
                            dScaledScores[i, j] = 0;
                        }
                    }
                }

                var dScores = MatrixScale(dScaledScores, scale);

                var dQ_head = MatrixMultiply(dScores, K_head);

                var dScoresT = new float[seqLenK, seqLenQ];

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < seqLenK; j++)
                    {
                        dScoresT[j, i] = dScores[i, j];
                    }
                }

                var dK_head = MatrixMultiply(dScoresT, Q_head);

                for (int i = 0; i < seqLenQ; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        dQ_full[i, startIdx + j] += dQ_head[i, j];
                    }
                }

                for (int i = 0; i < seqLenK; i++)
                {
                    for (int j = 0; j < headDim; j++)
                    {
                        dK_full[i, startIdx + j] += dK_head[i, j];
                        dV_full[i, startIdx + j] += dV_head[i, j];
                    }
                }
            }

            return (dQ_full, dK_full, dV_full);
        }

        public void BackpropLinearProjection(float[,] input, float[,] dOutput, float[,] weights, float[,] weightGrad, float[] biasGrad, float[,] dInput)
        {
            int seqLen = input.GetLength(0);
            int embeddingDim = input.GetLength(1);

            // weightGrad += input^T * dOutput  (GPU accelerated)
            // input is [seqLen, embeddingDim], dOutput is [seqLen, embeddingDim]
            // input^T is [embeddingDim, seqLen], result is [embeddingDim, embeddingDim]
            // But we need to transpose input first. We can use:
            // (input^T * dOutput)[k, j] = sum_i input[i, k] * dOutput[i, j]
            // This is the same as MatrixMultiplyTranspose(input^T ... ) - let's just build the transpose
            // Actually: MatrixMultiplyTranspose(A, B) computes A * B^T.
            // We want input^T * dOutput. Note: (input^T * dOutput) = (dOutput^T * input)^T
            // Simpler: just allocate a transposed copy.
            var inputT = new float[embeddingDim, seqLen];
            for (int i = 0; i < seqLen; i++)
                for (int k = 0; k < embeddingDim; k++)
                    inputT[k, i] = input[i, k];

            var wGradContrib = MatrixMultiply(inputT, dOutput);

            // Accumulate into weightGrad
            for (int k = 0; k < embeddingDim; k++)
                for (int j = 0; j < embeddingDim; j++)
                    weightGrad[k, j] += wGradContrib[k, j];

            // biasGrad[j] += sum_i dOutput[i, j]
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embeddingDim; j++)
                    biasGrad[j] += dOutput[i, j];

            // dInput += dOutput * weights^T  (GPU accelerated)
            // dOutput is [seqLen, embeddingDim], weights is [embeddingDim, embeddingDim]
            // MatrixMultiplyTranspose(dOutput, weights) computes dOutput * weights^T
            var dInputContrib = MatrixMultiplyTranspose(dOutput, weights);

            // Accumulate into dInput
            for (int i = 0; i < seqLen; i++)
                for (int k = 0; k < embeddingDim; k++)
                    dInput[i, k] += dInputContrib[i, k];
        }

        public (float[,] output, float[] means, float[] variances, float[,] normalized) LayerNormForward(float[,] input, float[] gamma, float[] beta, float epsilon = 1e-5f)
        {
            int batchSize = input.GetLength(0);
            int features = input.GetLength(1);

            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufBeta = _accelerator.Allocate1D<float>(features);
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufMeans = _accelerator.Allocate1D<float>(batchSize);
            var bufVariances = _accelerator.Allocate1D<float>(batchSize);

            try
            {
                bufIn.CopyFromCPU(input);
                bufGamma.CopyFromCPU(gamma);
                bufBeta.CopyFromCPU(beta);

                _layerNormForwardKernel(new Index1D(batchSize),
                    bufIn.View, bufGamma.View, bufBeta.View,
                    bufOut.View, bufMeans.View, bufVariances.View,
                    epsilon);

                var output = new float[batchSize, features];
                var means = new float[batchSize];
                var variances = new float[batchSize];
                bufOut.CopyToCPU(output);
                bufMeans.CopyToCPU(means);
                bufVariances.CopyToCPU(variances);

                // Compute normalized on CPU from means/variances (avoids extra GPU buffer)
                var normalized = new float[batchSize, features];
                for (int i = 0; i < batchSize; i++)
                {
                    float stdDev = MathF.Sqrt(variances[i] + epsilon);
                    for (int j = 0; j < features; j++)
                    {
                        normalized[i, j] = (input[i, j] - means[i]) / stdDev;
                    }
                }

                return (output, means, variances, normalized);
            }
            finally
            {
                bufIn.Dispose(); bufGamma.Dispose(); bufBeta.Dispose();
                bufOut.Dispose(); bufMeans.Dispose(); bufVariances.Dispose();
            }
        }

        public (float[,] dInput, float[] dGamma, float[] dBeta) LayerNormBackward(float[,] dOut, float[,] normalized, float[] gamma, float[,] input, float[] mean, float[] variance, float epsilon = 1e-5f)
        {
            int batchSize = dOut.GetLength(0);
            int features = dOut.GetLength(1);

            var bufDOut = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufNormalized = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufGamma = _accelerator.Allocate1D<float>(features);
            var bufInput = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufMean = _accelerator.Allocate1D<float>(batchSize);
            var bufVariance = _accelerator.Allocate1D<float>(batchSize);
            var bufDInput = _accelerator.Allocate2DDenseX<float>(new Index2D(batchSize, features));
            var bufDGamma = _accelerator.Allocate1D<float>(features);
            var bufDBeta = _accelerator.Allocate1D<float>(features);

            try
            {
                bufDOut.CopyFromCPU(dOut);
                bufNormalized.CopyFromCPU(normalized);
                bufGamma.CopyFromCPU(gamma);
                bufInput.CopyFromCPU(input);
                bufMean.CopyFromCPU(mean);
                bufVariance.CopyFromCPU(variance);

                // Zero out accumulation buffers
                bufDGamma.CopyFromCPU(new float[features]);
                bufDBeta.CopyFromCPU(new float[features]);
                bufDInput.CopyFromCPU(new float[batchSize, features]);

                _layerNormBackwardKernel(new Index1D(batchSize),
                    bufDOut.View, bufNormalized.View, bufGamma.View,
                    bufInput.View, bufMean.View, bufVariance.View,
                    bufDInput.View, bufDGamma.View, bufDBeta.View,
                    epsilon);

                var dInput = new float[batchSize, features];
                var dGamma = new float[features];
                var dBeta = new float[features];
                bufDInput.CopyToCPU(dInput);
                bufDGamma.CopyToCPU(dGamma);
                bufDBeta.CopyToCPU(dBeta);

                return (dInput, dGamma, dBeta);
            }
            finally
            {
                bufDOut.Dispose(); bufNormalized.Dispose(); bufGamma.Dispose();
                bufInput.Dispose(); bufMean.Dispose(); bufVariance.Dispose();
                bufDInput.Dispose(); bufDGamma.Dispose(); bufDBeta.Dispose();
            }
        }

        public float MatrixSquaredNorm(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            // For reduction operations, it's more efficient to do a simple GPU element-wise
            // square and then reduce on CPU, or just do it on CPU for moderate sizes.
            // Using the existing pattern of allocate-try-finally:
            var bufIn = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                bufIn.CopyFromCPU(matrix);

                // Scale in-place to square: we can't easily do a full reduction in ILGPU
                // with the existing kernel pattern, so copy back and sum on CPU.
                // This still benefits from GPU if used as part of a larger pipeline.
                var data = new float[rows, cols];
                bufIn.CopyToCPU(data);

                float sum = 0;
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        sum += data[i, j] * data[i, j];
                return sum;
            }
            finally
            {
                bufIn.Dispose();
            }
        }

        public void MatrixScaleInPlace(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                buf.CopyFromCPU(matrix);
                _matScaleInPlaceKernel(new Index2D(rows, cols), buf.View, scale);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public void VectorScaleInPlace(float[] vector, float scale)
        {
            int n = vector.Length;

            var buf = _accelerator.Allocate1D<float>(n);
            try
            {
                buf.CopyFromCPU(vector);
                _vecScaleInPlaceKernel(new Index1D(n), buf.View, scale);
                buf.CopyToCPU(vector);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public void MatrixUpdate(float[,] weights, float[,] gradients, float learningRate)
        {
            int rows = weights.GetLength(0);
            int cols = weights.GetLength(1);

            var bufW = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            var bufG = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));
            try
            {
                bufW.CopyFromCPU(weights);
                bufG.CopyFromCPU(gradients);
                _matUpdateKernel(new Index2D(rows, cols), bufW.View, bufG.View, learningRate);
                bufW.CopyToCPU(weights);
            }
            finally
            {
                bufW.Dispose(); bufG.Dispose();
            }
        }

        public void VectorUpdate(float[] weights, float[] gradients, float learningRate)
        {
            int n = weights.Length;

            var bufW = _accelerator.Allocate1D<float>(n);
            var bufG = _accelerator.Allocate1D<float>(n);
            try
            {
                bufW.CopyFromCPU(weights);
                bufG.CopyFromCPU(gradients);
                _vecUpdateKernel(new Index1D(n), bufW.View, bufG.View, learningRate);
                bufW.CopyToCPU(weights);
            }
            finally
            {
                bufW.Dispose(); bufG.Dispose();
            }
        }

    

        public float[,] EmbedTokensWithPosition(float[,] tokenEmbedding, int[] tokenIds, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            int vocabSize = tokenEmbedding.GetLength(0);

            var bufEmb = _accelerator.Allocate2DDenseX<float>(new Index2D(vocabSize, embeddingDim));
            var bufIds = _accelerator.Allocate1D<int>(seqLen);
            var bufPos = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                bufEmb.CopyFromCPU(tokenEmbedding);
                bufIds.CopyFromCPU(tokenIds);
                bufPos.CopyFromCPU(positionalEncoding);

                _embedTokensKernel(new Index2D(seqLen, embeddingDim), bufEmb.View, bufIds.View, bufPos.View, bufOut.View);

                var result = new float[seqLen, embeddingDim];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufEmb.Dispose(); bufIds.Dispose(); bufPos.Dispose(); bufOut.Dispose();
            }
        }

        public float[,] AddBiasAndPositionalEncoding(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            var bufProj = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufBias = _accelerator.Allocate1D<float>(embeddingDim);
            var bufPos = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufOut = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                bufProj.CopyFromCPU(projected);
                bufBias.CopyFromCPU(bias);
                bufPos.CopyFromCPU(positionalEncoding);

                _addBiasPosKernel(new Index2D(seqLen, embeddingDim), bufProj.View, bufBias.View, bufPos.View, bufOut.View);

                var result = new float[seqLen, embeddingDim];
                bufOut.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufProj.Dispose(); bufBias.Dispose(); bufPos.Dispose(); bufOut.Dispose();
            }
        }

        public (float loss, float[,] dLogits) CrossEntropyLossAndGradient(float[,] logits, int[] targets, int effectiveLen)
        {
            int outputDim = logits.GetLength(1);
            var dLogits = new float[logits.GetLength(0), outputDim];
            float invLen = 1.0f / effectiveLen;
            float loss = 0;

            // Softmax + cross-entropy has row-level sequential dependencies (max, sum, log)
            // that make it awkward to kernel without shared memory. The compute is O(seqLen × vocabDim)
            // which is small relative to MHA. CPU-side is fine - the GPU transfers would dominate.
            for (int i = 0; i < effectiveLen; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < outputDim; j++)
                    max = Math.Max(max, logits[i, j]);

                float sum = 0;
                var probs = new float[outputDim];
                for (int j = 0; j < outputDim; j++)
                {
                    probs[j] = MathF.Exp(logits[i, j] - max);
                    sum += probs[j];
                }
                for (int j = 0; j < outputDim; j++)
                    probs[j] /= sum;

                int targetToken = targets[i];
                loss += -MathF.Log(probs[targetToken] + 1e-10f);

                for (int j = 0; j < outputDim; j++)
                {
                    dLogits[i, j] = probs[j] * invLen;
                    if (j == targetToken)
                        dLogits[i, j] -= invLen;
                }
            }

            loss /= effectiveLen;
            return (loss, dLogits);
        }

        public (float loss, float[,] dOutput) MSELossAndGradient(float[,] predictions, float[,] targets, int effectiveLen)
        {
            int outputDim = predictions.GetLength(1);
            var dOutput = new float[predictions.GetLength(0), outputDim];
            float invLen = 1.0f / (effectiveLen * outputDim);
            float loss = 0;

            // Same rationale as CrossEntropy - reduction-heavy, CPU is fine.
            for (int i = 0; i < effectiveLen; i++)
            {
                float rowLoss = 0;
                for (int j = 0; j < outputDim; j++)
                {
                    float diff = predictions[i, j] - targets[i, j];
                    rowLoss += diff * diff;
                    dOutput[i, j] = 2.0f * diff * invLen;
                }
                loss += rowLoss;
            }

            loss /= (effectiveLen * outputDim);
            return (loss, dOutput);
        }

        public float[,] BackpropOutputProjection(float[,] dLogits, float[,] input, float[,] weights, float[,] weightGrad, float[] biasGrad, int seqLen, int outputDim, int embeddingDim)
        {
            var dX = MatrixMultiply(dLogits, weights);

            var dLogitsT = new float[outputDim, seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int v = 0; v < outputDim; v++)
                {
                    dLogitsT[v, i] = dLogits[i, v];
                }
            }
            var wGradContrib = MatrixMultiply(dLogitsT, input);

            for (int v = 0; v < outputDim; v++)
            {
                for (int e = 0; e < embeddingDim; e++)
                {
                    weightGrad[v, e] += wGradContrib[v, e];
                }
            }

            for (int i = 0; i < seqLen; i++)
            {
                for (int v = 0; v < outputDim; v++)
                {
                    biasGrad[v] += dLogits[i, v];
                }
            }

            return dX;
        }

        public void BackpropInputProjection(float[,] dX, float[,] continuousInput, float[,] weightGrad, float[] biasGrad, int seqLen, int embeddingDim, int inputFeatureDim)
        {
            var dXT = new float[embeddingDim, seqLen];

            for (int i = 0; i < seqLen; i++)
            {
                for (int e = 0; e < embeddingDim; e++)
                {
                    dXT[e, i] = dX[i, e];
                }
            }

            var wGradContrib = MatrixMultiply(dXT, continuousInput);

            for (int e = 0; e < embeddingDim; e++)
            {
                for (int f = 0; f < inputFeatureDim; f++)
                {
                    weightGrad[e, f] += wGradContrib[e, f];
                }
            }

            for (int i = 0; i < seqLen; i++)
            {
                for (int e = 0; e < embeddingDim; e++)
                {
                    biasGrad[e] += dX[i, e];
                }
            }
        }

        public void AccumulateTokenEmbeddingGrad(float[,] embeddingGrad, float[,] dX, int[] tokenIds, int seqLen, int embeddingDim)
        {
            for (int i = 0; i < seqLen; i++)
            {
                int tokenId = tokenIds[i];

                for (int j = 0; j < embeddingDim; j++)
                {
                    embeddingGrad[tokenId, j] += dX[i, j];
                }
            }
        }

    
        public float VectorSquaredNorm(float[] vector)
        {
            float sum = 0;
            for (int i = 0; i < vector.Length; i++)
            {
                sum += vector[i] * vector[i];
            }
            return sum;
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

      

        public void SigmoidInPlace(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            var buf = _accelerator.Allocate2DDenseX<float>(new Index2D(rows, cols));

            try
            {
                buf.CopyFromCPU(matrix);
                _sigmoidInPlaceKernel(new Index2D(rows, cols), buf.View);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public void Dispose()
        {
            foreach (var v in _matMulCache.Values)
            {
                v.a.Dispose();
                v.b.Dispose();
                v.c.Dispose();
            }
            foreach (var v in _outGradCache.Values)
            {
                v.cost.Dispose();
                v.der.Dispose();
                v.grad.Dispose();
            }
            foreach (var v in _hidGradCache.Values)
            {
                v.pre.Dispose();
                v.der.Dispose();
                v.delta.Dispose();
            }
            foreach (var v in _updWCache.Values)
            {
                v.w.Dispose();
                v.d.Dispose();
                v.pa.Dispose();
            }
            foreach (var v in _updBCache.Values)
            {
                v.b.Dispose();
                v.d.Dispose();
            }
            foreach (var v in _dotTransposedCache.Values)
            {
                v.mat.Dispose();
                v.vec.Dispose();
                v.res.Dispose();
            }
            _accelerator.Dispose();
        }

        private static void ApplyContextTypeEmbeddingKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> contextHidden, ArrayView2D<float, Stride2D.DenseX> typeEmbedding, ArrayView1D<int, Stride1D.Dense> typeIndices, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int i = idx.X;
            int d = idx.Y;
            int t = typeIndices[i];
            result[idx] = contextHidden[idx] + typeEmbedding[t, d];
        }

        private static void ComputeTimeDiffMatrixKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> result, ArrayView1D<float, Stride1D.Dense> keyArrivalTimes)
        {
            int p = idx.X;
            int s = idx.Y;
            result[idx] = XMath.Abs((float)p - keyArrivalTimes[s]);
        }

        private static void MeanPoolRowsKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> hidden, ArrayView1D<int, Stride1D.Dense> storyOffsets, ArrayView1D<int, Stride1D.Dense> storyCounts, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int s = idx.X;
            int d = idx.Y;

            int start = storyOffsets[s];
            int count = storyCounts[s];

            if (count <= 0)
            {
                result[idx] = 0f;
                return;
            }

            float sum = 0f;
            for (int t = start; t < start + count; t++)
                sum += hidden[t, d];

            result[idx] = sum / count;
        }

        private static void EmbedWithBiasAndPositionalKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> projected, ArrayView1D<float, Stride1D.Dense> bias, ArrayView2D<float, Stride2D.DenseX> positionalEncoding, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int j = idx.Y;
            result[idx] = projected[idx] + bias[j] + positionalEncoding[idx];
        }

        private static void ComputeMemoryAttentionScoresKernel(Index1D s, ArrayView2D<float, Stride2D.DenseX> priceHidden, ArrayView2D<float, Stride2D.DenseX> contextHidden, ArrayView1D<float, Stride1D.Dense> scores)
        {
            int embDim = (int)priceHidden.Extent.Y;
            int lastPos = (int)priceHidden.Extent.X - 1;

            float dot = 0f;
            for (int d = 0; d < embDim; d++)
                dot += priceHidden[lastPos, d] * contextHidden[s, d];

            scores[s] = dot;
        }

        private static void ProjectOutputBatchKernel(Index2D idx, ArrayView2D<float, Stride2D.DenseX> hidden, ArrayView2D<float, Stride2D.DenseX> outputProjection, ArrayView1D<float, Stride1D.Dense> outputBias, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int i = idx.X;
            int j = idx.Y;
            int embDim = (int)hidden.Extent.Y;

            float sum = outputBias[j];
            for (int k = 0; k < embDim; k++)
                sum += outputProjection[j, k] * hidden[i, k];

            result[idx] = sum;
        }

        public void ApplyContextTypeEmbedding(float[,] contextHidden, float[,] typeEmbedding, int[] typeIndices)
        {
            int n = contextHidden.GetLength(0);
            int embDim = contextHidden.GetLength(1);

            var bufContext = _accelerator.Allocate2DDenseX<float>(new Index2D(n, embDim));
            var bufType = _accelerator.Allocate2DDenseX<float>(new Index2D(typeEmbedding.GetLength(0), embDim));
            var bufIndices = _accelerator.Allocate1D<int>(n);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(n, embDim));

            try
            {
                bufContext.CopyFromCPU(contextHidden);
                bufType.CopyFromCPU(typeEmbedding);
                bufIndices.CopyFromCPU(typeIndices);

                _applyContextTypeEmbeddingKernel(new Index2D(n, embDim), bufContext.View, bufType.View, bufIndices.View, bufResult.View);

                bufResult.CopyToCPU(contextHidden);
            }
            finally
            {
                bufContext.Dispose();
                bufType.Dispose();
                bufIndices.Dispose();
                bufResult.Dispose();
            }
        }

        public float[,] ComputeTimeDiffMatrix(int priceSeqLen, float[] keyArrivalTimes)
        {
            int numKeys = keyArrivalTimes.Length;

            var bufKeys = _accelerator.Allocate1D<float>(numKeys);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(priceSeqLen, numKeys));

            try
            {
                bufKeys.CopyFromCPU(keyArrivalTimes);

                _computeTimeDiffMatrixKernel(new Index2D(priceSeqLen, numKeys), bufResult.View, bufKeys.View);

                var result = new float[priceSeqLen, numKeys];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufKeys.Dispose();
                bufResult.Dispose();
            }
        }

        public float[,] MeanPoolRows(float[,] hidden, int[] storyOffsets, int[] storyCounts, int numStories, int embeddingDim)
        {
            var bufHidden = _accelerator.Allocate2DDenseX<float>(new Index2D(hidden.GetLength(0), embeddingDim));
            var bufOffsets = _accelerator.Allocate1D<int>(numStories);
            var bufCounts = _accelerator.Allocate1D<int>(numStories);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(numStories, embeddingDim));

            try
            {
                bufHidden.CopyFromCPU(hidden);
                bufOffsets.CopyFromCPU(storyOffsets);
                bufCounts.CopyFromCPU(storyCounts);

                _meanPoolRowsKernel(new Index2D(numStories, embeddingDim), bufHidden.View, bufOffsets.View, bufCounts.View, bufResult.View);

                var result = new float[numStories, embeddingDim];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufHidden.Dispose();
                bufOffsets.Dispose();
                bufCounts.Dispose();
                bufResult.Dispose();
            }
        }

        public float[,] EmbedWithBiasAndPositional(float[,] projected, float[] bias, float[,] positionalEncoding, int seqLen, int embeddingDim)
        {
            var bufProj = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufBias = _accelerator.Allocate1D<float>(embeddingDim);
            var bufPos = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embeddingDim));

            try
            {
                bufProj.CopyFromCPU(projected);
                bufBias.CopyFromCPU(bias);
                bufPos.CopyFromCPU(positionalEncoding);

                _embedWithBiasAndPositionalKernel(new Index2D(seqLen, embeddingDim), bufProj.View, bufBias.View, bufPos.View, bufResult.View);

                var result = new float[seqLen, embeddingDim];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufProj.Dispose();
                bufBias.Dispose();
                bufPos.Dispose();
                bufResult.Dispose();
            }
        }

        public float[] ComputeMemoryAttentionScores(float[,] priceHidden, int lastPos, float[,] contextHidden, int totalCtx, float scale)
        {
            int embDim = priceHidden.GetLength(1);

            var bufPrice = _accelerator.Allocate2DDenseX<float>(new Index2D(priceHidden.GetLength(0), embDim));
            var bufContext = _accelerator.Allocate2DDenseX<float>(new Index2D(totalCtx, embDim));
            var bufScores = _accelerator.Allocate1D<float>(totalCtx);

            try
            {
                bufPrice.CopyFromCPU(priceHidden);
                bufContext.CopyFromCPU(contextHidden);

                _computeMemoryAttentionScoresKernel(new Index1D(totalCtx), bufPrice.View, bufContext.View, bufScores.View);

                var scores = new float[totalCtx];
                bufScores.CopyToCPU(scores);

                // Apply scale on CPU (simple element-wise)
                for (int i = 0; i < totalCtx; i++)
                    scores[i] *= scale;

                return scores;
            }
            finally
            {
                bufPrice.Dispose();
                bufContext.Dispose();
                bufScores.Dispose();
            }
        }

        public float[,] ProjectOutputBatch(float[,] hidden, float[,] outputProjection, float[] outputBias, int seqLen, int outputDim)
        {
            int embDim = hidden.GetLength(1);

            var bufHidden = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, embDim));
            var bufProj = _accelerator.Allocate2DDenseX<float>(new Index2D(outputDim, embDim));
            var bufBias = _accelerator.Allocate1D<float>(outputDim);
            var bufResult = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLen, outputDim));

            try
            {
                bufHidden.CopyFromCPU(hidden);
                bufProj.CopyFromCPU(outputProjection);
                bufBias.CopyFromCPU(outputBias);

                _projectOutputBatchKernel(new Index2D(seqLen, outputDim), bufHidden.View, bufProj.View, bufBias.View, bufResult.View);

                var result = new float[seqLen, outputDim];
                bufResult.CopyToCPU(result);
                return result;
            }
            finally
            {
                bufHidden.Dispose();
                bufProj.Dispose();
                bufBias.Dispose();
                bufResult.Dispose();
            }
        }

        public float[,] FFNForwardBatch(float[,] input, int seqLen, int outputDim, Func<float[], float[]> forwardPassFn)
        {
            // FFN with arbitrary Func cannot be GPU-accelerated (function pointer not supported)
            // Fall back to CPU implementation
            var result = new float[seqLen, outputDim];

            for (int i = 0; i < seqLen; i++)
            {
                var row = new float[input.GetLength(1)];
                for (int j = 0; j < input.GetLength(1); j++)
                    row[j] = input[i, j];

                var outRow = forwardPassFn(row);

                for (int j = 0; j < outputDim; j++)
                    result[i, j] = outRow[j];
            }

            return result;
        }

        #region ContentAwareDecayForward
        public (float[,,] decayBias, ContentAwareDecayCache cache) ContentAwareDecayForward(float[,] queryEmbeddings, float[,] keyEmbeddings, float[,] timeDiffs, float[] keyTimesFromRef, ContentAwareDecayNetwork network, bool isTraining = false, Random dropoutRng = null)
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

            bool useMemAttnDrop = isTraining && network.MemoryAttentionDropout > 0 && dropoutRng != null;
            bool useMLPDrop = isTraining && network.MLPDropout > 0 && dropoutRng != null;
            int dropoutSeed = dropoutRng?.Next() ?? 12345;

            int flatSize = queryLen * keyLen;

            var bufQueryEmb = _accelerator.Allocate2DDenseX<float>(new Index2D(queryLen, contentDim));
            var bufKeyEmb = _accelerator.Allocate2DDenseX<float>(new Index2D(keyLen, contentDim));
            var bufTimeDiffs = _accelerator.Allocate2DDenseX<float>(new Index2D(queryLen, keyLen));
            var bufKeyTimes = _accelerator.Allocate1D<float>(keyLen);
            var bufQProjW = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, projDim, contentDim));
            var bufQProjB = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, projDim));
            var bufKProjW = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, projDim, contentDim));
            var bufKProjB = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, projDim));
            var bufTimeLogFreq = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, numBases));
            var bufTimeProj = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, projDim, rawDim));
            var bufTimeProjB = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, projDim));
            var bufMemAttnOutW = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, projDim, projDim));
            var bufMemAttnOutB = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, projDim));
            var bufW1 = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, hiddenDim, mlpInputDim));
            var bufB1 = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, hiddenDim));
            var bufW2 = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads, hiddenDim));
            var bufB2 = _accelerator.Allocate1D<float>(numHeads);
            var bufLogBaseRate = _accelerator.Allocate1D<float>(numHeads);
            var bufQueryProj = _accelerator.Allocate2DDenseX<float>(new Index2D(numHeads * projDim, queryLen));
            var bufKeyProj = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, projDim));
            var bufTimeRaw = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, rawDim));
            var bufTimeEnc = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, projDim));
            var bufMemAttnQ = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, projDim));
            var bufMemAttnK = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, projDim));
            var bufMemAttnScores = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, keyLen));
            var bufMemAttnWeights = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, keyLen));
            var bufMemAttnDropMask = useMemAttnDrop
                ? _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, keyLen)) : null;
            var bufMemAttnOut = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, projDim));
            var bufRefinedKey = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, keyLen, projDim));
            var bufMLPInput = _accelerator.Allocate3DDenseXY<float>(new Index3D(flatSize, numHeads, mlpInputDim));
            var bufMLPPreAct = _accelerator.Allocate3DDenseXY<float>(new Index3D(flatSize, numHeads, hiddenDim));
            var bufMLPHidden = _accelerator.Allocate3DDenseXY<float>(new Index3D(flatSize, numHeads, hiddenDim));
            var bufMLPDropMask = useMLPDrop
                ? _accelerator.Allocate3DDenseXY<float>(new Index3D(flatSize, numHeads, hiddenDim)) : null;
            var bufGateLogits = _accelerator.Allocate3DDenseXY<float>(new Index3D(queryLen, keyLen, numHeads));
            var bufGates = _accelerator.Allocate3DDenseXY<float>(new Index3D(queryLen, keyLen, numHeads));
            var bufDecayBias = _accelerator.Allocate3DDenseXY<float>(new Index3D(queryLen, keyLen, numHeads));

            try
            {
                bufQueryEmb.CopyFromCPU(queryEmbeddings);
                bufKeyEmb.CopyFromCPU(keyEmbeddings);
                bufTimeDiffs.CopyFromCPU(timeDiffs);
                bufKeyTimes.CopyFromCPU(keyTimesFromRef);
                bufQProjW.CopyFromCPU(network.QueryProjection);
                bufQProjB.CopyFromCPU(network.QueryProjectionBias);
                bufKProjW.CopyFromCPU(network.KeyProjection);
                bufKProjB.CopyFromCPU(network.KeyProjectionBias);
                bufTimeLogFreq.CopyFromCPU(network.TimeLogFreq);
                bufTimeProj.CopyFromCPU(network.TimeProj);
                bufTimeProjB.CopyFromCPU(network.TimeProjBias);
                bufMemAttnOutW.CopyFromCPU(network.MemAttnOutputW);
                bufMemAttnOutB.CopyFromCPU(network.MemAttnOutputB);
                bufW1.CopyFromCPU(network.W1);
                bufB1.CopyFromCPU(network.B1);
                bufW2.CopyFromCPU(network.W2);
                bufB2.CopyFromCPU(network.B2);
                bufLogBaseRate.CopyFromCPU(network.LogBaseDecayRate);

                _decayProjectQueriesKernel(new Index2D(numHeads, queryLen),
                    bufQueryEmb.View, bufQProjW.View, bufQProjB.View, bufQueryProj.View);
                _decayProjectKeysKernel(new Index2D(numHeads, keyLen),
                    bufKeyEmb.View, bufKProjW.View, bufKProjB.View, bufKeyProj.View);

                _decayTimeEncodingRawKernel(new Index2D(numHeads, keyLen),
                    bufKeyTimes.View, bufTimeLogFreq.View, bufTimeRaw.View);
                _decayTimeEncodingProjKernel(new Index2D(numHeads, keyLen),
                    bufTimeRaw.View, bufTimeProj.View, bufTimeProjB.View, bufTimeEnc.View);

                _decayMemAttnQKInputKernel(new Index2D(numHeads, keyLen),
                    bufKeyProj.View, bufTimeEnc.View, bufMemAttnQ.View);
                _decayMemAttnQKInputKernel(new Index2D(numHeads, keyLen),
                    bufKeyProj.View, bufTimeEnc.View, bufMemAttnK.View);

                float memScale = 1.0f / MathF.Sqrt(projDim);
                _decayMemAttnScoresKernel(new Index2D(numHeads, keyLen),
                    bufMemAttnQ.View, bufMemAttnK.View, bufMemAttnScores.View, memScale);

                _decayMemAttnSoftmaxKernel(new Index2D(numHeads, keyLen),
                    bufMemAttnScores.View, bufMemAttnWeights.View);

                if (useMemAttnDrop)
                {
                    float keepProb = 1.0f - network.MemoryAttentionDropout;
                    float scale = 1.0f / keepProb;
                    _decayMemAttnDropoutKernel(new Index3D(numHeads, keyLen, keyLen),
                        bufMemAttnWeights.View, bufMemAttnDropMask.View,
                        bufMemAttnWeights.View, keepProb, scale, dropoutSeed);
                }

                _decayMemAttnWeightedSumKernel(new Index2D(numHeads, keyLen),
                    bufMemAttnWeights.View, bufKeyProj.View, bufMemAttnOut.View);

                // FIX: pass bufKeyProj.View so the kernel adds the residual connection
                _decayMemAttnOutputProjKernel(new Index2D(numHeads, keyLen),
                    bufMemAttnOut.View,
                    bufMemAttnOutW.View,
                    bufMemAttnOutB.View,
                    bufKeyProj.View,
                    bufRefinedKey.View);

                var queryProj3D = new float[numHeads, queryLen, projDim];
                var queryProjFlat = new float[numHeads * projDim, queryLen];
                bufQueryProj.CopyToCPU(queryProjFlat);
                for (int h = 0; h < numHeads; h++)
                    for (int q = 0; q < queryLen; q++)
                        for (int p = 0; p < projDim; p++)
                            queryProj3D[h, q, p] = queryProjFlat[h * projDim + p, q];

                var bufQueryProj3D = _accelerator.Allocate3DDenseXY<float>(new Index3D(numHeads, queryLen, projDim));
                bufQueryProj3D.CopyFromCPU(queryProj3D);

                _decayMLPInputKernel(new Index3D(flatSize, numHeads, 1),
                    bufQueryProj3D.View, bufRefinedKey.View, bufTimeDiffs.View, bufMLPInput.View);

                _decayMLPHiddenKernel(new Index3D(flatSize, numHeads, hiddenDim),
                    bufMLPInput.View, bufW1.View, bufB1.View, bufMLPPreAct.View, bufMLPHidden.View);

                if (useMLPDrop)
                {
                    float keepProb = 1.0f - network.MLPDropout;
                    float scale = 1.0f / keepProb;
                    _decayMLPDropoutKernel(new Index3D(flatSize, numHeads, hiddenDim),
                        bufMLPHidden.View, bufMLPDropMask.View, keepProb, scale, dropoutSeed + 1);
                }

                _decayMLPOutputKernel(new Index2D(queryLen, keyLen),
                    bufMLPHidden.View, bufW2.View, bufB2.View, bufGateLogits.View, bufGates.View);

                _decayFinalBiasKernel(new Index2D(queryLen, keyLen),
                    bufGates.View, bufTimeDiffs.View, bufLogBaseRate.View, bufDecayBias.View);

                var decayBias = new float[queryLen, keyLen, numHeads];
                bufDecayBias.CopyToCPU(decayBias);

                var cache = new ContentAwareDecayCache
                {
                    QueryEmbeddings = queryEmbeddings,
                    KeyEmbeddings = keyEmbeddings,
                    TimeDiffs = timeDiffs,
                    KeyTimesFromRef = keyTimesFromRef,
                    QueryProj = queryProj3D,
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
                    MemAttnDropoutMask = useMemAttnDrop ? new float[numHeads, keyLen, keyLen] : null,
                    MLPDropoutMask = useMLPDrop ? new float[queryLen, keyLen, numHeads, hiddenDim] : null
                };

                bufKeyProj.CopyToCPU(cache.KeyProj);
                bufTimeRaw.CopyToCPU(cache.TimeRawFeatures);
                bufTimeEnc.CopyToCPU(cache.TimeEncoding);
                bufMemAttnQ.CopyToCPU(cache.MemAttnQInput);
                bufMemAttnK.CopyToCPU(cache.MemAttnKInput);
                bufMemAttnWeights.CopyToCPU(cache.MemAttnWeights);
                bufMemAttnOut.CopyToCPU(cache.MemAttnOutput);
                bufRefinedKey.CopyToCPU(cache.RefinedKey);
                bufGateLogits.CopyToCPU(cache.GateLogits);
                bufGates.CopyToCPU(cache.Gates);

                var mlpInputFlat = new float[flatSize, numHeads, mlpInputDim];
                var mlpPreActFlat = new float[flatSize, numHeads, hiddenDim];
                var mlpHiddenFlat = new float[flatSize, numHeads, hiddenDim];
                bufMLPInput.CopyToCPU(mlpInputFlat);
                bufMLPPreAct.CopyToCPU(mlpPreActFlat);
                bufMLPHidden.CopyToCPU(mlpHiddenFlat);

                for (int qi = 0; qi < queryLen; qi++)
                    for (int si = 0; si < keyLen; si++)
                    {
                        int flatIdx = qi * keyLen + si;
                        for (int h = 0; h < numHeads; h++)
                        {
                            for (int k = 0; k < mlpInputDim; k++)
                                cache.MLPInput[qi, si, h, k] = mlpInputFlat[flatIdx, h, k];
                            for (int j = 0; j < hiddenDim; j++)
                            {
                                cache.MLPHiddenPreAct[qi, si, h, j] = mlpPreActFlat[flatIdx, h, j];
                                cache.MLPHidden[qi, si, h, j] = mlpHiddenFlat[flatIdx, h, j];
                            }
                        }
                    }

                if (useMemAttnDrop)
                    bufMemAttnDropMask.CopyToCPU(cache.MemAttnDropoutMask);

                if (useMLPDrop)
                {
                    var mlpDropMaskFlat = new float[flatSize, numHeads, hiddenDim];
                    bufMLPDropMask.CopyToCPU(mlpDropMaskFlat);
                    for (int qi = 0; qi < queryLen; qi++)
                        for (int si = 0; si < keyLen; si++)
                        {
                            int flatIdx = qi * keyLen + si;
                            for (int h = 0; h < numHeads; h++)
                                for (int j = 0; j < hiddenDim; j++)
                                    cache.MLPDropoutMask[qi, si, h, j] = mlpDropMaskFlat[flatIdx, h, j];
                        }
                }

                bufQueryProj3D.Dispose();
                return (decayBias, cache);
            }
            finally
            {
                bufQueryEmb.Dispose();
                bufKeyEmb.Dispose();
                bufTimeDiffs.Dispose();
                bufKeyTimes.Dispose();
                bufQProjW.Dispose();
                bufQProjB.Dispose();
                bufKProjW.Dispose();
                bufKProjB.Dispose();
                bufTimeLogFreq.Dispose();
                bufTimeProj.Dispose();
                bufTimeProjB.Dispose();
                bufMemAttnOutW.Dispose();
                bufMemAttnOutB.Dispose();
                bufW1.Dispose();
                bufB1.Dispose();
                bufW2.Dispose();
                bufB2.Dispose();
                bufLogBaseRate.Dispose();
                bufQueryProj.Dispose();
                bufKeyProj.Dispose();
                bufTimeRaw.Dispose();
                bufTimeEnc.Dispose();
                bufMemAttnQ.Dispose();
                bufMemAttnK.Dispose();
                bufMemAttnScores.Dispose();
                bufMemAttnWeights.Dispose();
                bufMemAttnDropMask?.Dispose();
                bufMemAttnOut.Dispose();
                bufRefinedKey.Dispose();
                bufMLPInput.Dispose();
                bufMLPPreAct.Dispose();
                bufMLPHidden.Dispose();
                bufMLPDropMask?.Dispose();
                bufGateLogits.Dispose();
                bufGates.Dispose();
                bufDecayBias.Dispose();
            }
        }

        #endregion

        #region Content aware cross attention forward 
        public float[,] ContentAwareCrossAttentionForward(float[,] Q, float[,] K, float[,] V, int numHeads, float scale,  float[,,] decayBias, out float[][,] attentionWeights, out float[][,] scoresPreSoftmax)
        {
            int seqLenQ = Q.GetLength(0);
            int seqLenK = K.GetLength(0);
            int embeddingDim = Q.GetLength(1);
            int headDim = embeddingDim / numHeads;

            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dim must be divisible by numHeads");
            }

            attentionWeights = new float[numHeads][,];
            scoresPreSoftmax = new float[numHeads][,];

            var concatenated = new float[seqLenQ, embeddingDim];


            var bufQ = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));
            var bufK = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufV = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenK, embeddingDim));
            var bufConcatenated = _accelerator.Allocate2DDenseX<float>(new Index2D(seqLenQ, embeddingDim));


            var bufQ_head = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, headDim));
            var bufK_head = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenK, headDim));
            var bufV_head = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenK, headDim));
            var bufScores = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));
            var bufScoresWithBias = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));
            var bufWeights = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));
            var bufHeadOutput = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, headDim));
            var bufDecayBias = _accelerator.Allocate3DDenseXY<float>(new Index3D(1, seqLenQ, seqLenK));

            try
            {

                bufQ.CopyFromCPU(Q);
                bufK.CopyFromCPU(K);
                bufV.CopyFromCPU(V);


                for (int head = 0; head < numHeads; head++)
                {
                    int startIdx = head * headDim;

                    bufWeights.MemSetToZero();

                    _extractHeadQKVKernel(new Index2D(seqLenQ, headDim), bufQ.View, bufQ_head.View, headDim, startIdx);
                    _extractHeadQKVKernel(new Index2D(seqLenK, headDim), bufK.View, bufK_head.View, headDim, startIdx);
                    _extractHeadQKVKernel(new Index2D(seqLenK, headDim), bufV.View, bufV_head.View, headDim, startIdx);


                    _contentAwareScoresKernel(new Index2D(seqLenQ, seqLenK), bufQ_head.View, bufK_head.View, bufScores.View, scale);


                    if (decayBias != null)
                    {
                        var decayBiasHead = new float[1, seqLenQ, seqLenK];
                        for (int i = 0; i < seqLenQ; i++)
                            for (int j = 0; j < seqLenK; j++)
                                decayBiasHead[0, i, j] = decayBias[i, j, head];

                        bufDecayBias.CopyFromCPU(decayBiasHead);


                        _addDecayBiasKernel(new Index2D(seqLenQ, seqLenK), bufScores.View, bufDecayBias.View, bufScoresWithBias.View);
                    }
                    else
                    {

                        var scores = new float[1, seqLenQ, seqLenK];

                        bufScores.CopyTo(bufScoresWithBias);

                    }
                    //_contentAwareSoftmaxKernel(seqLenQ, bufScoresWithBias.View, bufWeights.View);
                    // _contentAwareSoftmaxKernel(new Index2D(seqLenQ, seqLenK), bufScoresWithBias.View, bufWeights.View);
                    _contentAwareSoftmaxKernel(new Index1D(seqLenQ), bufScoresWithBias.View, bufWeights.View);
                    _contentAwareWeightedSumKernel(new Index2D(seqLenQ, headDim), bufWeights.View, bufV_head.View, bufHeadOutput.View);

                    _assembleHeadOutputKernel(new Index2D(seqLenQ, headDim), bufHeadOutput.View, bufConcatenated.View, headDim, startIdx);


                    var weightsHead = new float[1, seqLenQ, seqLenK];
                    var scoresHead = new float[1, seqLenQ, seqLenK];

                    bufWeights.CopyToCPU(weightsHead);
                    bufScoresWithBias.CopyToCPU(scoresHead);


                    attentionWeights[head] = new float[seqLenQ, seqLenK];
                    scoresPreSoftmax[head] = new float[seqLenQ, seqLenK];

                    for (int i = 0; i < seqLenQ; i++)
                    {
                        for (int j = 0; j < seqLenK; j++)
                        {
                            attentionWeights[head][i, j] = weightsHead[0, i, j];
                            scoresPreSoftmax[head][i, j] = scoresHead[0, i, j];
                        }
                    }
                }

                bufConcatenated.CopyToCPU(concatenated);

                return concatenated;
            }
            finally
            {
                bufQ.Dispose();
                bufK.Dispose();
                bufV.Dispose();
                bufConcatenated.Dispose();
                bufQ_head.Dispose();
                bufK_head.Dispose();
                bufV_head.Dispose();
                bufScores.Dispose();
                bufScoresWithBias.Dispose();
                bufWeights.Dispose();
                bufHeadOutput.Dispose();
                bufDecayBias.Dispose();
            }
        }
        #endregion

        public float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V,  float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, int PriceEmbeddingDim,  int PriceNumHeads, bool isTraining = false, Random dropoutRng = null)
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
                var (bias, decayCache) = ContentAwareDecayForward(
                    queryEmbeddings,
                    keyEmbeddings,
                    timeDiffs,
                    keyTimesFromRef,
                    block.DecayNetwork,
                    isTraining,
                    dropoutRng
                );
                decayBias = bias;
                bc.DecayCache = decayCache;
            }

            float[][,] attentionWeights;
            float[][,] scoresPreSoftmax;

            var output = ContentAwareCrossAttentionForward(
                Q,
                K,
                V,
                nh,
                scale,
                decayBias,
                out attentionWeights,
                out scoresPreSoftmax
            );

            bc.CrossAttentionWeights = attentionWeights;
            bc.CrossScoresPreSoftmax = scoresPreSoftmax;

            return output;
        }

      


        public void Matrix3DScaleInPlace(float[,,] matrix, float scale)
        {
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);

            var buf = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));
            try
            {
                buf.CopyFromCPU(matrix);
                _mat3DScaleInPlaceKernel(new Index3D(d0, d1, d2), buf.View, scale);
                buf.CopyToCPU(matrix);
            }
            finally
            {
                buf.Dispose();
            }
        }

        public float MatrixSquaredNorm3D(float[,,] matrix)
        {
            int d0 = matrix.GetLength(0);
            int d1 = matrix.GetLength(1);
            int d2 = matrix.GetLength(2);

            // Mirrors the existing MatrixSquaredNorm pattern (line ~1639):
            // ILGPU doesn't expose a single-pass parallel reduction with the existing kernel
            // convention, so we upload to verify transfer overhead is accounted for in the
            // pipeline, then copy back and reduce on the CPU - consistent with the 2D version.
            var buf = _accelerator.Allocate3DDenseXY<float>(new Index3D(d0, d1, d2));
            try
            {
                buf.CopyFromCPU(matrix);
                var data = new float[d0, d1, d2];
                buf.CopyToCPU(data);

                float sum = 0;
                for (int i = 0; i < d0; i++)
                    for (int j = 0; j < d1; j++)
                        for (int k = 0; k < d2; k++)
                            sum += data[i, j, k] * data[i, j, k];
                return sum;
            }
            finally
            {
                buf.Dispose();
            }
        }

        public (float[,] dQ, float[,] dK, float[,] dV) MultiHeadAttentionBackward(float[,] Q, float[,] K, float[,] V, float[,] dConcatenated, int numHeads, float scale, bool[,] mask)
        {
            throw new NotImplementedException();
        }

        public float[,] ContentAwareCrossAttentionWithCache(float[,] Q, float[,] K, float[,] V, float[,] timeDiffs, float[] keyTimesFromRef, float[,] queryEmbeddings, float[,] keyEmbeddings, TacamtBlock block, BlockCache bc, int PriceEmbeddingDim, int PriceNumHeads, bool enableDecayBias = true, bool isTraining = false, Random dropoutRng = null)
        {
            throw new NotImplementedException();
        }

        public float[] ProjectGlobalFeatures(float[] globalFeatures, float[,] projection, float[] bias)
        {
            throw new NotImplementedException();
        }

        public float[,] EmbedTokenIds(int[] tokenIds, float[,] embedding, int embeddingDim)
        {
            throw new NotImplementedException();
        }

        public float[] MeanPoolRows(float[,] matrix)
        {
            throw new NotImplementedException();
        }

        public (float[,] contextHidden, float[] contextTimes, int numGlobal, int numNews) BuildMmtacContext(float[,] newsHidden, float[] newsTimes, float[] globalToken, float[,] contextTypeEmbedding)
        {
            throw new NotImplementedException();
        }

        public (float[,] regression, float[,] range, float[,] quality, float[,] direction, float[,] midDirection, float[,] confidence, float[,] regressionLogits, float[] rangeLogits, float[] qualityLogits) ProjectMmtacOutputHeads(float[,] hidden, float[,] regressionProjection, float[] regressionBias, float[,] rangeProjection, float[] rangeBias, float[,] qualityProjection, float[] qualityBias, float[,] directionProjection, float[] directionBias, float[,] midDirectionProjection, float[] midDirectionBias, float[,] confidenceProjection, float[] confidenceBias, bool useConfidenceHead)
        {
            throw new NotImplementedException();
        }

        public float[] SoftmaxVector(float[] scores)
        {
            throw new NotImplementedException();
        }

        public (float[,] dQ, float[,] dK, float[,] dV, float[,,] dDecayBias) BackpropTimeDecayedAttention(float[,] q, float[,] k, float[,] v, float[,] dOutput, float[][,] attentionWeights, float[,] timeDiffs, int embeddingDim, int numHeads)
        {
            throw new NotImplementedException();
        }

        public string[] PreTokenize(string text)
        {
            throw new NotImplementedException();
        }

        public Dictionary<string, int> GetWordFrequencies(string[] texts, bool lowerCase)
        {
            throw new NotImplementedException();
        }

        public HashSet<string> BuildCharacterVocabulary(Dictionary<string, int> wordFreqs)
        {
            throw new NotImplementedException();
        }

        public List<string> ApplyMerge(List<string> word, string left, string right)
        {
            throw new NotImplementedException();
        }

        public List<int> EncodeWord(string word, Dictionary<(string, string), int> mergePriority, Dictionary<string, int> vocabToId, int unkTokenId)
        {
            throw new NotImplementedException();
        }

        public Dictionary<(string left, string right), int> CountPairFrequencies(Dictionary<List<string>, int> words)
        {
            throw new NotImplementedException();
        }

        public ((string left, string right) pair, int frequency) SelectBestPair(Dictionary<(string left, string right), int> pairCounts, int minFrequency)
        {
            throw new NotImplementedException();
        }

        public Dictionary<List<string>, int> ApplyMergeToVocabulary(Dictionary<List<string>, int> words, string left, string right)
        {
            throw new NotImplementedException();
        }

        public string DecodeTokens(int[] tokenIds, Dictionary<int, string> idToVocab, string unkToken, bool skipSpecialTokens)
        {
            throw new NotImplementedException();
        }

        public int[] PadOrTruncate(int[] tokenIds, int maxLength, bool addSpecialTokens, int padTokenId, int endTokenId)
        {
            throw new NotImplementedException();
        }

        public void ApplyRotaryPositionEmbeddingHeadInPlace(float[,] matrix, int startCol, int headDim, float baseTheta, bool inverse)
        {
            throw new NotImplementedException();
        }

        public void ApplyRotaryPositionEmbeddingInPlace(float[,] matrix, int numHeads, float baseTheta, bool inverse)
        {
            throw new NotImplementedException();
        }
    }
}
