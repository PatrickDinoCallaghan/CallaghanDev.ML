using CallaghanDev.ML.Enums;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using System;

namespace CallaghanDev.ML.AccelerationManagers.GPU
{
    public partial class AccelerationGPU : IAccelerationManager, IDisposable
    {
        private const bool AlwaysParallel = false; // Just for testing purposes. Forgot to turn it off.

        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly AccelerationMutliThreadCPU _mutliThreadCPU;

        // Transformer workloads in this project are often many medium-sized operations
        // (for example 60-120 timesteps) rather than a few giant GEMMs. The previous
        // thresholds silently routed those operations back to CPU. Keep the threshold
        // low enough that CUDA/OpenCL is actually used for transformer hot paths, while
        // still avoiding launch overhead for tiny scalar work.
        private const long GPU_ELEMENTWISE_THRESHOLD = 4_096;
        private const long GPU_MATMUL_OP_THRESHOLD = 32_768;

        private static bool ShouldUseGpu(long workUnits, long threshold = GPU_ELEMENTWISE_THRESHOLD)
        {
            if (AlwaysParallel)
            {
                return true;
            }
            return workUnits >= threshold;
        }

        public AccelerationGPU(AccelerationType accelerationType, int deviceIndex = 0)
        {
            _context = Context.Create(builder =>
            {
                builder.EnableAlgorithms();
                builder.AllAccelerators();
            });

            try
            {
                if (accelerationType == AccelerationType.GPU)
                {
                    _accelerator = _context.CreateCLAccelerator(deviceIndex);
                }
                else if (accelerationType == AccelerationType.CUDA)
                {
                    _accelerator = _context.CreateCudaAccelerator(deviceIndex);
                }
                else
                {
                    throw new NotSupportedException($"AccelerationGPU requires GPU or CUDA acceleration type. Got {accelerationType}.");
                }
            }
            catch
            {
                _context.Dispose();
                throw;
            }

            _mutliThreadCPU = new AccelerationMutliThreadCPU();

            InitSharedTensorKernels();
            InitNeuralNetworkKernels();
            InitTransformerCoreKernels();
            InitTransformerTrainingKernels();
            InitTransformerSpecificKernels();
            InitTokenizerAccelerationKernels();
            InitRotaryPositionEmbeddingKernels();
        }

        public void Dispose()
        {
            DisposeSharedTensorBuffers();
            DisposeNeuralNetworkBuffers();
            DisposeTransformerCoreBuffers();
            DisposeTransformerTrainingBuffers();
            DisposeTransformerSpecificBuffers();
            DisposeTokenizerAccelerationBuffers();
            DisposeRotaryPositionEmbeddingBuffers();

            _mutliThreadCPU.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
        }
    }
}
