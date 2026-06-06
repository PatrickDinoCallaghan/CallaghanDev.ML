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
        
        private static readonly long GPU_ELEMENTWISE_THRESHOLD = GetGpuThreshold("CALLAGHANDEV_ML_GPU_ELEMENTWISE_THRESHOLD", 65_536);
        private static readonly long GPU_MATMUL_OP_THRESHOLD = GetGpuThreshold("CALLAGHANDEV_ML_GPU_MATMUL_OP_THRESHOLD", 262_144);

        private static long GetGpuThreshold(string environmentVariableName, long defaultValue)
        {
            string value = Environment.GetEnvironmentVariable(environmentVariableName);
            if (!string.IsNullOrWhiteSpace(value) && long.TryParse(value, out long parsed) && parsed >= 0)
            {
                return parsed;
            }

            return defaultValue;
        }

        private static bool ShouldUseGpu(long workUnits) => ShouldUseGpu(workUnits, GPU_ELEMENTWISE_THRESHOLD);

        private static bool ShouldUseGpu(long workUnits, long threshold)
        {
            if (AlwaysParallel)
            {
                return true;
            }
            return workUnits >= threshold;
        }
        /*
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
        */

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
                    int count = 0;
                    foreach (CLDevice device in _context.GetCLDevices())
                    {
                        Console.WriteLine($"OpenCL index {count}: {device}");
                        count++;
                    }

                    if (deviceIndex < 0 || deviceIndex >= count)
                    {
                        throw new NotSupportedException($"OpenCL device index {deviceIndex} is not available. Found {count} OpenCL device(s). Remember: AccelerationType.GPU maps to OpenCL, not CUDA.");
                    }

                    _accelerator = _context.CreateCLAccelerator(deviceIndex);
                }
                else if (accelerationType == AccelerationType.CUDA)
                {
                    int count = 0;
                    foreach (CudaDevice device in _context.GetCudaDevices())
                    {
                        Console.WriteLine($"CUDA index {count}: {device}");
                        count++;
                    }

                    if (deviceIndex < 0 || deviceIndex >= count)
                    {
                        throw new NotSupportedException($"CUDA device index {deviceIndex} is not available. Found {count} CUDA device(s). Your NVIDIA RTX A1000 is CUDA index 0.");
                    }

                    _accelerator = _context.CreateCudaAccelerator(deviceIndex);
                }
                else
                {
                    throw new NotSupportedException($"AccelerationGPU requires GPU/OpenCL or CUDA acceleration type. Got {accelerationType}.");
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
            DisposeDeviceResidentCaches();

            _mutliThreadCPU.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
        }
    }
}
