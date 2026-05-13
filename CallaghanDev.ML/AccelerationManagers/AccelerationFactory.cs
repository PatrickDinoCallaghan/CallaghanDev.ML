using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.AccelerationManagers
{  
    /// <summary>
    /// Factory that instantiates the correct <see cref="IAccelerationManager"/>
    /// from a <see cref="RuntimeConfig"/>, removing the copy-pasted if/else
    /// chains that previously lived in each model constructor.
    /// </summary>
    public static class AccelerationFactory
    {
        public static IAccelerationManager Create(RuntimeConfig runtime, bool WithLogging = true)
        {
            if (runtime == null)
            {
                throw new ArgumentNullException(nameof(runtime));
            }

            return Create(runtime.AccelerationType, runtime.AccelerationDeviceId, WithLogging);
        }

        public static IAccelerationManager Create(AccelerationType type, int AccelerationDeviceId = 0,  bool WithLogging= false)
        {
            switch (type)
            {
                case AccelerationType.GPU:
                case AccelerationType.CUDA:
                    return WithLogging == true ? new AccelerationGPU(type, AccelerationDeviceId) : new AccelerationGPU(type, AccelerationDeviceId).WithDebugLogging(type.ToString());

                case AccelerationType.CPU:
                    return WithLogging == true ? new AccelerationCPU() : new AccelerationCPU().WithDebugLogging(type.ToString());

                case AccelerationType.MultiThreadCPU:
                    return WithLogging == true ? new AccelerationMutliThreadCPU() : new AccelerationMutliThreadCPU().WithDebugLogging(type.ToString());
                default:
                    throw new NotSupportedException($"Unsupported AccelerationType: {type}");
            }
        }
    }
}
