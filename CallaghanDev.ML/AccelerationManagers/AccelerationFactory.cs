using CallaghanDev.ML.AccelerationManagers.GPU;
using CallaghanDev.ML.Enums;
using CallaghanDev.ML.Transformers.Configuration;
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
        public static IAccelerationManager Create(RuntimeConfig runtime)
        {
            if (runtime == null)
            {
                throw new ArgumentNullException(nameof(runtime));
            }

            switch (runtime.AccelerationType)
            {
                case AccelerationType.GPU:
                case AccelerationType.CUDA:
                    return new AccelerationGPU(runtime.AccelerationType, runtime.AccelerationDeviceId);

                case AccelerationType.CPU:
                    return new AccelerationCPU();

                case AccelerationType.MultiThreadCPU:
                    return new AccelerationMutliThreadCPU();

                default:
                    throw new NotSupportedException($"Unsupported AccelerationType: {runtime.AccelerationType}");
            }
        }
    }
}
