using System.Diagnostics;
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace CallaghanDev.ML.AccelerationManagers
{
    public static class AccelerationDebugExtensions
    {
        public static IAccelerationManager WithDebugLogging(this IAccelerationManager inner, string accelerationName)
        {
#if DEBUG
            return DebugAccelerationProxy<IAccelerationManager>.Create(inner, accelerationName);
#else
        return inner;
#endif
        }
    }

    public sealed class DebugAccelerationProxy<T> : DispatchProxy where T : class
    {
        private T _inner = null!;
        private string _accelerationName = "";

        public static T Create(T inner, string accelerationName)
        {
            object proxy = Create<T, DebugAccelerationProxy<T>>();

            var debugProxy = (DebugAccelerationProxy<T>)proxy;
            debugProxy._inner = inner;
            debugProxy._accelerationName = accelerationName;

            return (T)proxy;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
#if DEBUG
            Debug.WriteLine($"Acceleration type: {_accelerationName} | Method: {targetMethod?.Name}");
#endif

            try
            {
                return targetMethod!.Invoke(_inner, args);
            }
            catch (TargetInvocationException ex) when (ex.InnerException != null)
            {
                ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }
    }
}
