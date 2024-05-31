using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    /// <summary>
    /// Represents a container for managing a generic asynchronous task.
    /// Provides properties to hold a TaskCompletionSource and its corresponding task.
    /// Includes methods to initialize and reset the task container.
    /// Configured to run task continuations asynchronously.
    /// </summary>
    public class TaskContainer<T>
    {
        public TaskCompletionSource<T> Tcs { get; set; }
        public Task<T> Task { get; set; }

        public TaskContainer()
        {
            Tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            Task = Tcs.Task;
        }

        public void Reset()
        {
            Tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
            Task = Tcs.Task;
        }
    }
}
