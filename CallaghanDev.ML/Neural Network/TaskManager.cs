using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    /// <summary>
    /// Manages asynchronous tasks using generic task containers.
    /// Provides methods to create, retrieve, and reset task containers.
    /// Executes tasks asynchronously and handles their results or exceptions.
    /// - Ensures thread-safe management of tasks using a concurrent dictionary.
    /// - Ensures consistent handling of task results and exceptions. 
    /// - Simplifies task management by using a centralized TaskManager to handle task lifecycle.
    /// - Efficiency: Reusing task containers avoids the overhead of creating new tasks repeatedly.
    /// </summary>
    public class TaskManager
    {
        private ConcurrentDictionary<string, object> Containers = new ConcurrentDictionary<string, object>();

        public TaskContainer<T> GetOrCreateContainer<T>(string taskName)
        {
            return (TaskContainer<T>)Containers.GetOrAdd(taskName, _ => new TaskContainer<T>());
        }

        public void SetOrCreateTask<T>(TaskContainer<T> container, Func<T> func)
        {
            if (container.Task.IsCompleted || container.Task.Status == TaskStatus.Faulted || container.Task.Status == TaskStatus.Canceled)
            {
                container.Reset();
            }

            // Run the function and set the result asynchronously
            Task.Run(() =>
            {
                try
                {
                    var result = func();
                    container.Tcs.TrySetResult(result);
                }
                catch (Exception ex)
                {
                    container.Tcs.TrySetException(ex);
                }
            });
        }

        public void WaitForAll(params TaskContainer<object>[] containers)
        {
            Task.WaitAll(containers.Select(c => c.Task).ToArray());
        }


        public void WaitForAll<T>(params TaskContainer<T>[] containers)
        {
            Task.WaitAll(containers.Select(c => c.Task).ToArray());
        }
    }
}
