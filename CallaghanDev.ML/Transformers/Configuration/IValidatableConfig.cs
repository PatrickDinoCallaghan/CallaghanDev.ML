using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.Configuration
{
    /// <summary>
    /// Base interface for configs that can validate themselves.
    /// </summary>
    public interface IValidatableConfig
    {
        void Validate();
    }

}
