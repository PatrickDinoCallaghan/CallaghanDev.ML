using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Transformers.Configuration
{
    /// <summary>
    /// Common helper methods for validation.
    /// </summary>
    public abstract class ConfigBase : IValidatableConfig
    {
        public abstract void Validate();

        protected static void RequirePositive(int value, string name)
        {
            if (value <= 0)
            {
                throw new ArgumentException($"{name} must be positive, got {value}.");
            }
        }

        protected static void RequirePositive(float value, string name)
        {
            if (value <= 0f)
            {
                throw new ArgumentException($"{name} must be positive, got {value}.");
            }
        }

        protected static void RequireNonNegative(float value, string name)
        {
            if (value < 0f)
            {
                throw new ArgumentException($"{name} must be non-negative, got {value}.");
            }
        }

        protected static void RequireInRangeZeroToOneExclusiveUpper(float value, string name)
        {
            if (value < 0f || value >= 1f)
            {
                throw new ArgumentException($"{name} must be in [0, 1), got {value}.");
            }
        }

        protected static void RequireDivisible(int numerator, int denominator, string numeratorName, string denominatorName)
        {
            if (denominator == 0 || numerator % denominator != 0)
            {
                throw new ArgumentException($"{numeratorName} ({numerator}) must be divisible by {denominatorName} ({denominator}).");
            }
        }

        protected static void Require(bool condition, string message)
        {
            if (!condition)
            {
                throw new ArgumentException(message);
            }
        }
    }
}
