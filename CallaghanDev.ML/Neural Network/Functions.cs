using CallaghanDev.ML.Enums;
using ILGPU.Algorithms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public static class Functions
    {
        public delegate float ActivationFunction(float x);


        public static ActivationFunction GetActivationFunction(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.None:
                    return x => x;
                case ActivationType.Sigmoid:
                    return x =>
                    {
                        float e = XMath.Exp(-x);
                        return 1 / (1.0f + e);
                    };
                case ActivationType.Tanh:
                    return x =>
                    {
                        return XMath.Tanh(x);
                    };
                case ActivationType.Relu:
                    return x =>
                    {
                        return 0 >= x ? 0 : x;
                    };
                case ActivationType.Leakyrelu:
                    return x =>
                    {
                        return 0 >= x ? 0.01f * x : x;
                    };

                case ActivationType.Swish:
                    return x =>
                    {
                        return x*GetActivationFunction(ActivationType.Sigmoid)(x);
                    };
                default: // ActivationType.Sigmoid:
                    return x =>
                    {
                        float e = XMath.Exp(x);
                        return e / (1.0f + e);
                    };
            }
        }

        public static ActivationFunction GetActivationDerivative(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.None:

                    return x => 1;

                case ActivationType.Sigmoid:
                    return x =>
                    {
                        return GetActivationFunction(type)(x)*(1- GetActivationFunction(type)(x));
                    };
                case ActivationType.Tanh:
                    return x =>
                    {
                        return 1 / (XMath.Pow(XMath.Cosh(x), 2));
                    };
                case ActivationType.Relu:
                    return x =>
                    {
                        return x >= 0 ? 1 : 0;
                    };
                case ActivationType.Leakyrelu:
                    return x =>
                    {
                        return 0 >= x ? 0.1f : 1;
                    };
                case ActivationType.Swish:
                    return x =>
                    {
                        float e = XMath.Exp(x);
                        return  (float)(e*(x+e+1))/(MathF.Pow((e+1), 2));
                    };
                default: //ActivationType.Sigmoid:
                    return x =>
                    {
                        float k = XMath.Exp(-x);
                        return k * (1.0f - k);
                    };
            }
        }
    }
}
