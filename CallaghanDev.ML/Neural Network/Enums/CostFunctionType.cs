﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML.Enums
{
    public enum CostFunctionType
    {
        mse,
        ZeroWeightedMSE,
        binaryCrossEntropy,
        mae,
        categoricalCrossEntropy,
        huberLoss,
    }
}
