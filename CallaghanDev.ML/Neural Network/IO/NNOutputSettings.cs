using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CallaghanDev.ML
{
    public struct NNOutputSettings
    {
        private string _Label;
        public NNOutputSettings(string InLabel)
        {
            _Label = InLabel;
        }
        public string Label { get { return _Label; } }
    }
}
