using System.Text;
using System.Text.RegularExpressions;
using MathNet.Symbolics;

namespace CallaghanDev.ML.Interpolation
{
    public class LagrangePolynomial : CallaghanDev.Utilities.Math.Polynomial
    {
        public Func<double, double> FindLagrangePolynomial(List<(double x, double y)> points)
        {
            int n = points.Count;

            // Method to compute the Lagrange basis polynomial ℓ_i(x)
            Func<double, int, double> L = (x, i) =>
            {
                double result = 1.0;
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        result *= (x - points[j].x) / (points[i].x - points[j].x);
                    }
                }
                return result;
            };

            // Method to compute the Lagrange polynomial L(x)
            return (x) =>
            {
                double result = 0.0;
                for (int i = 0; i < n; i++)
                {
                    result += points[i].y * L(x, i);
                }
                return result;
            };
        }
        public string GetLagrangePolynomialString(List<(double x, double y)> points)
        {
            int n = points.Count;
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < n; i++)
            {
                if (i > 0)
                {
                    sb.Append(" + ");
                }
                sb.Append($"{points[i].y} * (");

                bool firstTerm = true;
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        if (!firstTerm)
                        {
                            sb.Append(" * ");
                        }
                        firstTerm = false;
                        sb.Append($"(x - {points[j].x})");
                    }
                }

                sb.Append(") / (");

                firstTerm = true;
                for (int j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        if (!firstTerm)
                        {
                            sb.Append(" * ");
                        }
                        firstTerm = false;
                        sb.Append($"{points[i].x - points[j].x}");
                    }
                }

                sb.Append(")");
            }
            //SimplifyPolynomial
            return (sb.ToString());
        }
    }
}
