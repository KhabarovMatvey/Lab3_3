using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DecisionTreeExample;
using ScottPlot;

namespace DecisionTreeBenchmark
{
    public class Benchmark
    {
        public static void Main()
        {
            int minSamples = 100;   
            int maxSamples = 5000;  
            int step = 100;         

            int smoothWindow = 5;  

            int[] sampleSizes = Enumerable.Range(0, (maxSamples - minSamples) / step + 1)
                                          .Select(i => minSamples + i * step)
                                          .ToArray();

            Console.WriteLine($"Диапазон: {minSamples}–{maxSamples}, шаг: {step}\n");

            List<double> treeTimes = new();
            Random rnd = new();

            foreach (int n in sampleSizes)
            {
                Console.WriteLine($"Тест для {n} элементов...");

                var X = new List<double[]>();
                var y = new List<string>();

                for (int i = 0; i < n; i++)
                {
                    double a = rnd.NextDouble() * 100;
                    double b = rnd.NextDouble() * 100;
                    string label = (a + b > 100) ? "Yes" : "No";
                    X.Add(new double[] { a, b });
                    y.Add(label);
                }

                var tree = new DecisionTree();
                var sw = Stopwatch.StartNew();
                tree.Fit(X, y);
                sw.Stop();

                treeTimes.Add(sw.Elapsed.TotalMilliseconds);
            }

            double[] smoothed = Smooth(treeTimes.ToArray(), smoothWindow);

            var plt = new ScottPlot.Plot();
            double[] xs = Array.ConvertAll(sampleSizes, s => (double)s);

            var line = plt.Add.Scatter(xs, smoothed);
            line.LegendText = "Decision Tree (Fit Time)";
            line.Color = Colors.Blue;
            line.LineWidth = 2.5f;
            line.MarkerShape = MarkerShape.None; 

            plt.Title("Производительность Decision Tree");
            plt.XLabel("Количество элементов");
            plt.YLabel("Время обучения (мс)");
            plt.ShowLegend();
            plt.Grid.IsVisible = true;

            string outputPath = "decision_tree_performance.png";
            plt.SavePng(outputPath, 900, 600);

            Console.WriteLine($"\n✅ График сохранён: {outputPath}");
        }

        static double[] Smooth(double[] values, int window)
        {
            if (window < 2)
                return values;

            double[] result = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                int start = Math.Max(0, i - window / 2);
                int end = Math.Min(values.Length - 1, i + window / 2);
                double avg = 0;
                for (int j = start; j <= end; j++)
                    avg += values[j];
                result[i] = avg / (end - start + 1);
            }
            return result;
        }
    }
}
