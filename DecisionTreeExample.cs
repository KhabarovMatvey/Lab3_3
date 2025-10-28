using System;
using System.Collections.Generic;
using System.Linq;

namespace DecisionTreeExample
{
    public class DecisionNode
    {
        public int? FeatureIndex;
        public double? Threshold;
        public string? Label;       
        public DecisionNode? Left;
        public DecisionNode? Right;
    }

    public class DecisionTree
    {
        public DecisionNode Root;
        public int MaxDepth = 5;

        public void Fit(List<double[]> X, List<string> y)
        {
            Root = BuildTree(X, y, 0);
        }

        public string Predict(double[] sample)
        {
            var node = Root;
            while (node.Label == null)
            {
                if (sample[node.FeatureIndex.Value] <= node.Threshold.Value)
                    node = node.Left;
                else
                    node = node.Right;
            }
            return node.Label;
        }

        // Построение дерева рекурсивно
        private DecisionNode BuildTree(List<double[]> X, List<string> y, int depth)
        {
            if (y.Distinct().Count() == 1)
                return new DecisionNode { Label = y.First() };

            if (depth >= MaxDepth || X.Count <= 1)
                return new DecisionNode { Label = MostCommonLabel(y) };

            int bestFeature;
            double bestThreshold;
            double bestGain;
            FindBestSplit(X, y, out bestFeature, out bestThreshold, out bestGain);

            if (bestGain == 0)
                return new DecisionNode { Label = MostCommonLabel(y) };

            var (leftX, leftY, rightX, rightY) = SplitDataset(X, y, bestFeature, bestThreshold);

            return new DecisionNode
            {
                FeatureIndex = bestFeature,
                Threshold = bestThreshold,
                Left = BuildTree(leftX, leftY, depth + 1),
                Right = BuildTree(rightX, rightY, depth + 1)
            };
        }

        private void FindBestSplit(List<double[]> X, List<string> y,
                                   out int bestFeature, out double bestThreshold, out double bestGain)
        {
            bestFeature = 0;
            bestThreshold = 0;
            bestGain = 0;
            double baseEntropy = Entropy(y);

            for (int feature = 0; feature < X[0].Length; feature++)
            {
                var thresholds = X.Select(x => x[feature]).Distinct().OrderBy(v => v).ToList();
                foreach (var t in thresholds)
                {
                    var (_, leftY, _, rightY) = SplitDataset(X, y, feature, t);
                    if (leftY.Count == 0 || rightY.Count == 0) continue;

                    double gain = baseEntropy - WeightedEntropy(leftY, rightY);
                    if (gain > bestGain)
                    {
                        bestGain = gain;
                        bestFeature = feature;
                        bestThreshold = t;
                    }
                }
            }
        }

        private (List<double[]>, List<string>, List<double[]>, List<string>)
            SplitDataset(List<double[]> X, List<string> y, int feature, double threshold)
        {
            var leftX = new List<double[]>();
            var rightX = new List<double[]>();
            var leftY = new List<string>();
            var rightY = new List<string>();

            for (int i = 0; i < X.Count; i++)
            {
                if (X[i][feature] <= threshold)
                {
                    leftX.Add(X[i]);
                    leftY.Add(y[i]);
                }
                else
                {
                    rightX.Add(X[i]);
                    rightY.Add(y[i]);
                }
            }

            return (leftX, leftY, rightX, rightY);
        }

        private double WeightedEntropy(List<string> left, List<string> right)
        {
            int total = left.Count + right.Count;
            return (left.Count / (double)total) * Entropy(left)
                 + (right.Count / (double)total) * Entropy(right);
        }

        private double Entropy(List<string> labels)
        {
            var counts = labels.GroupBy(l => l).Select(g => g.Count());
            double entropy = 0.0;
            foreach (var count in counts)
            {
                double p = count / (double)labels.Count;
                entropy -= p * Math.Log(p, 2);
            }
            return entropy;
        }

        private string MostCommonLabel(List<string> y)
        {
            return y.GroupBy(l => l).OrderByDescending(g => g.Count()).First().Key;
        }
    }

}
