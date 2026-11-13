using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace NeuralNetworks_1;

class DigitSample
{
    public int Label { get; }
    public byte[] Pixels { get; }

    public DigitSample(int label, byte[] pixels)
    {
        Label = label;
        Pixels = pixels;
    }
}

static class MnistData
{
    public static List<DigitSample> Load(string path)
    {
        List<DigitSample> samples = new List<DigitSample>();
        using StreamReader reader = new StreamReader(path);
        string? line;
        int lineNumber = 0;

        while ((line = reader.ReadLine()) != null)
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] values = line.Split(',');
            if (values.Length != 785)
                throw new InvalidDataException("Zeile " + lineNumber + ": Es werden 1 Label und 784 Pixel erwartet.");

            int label = int.Parse(values[0], CultureInfo.InvariantCulture);
            if (label < 0 || label > 9)
                throw new InvalidDataException("Zeile " + lineNumber + ": Das Label muss 0 bis 9 sein.");

            byte[] pixels = new byte[784];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = byte.Parse(values[i + 1], CultureInfo.InvariantCulture);
            samples.Add(new DigitSample(label, pixels));
        }

        if (samples.Count == 0)
            throw new InvalidDataException("Die Datei enthält keine Bilder.");
        return samples;
    }

    public static void FillInputs(DigitSample sample, double[] inputs)
    {
        if (inputs.Length != 784)
            throw new ArgumentException("MNIST braucht 784 Input-Knoten.");
        for (int i = 0; i < 784; i++)
            inputs[i] = 0.01 + sample.Pixels[i] / 255.0 * 0.99;
    }

    public static void FillTargets(DigitSample sample, double[] targets)
    {
        if (targets.Length != 10)
            throw new ArgumentException("MNIST braucht 10 Output-Knoten.");
        for (int i = 0; i < 10; i++)
            targets[i] = 0.01;
        targets[sample.Label] = 0.99;
    }

    public static int BestOutput(double[] outputs)
    {
        int best = 0;
        for (int i = 1; i < outputs.Length; i++)
            if (outputs[i] > outputs[best])
                best = i;
        return best;
    }
}
