using System;
using System.IO;
using System.Globalization;

namespace NeuralNetworks_1;

class nn3S
{
    private readonly double[,] wih;
    private readonly double[,] who;
    private readonly int inodes;
    private readonly int hnodes;
    private readonly int onodes;

    private readonly double[] hidden_inputs;
    private readonly double[] hidden_outputs;
    private readonly double[] final_inputs;
    private readonly double[] final_outputs;
    private readonly double[] error_hidden;
    private readonly double[] error_output;
    private readonly double[] output_delta;
    private readonly double[] hidden_delta;

    private double learningRate = 0.21;

    public double[,] WIH => wih;
    public double[,] WHO => who;
    public double[] Hidden_inputs => hidden_inputs;
    public double[] Hidden_outputs => hidden_outputs;
    public double[] Final_inputs => final_inputs;
    public double[] Final_outputs => final_outputs;
    public double[] Error_hidden => error_hidden;
    public double[] Error_output => error_output;
    public int InputNodes => inodes;
    public int HiddenNodes => hnodes;
    public int OutputNodes => onodes;
    public double LastLoss { get; private set; }
    public bool HasTrained { get; private set; }

    public double LearningRate
    {
        get => learningRate;
        set
        {
            if (!double.IsFinite(value) || value <= 0)
                throw new ArgumentOutOfRangeException(nameof(value), "Die Learning Rate muss positiv sein.");
            learningRate = value;
        }
    }

    public nn3S(int inodes, int hnodes, int onodes, int? seed = null)
    {
        if (inodes < 1 || hnodes < 1 || onodes < 1)
            throw new ArgumentException("Alle Knotenzahlen müssen positiv sein.");

        this.inodes = inodes;
        this.hnodes = hnodes;
        this.onodes = onodes;

        wih = new double[inodes, hnodes];
        who = new double[hnodes, onodes];
        hidden_inputs = new double[hnodes];
        hidden_outputs = new double[hnodes];
        final_inputs = new double[onodes];
        final_outputs = new double[onodes];
        error_hidden = new double[hnodes];
        error_output = new double[onodes];
        output_delta = new double[onodes];
        hidden_delta = new double[hnodes];

        createWeightMatrizes(seed.HasValue ? new Random(seed.Value) : Random.Shared);
    }

    private void createWeightMatrizes(Random random)
    {
        double inputScale = 1.0 / Math.Sqrt(inodes);
        double hiddenScale = 1.0 / Math.Sqrt(hnodes);

        for (int input = 0; input < inodes; input++)
            for (int hidden = 0; hidden < hnodes; hidden++)
                wih[input, hidden] = (random.NextDouble() * 2.0 - 1.0) * inputScale;

        for (int hidden = 0; hidden < hnodes; hidden++)
            for (int output = 0; output < onodes; output++)
                who[hidden, output] = (random.NextDouble() * 2.0 - 1.0) * hiddenScale;
    }

    public void queryNN(double[] inputs)
    {
        if (inputs.Length != inodes)
            throw new ArgumentException("Das Netz erwartet " + inodes + " Eingabewerte.");

        for (int hidden = 0; hidden < hnodes; hidden++)
        {
            double sum = 0;
            for (int input = 0; input < inodes; input++)
                sum += wih[input, hidden] * inputs[input];
            hidden_inputs[hidden] = sum;
            hidden_outputs[hidden] = 1.0 / (1.0 + Math.Exp(-sum));
        }

        for (int output = 0; output < onodes; output++)
        {
            double sum = 0;
            for (int hidden = 0; hidden < hnodes; hidden++)
                sum += who[hidden, output] * hidden_outputs[hidden];
            final_inputs[output] = sum;
            final_outputs[output] = 1.0 / (1.0 + Math.Exp(-sum));
        }
    }

    public void train(double[] inputs, double[] targets)
    {
        if (targets.Length != onodes)
            throw new ArgumentException("Das Netz erwartet " + onodes + " Zielwerte.");

        queryNN(inputs);
        LastLoss = 0;

        for (int output = 0; output < onodes; output++)
        {
            error_output[output] = targets[output] - final_outputs[output];
            LastLoss += 0.5 * error_output[output] * error_output[output];
            output_delta[output] =
                error_output[output] * final_outputs[output] * (1.0 - final_outputs[output]);
        }

        for (int hidden = 0; hidden < hnodes; hidden++)
        {
            double sum = 0;
            for (int output = 0; output < onodes; output++)
                sum += who[hidden, output] * error_output[output];
            error_hidden[hidden] = sum;
            hidden_delta[hidden] =
                sum * hidden_outputs[hidden] * (1.0 - hidden_outputs[hidden]);
        }

        for (int hidden = 0; hidden < hnodes; hidden++)
            for (int output = 0; output < onodes; output++)
                who[hidden, output] += learningRate * hidden_outputs[hidden] * output_delta[output];

        for (int input = 0; input < inodes; input++)
            for (int hidden = 0; hidden < hnodes; hidden++)
                wih[input, hidden] += learningRate * inputs[input] * hidden_delta[hidden];

        HasTrained = true;
    }

    public void calculateErrors(double[] targets)
    {
        if (targets.Length != onodes)
            throw new ArgumentException("Die Anzahl der Zielwerte passt nicht zum Netz.");

        for (int output = 0; output < onodes; output++)
            error_output[output] = targets[output] - final_outputs[output];

        for (int hidden = 0; hidden < hnodes; hidden++)
        {
            double sum = 0;
            for (int output = 0; output < onodes; output++)
                sum += who[hidden, output] * error_output[output];
            error_hidden[hidden] = sum;
        }
    }

    public void SaveWeights(string path)
    {
        using BinaryWriter writer = new BinaryWriter(File.Create(path));
        writer.Write("NN3S1");
        writer.Write(inodes);
        writer.Write(hnodes);
        writer.Write(onodes);
        writer.Write(learningRate);
        for (int input = 0; input < inodes; input++)
            for (int hidden = 0; hidden < hnodes; hidden++)
                writer.Write(wih[input, hidden]);
        for (int hidden = 0; hidden < hnodes; hidden++)
            for (int output = 0; output < onodes; output++)
                writer.Write(who[hidden, output]);
    }

    public void SaveMatricesAsCsv(string folder, string name)
    {
        SaveMatrixAsCsv(wih, Path.Combine(folder, name + "-WIH.csv"));
        SaveMatrixAsCsv(who, Path.Combine(folder, name + "-WHO.csv"));
    }

    private static void SaveMatrixAsCsv(double[,] matrix, string path)
    {
        using StreamWriter writer = new StreamWriter(path);
        for (int row = 0; row < matrix.GetLength(0); row++)
        {
            for (int column = 0; column < matrix.GetLength(1); column++)
            {
                if (column > 0) writer.Write(",");
                writer.Write(matrix[row, column].ToString("R", CultureInfo.InvariantCulture));
            }
            writer.WriteLine();
        }
    }

    public static nn3S LoadWeights(string path)
    {
        using BinaryReader reader = new BinaryReader(File.OpenRead(path));
        if (reader.ReadString() != "NN3S1")
            throw new InvalidDataException("Das ist keine Gewichtsdatei dieses Programms.");

        int inputs = reader.ReadInt32();
        int hidden = reader.ReadInt32();
        int outputs = reader.ReadInt32();
        if (inputs < 1 || inputs > 10000 || hidden < 1 || hidden > 10000 || outputs < 1 || outputs > 10000)
            throw new InvalidDataException("Ungültige Knotenzahlen in der Gewichtsdatei.");

        nn3S network = new nn3S(inputs, hidden, outputs);
        network.LearningRate = reader.ReadDouble();
        for (int input = 0; input < inputs; input++)
            for (int node = 0; node < hidden; node++)
                network.wih[input, node] = reader.ReadDouble();
        for (int node = 0; node < hidden; node++)
            for (int output = 0; output < outputs; output++)
                network.who[node, output] = reader.ReadDouble();
        network.HasTrained = true;
        return network;
    }
}
