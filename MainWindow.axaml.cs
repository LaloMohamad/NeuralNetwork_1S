using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Platform.Storage;

namespace NeuralNetworks_1;

public partial class MainWindow : Window
{
    private nn3S? network;
    private CancellationTokenSource? cancellation;
    private bool skipRemainingImages;
    private readonly ObservableCollection<nodeRow> rows = new ObservableCollection<nodeRow>();
    private string lastTrainingFile = "network";
    private bool isFashionNetwork;
    private bool loadedWeightsOnly;

    public MainWindow()
    {
        InitializeComponent();
        networkDataGrid.ItemsSource = rows;

        string? dataRoot = Environment.GetEnvironmentVariable("NN_DATA_DIR");
        if (string.IsNullOrWhiteSpace(dataRoot))
            dataRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
                "Softwaretechnik", "Skripte", "alte", "projekte",
                "Neuronale netze", "teil 3");
        string dataFolder = Path.Combine(dataRoot, "mnist db - NN");
        string trainFile = Path.Combine(dataFolder, "mnist_train.txt");
        string queryFile = Path.Combine(dataFolder, "mnist_test_10.txt");
        if (File.Exists(trainFile)) trainingFileTextBox.Text = trainFile;
        if (File.Exists(queryFile)) queryFileTextBox.Text = queryFile;

        Closing += (_, e) =>
        {
            if (cancellation != null)
            {
                cancellation.Cancel();
                e.Cancel = true;
                statusTextBlock.Text = "Lauf wird abgebrochen. Danach kannst du schließen.";
                return;
            }
            if (network?.HasTrained == true && !loadedWeightsOnly)
            {
                try { SaveCurrentWeights(); }
                catch (Exception ex)
                {
                    e.Cancel = true;
                    statusTextBlock.Text = "Gewichte konnten nicht gespeichert werden: " + ex.Message;
                }
            }
        };
    }

    private bool ReadSettings(out int inputs, out int hidden, out int outputs, out int epochs, out double rate)
    {
        bool inputsValid = int.TryParse(inputTextBox.Text, out inputs);
        bool hiddenValid = int.TryParse(hiddenTextBox.Text, out hidden);
        bool outputsValid = int.TryParse(outputTextBox.Text, out outputs);
        bool epochsValid = int.TryParse(epochsTextBox.Text, out epochs);
        bool rateValid = double.TryParse((learningRateTextBox.Text ?? "").Replace(',', '.'),
            NumberStyles.Float, CultureInfo.InvariantCulture, out rate);
        bool valid = inputsValid && hiddenValid && outputsValid && epochsValid && rateValid;
        if (!valid || inputs < 1 || hidden < 1 || outputs < 1 || epochs < 1
            || !double.IsFinite(rate) || rate <= 0)
        {
            statusTextBlock.Text = "Knotenzahlen und Epochs müssen positive ganze Zahlen sein; Learning Rate positiv.";
            return false;
        }
        return true;
    }

    private void createButton_Click(object? sender, RoutedEventArgs e)
    {
        if (!ReadSettings(out int inputs, out int hidden, out int outputs, out _, out double rate))
            return;

        network = new nn3S(inputs, hidden, outputs);
        network.LearningRate = rate;
        isFashionNetwork = false;
        loadedWeightsOnly = false;
        rows.Clear();
        trainingAccuracyTextBox.Text = "-";
        performanceTextBox.Text = "-";
        lossTextBox.Text = "-";
        statusTextBlock.Text = "Neues Netz: " + inputs + "-" + hidden + "-" + outputs + ".";
    }

    private async void browseTrainingButton_Click(object? sender, RoutedEventArgs e)
    {
        string? path = await PickTextFile();
        if (path != null) trainingFileTextBox.Text = path;
    }

    private async void browseQueryButton_Click(object? sender, RoutedEventArgs e)
    {
        string? path = await PickTextFile();
        if (path != null) queryFileTextBox.Text = path;
    }

    private async Task<string?> PickTextFile()
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "MNIST-TXT-Datei auswählen",
            AllowMultiple = false
        });
        return files.Count > 0 ? files[0].Path.LocalPath : null;
    }

    private void SetBusy(bool busy)
    {
        createButton.IsEnabled = !busy;
        trainButton.IsEnabled = !busy;
        queryButton.IsEnabled = !busy;
        saveButton.IsEnabled = !busy;
        loadButton.IsEnabled = !busy;
        showImagesCheckBox.IsEnabled = !busy;
        cancelButton.IsEnabled = busy;
    }

    private void cancelButton_Click(object? sender, RoutedEventArgs e) => cancellation?.Cancel();

    private async void trainButton_Click(object? sender, RoutedEventArgs e)
    {
        if (network == null)
        {
            statusTextBlock.Text = "Zuerst Create klicken oder Gewichte laden.";
            return;
        }
        if (!ReadSettings(out int inputs, out int hidden, out int outputs,
            out int epochs, out double rate))
            return;
        if (network.InputNodes != inputs || network.HiddenNodes != hidden
            || network.OutputNodes != outputs)
        {
            statusTextBlock.Text = "Knotenzahlen geändert: Bitte zuerst Create klicken.";
            return;
        }
        if (network.InputNodes != 784 || network.OutputNodes != 10)
        {
            statusTextBlock.Text = "MNIST benötigt 784 Input- und 10 Output-Knoten.";
            return;
        }
        string path = trainingFileTextBox.Text ?? "";
        if (!File.Exists(path))
        {
            statusTextBlock.Text = "Trainingsdatei nicht gefunden.";
            return;
        }

        network.LearningRate = rate;
        lastTrainingFile = Path.GetFileNameWithoutExtension(path);
        isFashionNetwork = lastTrainingFile.Contains("fashion", StringComparison.OrdinalIgnoreCase);
        loadedWeightsOnly = false;
        skipRemainingImages = false;
        trainingAccuracyTextBox.Text = "-";
        performanceTextBox.Text = "-";
        lossTextBox.Text = "-";
        cancellation = new CancellationTokenSource();
        SetBusy(true);
        try
        {
            statusTextBlock.Text = "Lese Trainingsdatei ...";
            List<DigitSample> samples = await Task.Run(() => MnistData.Load(path));
            List<double> losses;

            if (showImagesCheckBox.IsChecked == true)
                losses = await TrainWithImages(samples, epochs, cancellation.Token);
            else
            {
                IProgress<(int Epoch, int Index)> progress = new Progress<(int Epoch, int Index)>(p =>
                    statusTextBlock.Text = "Training: Epoche " + p.Epoch + "/" + epochs
                        + ", Bild " + p.Index + "/" + samples.Count);
                losses = await Task.Run(() => TrainAll(
                    network, samples, epochs,
                    (epoch, index) => progress.Report((epoch, index)), cancellation.Token));
            }

            lossTextBox.Text = Format(losses[losses.Count - 1]);
            statusTextBlock.Text = "Prüfe das Ergebnis auf den Trainingsbildern ...";
            var trainingResult = await Task.Run(() => QueryAll(
                network, samples, cancellationToken: cancellation.Token));
            trainingAccuracyTextBox.Text = trainingResult.Correct + " / "
                + trainingResult.Total + " = "
                + Format(100.0 * trainingResult.Correct / trainingResult.Total) + " %";
            ShowSample(samples[samples.Count - 1], true);
            SaveCurrentWeights();
            statusTextBlock.Text = "Training fertig: " + samples.Count + " Bilder, "
                + epochs + " Epoche(n). Trainingstreffer: "
                + trainingAccuracyTextBox.Text + ". Gewichte gespeichert.";
        }
        catch (OperationCanceledException)
        {
            statusTextBlock.Text = "Training abgebrochen.";
        }
        catch (Exception ex)
        {
            statusTextBlock.Text = "Fehler: " + ex.Message;
        }
        finally
        {
            SetBusy(false);
            cancellation?.Dispose();
            cancellation = null;
        }
    }

    private static List<double> TrainAll(nn3S network, IReadOnlyList<DigitSample> samples,
        int epochs, Action<int, int>? progress, CancellationToken token)
    {
        double[] inputs = new double[784];
        double[] targets = new double[10];
        List<double> losses = new List<double>();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double loss = 0;
            for (int index = 0; index < samples.Count; index++)
            {
                token.ThrowIfCancellationRequested();
                MnistData.FillInputs(samples[index], inputs);
                MnistData.FillTargets(samples[index], targets);
                network.train(inputs, targets);
                loss += network.LastLoss;

                if (index % 500 == 0 || index == samples.Count - 1)
                    progress?.Invoke(epoch + 1, index + 1);
            }
            losses.Add(loss / samples.Count);
        }

        return losses;
    }

    private static (int Correct, int Total) QueryAll(nn3S network,
        IReadOnlyList<DigitSample> samples, Action<int, int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        double[] inputs = new double[784];
        int correct = 0;

        for (int index = 0; index < samples.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            MnistData.FillInputs(samples[index], inputs);
            network.queryNN(inputs);
            if (MnistData.BestOutput(network.Final_outputs) == samples[index].Label)
                correct++;

            if (index % 500 == 0 || index == samples.Count - 1)
                progress?.Invoke(index + 1, correct);
        }

        return (correct, samples.Count);
    }

    private async Task<List<double>> TrainWithImages(
        IReadOnlyList<DigitSample> samples, int epochs, CancellationToken token)
    {
        if (network == null) throw new InvalidOperationException();
        double[] inputs = new double[784];
        double[] targets = new double[10];
        List<double> losses = new List<double>();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double loss = 0;
            for (int index = 0; index < samples.Count; index++)
            {
                token.ThrowIfCancellationRequested();
                MnistData.FillInputs(samples[index], inputs);
                MnistData.FillTargets(samples[index], targets);
                network.train(inputs, targets);
                loss += network.LastLoss;
                if (!skipRemainingImages)
                {
                    ShowSample(samples[index], true);
                    statusTextBlock.Text = "Training: Epoche " + (epoch + 1) + "/" + epochs
                        + ", Bild " + (index + 1) + "/" + samples.Count;
                    await ShowNextDialog(index + 1, samples.Count);
                }
            }
            losses.Add(loss / samples.Count);
        }
        return losses;
    }

    private async void queryButton_Click(object? sender, RoutedEventArgs e)
    {
        if (network == null)
        {
            statusTextBlock.Text = "Zuerst Create klicken oder Gewichte laden.";
            return;
        }
        if (network.InputNodes != 784 || network.OutputNodes != 10)
        {
            statusTextBlock.Text = "MNIST benötigt 784 Input- und 10 Output-Knoten.";
            return;
        }
        string path = queryFileTextBox.Text ?? "";
        if (!File.Exists(path))
        {
            statusTextBlock.Text = "Query-Datei nicht gefunden.";
            return;
        }

        skipRemainingImages = false;
        cancellation = new CancellationTokenSource();
        SetBusy(true);
        try
        {
            statusTextBlock.Text = "Lese Query-Datei ...";
            List<DigitSample> samples = await Task.Run(() => MnistData.Load(path));
            int correct;

            if (showImagesCheckBox.IsChecked == true)
                correct = await QueryWithImages(samples, cancellation.Token);
            else
            {
                IProgress<(int Index, int Correct)> progress = new Progress<(int Index, int Correct)>(p =>
                {
                    performanceTextBox.Text = p.Correct + " / " + p.Index + " = "
                        + Format(100.0 * p.Correct / p.Index) + " %";
                    statusTextBlock.Text = "Query: " + p.Index + "/" + samples.Count;
                });
                var result = await Task.Run(() => QueryAll(
                    network, samples,
                    (index, hits) => progress.Report((index, hits)), cancellation.Token));
                correct = result.Correct;
            }

            ShowSample(samples[samples.Count - 1], false);
            performanceTextBox.Text = correct + " / " + samples.Count + " = "
                + Format(100.0 * correct / samples.Count) + " %";
            statusTextBlock.Text = "Query fertig.";
        }
        catch (OperationCanceledException)
        {
            statusTextBlock.Text = "Query abgebrochen.";
        }
        catch (Exception ex)
        {
            statusTextBlock.Text = "Fehler: " + ex.Message;
        }
        finally
        {
            SetBusy(false);
            cancellation?.Dispose();
            cancellation = null;
        }
    }

    private async Task<int> QueryWithImages(IReadOnlyList<DigitSample> samples, CancellationToken token)
    {
        if (network == null) throw new InvalidOperationException();
        double[] inputs = new double[784];
        int correct = 0;
        for (int index = 0; index < samples.Count; index++)
        {
            token.ThrowIfCancellationRequested();
            MnistData.FillInputs(samples[index], inputs);
            network.queryNN(inputs);
            if (MnistData.BestOutput(network.Final_outputs) == samples[index].Label)
                correct++;
            if (!skipRemainingImages)
            {
                ShowSample(samples[index], false);
                performanceTextBox.Text = correct + " / " + (index + 1) + " = "
                    + Format(100.0 * correct / (index + 1)) + " %";
                await ShowNextDialog(index + 1, samples.Count);
            }
        }
        return correct;
    }

    private async Task ShowNextDialog(int index, int total)
    {
        var dialog = new Window
        {
            Title = "Bild " + index + " / " + total,
            Width = 390,
            Height = 150,
            WindowStartupLocation = WindowStartupLocation.CenterOwner
        };
        var next = new Button { Content = "Weiter", HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center };
        var skip = new Button { Content = "Rest ohne Anzeige",
            HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center };
        var stop = new Button { Content = "Abbrechen", HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Center };
        next.Click += (_, _) => dialog.Close();
        skip.Click += (_, _) => { skipRemainingImages = true; dialog.Close(); };
        stop.Click += (_, _) => { cancellation?.Cancel(); dialog.Close(); };
        dialog.Content = new StackPanel
        {
            Margin = new Thickness(16),
            Spacing = 10,
            Children =
            {
                new TextBlock { Text = "Im Hauptfenster ist das aktuelle Bild zu sehen." },
                new StackPanel { Orientation = Avalonia.Layout.Orientation.Horizontal, Spacing = 12,
                    Children = { next, skip, stop } }
            }
        };
        await dialog.ShowDialog(this);
    }

    private void ShowSample(DigitSample sample, bool showErrors)
    {
        if (network == null) return;
        double[] inputs = new double[784];
        MnistData.FillInputs(sample, inputs);
        network.queryNN(inputs);
        if (showErrors)
        {
            double[] targets = new double[10];
            MnistData.FillTargets(sample, targets);
            network.calculateErrors(targets);
        }
        int prediction = MnistData.BestOutput(network.Final_outputs);
        labelTextBlock.Text = "Richtiges Label: " + sample.Label;
        recognizedTextBlock.Text = "Erkannt: " + prediction;

        var bitmap = new WriteableBitmap(
            new PixelSize(28, 28), new Vector(96, 96),
            PixelFormats.Bgra8888, AlphaFormat.Opaque);
        using (var frame = bitmap.Lock())
        {
            byte[] row = new byte[28 * 4];
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    byte gray = sample.Pixels[y * 28 + x];
                    int offset = x * 4;
                    row[offset] = gray;
                    row[offset + 1] = gray;
                    row[offset + 2] = gray;
                    row[offset + 3] = 255;
                }
                Marshal.Copy(row, 0, IntPtr.Add(frame.Address, y * frame.RowBytes), row.Length);
            }
        }
        var old = digitImage.Source as IDisposable;
        digitImage.Source = bitmap;
        old?.Dispose();

        rows.Clear();
        int count = Math.Max(inputs.Length,
            Math.Max(network.HiddenNodes, network.OutputNodes));
        for (int i = 0; i < count; i++)
        {
            rows.Add(new nodeRow
            {
                inputValue = i < inputs.Length ? i + ": " + Format(inputs[i]) : i.ToString(),
                weightsIH = i < network.InputNodes ? FormatWeights(network.WIH, i) : "",
                inputHidden = i < network.HiddenNodes ? Format(network.Hidden_inputs[i]) : "",
                outputHidden = i < network.HiddenNodes ? Format(network.Hidden_outputs[i]) : "",
                weightsHO = i < network.HiddenNodes ? FormatWeights(network.WHO, i) : "",
                errorHidden = showErrors && i < network.HiddenNodes ? Format(network.Error_hidden[i]) : "",
                inputOutput = i < network.OutputNodes ? Format(network.Final_inputs[i]) : "",
                outputLayer = i < network.OutputNodes ? Format(network.Final_outputs[i]) : "",
                target = showErrors && i < 10 ? (i == sample.Label ? "0.99" : "0.01") : "",
                errorOutput = showErrors && i < 10 ? Format(network.Error_output[i]) : "",
                learningRate = Format(network.LearningRate)
            });
        }
    }

    private static string Format(double number) =>
        number.ToString("0.###", CultureInfo.CurrentCulture);

    private static string FormatWeights(double[,] weights, int source)
    {
        int count = Math.Min(2, weights.GetLength(1));
        string[] values = new string[count];
        for (int i = 0; i < count; i++)
            values[i] = weights[source, i].ToString("0.##", CultureInfo.CurrentCulture);
        return string.Join("|", values) + (weights.GetLength(1) > count ? "…" : "");
    }

    private string SaveCurrentWeights()
    {
        if (network == null) throw new InvalidOperationException("Kein Netz vorhanden.");
        string folder = isFashionNetwork
            ? Path.Combine(Directory.GetCurrentDirectory(), "projects", "Project6", "6.3", "weights")
            : Path.Combine(Directory.GetCurrentDirectory(), "projects", "Project3", "weights");
        Directory.CreateDirectory(folder);
        string name = lastTrainingFile + "-" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff");
        string path = Path.Combine(folder, name + ".nnw");
        int suffix = 2;
        while (File.Exists(path))
            path = Path.Combine(folder, name + "-" + suffix++ + ".nnw");
        network.SaveWeights(path);
        weightsPathTextBox.Text = path;
        return path;
    }

    private void saveButton_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            string path = SaveCurrentWeights();
            statusTextBlock.Text = "Gewichte gespeichert: " + path;
        }
        catch (Exception ex) { statusTextBlock.Text = "Fehler: " + ex.Message; }
    }

    private async void loadButton_Click(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Gewichtsdatei auswählen",
            AllowMultiple = false
        });
        if (files.Count == 0) return;

        try
        {
            string path = files[0].Path.LocalPath;
            LoadWeightsFromPath(path);
        }
        catch (Exception ex) { statusTextBlock.Text = "Fehler: " + ex.Message; }
    }

    private void LoadWeightsFromPath(string path)
    {
        network = nn3S.LoadWeights(path);
        loadedWeightsOnly = true;
        isFashionNetwork = path.Contains("Project6", StringComparison.OrdinalIgnoreCase)
            || Path.GetFileName(path).Contains("fashion", StringComparison.OrdinalIgnoreCase);
        if (isFashionNetwork)
        {
            string? dataRoot = Environment.GetEnvironmentVariable("NN_DATA_DIR");
            if (string.IsNullOrWhiteSpace(dataRoot))
                dataRoot = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
                    "Softwaretechnik", "Skripte", "alte", "projekte",
                    "Neuronale netze", "teil 3");
            string fashionTest = Path.Combine(dataRoot, "mnist db - zalando fashion",
                "mnist_fashion_test.txt");
            if (File.Exists(fashionTest)) queryFileTextBox.Text = fashionTest;
        }
        inputTextBox.Text = network.InputNodes.ToString();
        hiddenTextBox.Text = network.HiddenNodes.ToString();
        outputTextBox.Text = network.OutputNodes.ToString();
        learningRateTextBox.Text = Format(network.LearningRate);
        weightsPathTextBox.Text = path;
        rows.Clear();
        trainingAccuracyTextBox.Text = "-";
        performanceTextBox.Text = "-";
        lossTextBox.Text = "-";
        statusTextBlock.Text = "Gewichte geladen. Du kannst sofort Testen ausführen.";
    }

}

public class nodeRow
{
    public string inputValue { get; set; } = "";
    public string weightsIH { get; set; } = "";
    public string inputHidden { get; set; } = "";
    public string outputHidden { get; set; } = "";
    public string weightsHO { get; set; } = "";
    public string errorHidden { get; set; } = "";
    public string inputOutput { get; set; } = "";
    public string outputLayer { get; set; } = "";
    public string target { get; set; } = "";
    public string errorOutput { get; set; } = "";
    public string learningRate { get; set; } = "";
}
