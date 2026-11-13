using System;

namespace NeuralNetworks_1;
class nnMath
{
    public double[] matrixMult (double[,] WeightedMatrix, double[] InputMatrix)
    {
        if (WeightedMatrix.GetLength(0) != InputMatrix.Length)
            throw new ArgumentException("Matrix und Vektor haben unpassende Größen.");

        double[] result = new double[WeightedMatrix.GetLength(1)];
        for (int i = 0; i < result.Length; i++)
        {
            for (int j = 0; j < WeightedMatrix.GetLength(0); j++)
            {
                result[i] += WeightedMatrix[j,i] * InputMatrix[j];
            }
        }
        return result;
    }
    public double[] activationFunction(double[] WeightedInputs)
    {
        double[] result = new double[WeightedInputs.Length];

        for (int i = 0; i < WeightedInputs.Length; i++)
        {
            result[i] = 1.0/(1.0 + Math.Exp(-WeightedInputs[i]));
        }
        return result;
    }

    public double[,] fullMatrixMult(double[] vec1, double[] vec2)
    {
        double [,] result = new double [vec1.Length, vec2.Length];
        for (int i = 0; i < vec1.Length; i++)
        {
            for (int j = 0; j < vec2.Length; j++)
            {
                result [i,j] = vec1[i] * vec2[j];
            }
        }
        return result;

    }
    public double[,] matrixSum(double[,] mat1, double[,] mat2)
    {
        if (mat1.GetLength(0) != mat2.GetLength(0) || mat1.GetLength(1) != mat2.GetLength(1))
            throw new ArgumentException("Die Matrizen müssen gleich groß sein.");
        double [,] result = new double [mat1.GetLength(0), mat1.GetLength(1)]; 
        for (int i = 0; i < mat1.GetLength(0); i++) 
        {
            for (int j = 0; j < mat1.GetLength(1); j++)
            {
                result [i, j] = mat1[i, j] + mat2[i, j];
            }
        }
        return result;
    }
    public double[] vectorMult(double[] vec1, double[] vec2)
    {
        if (vec1.Length != vec2.Length)
            throw new ArgumentException("Die Vektoren müssen gleich lang sein.");
        double [] result = new double [vec1.Length];
        for (int i = 0; i < vec1.Length; i++)
        {
            result [i] = vec1[i] * vec2[i];
        }
        return result;
    }
    public double[,] transpose(double[,] inputMat)
    {
        double [,] result = new double [inputMat.GetLength(1), inputMat.GetLength(0)];
        for (int i = 0; i < inputMat.GetLength(0); i++)
        {
            for (int j = 0; j < inputMat.GetLength(1); j++)
            {
                result[j,i] = inputMat[i,j];
            }
        }
        return result;
    }
    public double[,] matrixScale(double[,] inputMat, double scale)
    {
        double [,] result = new double [inputMat.GetLength(0), inputMat.GetLength(1)]; 
        for (int i = 0; i < inputMat.GetLength(0); i++)
        {
            for (int j = 0; j < inputMat.GetLength(1); j++)
            {
                result [i, j] = inputMat[i, j] * scale;
            }
        }
        return result;
    }
}
