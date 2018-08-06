using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LearningAI
{
    class Network
    {
        public static Random rnd = new Random(); // used to initialize weights

        // number of node on each layer + number of layer
        private int nInput;
        private List<int> nHidden;
        private int nOutput;
        private int nLayers;

        // value of each node
        private List<double> iNodes;
        private List<List<double>> hNodes;
        private List<double> oNodes;

        // weights between each node
        private List<List<double>> ihWeights; // [index of the input node][index of the hidden node on first hidden layer]
        private List<List<List<double>>> hhWeights; // [index of the layer of the hidden node layer][index of the left node][index of the right node]
        private List<List<double>> hoWeights; // [index of the hidden node on the last hidden layer][index of the output node]

        // biase of each node
        private List<List<double>> hBiases;
        private List<double> oBiases;

        public Network(int nInput, List<int> nHidden, int nOutput)
        {
            this.nInput = nInput;
            this.nHidden = nHidden;
            this.nOutput = nOutput;
            this.nLayers = nInput + nHidden.Sum() + nOutput;

            this.iNodes = new List<double>();
            this.hNodes = new List<List<double>>();
            this.oNodes = new List<double>();

            this.ihWeights = new List<List<double>>();
            this.hhWeights = new List<List<List<double>>>();
            this.hoWeights = new List<List<double>>();
            
            this.hBiases = new List<List<double>>();
            this.oBiases = new List<double>();
        }

        public List<List<double>> GetWeights()
        {
            List<List<double>> weights = new List<List<double>>();
            weights.Add(new List<double>());
            for(int i = 0; i < ihWeights.Count; i++)
            {
                for(int j = 0; j < ihWeights[i].Count; j++)
                {
                    weights.ElementAt(0).Insert(j, ihWeights[i][j]);
                }
            }

            if(nHidden.Count > 1)
            {
                for (int i = 0; i < nHidden.Count - 1; i++)
                {
                    weights.Insert(i + 1, new List<double>());
                    for (int j = 0; j < hhWeights[i].Count; j++)
                    {
                        for (int k = 0; k < hhWeights[i][j].Count; k++)
                        {
                            weights.ElementAt(i + 1).Insert(k, hhWeights[i][j][k]);
                        }
                    }
                }
            }

            weights.Add(new List<double>());
            for (int i = 0; i < hoWeights.Count; i++)
            {
                for (int j = 0; j < hoWeights[i].Count; j++)
                {
                    weights.ElementAt(weights.Count - 1).Insert(j, hoWeights[i][j]);
                }
            }

            return weights;
        }

        public void Initialize()
        {
            // initialization of node values (initialization to 0, except for the inputs which will get a value later)
            for (int i = 0; i < nInput; i++)
            {
                iNodes.Insert(i, 0);
            }

            for (int i = 0; i < nHidden.Count; i++)
            {
                hNodes.Insert(i, new List<double>());
                for (int j = 0; j < nHidden[i]; j++)
                {
                    hNodes.ElementAt(i).Insert(j, 0);
                }
            }

            for (int i = 0; i < nOutput; i++)
            {
                oNodes.Insert(i, 0);
            }

            // initialization of weights (initialization at a random value between 0 and 0.25)
            for (int i = 0; i < nInput; i++)
            {
                ihWeights.Insert(i, new List<double>());
                for(int j = 0; j < nHidden[0]; j++)
                {
                    ihWeights.ElementAt(i).Insert(j, rnd.NextDouble() * 0.25);
                }
            }

            for(int i = 0; i < nHidden.Count - 1; i++)
            {
                hhWeights.Insert(i, new List<List<double>>());
                for(int j = 0; j < nHidden[i]; j++)
                {
                    hhWeights.ElementAt(i).Insert(j, new List<double>());
                    for(int k = 0; k < nHidden[i + 1]; k++)
                    {
                        hhWeights.ElementAt(i).ElementAt(j).Insert(k, rnd.NextDouble() * 0.25);
                    }
                }
            }

            for(int i = 0; i < nHidden[nHidden.Count - 1]; i++)
            {
                hoWeights.Insert(i, new List<double>());
                for (int j = 0; j < nOutput; j++)
                {
                    hoWeights.ElementAt(i).Insert(j, rnd.NextDouble() * 0.25);
                }
            }
            
            // initialization of biases (initialized at 0 for now)
            for (int i = 0; i < nHidden.Count; i++)
            {
                hBiases.Insert(i, new List<double>());
                for(int j = 0; j < nHidden[i]; j++)
                {
                    hBiases.ElementAt(i).Insert(j, 0);
                }
            }

            for(int i = 0; i < nOutput; i++)
            {
                oBiases.Insert(i, 0);
            }
        }

        public List<double> ComputeOutput(List<double> inputs, List<double> target)
        {
            for (int i = 0; i < nInput; i++)
            {
                iNodes[i] = 0;
            }

            for (int i = 0; i < nHidden.Count; i++)
            {
                for (int j = 0; j < nHidden[i]; j++)
                {
                    hNodes[i][j] = 0;
                }
            }

            for (int i = 0; i < nOutput; i++)
            {
                oNodes[i] = 0;
            }

            if (inputs.Count == nInput)
            {
                // we enter the inputs into the machine
                for (int i = 0; i < nInput; i++)
                {
                    iNodes[i] = inputs[i];
                }

                // we calculate the value of each node on the first hidden layer using each node of the input layer
                for (int i = 0; i < nHidden[0]; i++)
                {
                    for(int j = 0; j < nInput; j++)
                    {
                        hNodes[0][i] += iNodes[j] * ihWeights[j][i];
                    }
                    hNodes[0][i] += hBiases[0][i];
                    hNodes[0][i] = 1 / (1 + Math.Pow(Math.E, -hNodes[0][i]));
                }

                // we calculate the value of each node on each hidden layer using each node of the previous hidden layer
                for (int i = 1; i < nHidden.Count; i++)
                {
                    for(int j = 0; j < nHidden[i]; j++)
                    {
                        for(int k = 0; k < nHidden[i - 1]; k++)
                        {
                            hNodes[i][j] += hNodes[i - 1][k] * hhWeights[i - 1][k][j];
                        }
                        hNodes[i][j] += hBiases[i][j];
                        hNodes[i][j] = 1 / (1 + Math.Pow(Math.E, -hNodes[i][j]));
                    }
                }

                // we calculate the value of each node on the output layer using each node of the last hidden layer
                for (int i = 0; i < nOutput; i++)
                {
                    for(int j = 0; j < nHidden[nHidden.Count - 1]; j++)
                    {
                        oNodes[i] += hNodes[nHidden.Count - 1][j] * hoWeights[j][i];
                    }
                    oNodes[i] += oBiases[i];
                    oNodes[i] = 1 / (1 + Math.Pow(Math.E, -oNodes[i]));
                }

                Backpropagation(oNodes, target);

                return oNodes;
            }
            else
            {
                throw new ArgumentException("Number of input invalid");
            }
        }

        public void Backpropagation(List<double> outputs, List<double> target)
        {
            List<List<List<double>>> modWeights = new List<List<List<double>>>();
            double localError = 0;
            double totalError = 0;
            double reverseActivation = 0;

            double learnRate = 1;
            
            for(int i = GetWeights().Count - 1; i >= 0; i--) // for each layer of weight starting from the end
            {
                modWeights.Insert(0, new List<List<double>>());
                localError = totalError;
                if(i == GetWeights().Count - 1) // weights between output layer and last hidden layer
                {
                    if(nHidden.Count > 0)
                    {
                        for (int j = 0; j < hoWeights.Count; j++) // for each node of the last hidden layer
                        {
                            modWeights.ElementAt(0).Insert(j, new List<double>());
                            for (int k = 0; k < hoWeights[j].Count; k++) // for each output node
                            {
                                localError = outputs[k] - target[k];
                                reverseActivation = outputs[k] * (1 - outputs[k]);

                                modWeights.ElementAt(0).ElementAt(j).Insert(k, localError * reverseActivation * hNodes[i - 1][j]);
                                totalError += localError * reverseActivation * hoWeights[j][k];
                            }
                        }
                    }
                    else
                    {
                        for (int j = 0; j < ihWeights.Count; j++) // for each node of the input layer
                        {
                            modWeights.ElementAt(0).Insert(j, new List<double>());
                            for (int k = 0; k < hoWeights[j].Count; k++) // for each output node
                            {
                                localError = outputs[k] - target[k];
                                reverseActivation = outputs[k] * (1 - outputs[k]);

                                modWeights.ElementAt(0).ElementAt(j).Insert(k, localError * reverseActivation * iNodes[j]);
                                totalError += localError * reverseActivation * hoWeights[j][k];
                            }
                        }
                    }
                }
                else if(i > 0) // weights between hidden layers
                {
                    for (int j = 0; j < hhWeights[i - 1].Count; j++) // for each node on the layer i
                    {
                        modWeights.ElementAt(0).Insert(j, new List<double>());
                        for (int k = 0; k < hhWeights[i- 1][j].Count; k++) // for each node on the layer i + 1
                        {
                            reverseActivation = hNodes[i][k] * (1 - hNodes[i][k]);

                            modWeights.ElementAt(0).ElementAt(j).Insert(k, localError * reverseActivation * hNodes[i - 1][j]);
                            totalError += localError * reverseActivation * hhWeights[i - 1][j][k];
                        }
                    }
                }
                else // weights between first hidden layer and input layer
                {
                    for (int j = 0; j < ihWeights.Count; j++) // for each node on the layer i
                    {
                        modWeights.ElementAt(0).Insert(j, new List<double>());

                        if(hhWeights.Count > 0)
                        {
                            for (int k = 0; k < hhWeights[i][j].Count; k++) // for each node on the layer i + 1
                            {
                                reverseActivation = hNodes[i][k] * (1 - hNodes[i][k]);

                                modWeights.ElementAt(0).ElementAt(j).Insert(k, localError * reverseActivation * iNodes[j]);
                            }
                        }
                        else
                        {
                            for (int k = 0; k < ihWeights[j].Count; k++) // for each node on the layer i + 1
                            {
                                reverseActivation = hNodes[i][k] * (1 - hNodes[i][k]);

                                modWeights.ElementAt(0).ElementAt(j).Insert(k, localError * reverseActivation * iNodes[j]);
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < modWeights.Count - 1; i++)
            {
                for(int j = 0; j < modWeights[i].Count - 1; j++)
                {
                    for(int k = 0; k < modWeights[i][j].Count - 1; k++)
                    {
                        if(i == 0)
                        {
                            ihWeights[j][k] -= learnRate * modWeights[i][j][k];
                        }
                        else if(i < modWeights.Count - 1)
                        {
                            hhWeights[i - 1][j][k] -= learnRate * modWeights[i][j][k];
                        }
                        else
                        {
                            hoWeights[j][k] -= learnRate * modWeights[i][j][k];
                        }
                    }
                }
            }
        }
    }
}
