using System;
using System.Collections.Generic;
using System.Timers;

namespace LearningAI
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Number of layers (min 4):");
            if(!int.TryParse(Console.ReadLine(), out int nLayers))
            {
                nLayers = 4;
            }
            int nInput = 0;
            List<int> nHidden = new List<int>();
            int nOutput = 0;

            for(int i = 0; i < nLayers; i++)
            {
                if(i == 0)
                {
                    Console.WriteLine("Number of inputs (min 1):");
                    if(!int.TryParse(Console.ReadLine(), out nInput))
                    {
                        nInput = 2;
                    }
                }
                else if(i < nLayers - 1)
                {
                    Console.WriteLine("Number of hidden nodes on layer " + (i + 1) + " (min 1):");
                    if(!int.TryParse(Console.ReadLine(), out int result))
                    {
                        result = 2;
                    }
                    nHidden.Insert(i - 1, result);
                }
                else
                {
                    Console.WriteLine("Number of outputs (min 1):");
                    if(!int.TryParse(Console.ReadLine(), out nOutput))
                    {
                        nOutput = 2;
                    }
                }
                
            }
            
            Network network = new Network(nInput, nHidden, nOutput);
            network.Initialize();

            List<double> inputs = new List<double>();
            for(int i = 0; i < nInput; i++)
            {
                Console.WriteLine("Enter input n°" + (i + 1) + ":");
                if(!double.TryParse(Console.ReadLine(), out double result))
                {
                    result = 1;
                }
                inputs.Insert(i, result);
            }

            List<double> target = new List<double>();
            for (int i = 0; i < nOutput; i++)
            {
                Console.WriteLine("Enter target for output n°" + (i + 1) + " (in interval O[  ]1):");
                if(!double.TryParse(Console.ReadLine(), out double result))
                {
                    result = 0.3;
                }
                target.Insert(i, result);
            }

            Console.WriteLine("Until error inferior at (default: 0.01):");
            if(!double.TryParse(Console.ReadLine(), out double goal))
            {
                goal = 0.01;
            }

            Console.WriteLine("Max number of try (default: 2000):");
            if(!int.TryParse(Console.ReadLine(), out int maxTry))
            {
                maxTry = 2000;
            }

            List<double> outputs = new List<double>();
            int count = 0;
            double totalError = 0;
            do
            {
                totalError = 0;
                Console.Clear();
                outputs = network.ComputeOutput(inputs, target);
                for (int i = 0; i < outputs.Count; i++)
                {
                    Console.WriteLine(i + ": " + outputs[i]);
                }

                Console.WriteLine();

                for (int i = 0; i < outputs.Count; i++)
                {
                    totalError += outputs[i] - target[i];
                }

                Console.WriteLine("Error total: " + totalError);
                Console.WriteLine("Number of try: " + count);

                count++;

                DateTime start = DateTime.Now;
                while (DateTime.Now.Subtract(start).Milliseconds < 100)
                {

                }
            } while (totalError >= goal && count < maxTry);

            Console.WriteLine("End");
            Console.ReadKey();
        }
    }
}
