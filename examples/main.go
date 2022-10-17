package main

import (
	"fmt"
	"github.com/CtrlAltDeplete/learn"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

func main() {
	rand.Seed(42)

	var xTrain, yTrain = learn.BatchDataset(
		mat.NewDense(4, 2, []float64{
			0, 0,
			0, 1,
			1, 0,
			1, 1,
		}),
		mat.NewDense(4, 1, []float64{
			0, 1, 1, 1,
		}),
		4,
		false)

	var network learn.Network
	network = append(network, learn.NewDense(2, 1))
	network = append(network, &learn.Activation{
		Activation:      learn.Sigmoid,
		ActivationPrime: learn.SigmoidPrime,
	})

	for _, batch := range xTrain {
		yPred := network.Predict(batch)
		fmt.Printf("Network(%f, %f) = %f\n", batch.At(0, 0), batch.At(0, 1), yPred.At(0, 0))
	}

	network.Train(xTrain, yTrain, learn.MeanSquaredError, learn.MeanSquaredGradient, 1000, 1, 1)

	for _, batch := range xTrain {
		yPred := network.Predict(batch)
		fmt.Printf("Network(%f, %f) = %f\n", batch.At(0, 0), batch.At(0, 1), yPred.At(0, 0))
	}
}
