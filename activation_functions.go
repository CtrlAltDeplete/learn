package learn

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type activationFunction func(x mat.Dense) mat.Dense

func sigmoid(_, _ int, v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

// Sigmoid is used to clamp values between (0, 1)
func Sigmoid(x mat.Dense) mat.Dense {
	x.Apply(sigmoid, &x)
	return x
}

func sigmoidPrime(_, _ int, v float64) float64 {
	return 1.0 - v
}

// SigmoidPrime is the derivative of the Sigmoid function
func SigmoidPrime(x mat.Dense) mat.Dense {
	sig := Sigmoid(x)
	var sigPrime mat.Dense
	sigPrime.Apply(sigmoidPrime, &sig)

	var output mat.Dense
	output.MulElem(&sig, &sigPrime)

	return output
}
