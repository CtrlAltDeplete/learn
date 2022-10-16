package learn

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type activationFunction func(x *mat.Dense) *mat.Dense

func sigmoid(_, _ int, v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}

// Sigmoid is used to clamp values between (0, 1)
func Sigmoid(input *mat.Dense) *mat.Dense {
	var output mat.Dense
	output.Apply(sigmoid, input)
	return &output
}

func sigmoidPrime(_, _ int, v float64) float64 {
	return 1.0 - v
}

// SigmoidPrime is the derivative of the Sigmoid function
func SigmoidPrime(input *mat.Dense) *mat.Dense {
	sig := Sigmoid(input)

	var sigPrime mat.Dense
	sigPrime.Apply(sigmoidPrime, sig)

	var output mat.Dense
	output.MulElem(sig, &sigPrime)

	return &output
}
