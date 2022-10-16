package learn

import "gonum.org/v1/gonum/mat"

type GradientFunction func(yPred, yTrue *mat.Dense) *mat.Dense

// MeanSquaredGradient is the derivative of the MeanSquaredError function
func MeanSquaredGradient(yPred, yTrue *mat.Dense) *mat.Dense {
	var grad = &mat.Dense{}
	grad.Sub(yTrue, yPred)

	//r, c := grad.Dims()
	//grad.Scale(2.0/float64(r*c), grad)

	grad.Scale(-2.0, grad)
	return grad
}
