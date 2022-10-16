package learn

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type ErrorFunction func(yPred, yTrue *mat.Dense) float64

// MeanSquaredError returns the average mean-squared difference between elements in two matrices
func MeanSquaredError(yPred, yTrue *mat.Dense) float64 {
	var err = &mat.Dense{}
	err.Sub(yTrue, yPred)
	err.Apply(square, err)

	total := 0.0
	r, c := err.Dims()
	for i := 0; i < r; i++ {
		row := err.RawRowView(i)
		for _, val := range row {
			total += val
		}
	}

	return total / float64(r*c)
}

func square(_, _ int, val float64) float64 {
	return math.Pow(val, 2)
}
