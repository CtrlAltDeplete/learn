package learn

import (
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name string
		args *mat.Dense
		want *mat.Dense
	}{
		{
			"happy path",
			mat.NewDense(2, 4, []float64{-100, -10, -1, -0.1, 0.1, 1, 10, 100}),
			mat.NewDense(2, 4, []float64{
				0.00000, 0.00005, 0.26894, 0.47502,
				0.52498, 0.73106, 0.99995, 1.00000,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Sigmoid(tt.args); !matricesAlmostEqual(got, tt.want, 0.0001) {
				t.Errorf("Sigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Fuzz_sigmoid(t *testing.F) {
	t.Add(0, 0, -32.0)
	t.Fuzz(func(t *testing.T, i int, j int, v float64) {
		response := sigmoid(i, j, v)
		if response < 0.0 || response > 1.0 {
			t.Errorf("sigmoid(%v) = %v", v, response)
		}
	})
}

func TestSigmoidPrime(t *testing.T) {
	tests := []struct {
		name string
		args *mat.Dense
		want *mat.Dense
	}{
		{
			"happy path",
			mat.NewDense(2, 4, []float64{-100, -10, -1, -0.1, 0.1, 1, 10, 100}),
			mat.NewDense(2, 4, []float64{
				0.00000, 0.00005, 0.19661, 0.24938,
				0.24938, 0.19661, 0.00005, 0.00000,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SigmoidPrime(tt.args); !matricesAlmostEqual(got, tt.want, 0.0001) {
				t.Errorf("SigmoidPrime() = %v, want %v", got, tt.want)
			}
		})
	}
}
