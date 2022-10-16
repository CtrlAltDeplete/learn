package learn

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"reflect"
	"testing"
)

func TestMeanSquaredError(t *testing.T) {
	type args struct {
		yPred *mat.Dense
		yTrue *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "happy path",
			args: args{
				yPred: mat.NewDense(4, 3, []float64{
					0.44279, 0.28073, 0.65765,
					0.41863, 0.33948, 0.64744,
					0.44360, 0.26871, 0.67976,
					0.41942, 0.32609, 0.66987,
				}),
				yTrue: mat.NewDense(4, 3, []float64{
					1.00000, 1.00000, 1.00000,
					0.90909, 0.90909, 0.90909,
					0.69697, 0.69697, 0.69697,
					0.60606, 0.60606, 0.60606,
				}),
			},
			want: 0.16197308716822795,
		},
		{
			name: "zeroes",
			args: args{
				yPred: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
				yTrue: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
			},
			want: 0.0,
		},
		{
			name: "ones",
			args: args{
				yPred: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
				yTrue: mat.NewDense(2, 2, []float64{
					1, 1,
					1, 1,
				}),
			},
			want: 1.0,
		},
		{
			name: "negative ones",
			args: args{
				yPred: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
				yTrue: mat.NewDense(2, 2, []float64{
					-1, -1,
					-1, -1,
				}),
			},
			want: 1.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MeanSquaredError(tt.args.yPred, tt.args.yTrue); math.Abs(got-tt.want) > 0.00001 {
				t.Errorf("MeanSquaredError() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMeanSquaredErrorPrime(t *testing.T) {
	type args struct {
		yPred *mat.Dense
		yTrue *mat.Dense
	}
	tests := []struct {
		name string
		args args
		want *mat.Dense
	}{
		{
			name: "happy path",
			args: args{
				yPred: mat.NewDense(4, 3, []float64{
					0.44279, 0.28073, 0.65765,
					0.41863, 0.33948, 0.64744,
					0.44360, 0.26871, 0.67976,
					0.41942, 0.32609, 0.66987,
				}),
				yTrue: mat.NewDense(4, 3, []float64{
					1.00000, 1.00000, 1.00000,
					0.90909, 0.90909, 0.90909,
					0.69697, 0.69697, 0.69697,
					0.60606, 0.60606, 0.60606,
				}),
			},
			want: mat.NewDense(4, 3, []float64{
				-0.09286833333333333, -0.11987833333333334, -0.057058333333333336,
				-0.08174333333333332, -0.09493499999999999, -0.04360833333333332,
				-0.042228333333333326, -0.07137666666666666, -0.0028683333333333243,
				-0.03110666666666667, -0.04666166666666667, 0.010634999999999986,
			}),
		},
		{
			name: "zeros",
			args: args{
				yPred: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
				yTrue: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
			},
			want: mat.NewDense(2, 2, []float64{
				0, 0,
				0, 0,
			}),
		},
		{
			name: "ones",
			args: args{
				yPred: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
				yTrue: mat.NewDense(2, 2, []float64{
					1, 1,
					1, 1,
				}),
			},
			want: mat.NewDense(2, 2, []float64{
				-0.5, -0.5,
				-0.5, -0.5,
			}),
		},
		{
			name: "negative ones",
			args: args{
				yPred: mat.NewDense(2, 2, []float64{
					0, 0,
					0, 0,
				}),
				yTrue: mat.NewDense(2, 2, []float64{
					-1, -1,
					-1, -1,
				}),
			},
			want: mat.NewDense(2, 2, []float64{
				0.5, 0.5,
				0.5, 0.5,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MeanSquaredErrorPrime(tt.args.yPred, tt.args.yTrue); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MeanSquaredErrorPrime() = %v, want %v", got, tt.want)
			}
		})
	}
}
