package learn

import (
	"encoding/json"
	"gonum.org/v1/gonum/mat"
	"reflect"
	"testing"
)

func TestDense_Forward(t *testing.T) {
	tests := []struct {
		name  string
		layer Dense
		args  *mat.Dense
		want  *mat.Dense
	}{
		{
			"happy path",
			Dense{
				Weights: mat.NewDense(2, 6, []float64{
					-0.12546, 0.45071, 0.23199, 0.09866, -0.34398, -0.34401,
					-0.44192, 0.36618, 0.10112, 0.20807, -0.47942, 0.46991,
				}),
				Biases: mat.NewDense(1, 6, []float64{
					0.33244, -0.28766, -0.31818, -0.31660, -0.19576, 0.02476,
				}),
			},
			mat.NewDense(1, 2, []float64{0, 0}),
			mat.NewDense(1, 6, []float64{0.33244, -0.28766, -0.31818, -0.31660, -0.19576, 0.02476}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.layer.Forward(tt.args); !matricesAlmostEqual(&got, tt.want, 0.00001) {
				t.Errorf("Dense Forward() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDense_Backward(t *testing.T) {
	type args struct {
		outputError  *mat.Dense
		learningRate float64
	}
	tests := []struct {
		name  string
		layer Dense
		args  args
		want  *mat.Dense
	}{
		{
			"happy path",
			Dense{
				Weights: mat.NewDense(2, 6, []float64{
					-0.12546, 0.45071, 0.23199, 0.09866, -0.34398, -0.34401,
					-0.44192, 0.36618, 0.10112, 0.20807, -0.47942, 0.46991,
				}),
				Biases: mat.NewDense(1, 6, []float64{
					0.33244, -0.28766, -0.31818, -0.31660, -0.19576, 0.02476,
				}),
				_input: *mat.NewDense(4, 2, []float64{
					0.00000, 0.00000,
					0.00000, 1.00000,
					0.33333, 0.00000,
					0.33333, 1.00000,
				}),
				_output: *mat.NewDense(4, 6, []float64{
					0.33244, -0.28766, -0.31818, -0.31660, -0.19576, 0.02476,
					-0.10947, 0.07852, -0.21706, -0.10852, -0.67517, 0.49467,
					0.29062, -0.13742, -0.24084, -0.28371, -0.31042, -0.08991,
					-0.15129, 0.22875, -0.13973, -0.07564, -0.78983, 0.38000,
				}),
			},
			args{
				mat.NewDense(4, 6, []float64{
					0.00441, 0.00298, -0.01271, -0.00234, -0.00262, -0.00522,
					0.00391, 0.00305, -0.01106, -0.00231, -0.00242, -0.00511,
					0.00255, 0.00380, -0.00640, -0.00236, -0.00167, -0.00582,
					0.00198, 0.00371, -0.00457, -0.00233, -0.00157, -0.00526,
				}),
				0.1,
			},
			mat.NewDense(4, 2, []float64{
				0.00031, -0.00382,
				0.00068, -0.00345,
				0.00225, -0.00280,
				0.00249, -0.00218,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.layer.Backward(tt.args.outputError, tt.args.learningRate); !matricesAlmostEqual(&got, tt.want, 0.0001) {
				t.Errorf("Dense Backward() = %v, want %v", got, tt.want)
			}
		})
	}
}

func FuzzDense_JSON(t *testing.F) {
	t.Add(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
	t.Fuzz(func(t *testing.T, i, j, k, l, m, n, o, p float64) {
		var err error
		var marshalled []byte
		var got Dense
		var want = Dense{
			Weights: mat.NewDense(1, 4, []float64{
				i, j, k, l,
			}),
			Biases: mat.NewDense(1, 4, []float64{
				m, n, o, p,
			}),
		}

		if marshalled, err = json.Marshal(&want); err != nil {
			t.Errorf("Dense Marshal() = %v", err)
		}

		if err = json.Unmarshal(marshalled, &got); err != nil {
			t.Errorf("Dense Unmarshal() = %v", err)
		}

		if !reflect.DeepEqual(want, got) {
			t.Errorf("Dense Marshal()/Unmarshal() = %v, want %v", got, want)
		}
	})
}

func TestActivation_Forward(t *testing.T) {
	tests := []struct {
		name  string
		layer Activation
		args  *mat.Dense
		want  *mat.Dense
	}{
		{
			"happy path",
			Activation{
				Activation:      Sigmoid,
				ActivationPrime: SigmoidPrime,
			},
			mat.NewDense(1, 3, []float64{-0.22983, -0.94082, 0.65286}),
			mat.NewDense(1, 3, []float64{0.44279, 0.28073, 0.65765}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.layer.Forward(tt.args); !matricesAlmostEqual(&got, tt.want, 0.0001) {
				t.Errorf("Activation Forward() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestActivation_Backward(t *testing.T) {
	tests := []struct {
		name  string
		layer Activation
		args  *mat.Dense
		want  *mat.Dense
	}{
		{
			name: "happy path",
			layer: Activation{
				Activation:      Sigmoid,
				ActivationPrime: SigmoidPrime,
				_input: *mat.NewDense(4, 3, []float64{
					-0.22983, -0.94082, 0.65286,
					-0.32841, -0.66559, 0.60780,
					-0.22656, -1.00116, 0.75266,
					-0.32514, -0.72593, 0.70760,
				}),
				_output: *mat.NewDense(4, 3, []float64{
					0.44279, 0.28073, 0.65765,
					0.41863, 0.33948, 0.64744,
					0.44360, 0.26871, 0.67976,
					0.41942, 0.32609, 0.66987,
				}),
			},
			args: mat.NewDense(4, 3, []float64{
				-0.09287, -0.11988, -0.05706,
				-0.08174, -0.09493, -0.04361,
				-0.04223, -0.07138, -0.00287,
				-0.03111, -0.04666, 0.01063,
			}),
			want: mat.NewDense(4, 3, []float64{
				-0.02291, -0.02421, -0.01285,
				-0.01989, -0.02129, -0.00995,
				-0.01042, -0.01403, -0.00062,
				-0.00757, -0.01025, 0.00235,
			}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.layer.Backward(tt.args, 0); !matricesAlmostEqual(&got, tt.want, 0.0001) {
				t.Errorf("Activation Backward() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestActivation_JSON(t *testing.T) {
	tests := []struct {
		name  string
		layer Activation
	}{
		{
			"happy path",
			Activation{
				Activation:      Sigmoid,
				ActivationPrime: SigmoidPrime,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			var marshalled []byte
			var got Activation

			if marshalled, err = json.Marshal(&tt.layer); err != nil {
				t.Errorf("Activation Marshal() = %v", err)
			}

			if err = json.Unmarshal(marshalled, &got); err != nil {
				t.Errorf("Activation Unmarshal() = %v", err)
			}

			if reflect.ValueOf(tt.layer.Activation).Pointer() != reflect.ValueOf(got.Activation).Pointer() {
				t.Errorf("Activation Marshal()/Unmarshal() = %v, want %v", got, tt.layer)
			}
		})
	}
}
