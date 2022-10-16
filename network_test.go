package learn

import (
	"encoding/json"
	"gonum.org/v1/gonum/mat"
	"reflect"
	"testing"
)

func TestNetwork_JSON(t *testing.T) {
	tests := []struct {
		name    string
		network Network
	}{
		{
			"happy path",
			Network{
				&Activation{
					Activation:      Sigmoid,
					ActivationPrime: SigmoidPrime,
				},
				&Dense{
					Weights: *mat.NewDense(2, 6, []float64{
						-0.12546, 0.45071, 0.23199, 0.09866, -0.34398, -0.34401,
						-0.44192, 0.36618, 0.10112, 0.20807, -0.47942, 0.46991,
					}),
					Biases: *mat.NewDense(1, 6, []float64{
						0.33244, -0.28766, -0.31818, -0.31660, -0.19576, 0.02476,
					}),
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			var marshalled []byte
			var got Network

			marshalled, err = json.Marshal(&tt.network)
			if err != nil {
				panic(err)
			}

			if err = json.Unmarshal(marshalled, &got); err != nil {
				panic(err)
			}

			if len(tt.network) != len(got) {
				t.Errorf("Marshal()/Unmarshal() = %v, want %v", got, tt.network)
			}

			for i := range tt.network {
				if reflect.TypeOf(tt.network[i]) != reflect.TypeOf(got[i]) {
					t.Errorf("Marshal()/Unmarshal()[%d] = %v, want %v", i, got[i], tt.network[i])
				}
			}
		})
	}
}
