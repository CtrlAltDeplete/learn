package learn

import (
	"gonum.org/v1/gonum/mat"
	"reflect"
	"testing"
)

func TestBatchDataset(t *testing.T) {
	type args struct {
		inputs    *mat.Dense
		batchSize int
		shuffle   bool
	}
	tests := []struct {
		name string
		args args
		want []*mat.Dense
	}{
		{
			"happy path",
			args{
				mat.NewDense(8, 2, []float64{0, 0, 0, 1, 1 / 3, 0, 1 / 3, 1, 2 / 3, 0, 2 / 3, 1, 1, 0, 1, 1}),
				4,
				false,
			},
			[]*mat.Dense{
				mat.NewDense(4, 2, []float64{0, 0, 0, 1, 1 / 3, 0, 1 / 3, 1}),
				mat.NewDense(4, 2, []float64{2 / 3, 0, 2 / 3, 1, 1, 0, 1, 1}),
			},
		},
		{
			"single item dataset",
			args{
				mat.NewDense(1, 1, []float64{0}),
				1,
				false,
			},
			[]*mat.Dense{
				mat.NewDense(1, 1, []float64{0}),
			},
		},
		{
			"cutoff data",
			args{
				mat.NewDense(3, 1, []float64{0, 1, 2}),
				2,
				false,
			},
			[]*mat.Dense{
				mat.NewDense(2, 1, []float64{0, 1}),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BatchDataset(tt.args.inputs, tt.args.batchSize, tt.args.shuffle); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("BatchDataset() = %v, want %v", got, tt.want)
			}
		})
	}
}
