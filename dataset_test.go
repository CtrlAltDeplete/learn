package learn

import (
	"gonum.org/v1/gonum/mat"
	"reflect"
	"testing"
)

func TestBatchDataset(t *testing.T) {
	type args struct {
		xs        *mat.Dense
		ys        *mat.Dense
		batchSize int
		shuffle   bool
	}
	tests := []struct {
		name  string
		args  args
		wantX []*mat.Dense
		wantY []*mat.Dense
	}{
		{
			"happy path",
			args{
				mat.NewDense(8, 2, []float64{0, 0, 0, 1, 1 / 3, 0, 1 / 3, 1, 2 / 3, 0, 2 / 3, 1, 1, 0, 1, 1}),
				mat.NewDense(8, 1, []float64{0, 1, 2, 3, 0, 1, 2, 3}),
				4,
				false,
			},
			[]*mat.Dense{
				mat.NewDense(4, 2, []float64{0, 0, 0, 1, 1 / 3, 0, 1 / 3, 1}),
				mat.NewDense(4, 2, []float64{2 / 3, 0, 2 / 3, 1, 1, 0, 1, 1}),
			},
			[]*mat.Dense{
				mat.NewDense(4, 1, []float64{0, 1, 2, 3}),
				mat.NewDense(4, 1, []float64{0, 1, 2, 3}),
			},
		},
		{
			"single item dataset",
			args{
				mat.NewDense(1, 1, []float64{0}),
				mat.NewDense(1, 1, []float64{0}),
				1,
				false,
			},
			[]*mat.Dense{
				mat.NewDense(1, 1, []float64{0}),
			},
			[]*mat.Dense{
				mat.NewDense(1, 1, []float64{0}),
			},
		},
		{
			"cutoff data",
			args{
				mat.NewDense(3, 1, []float64{0, 1, 2}),
				mat.NewDense(3, 1, []float64{2, 1, 0}),
				2,
				false,
			},
			[]*mat.Dense{
				mat.NewDense(2, 1, []float64{0, 1}),
			},
			[]*mat.Dense{
				mat.NewDense(2, 1, []float64{2, 1}),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotX, gotY := BatchDataset(tt.args.xs, tt.args.ys, tt.args.batchSize, tt.args.shuffle)
			if !reflect.DeepEqual(gotX, tt.wantX) {
				t.Errorf("BatchDataset() = %v, want %v", gotX, tt.wantX)
			}
			if !reflect.DeepEqual(gotY, tt.wantY) {
				t.Errorf("BatchDataset() = %v, want %v", gotY, tt.wantY)
			}
		})
	}
}
