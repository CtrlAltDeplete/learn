package learn

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

// BatchDataset splits the initial xs (each row being one input) into multiple matrices
func BatchDataset(xs, ys *mat.Dense, batchSize int, shuffle bool) ([]*mat.Dense, []*mat.Dense) {
	xr, xc := xs.Dims()
	yr, yc := ys.Dims()
	if xr != yr {
		panic(fmt.Errorf(""))
	}

	var indices []int

	if shuffle {
		indices = rand.Perm(xr)
	} else {
		indices = make([]int, xr)
		for i := range indices {
			indices[i] = i
		}
	}

	var xBatches []*mat.Dense
	for i := 0; i < xr-batchSize+1; i += batchSize {
		var batchData []float64
		for rowId := i; rowId < i+batchSize; rowId++ {
			rowData := xs.RawRowView(rowId)
			batchData = append(batchData, rowData...)
		}
		xBatches = append(xBatches, mat.NewDense(batchSize, xc, batchData))
	}

	var yBatches []*mat.Dense
	for i := 0; i < xr-batchSize+1; i += batchSize {
		var batchData []float64
		for rowId := i; rowId < i+batchSize; rowId++ {
			rowData := ys.RawRowView(rowId)
			batchData = append(batchData, rowData...)
		}
		yBatches = append(yBatches, mat.NewDense(batchSize, yc, batchData))
	}

	return xBatches, yBatches
}
