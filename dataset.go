package learn

import (
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

// BatchDataset splits the initial inputs (each row being one input) into multiple matrices
func BatchDataset(inputs *mat.Dense, batchSize int, shuffle bool) []*mat.Dense {
	var indices []int

	r, c := inputs.Dims()
	if shuffle {
		indices = rand.Perm(r)
	} else {
		indices = make([]int, r)
		for i := range indices {
			indices[i] = i
		}
	}

	var batches []*mat.Dense
	for i := 0; i < r-batchSize+1; i += batchSize {
		var batchData []float64
		for rowId := i; rowId < i+batchSize; rowId++ {
			rowData := inputs.RawRowView(rowId)
			batchData = append(batchData, rowData...)
		}
		batches = append(batches, mat.NewDense(batchSize, c, batchData))
	}

	return batches
}
