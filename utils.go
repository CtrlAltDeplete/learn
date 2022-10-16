package learn

import "gonum.org/v1/gonum/mat"

type serializedMatrix struct {
	Rows    int       `json:"rows"`
	Columns int       `json:"columns"`
	Data    []float64 `json:"data"`
}

func serializeDenseMatrix(dense *mat.Dense) serializedMatrix {
	r, c := dense.Dims()

	var data []float64
	for i := 0; i < r; i++ {
		data = append(data, dense.RawRowView(i)...)
	}

	return serializedMatrix{
		Rows:    r,
		Columns: c,
		Data:    data,
	}
}

func (s *serializedMatrix) DenseMatrix() *mat.Dense {
	return mat.NewDense(s.Rows, s.Columns, s.Data)
}
