package learn

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

func matricesAlmostEqual(got, want *mat.Dense, tolerance float64) bool {
	gotR, gotC := got.Dims()
	wantR, wantC := want.Dims()
	if gotR != wantR || gotC != wantC {
		return false
	}

	for i := 0; i < gotR; i++ {
		gotRow := got.RawRowView(i)
		wantRow := want.RawRowView(i)

		for j := 0; j < len(gotRow); j++ {
			gotVal := gotRow[j]
			wantVal := wantRow[j]

			if tolerance < math.Abs(gotVal-wantVal) {
				return false
			}
		}
	}

	return true
}
