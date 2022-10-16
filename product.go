package learn

func product(existingItems [][]float64, itemsToChooseFrom []float64) [][]float64 {
	var newItems [][]float64
	for _, existing := range existingItems {
		for _, item := range itemsToChooseFrom {
			newItems = append(newItems, append(existing, item))
		}
	}
	return newItems
}

// Product returns a slice containing all possible products of the slices passed in.
func Product(itemSlices [][]float64) [][]float64 {
	var existingItems [][]float64
	if len(itemSlices) < 1 {
		return existingItems
	}

	for i := 0; i < len(itemSlices[0]); i++ {
		existingItems = append(existingItems, []float64{itemSlices[0][i]})
	}
	if len(itemSlices) == 1 {
		return existingItems
	}

	for i := 1; i < len(itemSlices); i++ {
		existingItems = product(existingItems, itemSlices[i])
	}
	return existingItems
}
