package learn

import (
	"fmt"
	"reflect"
	"testing"
)

func Test_product(t *testing.T) {
	type args struct {
		existingItems     [][]float64
		itemsToChooseFrom []float64
	}
	tests := []struct {
		name string
		args args
		want [][]float64
	}{
		{
			"happy path",
			args{
				[][]float64{{0}, {1}},
				[]float64{2, 3},
			},
			[][]float64{{0, 2}, {0, 3}, {1, 2}, {1, 3}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := product(tt.args.existingItems, tt.args.itemsToChooseFrom); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("product() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestProduct(t *testing.T) {
	tests := []struct {
		name string
		args [][]float64
		want [][]float64
	}{
		{
			"happy path",
			[][]float64{{0, 1}, {2, 3}},
			[][]float64{{0, 2}, {0, 3}, {1, 2}, {1, 3}},
		},
		{
			"empty",
			[][]float64{},
			nil,
		},
		{
			"one item",
			[][]float64{{0}},
			[][]float64{{0}},
		},
		{
			"varying input lengths",
			[][]float64{{0, 1}, {2, 3, 4}},
			[][]float64{{0, 2}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {1, 4}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Product(tt.args); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Product() = %v, want %v", got, tt.want)
			}
		})
	}
}

func ExampleProduct() {
	var a = []float64{0, 1}
	var b = []float64{2, 3}
	var c = []float64{3, 4}

	var prod = Product([][]float64{a, b, c})
	fmt.Println(prod)
	/*
		[0 2 3]
		[0 2 4]
		[0 3 3]
		[0 3 4]
		[1 2 3]
		[1 2 4]
		[1 3 3]
		[1 3 4]
	*/
}
