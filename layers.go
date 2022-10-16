package learn

import (
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"reflect"
)

type Layer interface {
	Name() string
	Forward(input mat.Dense) mat.Dense
	Backward(outputError mat.Dense, learningRate float64) mat.Dense
}

// Dense a standard densely-connected neural network layer
type Dense struct {
	Weights mat.Dense
	Biases  mat.Dense

	_input  mat.Dense
	_output mat.Dense
}

func NewDense(inputSize, outputSize int) *Dense {
	var layer Dense
	var data []float64
	for i := 0; i < inputSize*outputSize; i++ {
		data = append(data, rand.Float64()-0.5)
	}

	layer.Weights = *mat.NewDense(inputSize, outputSize, data)

	data = []float64{}
	for i := 0; i < outputSize; i++ {
		data = append(data, rand.Float64()-0.5)
	}
	layer.Biases = *mat.NewDense(1, outputSize, data)

	return &layer
}

func (layer *Dense) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Weights serializedMatrix `json:"weights"`
		Biases  serializedMatrix `json:"biases"`
	}{
		serializeDenseMatrix(&layer.Weights),
		serializeDenseMatrix(&layer.Biases),
	})
}

func (layer *Dense) UnmarshalJSON(data []byte) error {
	var serialized = struct {
		Weights serializedMatrix `json:"weights"`
		Biases  serializedMatrix `json:"biases"`
	}{}

	if err := json.Unmarshal(data, &serialized); err != nil {
		return err
	}

	layer.Weights = *serialized.Weights.DenseMatrix()
	layer.Biases = *serialized.Biases.DenseMatrix()
	return nil
}

func (layer *Dense) Name() string {
	return "Dense"
}

func (layer *Dense) Forward(input mat.Dense) mat.Dense {
	layer._input = input
	layer._output.Mul(&layer._input, &layer.Weights)

	var rowData = layer.Biases.RawRowView(0)
	var data []float64

	r, c := layer._output.Dims()
	for i := 0; i < r; i++ {
		data = append(data, rowData...)
	}

	var biases = mat.NewDense(r, c, data)
	layer._output.Add(&layer._output, biases)
	return layer._output
}

func (layer *Dense) Backward(outputError mat.Dense, learningRate float64) mat.Dense {
	var inputError, weightsError mat.Dense

	inputError.Mul(&outputError, layer.Weights.T())
	weightsError.Mul(layer._input.T(), &outputError)

	weightsError.Scale(learningRate, &weightsError)
	layer.Weights.Sub(&layer.Weights, &weightsError)

	outputError.Scale(learningRate, &outputError)

	r, c := outputError.Dims()
	var colAverages []float64
	for i := 0; i < c; i++ {
		colAverages = append(colAverages, 0)
	}

	for i := 0; i < r; i++ {
		rawRow := outputError.RawRowView(i)
		for j, v := range rawRow {
			colAverages[j] += v
		}
	}

	for i := range colAverages {
		colAverages[i] /= float64(r)
	}

	var outputErrorAverages = mat.NewDense(1, len(colAverages), colAverages)
	layer.Biases.Sub(&layer.Biases, outputErrorAverages)
	return inputError
}

// Activation a layer for activation functions
type Activation struct {
	Activation      activationFunction
	ActivationPrime activationFunction

	_input  mat.Dense
	_output mat.Dense
}

func (layer *Activation) MarshalJSON() ([]byte, error) {
	var activation string

	switch reflect.ValueOf(layer.Activation).Pointer() {
	case reflect.ValueOf(Sigmoid).Pointer():
		activation = "Sigmoid"
	default:
		panic(fmt.Errorf("unknown activation %v", layer.Activation))
	}

	return json.Marshal(map[string]string{"activation": activation})
}

func (layer *Activation) UnmarshalJSON(data []byte) error {
	var serialized = map[string]string{}

	if err := json.Unmarshal(data, &serialized); err != nil {
		return err
	}

	switch serialized["activation"] {
	case "Sigmoid":
		layer.Activation = Sigmoid
		layer.ActivationPrime = SigmoidPrime
	default:
		return fmt.Errorf("unkown activation %v", serialized["activation"])
	}

	return nil
}

func (layer *Activation) Name() string {
	return "Activation"
}

func (layer *Activation) Forward(input mat.Dense) mat.Dense {
	layer._input = input
	layer._output = layer.Activation(input)
	return layer._output
}

func (layer *Activation) Backward(outputError mat.Dense, _ float64) mat.Dense {
	var inputError = layer.ActivationPrime(layer._input)
	inputError.MulElem(&inputError, &outputError)
	return inputError
}
