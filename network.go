package learn

import (
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
)

type Network []Layer

func NewNetwork(inputSize, outputSize, denseLayers, nodesPerLayer int) Network {
	var network = Network{NewDense(inputSize, nodesPerLayer)}
	for i := 0; i < denseLayers; i++ {
		network = append(network, NewDense(nodesPerLayer, nodesPerLayer))
	}
	network = append(network, NewDense(nodesPerLayer, outputSize))
	network = append(network, &Activation{
		Activation:      Sigmoid,
		ActivationPrime: SigmoidPrime,
	})
	return network
}

func (network *Network) MarshalJSON() ([]byte, error) {
	type layerWithType struct {
		LayerType string `json:"type"`
		Data      Layer  `json:"layer"`
	}

	var layers []layerWithType
	for i := range *network {
		layers = append(layers, layerWithType{
			(*network)[i].Name(),
			(*network)[i],
		})
	}

	return json.Marshal(layers)
}

func (network *Network) UnmarshalJSON(data []byte) error {
	type layerWithType struct {
		LayerType string          `json:"type"`
		Data      json.RawMessage `json:"layer"`
	}

	var layers []layerWithType
	if err := json.Unmarshal(data, &layers); err != nil {
		return err
	}

	for _, l := range layers {
		switch l.LayerType {
		case "Dense":
			var layer Dense
			if err := json.Unmarshal(l.Data, &layer); err != nil {
				return err
			}
			*network = append(*network, &layer)
		case "Activation":
			var layer Activation
			if err := json.Unmarshal(l.Data, &layer); err != nil {
				return err
			}
			*network = append(*network, &layer)
		default:
			return fmt.Errorf("unkown layer type: %s", l.LayerType)
		}
	}

	return nil
}

func (network *Network) Predict(x *mat.Dense) *mat.Dense {
	var yPred = *x
	for _, layer := range *network {
		yPred = layer.Forward(yPred)
	}
	return &yPred
}

func (network *Network) Train(xs []*mat.Dense, ys []*mat.Dense, errorFunc ErrorFunction, gradFunc GradientFunction,
	epochs int, learningRate float64, checkIn int) {
	for i := 1; i <= epochs; i++ {
		err := 0.0
		for j, batch := range xs {
			var yPred = network.Predict(batch)

			err += errorFunc(yPred, ys[j])
			errPrime := gradFunc(yPred, ys[j])

			for k := len(*network) - 1; k >= 0; k-- {
				*errPrime = (*network)[k].Backward(*errPrime, learningRate)
			}
		}

		if i%checkIn == 0 {
			fmt.Printf("epoch %d/%d\terror=%f\n", i, epochs, err)
		}
	}
}
