package ml

import "fmt"

// type Model interface{
//   AddLayer(Layer)
//   Train([][]float64, [][]float64)
// }

type Model struct {
  inputUnits int
  initialized bool
  layers []*Layer
  loss Loss
}

func NewModel(inputUnits int) *Model{
  result := Model{
    inputUnits: inputUnits,
    initialized: false,
  }

  return &result
}

func (m *Model) init() {
  inputUnits := m.inputUnits
  for i:=0; i<len(m.layers); i++ {
    m.layers[i].Init(inputUnits)
    inputUnits = m.layers[i].GetOutputUnits()
  }
  m.initialized = true
}

func (m *Model) SetLoss(loss Loss) {
  m.loss = loss
}

func (m *Model) AddLayer(layer *Layer) {
  fmt.Println("Before Total Layers: ", len(m.layers))
  m.layers = append(m.layers, layer)
  fmt.Println("After Total Layers: ", len(m.layers))
}

func (m *Model) Train(data [][]float64, labels [][]float64){
  if !m.initialized {
    m.init()
  }

  var trainData []float64

  for i:=0; i<len(data); i++ {
    // fmt.Println("Training Iteration #", i, " of ", len(data), " through ", len(m.layers), " layers")
    trainData = data[i]

    // Forward
    for j:=0; j<len(m.layers); j++ {
      trainData = m.layers[j].Forward(trainData)
    }

    loss := m.loss.Loss(labels[i], trainData)
    fmt.Println("Output: ", trainData)
    fmt.Println("Loss:", loss)

    // Back
    // de := m.loss.Dloss(labes[i], trainData)
    // for j:=len(m.layers)-1; j>=0; j-- {
    //   m.layers[j].BackProp(de)
    // }
  }



}
