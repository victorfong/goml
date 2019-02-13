package main

type Model interface{
  AddLayer(Layer)
  Train([][]float64, []float64)
}

type model struct {
  inputUnits int
  initialized bool
  layers []Layer
}

func NewModel(inputUnits int) Model{
  return &model{
    inputUnits: inputUnits,
    initialized: false,
  }
}

func (m model) init() {
  inputUnits := m.inputUnits
  for i:=0; i<len(m.layers); i++ {
    m.layers[i].Init(inputUnits)
    inputUnits = m.layers[i].GetOutputUnits()
  }
}

func (m model) AddLayer(layer Layer) {
  m.layers = append(m.layers, layer)
}

func (m model) Train([][]float64, []float64){
  if !m.initialized {
    m.init()
  }
}
