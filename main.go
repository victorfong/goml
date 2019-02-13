package main

func main() {
  // LoadMNIST()
  model := NewModel(28 * 28)
  model.AddLayer(NewLayer(64, ReLU{}))
}
