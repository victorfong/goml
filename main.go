package main

import "fmt"

func main() {
  trainImagesBytes,trainLabelsBytes, _, _ := LoadMNIST()

  model := NewModel(28 * 28)
  model.SetLoss(CrossEntropy{})

  model.AddLayer(NewLayer(64, ReLU{}))
  // model.AddLayer(NewLayer(64, ReLU{}))
  model.AddLayer(NewLayer(10, Softmax{}))

  trainImages := convertMatrix(trainImagesBytes)
  trainLabels := toCategorical(trainLabelsBytes)
  model.Train(trainImages[0:1], trainLabels)

  // c := CrossEntropy{}
  // a := []float64{
  //   0.0, 1.0, 0,
  // }
  //
  // b := []float64{
  //   0.228, 0.619, 0.153,
  // }
  //
  // output := c.Dloss(a, b)
  // fmt.Println("Cross Entropy:", output)

  // a := []float64{
  //   1.0, 2.0, 3.5,
  // }
  //
  // softmax := Softmax{}
  // output := softmax.Activate(a)
  // fmt.Println(output)


}

func convertMatrix(input [][]byte) [][]float64{
  result := make([][]float64, len(input))

  for i:=0; i<len(input); i++ {
    result[i] = make([]float64, len(input[i]))

    for j:=0; j<len(input[i]); j++ {
      result[i][j] = float64(input[i][j])
    }
  }

  return result
}

func toCategorical(input []byte) [][]float64{
  result := make([][]float64, len(input))

  for i:=0; i<len(input); i++ {
    result[i] = make([]float64, 10)
    result[i][input[i]] = 1.0
  }

  return result
}
