package main

// import "fmt"

func main() {
  trainImagesBytes, _, _, _ := LoadMNIST()
  model := NewModel(28 * 28)
  model.AddLayer(NewLayer(64, ReLU{}))
  // model.AddLayer(NewLayer(64, ReLU{}))
  model.AddLayer(NewLayer(10, Softmax{}))

  trainImages := convertMatrix(trainImagesBytes)
  model.Train(trainImages[0:1], trainImages)


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
