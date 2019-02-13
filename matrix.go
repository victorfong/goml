package main

// import "fmt"

func matrixMultiply(a [][]float64, b [][]float64, i int, j int) float64{
  var result float64
  for k := 0; k < len(a[i]); k++ {
    result += a[i][k] * b[k][j]
  }
  return result
}

func MatrixMultiply(a [][]float64, b [][]float64) [][]float64 {
  if len(a) == 0 || len(b) == 0{
    var nothing [][]float64
    return nothing
  }

  resultHeight := len(a)
  resultWidth := len(b[0])

  result := make([][]float64, resultHeight)

  for i := 0; i < resultHeight; i++ {
    result[i] = make([]float64, resultWidth)

    for j := 0; j < resultWidth; j++ {

      item := matrixMultiply(a, b, i, j)
      result[i][j] = item
    }
	}

  return result
}

// func main() {
//   a := [][]float64{
//     {1, 2, 3},
//     {4, 5, 6},
//   }
//
//   b := [][]float64{
//     {7, 8},
//     {9, 10},
//     {11, 12},
//   }
//
//   result := MatrixMultiply(a, b)
//   fmt.Println("result: ", result)
// }
