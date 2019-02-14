package ml

import (
    "fmt"
    "encoding/binary"
    "os"
)

func LoadMNIST() ([][]byte, []byte, [][]byte, []byte) {
  trainImages := loadImages("data/train-images-idx3-ubyte")
  trainLabels := loadLabels("data/train-labels-idx1-ubyte")

  testImages := loadImages("data/t10k-images-idx3-ubyte")
  testLabels := loadLabels("data/t10k-labels-idx1-ubyte")
  return trainImages, trainLabels, testImages, testLabels
}

func loadLabels(filename string) []byte{
  f, err := os.Open(filename)
  if err != nil {
    fmt.Println("Error: ", err)
  }

  data := make([]byte, 4)
  f.Read(data)
  magicNumber := binary.BigEndian.Uint32(data)
  if magicNumber != 2049 {
    fmt.Println("Something is wrong. Magic Number does not match")
  }

  data = make([]byte, 4)
  f.Read(data)
  numOfItems := binary.BigEndian.Uint32(data)

  // result := make([]byte, numOfItems)
  result := make([]byte, numOfItems)
  f.Read(result)
  return result
}

func loadImages(filename string) [][]byte{
  f, err := os.Open(filename)
  if err != nil {
    fmt.Println("Error: ", err)
  }

  data := make([]byte, 4)
  f.Read(data)
  fmt.Println("Bytes: ", data)
  magicNumber := binary.BigEndian.Uint32(data)
  fmt.Println("Magic Number: ", magicNumber)

  data = make([]byte, 4)
  f.Read(data)
  fmt.Println("Bytes: ", data)
  numOfItems := binary.BigEndian.Uint32(data)
  fmt.Println("Num of Items: ", numOfItems)

  data = make([]byte, 4)
  f.Read(data)
  fmt.Println("Bytes: ", data)
  width := binary.BigEndian.Uint32(data)
  fmt.Println("Width: ", width)

  data = make([]byte, 4)
  f.Read(data)
  fmt.Println("Bytes: ", data)
  height := binary.BigEndian.Uint32(data)
  fmt.Println("Height: ", height)

  result := make([][]byte, numOfItems)
  for i:=0;i < int(numOfItems);i++ {
    result[i] = make([]byte, width * height)
    f.Read(result[i])
  }
  return result
}
