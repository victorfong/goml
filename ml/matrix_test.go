package ml_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
  . "github.com/victorfong/goml/ml"
)

var _ = Describe("Matrix Operations", func() {

	BeforeEach(func() {
		// do nothing yet
	})

	Describe("Multiplication", func() {
		Context("when 2 matrices multiplies", func() {

			It("should return product", func() {
          a := [][]float64{
            {1, 2, 3},
            {4, 5, 6},
          }

          b := [][]float64{
            {7, 8},
            {9, 10},
            {11, 12},
          }

          result := MatrixMultiply(a, b)

          expectedResult := [][]float64{
            {58, 64},
            {139, 154},
          }

          Expect(result).To(Equal(expectedResult))
			})
		})
	})
})
