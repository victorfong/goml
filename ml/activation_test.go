package ml_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
  . "github.com/victorfong/goml/ml"
)

var _ = Describe("Activation Operations", func() {

	BeforeEach(func() {
		// do nothing yet
	})

	Describe("Activation", func() {
		Context("When using Softmax activation function", func() {

			It("should activate correctly", func() {
          a := []float64{
            1.8658, 2.2292, 2.8204,
          }

          softmax := Softmax{}
          result := softmax.Activate(a)

          expectedResult := []float64{
            0.19857651019773828, 0.28559492698949396, 0.5158285628127679,
          }

          Expect(result).To(Equal(expectedResult))
			})

			It("should deActivate correctly", func() {
          a := []float64{
            1.8658, 2.2292, 2.8204,
          }

          softmax := Softmax{}
          result := softmax.DeActivate(a)

          expectedResult := []float64{
            0.15914387979542582, 0.2040304646673596, 0.2497494565992823,
          }

          Expect(result).To(Equal(expectedResult))
			})
		})



		Context("When using Sigmoid activation function", func() {

			It("should activate correctly", func() {
          a := []float64{
            0.3775, 0.39249999999999996,
          }

          sigmoid := Sigmoid{}
          result := sigmoid.Activate(a)

          expectedResult := []float64{
            0.5932699921071872, 0.596884378259767,
          }

          Expect(result).To(Equal(expectedResult))
			})

			It("should activate 1 by 3 matrix correctly", func() {
          a := []float64{
            2.73, 2.76, 4.001,
          }

          sigmoid := Sigmoid{}
          result := sigmoid.Activate(a)

          expectedResult := []float64{
            0.9387738371800919, 0.9404756340234984, 0.9820314442330851,
          }

          Expect(result).To(Equal(expectedResult))
			})
		})

    Context("When using derivative of Sigmoid activation function", func() {

			It("should calculate correctly", func() {
          a := []float64{
            0.75136507, 0.772928465,
          }

          sigmoid := Sigmoid{}
          result := sigmoid.DeActivate(a)

          expectedResult := []float64{
            0.18681560158389512, 0.17551005299274378,
          }

          Expect(result).To(Equal(expectedResult))
			})

			It("should calculate correctly", func() {
          a := []float64{
            2.73, 2.76, 4.001,
          }

          sigmoid := Sigmoid{}
          result := sigmoid.DeActivate(a)

          expectedResult := []float64{
            0.18681560158389512, 0.17551005299274378, 1.0,
          }

          Expect(result).To(Equal(expectedResult))
			})
		})

	})
})
