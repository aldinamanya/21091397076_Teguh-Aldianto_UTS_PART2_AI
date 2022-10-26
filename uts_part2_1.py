# TEGUH ALDIANTO
# 21091397076
# UTS PART 2 KECERDASAN BUATAN

#ketik import numpy lalu inisialisasi nu
import numpy as nu

#masukkan variabel yang sudah ditentukan (inputs 10 ; batch 6 = matriks 6*10)
inputs = [
    [1.0, 2.3, 3.2, 2.5, 13.3, 9.1, 12.3, 14.3, 12.2, 11.0],
		[2.2, 5.2, -1.2, 2.9, -2.1, 3.5, 6.5, 3.5, -12.4, 15.2],
    [-9.2, 1.1, 4.2, 1.5, -9.3, 24.2, 5.3, 6.2, 23.2, 35.0],
    [-5.3, 5.0, 9.3, -2.1, 0.3, -3.4, 8.3, 7.3, 36.2, 34.5],
    [3.9, 9.2, 0.2, -1.3, 13.2, 44.2, -4.8, 4.4, -2.2, 2.4],
		[-1.5, 2.7, 3.3,-0.8, 19.2, -9.1, 8.2, 12.5, 44.4, 4.6]
    ]

#panjang weights = panjang neuron (10) ; jumlah weights = jumlah neuron (5)
weights = [
    [0.2, 3.8, -0.5, 1.2, 23.3, 23.1, 22.0, 2.4, 4.5, 4.4],
		[0.5, -0.9, 0.2, -3.5, 0.3, 23.0, 0.2, 3.2, 23.0, 0.2],
    [-2.3, 3.2, 23.2, 3.2, 0.4, 0.5, -4.5, 4.9, 0.2, -9.4],
    [-5.4, 5.6, 4.5, 43.4, 6.4, 0.4, 9.0, 3.3, -3.4, 20.9],
		[-0.6, -0.7, 0.1, 20.7, -3.5, 0.5, 5.3, 6.5, 7.6, 0.8]
    ]

#masukkan bias pada layer 1 dengan jumlah 5 neuron
biases = [2.0, 3.4, 0.5, -5.4, -0.4]
#panjang weights = neuron layer 1 (5) ; jumlah weights = jumlah neuron layer 2 (3)
weights2 = [
    [0.1,-10.1, 0.5, 21.2, 40.3],
		[-0.5, 1.2,-0.3, 12.4, -9.3],
		[-0.4, 0.3,-1.4, -83.1, 1.3]
    ]

#masukkan bias pada layer 2 dengan jumlah 3 neuron
biases2 = [-1.4, 9.2,-20.5]
#pada baris ini untuk menghitung layer 1
layer1_outputs = nu.dot(inputs, nu.array(weights).T) + biases
#pada baris ini untuk menghitung layer 2
layer2_outputs = nu.dot(layer1_outputs, nu.array(weights2).T) + biases2
#print output layer 2
print(layer2_outputs)