 #numpy
 ## Topics
 1. ndarrays
 2. accessing individual values and subsets inside an n-dimensional array
 3. How broadcasting works with ndarrays
 4. Doing linear algebra in NumPy
 5. How to generate points you can plot on a chart
 6. How to manipulate images as ndarrays

## 1) the ndarray

![[1IUqnxX.png]]
* They are homogenous (every element is the same data type)
### Create a 1D array

Just like a DataFrame, ndarray's have a `shape` property that gives you the dimensions. the `ndim` property returns an integer representation of the dimensions

```
my_array = np.array([1.1, 9.2, 8.1, 4.7])
my_array.shape
my_array.ndim
```
ouput: `(4,)`; `1`

### Creating a 2D Array (aka a matrix)

```
array_2d = np.array([[1, 2, 3, 9], 
                     [5, 6, 7, 8]])
                    
```
This array has 2 rows and 4 columns. In Numpy, the dimensions are axes. The first axis has length 2 and the second axis has length 4. It has {array_2d.shape\[0]} rows and {array_2.shape\[1]} columns.

To access particular values, provide an index for each dimension.
`array_2d[1, 2]` accesses the third value in the second row. Use the `:` operator to access
an entire row. 
`array_2d[0,:]` -> \[1, 2, 3, 9]

### N-Dimensional Arrays (Tensors)

```python
mystery_array = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                        
                         [[7, 86, 6, 98],
                          [5, 1, 0, 4]],
                          
                          [[5, 36, 32, 48],
                           [97, 0, 27, 18]]])

# Note all the square brackets!
mystery_array.shape
# (3, 2, 4)
# Can think of this as 3 matrices, each containing 2 rows and 4 columns?. 3 groups of 2 vectors each containg 4 items
# Retrieving a (3,2) matrix with the values [[ 0,  4], [ 7,  5], [ 5, 97]]
# 1st matrix
mystery_array[0]

# array([[0, 1, 2, 3],
#        [4, 5, 6, 7]])

# 1st vector in the 1st matrix

mystery_array[0][0]
# array([0, 1, 2, 3])
# 1st 2 items in the first vector in the first matrix

mystery_array[:,:,0]

# array([[ 0,  4],
#        [ 7,  5],
#    
```

np.array\[index_of_outer_most_array, indexy_of_inner array, ... , index_of_inner_most_array]

`mystery_array[:,:,0]` retrieves all the first elements along the third axis.

## 2. Generating and Manipulating ndarrays

### 1. Use `.arange()` to create a vector with values ranging from 10 to 29.

`a = np.arange(10, 30)`
### 2. Use Python slicing techniques on the vector to:

	2.1 Create an array containing only the last 3 values of the vector
	2.2 Create a subset with only the 4th, 5th, and 6th values
	2.3 Create a subset containing all values except the first 12
	2.4 Create a subset containing every second number

```python
last_three = a[-3:]
print(f"{last_three=}")
print(last_three)

subset = a[3:6]
print("subset=", end="")
print(subset)

all_but_first_12 = a[12:]
print("all but first 12=", end=" ")
print(all_but_first_12)

evens = a[0::2]
print("evens = ", end="")
print(evens)
```
### 3. Reverse the order of the values in the vector.

```python
# reverse = a [::-1]
reverse = np.flip(a)
print("reverse = ", end='')
print(reverse)
	np.flip(a) == a[::-1]
```
### 4. Print all indices of the non-zero elements in the array \[6, 0, 9, 0, 0, 5, 0]

```python
arr = np.array([6, 0, 9, 0, 0, 5, 0])
# non_zero_idx = [arr.index(n) for n in arr if n != 0]
print(non_zero_idx)
non_zero = arr.nonzero()
print(non_zero[0])
```

### 5. Use NumPy to generate a 3x3x3 array with random numbers

```python
rng = np.random.default_rng()
arr = rng.random(size=(3, 3))
arr

from numpy.random import random
random_arr = random((3, 3, 3))
```
### 6. Use `.linspace()` to create a vector x of size 9 with values evenly spaced between 0 and 100 (inclusive)

```python
x = np.linspace(0, 100, num=9)
print(x)
x.shape
```

### 7. Use `.linspace()` to create another vector y size 9 with values between -3 and 3 (inclusive). Plot x and y on a linechart using Matplotlib

```python
y = np.linspace(-3, 3, 9)
y

plt.plot(x, y)
plt.show()
```
### 8. Use NumPy to generate an array called `noise` with shape 128x128x3 that has random values, then use Matplotlib's `.imshow()` to display the array as an image.

```python
arr = rng.random(size=(128, 128, 3))
plt.imshow(arr)

noise = np.random.random((128, 128, 3))
print(noise.shape)
plt.imshow(noise)
```

![[Untitled 1.png]]

The above is a 128 x 128 image. the R G B values of each pixel are determined by the inner most values
	\[
		\[
			\[
			   a, b, c
			]	  # 128 three value groups
		] # 128 groups of 128 three value groups
	]

## 3. Broadcasting, Scalars, and Matrix Multiplication

### Linear Algebra with Vectors

Take two vectors:

```python
v1 = np.array([4, 5, 2, 7])
v2 = np.array([2, 1, 3, 3])
```

**Addition**

`v1 + v2` -> `array([6, 6, 5, 10])`

Adds v1\[i] and v2\[i] for result, insteading of concatenating like added lists would.

**Multiplication**
`v1 * v2` -> `array([8, 5, 6, 21])`

### Broadcasting

Sometimes, you want to do an operation between an array and a single number. In mathematics, this single number is often called a **scalar**.

For example, you might want to multiply every value in your NumPy array by 2.

```
v1 * 2  # array([4, 5, 2, 7]) every element gets multiplied by 2
# result = array([8, 10, 4, 14])
```


### Matrix Multiplication

![[2020-10-12_17-01-09-7243f82f4dd88bec877e3206fb9d9add.png]]

What if we are not multiplying our dnarray by a single number, but by another vector or a 2-D array? We must [follow the rules of linear algebra](https://en.wikipedia.org/wiki/Matrix_multiplication#Illustration)

`matmul()` for multiplying matrices

multiply a `m x n` matrix and a `n x p` matrix will result in a `m x p` matrix
![[9196c0c24ad20c3b18582bc78785fa405d91c7c3.svg]]
![[7d3ce5d06e84e1a8575ce6f1d47a90d006baf628.svg]]
![[ee372c649dea0a05bf1ace77c9d6faf051d9cc8d.svg]]
>That is, the entry c i j ![{\displaystyle c_{ij}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/2a106b8753c0948250bbc2e03df3207799beaedb) of the product is obtained by multiplying term-by-term the entries of the ith row of **A** and the jth column of **B**, and summing these n products. In other words, c i j ![{\displaystyle c_{ij}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/2a106b8753c0948250bbc2e03df3207799beaedb) is the [dot product](https://en.wikipedia.org/wiki/Dot_product "Dot product") of the ith row of **A** and the jth column of **B**.

`c = np.matmul(a, b)`
`c = a@b`

## 4. Manipulating Images as ndarrays

We will use two libraries to help us work with image data (an array where each element is composed of 3 values: the red, green and blue values of the pixel)

* scipy
* PIL
* 
```
from scipy import misc, datasets  # image of raccoon
from PIL import Image  # for reading image files
```

`raccoon_img = misc.face()`
**`misc` is deprecated, use `datasets` instead.**

3 matrices stacked on top of each other, one for the red values, one for the green values, and one for the red values

Display it with Matplotlib's `.imshow()`

`plt.imshow(raccoon_img)`

#### [More about images](https://towardsdatascience.com/exploring-the-mnist-digits-dataset-7ff62631766a)

>## Image properties
>
>Every image has three main properties:
>
>- **Size** — This is the height and width of an image. It can be represented in centimeters, inches or even in pixels.
>- **Color space** — Examples are **RGB** and **HSV** color spaces.
>- **Channel** — This is an attribute of the color space. For example, RGB color space has three types of colors or attributes known as **_r_**_ed_, **_g_**_reen_ and **_b_**_lue_ (hence the name rgb).

a rgb image has 3 channels (red, green and blue) while a grayscale image has only 1 channel.

### challenge: convert the image to black and white

`y_linear = 0.2126*r_linear + 0.7152*g_linear + 0.0722*b_linear`

rgb values must be between 0 and 1 for this formula to work (srgb format) so we can divide every element by 255 to convert it to srgb

### challenge: flip the image

use `np.flip()` to reverse the array. this will cause the image to appear upside down as it now being rendered in the opposite order.

### challenge: rotate the image
`ndarray.reshape` <- **doesn't work**
maybe need to transpose the 1024 x 768 array?
when you call `ndarray.transpose(t)` where t is a tuple of ints representing the original axes, the dimensions of the transposed array are determined by their index in the tuple.
`arr1.shape` = (1024, 768, 3)
`arr1.transpose(1, 0, 2).shape` = (768, 1024, 3)
you can specify axes in `np.flip()` with a tuple of ints as well. to flip the image vertically, flip the transposed array along its vertical axis (0)

`plt.imshow(np.flip(rotated_img, axis=0))`

