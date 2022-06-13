
Welcome to our improved C++ implementation of the following paper:

Unmixing-Based Soft Color Segmentation for Image Manipulation

Yagiz Aksoy, Tunc Ozan Aydin, Aljosa Smolic and Marc Pollefeys, ACM Trans. Graph., 2017

# Dependencies

For this project, you need:

Ubuntu 14.04 or 16.04
OpenCV 3.x

You can also install the additional dependencies via terminal using:

`sudo apt-get install libpthread-stubs0-dev libboost-all-dev`

# Installation

Here are the steps to clone and build the project:

1. Clone the code
2. Build the project using the following commands:

- Create a build directory using `mkdir build`
- Navigate to the build directory using `cd build`
- Build the project using `cmake ..`
- Lastly, compile using `make`

# Running the code

Use the following command to run:

`./SoftSegmentation <path/to/image/> <tau_parameterer(optional)> <results directory>`

For example:
`./SoftSegmentation ../4_small.png 15 results`

The results will be saved in the specified results folder.

You can try a tau parameter of 11 for good results.

Repository maintained by @qubilyan.