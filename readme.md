# Fast hough transform using GPU

### Dependencies
- OpenCV
- CUDA
- OpenMP

### Building
When in the main folder:
```
$ mkdir build && cd build
$ cmake build ..
$ cmake --build .
```

### Running
Currently, there are three programs:
- hough_gpu_streams: using streams
- hough_gpu_managed: using managed cuda allocation
- hough_seq: using basic cpu implementation

Execution looks like this:
```
./program_name filename threshold
```
where:
- `program_name`: name of a selected program.
- `filename`: an image file located in the `pictures` folder.
- `threshold`: value used while filling the accumulator.

Every program is compared with sequential OpenCV implementation.
