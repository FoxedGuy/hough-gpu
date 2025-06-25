# Fast hough transform using GPU

## Dependencies
- OpenCV
- CUDA
- OpenMP

## Building
When in the project folder:
```
$ mkdir build && cd build
$ cmake build ..
$ cmake --build .
```

## Running
Currently, there are four programs:
- hough_gpu_streams: line detection using gpu and streams
- hough_seq: line detection using cpu
- hough_circles_gpu: circles detection within radius range using gpu (and cpu)

### Execution
#### Lines
```
./program_name filename threshold experiments
```
where:
- `program_name`: name of a selected program.
- `filename`: an image file located in the `pictures` folder.
- `threshold`: value used while filling the accumulator (inclusive).
- `experiments`: number of experiments.

#### Circles
```
./program_name filename min_radius max_radius threshold experiments
```
where:
- `program_name`: name of a selected program.
- `filename`: an image file located in the `pictures` folder.
- `min_radius`: minimal radius (inclusive).
- `max_radius`: maximal radius (inclusive).
- `threshold`: value used while filling the accumulator (inclusive).
- `experiments`: number of experiments.


