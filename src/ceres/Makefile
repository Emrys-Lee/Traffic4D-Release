all: ceres1 ceres2
ceres1: ceres_reconstruct.cc
			/usr/bin/clang++-6.0 -std=c++11 -g -O0 -Wno-deprecated -shared -o ceres_reconstruct.so ceres_reconstruct.cc -I /usr/include/eigen3 -I /usr/local/include -I /usr/local/include/pybind11/ -lgomp -lglog -lceres -lcholmod -fPIC -L /usr/local/lib -L /usr/lib/x86_64-linux-gnu/ $(shell python3-config --libs --cflags)
ceres2: ceres_spline.cc
			/usr/bin/clang++-6.0 -std=c++11 -g -O0 -Wno-deprecated -shared -o ceres_spline.so ceres_spline.cc -I /usr/include/eigen3 -I /usr/local/include -I /usr/local/include/pybind11/ -lgomp -lglog -lceres -lcholmod -fPIC -L /usr/local/lib -L /usr/lib/x86_64-linux-gnu/ $(shell python3-config --libs --cflags)
# -L ~/anaconda3/lib

clean:
	rm *.so