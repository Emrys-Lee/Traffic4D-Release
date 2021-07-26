#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <glog/logging.h>
#include <iterator>
#include <unordered_set>
#include <cmath>
#include <math.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define EPS T(0.00001)

using namespace pybind11::literals;
namespace py = pybind11;
using namespace Eigen;

struct FittingResidual {
    FittingResidual(double *rt_in, double* time_in, double coeff_3_in): rt(rt_in), time(time_in), coeff_3(coeff_3_in){}
    double *rt;
    double *time;
    double coeff_3;

    template<typename T>
    bool operator()(const T* const spline, T *residual) const {
        // parameter to be optimized
        // residual size 2
        residual[0] = spline[0]*time[0]*time[0]*time[0] + spline[1]*time[0]*time[0] +\
                      spline[2]*time[0] + spline[3] - rt[3];
        residual[1] = spline[4]*time[0]*time[0]*time[0] + spline[5]*time[0]*time[0] +\
                      spline[6]*time[0] + spline[7]  - rt[4];
        auto vx = 3.0*spline[0]*time[0]*time[0] + 2*time[0]*spline[1] + spline[2];
        auto vy = 3.0*spline[4]*time[0]*time[0] + 2*time[0]*spline[5] + spline[6];
        auto v_norm = sqrt(vx*vx + vy*vy);
        vx /= v_norm, vy /= v_norm;
        residual[2] = 1.0*(vx - rt[0]);
        residual[3] = 1.0*(vy - rt[1]);
        residual[4] = coeff_3*spline[0];
        residual[5] = coeff_3*spline[4];
        residual[6] = 0.0*spline[1];
        residual[7] = 0.0*spline[5];
        // auto grad_square = (spline[0]*time[0]+spline[1])*(spline[0]*time[0]+spline[1]) +\
        //          (spline[3]*time[0]+spline[4])*(spline[3]*time[0]+spline[4]);
        // auto cross = (spline[0]*time[0]+spline[1])*spline[3]-(spline[3]*time[0]+spline[4])*spline[0];
        return true;
    }
};


py::object spline_fitting(py::buffer rt_, py::buffer time_, py::buffer spline_, py::buffer cost_,
                int num_obj, int num_iter, double coeff_3){
    auto buf0 = rt_.request(); // num_obj*6
    double *rt = static_cast<double *>(buf0.ptr);
    auto buf1 = time_.request();
    double *time = static_cast<double *>(buf1.ptr);
    auto buf2 = spline_.request();
    double *spline = static_cast<double *>(buf2.ptr); // 6
    auto buf3 = cost_.request();
    double *cost = static_cast<double *>(buf3.ptr);
    double *rt_i, *time_i;
    ceres::Problem problem;
    for (int i = 0; i <num_obj; i++){
        rt_i = rt + 6*i;
        time_i = time + i;
        ceres::CostFunction *cost_function;
        cost_function = new ceres::AutoDiffCostFunction <FittingResidual, 8, 8>(
                      new FittingResidual(rt_i, time_i, coeff_3)
                );
        problem.AddResidualBlock(cost_function, NULL, spline);

    }
    ceres::Solver::Options options;
    options.max_num_iterations = num_iter;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  //ceres::SPARSE_SCHUR;  //ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout<<summary.FullReport()<<"\n";
    cost[0] = summary.final_cost;
    // return optimized values
    py::array_t <double> x_spline = py::array_t <double>(
            py::buffer_info(
                    spline,
                    sizeof(double), //itemsize
                    py::format_descriptor <double>::format(),
                    1, // ndim
                    std::vector<size_t>{(unsigned long) (8)}, // shape
                    std::vector<size_t>{sizeof(double)} // strides
            )
    );
    py::array_t <double> x_cost = py::array_t <double>(
            py::buffer_info(
                    cost,
                    sizeof(double), //itemsize
                    py::format_descriptor <double>::format(),
                    1, // ndim
                    std::vector<size_t>{(unsigned long) (1)}, // shape
                    std::vector<size_t>{sizeof(double)} // strides
            )
    );
    py::list outputs;
    outputs.append(x_spline);
    outputs.append(x_cost);
    return outputs;
}


PYBIND11_PLUGIN(ceres_spline) {
        py::module m("ceres_spline", "Python bindings to the Ceres-Solver minimizer.");
        // google::InitGoogleLogging("ceres_spline");
        m.def("spline_fitting", &spline_fitting, "Fitting a spline to a car trajectory");
        return m.ptr();
}
