#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <glog/logging.h>
#include <iterator>
#include <unordered_set>
#include <cmath>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define EPS T(0.00001)

using namespace pybind11::literals;
namespace py = pybind11;
using namespace Eigen;

// use 13 dof features to do PCA
struct ReprojectionResidualFixedFNewPCA {
    ReprojectionResidualFixedFNewPCA(double *locs_2d_in, Matrix <double, 13, 5> &pca_comps_in,
                                     Matrix <double, 13, 1> &mean_shape_in, double cad_to_meter_in, double *img_size_in,
                                     double focal_in, int *valid_mask_per_obj_, double reprojection_scale_factor_in, double plane_scale_factor_in)
            : pca_comps(pca_comps_in), mean_shape(mean_shape_in), locs_2d(locs_2d_in),
            cad_to_meter(cad_to_meter_in), img_size(img_size_in), focal(focal_in),
            valid_mask_per_obj(valid_mask_per_obj_), reprojection_scale_factor(reprojection_scale_factor_in),
            plane_scale_factor(plane_scale_factor_in){}

    Matrix <double, 13, 5> &pca_comps;
    Matrix <double, 13, 1> &mean_shape;
    double *locs_2d;
    double cad_to_meter;
    double *img_size; // h, w
    double focal;
    int *valid_mask_per_obj;  // length of 12: 0 or 1
    double reprojection_scale_factor;
    double plane_scale_factor;

    template<typename T>
    bool operator()(const T *const rvec, const T *const tvec,
                    const T *const pca_coeff, const T *const plane, T *residual) const {
        // return the reprojection error
        Matrix <T, 5, 1> pca_coeff_;
        pca_coeff_<<pca_coeff[0], pca_coeff[1], pca_coeff[2], pca_coeff[3], pca_coeff[4];
        Matrix <T, 13, 1> shape_3d = mean_shape.cast <T>() + pca_comps.cast <T>() * pca_coeff_;

        Matrix <T, 12, 4> shape_full;
        T base_front_length = shape_3d[0] * T(cad_to_meter);
        T base_back_length = shape_3d[1] * T(cad_to_meter);
        T base_half_width = shape_3d[2] * T(cad_to_meter);
        T base_height = shape_3d[3] * T(cad_to_meter);
        T top_front_length = shape_3d[4] * T(cad_to_meter);
        T top_back_length = shape_3d[5] * T(cad_to_meter);
        T top_half_width = shape_3d[6] * T(cad_to_meter);
        T top_height = shape_3d[7] * T(cad_to_meter);
        T mid_front_length = shape_3d[8] * T(cad_to_meter);
        T mid_back_length = shape_3d[9] * T(cad_to_meter);
        T mid_front_height = shape_3d[10] * T(cad_to_meter);
        T mid_back_height = shape_3d[11] * T(cad_to_meter);
        T mid_half_width = shape_3d[12] * T(cad_to_meter);

        //# base four keypoints
        shape_full.row(0) = Matrix <T, 4, 1>(base_front_length, base_height, -base_half_width, T(1));
        shape_full.row(1) = Matrix <T, 4, 1>(base_front_length, base_height, base_half_width, T(1));
        shape_full.row(2) = Matrix <T, 4, 1>(base_back_length, base_height, -base_half_width, T(1));
        shape_full.row(3) = Matrix <T, 4, 1>(base_back_length, base_height, base_half_width, T(1));

        //# middle four keypoints
        shape_full.row(4) = Matrix <T, 4, 1>(mid_front_length, mid_front_height, -mid_half_width, T(1));
        shape_full.row(5) = Matrix <T, 4, 1>(mid_front_length, mid_front_height, mid_half_width, T(1));
        shape_full.row(6) = Matrix <T, 4, 1>(mid_back_length, mid_back_height, -mid_half_width, T(1));
        shape_full.row(7) = Matrix <T, 4, 1>(mid_back_length, mid_back_height, mid_half_width, T(1));

        //# top four keypoints
        shape_full.row(8) = Matrix <T, 4, 1>(top_front_length, top_height, -top_half_width, T(1));
        shape_full.row(9) = Matrix <T, 4, 1>(top_front_length, top_height, top_half_width, T(1));
        shape_full.row(10) = Matrix <T, 4, 1>(top_back_length, top_height, -top_half_width, T(1));
        shape_full.row(11) = Matrix <T, 4, 1>(top_back_length, top_height, top_half_width, T(1));
        Matrix <T, 4, 12> P = shape_full.transpose();

        Matrix <T, 3, 3> R;
        Matrix <T, 4, 4> Rt;
        T theta = sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);
        Matrix <T, 3, 3> omega;
        omega<<T(0), -rvec[2], rvec[1], rvec[2], T(0), -rvec[0], -rvec[1], rvec[0], T(0);
        R = (sin(theta) / theta) * omega + ((T(1) - cos(theta)) / (theta * theta)) * (omega * omega);
        R(0, 0) += T(1), R(1, 1) += T(1), R(2, 2) += T(1);

        /*
        Matrix<T,1,3> rvec1;
        rvec1(0,0) = T(1.0)*rvec[0];
        rvec1(0,1) = T(-1.0)*rvec[1];
        rvec1(0,2) = T(-1.0)*rvec[2];
        Rt.setZero();
        ceres::AngleAxisToRotationMatrix(rvec1.data(), R.data());
        */
        Rt(0, 0) = R(0, 0), Rt(0, 1) = R(0, 1), Rt(0, 2) = R(0, 2);
        Rt(1, 0) = R(1, 0), Rt(1, 1) = R(1, 1), Rt(1, 2) = R(1, 2);
        Rt(2, 0) = R(2, 0), Rt(2, 1) = R(2, 1), Rt(2, 2) = R(2, 2);
        Rt(0, 3) = tvec[0], Rt(1, 3) = tvec[1], Rt(2, 3) = tvec[2];
        Rt(3, 3) = T(1);

        Matrix <T, 3, 4> I;
        I.setIdentity();

        Matrix <T, 3, 3> K;
        K<<T(focal), T(0), T(img_size[1] / 2.0), T(0), T(focal), T(img_size[0] / 2.0), T(0), T(0), T(1);

        Matrix <T, 3, 12> P_cam = I * Rt * P;
        Matrix <T, 3, 12> p_homo = K * P_cam;

        float valid_sum = 0.0f;
        for (int i = 0; i<12; i++) {
            valid_sum += float(valid_mask_per_obj[i]);
        }

        T scale_inlier = T(1.0);
        T scale_outlier = T(1.0);
        T scale_factor = T(reprojection_scale_factor) / (T(valid_sum) * scale_inlier + T(12 - valid_sum) * scale_outlier);
        T p[24];
        for (int i = 0; i<12; i++) {
            p[2 * i] = p_homo(0, i) / p_homo(2, i);
            p[2 * i + 1] = p_homo(1, i) / p_homo(2, i);
        }

        T *locs_2d_est = p; // 12 x 2
        for (int i = 0; i<24; i++) {
            if (valid_mask_per_obj[i / 2] == 1)
                residual[i] = scale_factor * scale_inlier * (T(locs_2d[i]) - locs_2d_est[i]);
            else
                residual[i] = scale_factor * scale_outlier * (T(locs_2d[i]) - locs_2d_est[i]);
        }

        // four wheels should lie on the same plane
        T plane_scale = T(plane_scale_factor);
        for (int i = 0; i<4; i++) {
            T bottom_plane_dis = abs(plane[0] * P_cam(0, 0 + i) + plane[1] * P_cam(1, 0 + i)
                                     + plane[2] * P_cam(2, 0 + i) + plane[3]);
            bottom_plane_dis = bottom_plane_dis / sqrt(plane[0] * plane[0] + plane[1] * plane[1]
                                                       + plane[2] * plane[2]);
            residual[24 + i] = plane_scale * bottom_plane_dis;
        }

        return true;
    }
};


py::object optimize_pca_per_vehicle_new_pca(double focal, py::buffer rt_, py::buffer pca_, int num_obj, int num_veh,
                                            py::buffer pca_comps_, py::buffer mean_shape_, py::buffer locs_2d_,
                                            double cad_to_meter, py::buffer img_size_, py::buffer plane_,
                                            int num_iter, py::buffer mapping_, py::buffer kp_valid_mask_,
                                            double reprojection_scale_factor, double plane_scale_factor) {
    auto buf0 = rt_.request();    // num_obj x 6
    double *rt = static_cast<double *>(buf0.ptr);
    auto buf1 = pca_.request();  // num_veh x 5
    double *pca_start = static_cast<double *>(buf1.ptr);
    auto buf2 = pca_comps_.request();
    double *pca_comps = static_cast<double *>(buf2.ptr);
    auto buf3 = mean_shape_.request();
    double *mean_shape = static_cast<double *>(buf3.ptr);
    auto buf4 = locs_2d_.request();
    double *locs_2d_start = static_cast<double *>(buf4.ptr); // n x 12 x 2
    auto buf5 = img_size_.request();
    double *img_size = static_cast<double *>(buf5.ptr);
    auto buf6 = plane_.request();
    double *plane0 = static_cast<double *>(buf6.ptr);
    auto buf7 = mapping_.request();
    int *mapping = static_cast<int *>(buf7.ptr);
    auto buf8 = kp_valid_mask_.request();
    int *valid_mask = static_cast<int *>(buf8.ptr);

    double *locs_2d, *rvec, *tvec, *pca_coeff;

    Matrix<double, 5, 13>pca_comps_in = Map<Matrix<double, 5, 13>>(pca_comps);
    Matrix <double, 13, 5> pca_comps_in_t = pca_comps_in.transpose();
    Matrix<double, 13, 1>mean_shape_in = Map<Matrix<double, 13, 1>>(mean_shape);
    ceres::Problem problem;
    for (int i = 0; i<num_obj; i++) {
        rvec = rt + 6 * i;
        tvec = rvec + 3;
        int pca_index = *(mapping + i);
        pca_coeff = pca_start + pca_index * 5;
        locs_2d = locs_2d_start + i * 24;
        int *valid_mask_per_obj = valid_mask + i * 12;
        ceres::CostFunction *cost_function;
        cost_function = new ceres::AutoDiffCostFunction <ReprojectionResidualFixedFNewPCA, 28, 3, 3, 5, 4>(
                new ReprojectionResidualFixedFNewPCA(locs_2d,
                                                     pca_comps_in_t,
                                                     mean_shape_in,
                                                     cad_to_meter,
                                                     img_size,
                                                     focal,
                                                     valid_mask_per_obj,
                                                     reprojection_scale_factor, plane_scale_factor)
        );
        problem.AddResidualBlock(cost_function, NULL, rvec, tvec, pca_coeff, plane0);
        for (int j = 0; j<5; j++) {
            problem.SetParameterLowerBound(pca_coeff, j, -0.15);
            problem.SetParameterUpperBound(pca_coeff, j, 0.15);
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = num_iter;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  //ceres::SPARSE_SCHUR;  //ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout<<summary.FullReport()<<"\n";
    // return optimized values
    py::array_t <double> x_rt = py::array_t <double>(
            py::buffer_info(
                    rt,
                    sizeof(double), //itemsize
                    py::format_descriptor <double>::format(),
                    1, // ndim
                    std::vector<size_t>{(unsigned long) (num_obj * 6)}, // shape
                    std::vector<size_t>{sizeof(double)} // strides
            )
    );

    py::array_t <double> x_pca = py::array_t <double>(
            py::buffer_info(
                    pca_start,
                    sizeof(double), //itemsize
                    py::format_descriptor <double>::format(),
                    1, // ndim
                    std::vector<size_t>{(unsigned long) (num_veh * 5)}, // shape
                    std::vector<size_t>{sizeof(double)} // strides
            )
    );

    py::array_t <double> plane = py::array_t <double>(
            py::buffer_info(
                    plane0,
                    sizeof(double), //itemsize
                    py::format_descriptor <double>::format(),
                    1, // ndim
                    std::vector<size_t>{(unsigned long) (4)}, // shape
                    std::vector<size_t>{sizeof(double)} // strides
            )
    );
    py::list outputs;
    outputs.append(x_rt);
    outputs.append(x_pca);
    outputs.append(plane);
    return outputs;
}


PYBIND11_PLUGIN(ceres_reconstruct) {
        py::module m("ceres_reconstruct", "Python bindings to the Ceres-Solver minimizer.");
        // google::InitGoogleLogging("ceres_reconstruct");
        m.def("optimize_pca_per_vehicle_new_pca", &optimize_pca_per_vehicle_new_pca, "Optimizes with pca and rt with ransac");
        return m.ptr();
}
