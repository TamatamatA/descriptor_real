#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "eigenvectors_R_data.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <omp.h>

namespace py = pybind11;

const double PI = 3.14159265358979323846;

#ifndef DESCRIPTOR_MODULE_NAME
#define DESCRIPTOR_MODULE_NAME descriptor_all_R
#endif

enum class DerivCacheMode {
    Auto,
    On,
    Off
};

inline std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

inline DerivCacheMode get_deriv_cache_mode() {
    const char* env = std::getenv("DESCRIPTOR_DERIV_CACHE");
    if (env == nullptr) return DerivCacheMode::Auto;

    const std::string mode = to_lower_ascii(std::string(env));
    if (mode == "on" || mode == "1" || mode == "true") return DerivCacheMode::On;
    if (mode == "off" || mode == "0" || mode == "false") return DerivCacheMode::Off;
    return DerivCacheMode::Auto;
}


// ============================================================
// basic structs
// ============================================================

struct Vec3 {
    double x, y, z;
    Vec3() : x(0.0), y(0.0), z(0.0) {}
    Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
};

struct Mat3 {
    double a[3][3];

    Mat3() {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                a[i][j] = 0.0;
            }
        }
    }
};

struct PairList {
    std::vector<int> pair_i;
    std::vector<int> pair_j;

    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<double> dz;

    std::vector<double> dist;
    std::vector<double> cos_theta;
    std::vector<double> sin_theta;

    std::vector<double> cos_phi;
    std::vector<double> sin_phi;
};

struct RadialParam {
    double beta = 0.0;
    double r_n = 0.0;
};



template<int LMAX>
struct PairAngularCache {
    static constexpr int NLM = (LMAX + 1) * (LMAX + 1);

    int i = -1;
    int j = -1;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double r = 0.0;

    double cos_theta = 1.0;
    double sin_theta = 0.0;
    double cos_phi = 1.0;
    double sin_phi = 0.0;

    // +dr only. -dr is derived via parity:
    //   ylm_minus[idx]      = parity[l] * ylm[idx]      where parity[l] = (-1)^l
    //   dYdtheta_minus[idx] = -parity[l] * dYdtheta[idx]
    std::array<double, NLM> ylm;
    std::array<double, NLM> dYdtheta;
};

struct CellList {
    int Mx = 1;
    int My = 1;
    int Mz = 1;

    int rx = 1;
    int ry = 1;
    int rz = 1;

    std::vector<Vec3> frac_pos;
    std::vector<std::vector<int>> buckets;
};

struct ProductTerm {
    std::vector<int> radial_ids;
    std::vector<int> flat_idx;
};

struct CellKey3D {
    int ix;
    int iy;
    int iz;
    bool operator==(const CellKey3D& other) const {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
};

struct CellKey3DHash {
    std::size_t operator()(const CellKey3D& k) const noexcept {
        std::size_t h1 = std::hash<int>{}(k.ix);
        std::size_t h2 = std::hash<int>{}(k.iy);
        std::size_t h3 = std::hash<int>{}(k.iz);
        return h1 ^ (h2 << 1) ^ (h3 << 7);
    }
};

struct MappedTerm {
    int prod_id;
    double coeff;
};

using FeatureMapping = std::vector<std::vector<MappedTerm>>;

// ============================================================
// forward declarations
// ============================================================

// linear algebra / geometry
inline Vec3 vecmat(const Mat3& M, const Vec3& v);
inline Mat3 invert3x3(const Mat3& m);
inline double norm3(const Vec3& v);
inline int mod_int(int x, int m);
inline int cell_index_3d(int ix, int iy, int iz, int My, int Mz);
inline Vec3 wrap_fractional(const Vec3& s);
inline Vec3 minimum_image_disp(
    const Vec3& si,
    const Vec3& sj,
    const Mat3& lattice
);

// math
inline double factorial_int(int n);
inline double radial_gaussian_cutoff(
    double r,
    double r_n,
    double beta_n,
    double r_c
);
inline double radial_gaussian_cutoff_derivative(
    double r,
    double r_n,
    double beta_n,
    double r_c
);
inline double sph_harm_normalization(int l, int m_abs);
inline int sph_harm_flat_index(int l, int m);

// spherical harmonics helpers
template<int LMAX>
inline void compute_cos_sin_mphi(
    double cos_phi,
    double sin_phi,
    double (&cos_m)[LMAX + 1],
    double (&sin_m)[LMAX + 1]
);

template<int LMAX>
inline void compute_associated_legendre_all(
    double cos_theta,
    double sin_theta,
    double P[LMAX + 1][LMAX + 1]
);

template<int LMAX>
inline void compute_sph_harm_all(
    double cos_theta,
    double sin_theta,
    double cos_phi,
    double sin_phi,
    double (&ylm_re)[(LMAX + 1) * (LMAX + 1)],
    double (&ylm_im)[(LMAX + 1) * (LMAX + 1)]
);

inline void compute_spherical_info_from_dr(
    const Vec3& dr,
    double& dist,
    double& cos_theta,
    double& sin_theta,
    double& cos_phi,
    double& sin_phi
);

inline double dY_dphi_from_Ylm(
    int l,
    int m,
    const double* ylm_block
);

inline double dY_dtheta_from_Ylm(
    int l,
    int m,
    double cos_theta,
    double sin_theta,
    double Ylm,
    const double* ylm_block
);


// ============================================================
// basic linear algebra / geometry
// ============================================================

inline Vec3 vecmat(const Vec3& v, const Mat3& M) {
    return Vec3(
        v.x * M.a[0][0] + v.y * M.a[1][0] + v.z * M.a[2][0],
        v.x * M.a[0][1] + v.y * M.a[1][1] + v.z * M.a[2][1],
        v.x * M.a[0][2] + v.y * M.a[1][2] + v.z * M.a[2][2]
    );
}

inline Mat3 invert3x3(const Mat3& m) {
    Mat3 inv{};

    double det =
          m.a[0][0] * (m.a[1][1] * m.a[2][2] - m.a[1][2] * m.a[2][1])
        - m.a[0][1] * (m.a[1][0] * m.a[2][2] - m.a[1][2] * m.a[2][0])
        + m.a[0][2] * (m.a[1][0] * m.a[2][1] - m.a[1][1] * m.a[2][0]);

    if (std::abs(det) < 1e-15) {
        throw std::runtime_error("invert3x3: singular matrix");
    }

    const double invdet = 1.0 / det;

    inv.a[0][0] =  (m.a[1][1] * m.a[2][2] - m.a[1][2] * m.a[2][1]) * invdet;
    inv.a[0][1] = -(m.a[0][1] * m.a[2][2] - m.a[0][2] * m.a[2][1]) * invdet;
    inv.a[0][2] =  (m.a[0][1] * m.a[1][2] - m.a[0][2] * m.a[1][1]) * invdet;

    inv.a[1][0] = -(m.a[1][0] * m.a[2][2] - m.a[1][2] * m.a[2][0]) * invdet;
    inv.a[1][1] =  (m.a[0][0] * m.a[2][2] - m.a[0][2] * m.a[2][0]) * invdet;
    inv.a[1][2] = -(m.a[0][0] * m.a[1][2] - m.a[0][2] * m.a[1][0]) * invdet;

    inv.a[2][0] =  (m.a[1][0] * m.a[2][1] - m.a[1][1] * m.a[2][0]) * invdet;
    inv.a[2][1] = -(m.a[0][0] * m.a[2][1] - m.a[0][1] * m.a[2][0]) * invdet;
    inv.a[2][2] =  (m.a[0][0] * m.a[1][1] - m.a[0][1] * m.a[1][0]) * invdet;

    return inv;
}

inline double norm3(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline int mod_int(int x, int m) {
    int r = x % m;
    return (r < 0) ? (r + m) : r;
}

inline int cell_index_3d(int ix, int iy, int iz, int My, int Mz) {
    return (ix * My + iy) * Mz + iz;
}

inline Vec3 wrap_fractional(const Vec3& s) {
    auto wrap01 = [](double x) {
        x -= std::floor(x);
        if (x >= 1.0) x -= 1.0;
        if (x < 0.0)  x += 1.0;
        return x;
    };
    return Vec3(wrap01(s.x), wrap01(s.y), wrap01(s.z));
}

inline Vec3 minimum_image_disp(
    const Vec3& si,
    const Vec3& sj,
    const Mat3& lattice
) {
    Vec3 ds(sj.x - si.x, sj.y - si.y, sj.z - si.z);

    ds.x -= std::round(ds.x);
    ds.y -= std::round(ds.y);
    ds.z -= std::round(ds.z);

    return vecmat(ds, lattice);
}

inline Vec3 lattice_vector(const Mat3& lattice, int col) {
    return Vec3(lattice.a[0][col], lattice.a[1][col], lattice.a[2][col]);
}

inline Vec3 frac_to_cart(const Mat3& lattice, int i, int j, int k) {
    return Vec3(
        lattice.a[0][0] * i + lattice.a[0][1] * j + lattice.a[0][2] * k,
        lattice.a[1][0] * i + lattice.a[1][1] * j + lattice.a[1][2] * k,
        lattice.a[2][0] * i + lattice.a[2][1] * j + lattice.a[2][2] * k
    );
}
/*
inline Vec3 minimum_image_disp(
    const Vec3& si,
    const Vec3& sj,
    const Mat3& lattice
) {
    Vec3 ds(sj.x - si.x, sj.y - si.y, sj.z - si.z);

    Vec3 best_cart;
    double best_d2 = 1e300;

    for (int ax = -1; ax <= 1; ++ax) {
        for (int ay = -1; ay <= 1; ++ay) {
            for (int az = -1; az <= 1; ++az) {
                Vec3 trial_frac(
                    ds.x + static_cast<double>(ax),
                    ds.y + static_cast<double>(ay),
                    ds.z + static_cast<double>(az)
                );

                Vec3 trial_cart = matvec(lattice, trial_frac);
                double d2 = trial_cart.x * trial_cart.x
                          + trial_cart.y * trial_cart.y
                          + trial_cart.z * trial_cart.z;

                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_cart = trial_cart;
                }
            }
        }
    }

    return best_cart;
}*/

// ============================================================
// math utilities
// ============================================================

inline double factorial_int(int n) {
    if (n <= 1) return 1.0;
    double v = 1.0;
    for (int i = 2; i <= n; ++i) {
        v *= static_cast<double>(i);
    }
    return v;
}

inline double radial_gaussian_cutoff(
    double r,
    double r_n,
    double beta_n,
    double r_c
) {
    if (r >= r_c) return 0.0;

    const double dr = r - r_n;
    const double gauss = std::exp(-beta_n * dr * dr);
    const double cutoff = 0.5 * (std::cos(M_PI * r / r_c) + 1.0);
    return gauss * cutoff;
}

inline double radial_gaussian_cutoff_derivative(
    double r,
    double r_n,
    double beta_n,
    double r_c
) {
    if (r >= r_c) return 0.0;

    const double dr = r - r_n;
    const double gauss = std::exp(-beta_n * dr * dr);
    const double cutoff = 0.5 * (std::cos(M_PI * r / r_c) + 1.0);
    const double dgauss = -2.0 * beta_n * dr * gauss;
    const double dcutoff = -0.5 * (M_PI / r_c) * std::sin(M_PI * r / r_c);

    return dgauss * cutoff + gauss * dcutoff;
}

// Compute f and fp together to avoid recomputing exp/cos/sin.
inline void radial_gaussian_cutoff_and_deriv(
    double r,
    double r_n,
    double beta_n,
    double r_c,
    double& f_out,
    double& fp_out
) {
    if (r >= r_c) {
        f_out = 0.0;
        fp_out = 0.0;
        return;
    }
    const double dr = r - r_n;
    const double gauss = std::exp(-beta_n * dr * dr);
    const double angle = M_PI * r / r_c;
    const double cos_a = std::cos(angle);
    const double sin_a = std::sin(angle);
    const double cutoff = 0.5 * (cos_a + 1.0);
    const double dgauss = -2.0 * beta_n * dr * gauss;
    const double dcutoff = -0.5 * (M_PI / r_c) * sin_a;
    f_out  = gauss * cutoff;
    fp_out = dgauss * cutoff + gauss * dcutoff;
}

inline double sph_harm_normalization(int l, int m_abs) {
    return std::sqrt(
        (2.0 * l + 1.0) / (4.0 * PI) *
        factorial_int(l - m_abs) / factorial_int(l + m_abs)
    );
}

inline int sph_harm_flat_index(int l, int m) {
    return l * l + (m + l);
}


// ============================================================
// spherical harmonics
// ============================================================

// Precomputed normalization coefficients for compute_sph_harm_all.
// norm_m0[l]   = sph_harm_normalization(l, 0)
// norm_m[l][m] = M_SQRT2 * sph_harm_normalization(l, m)  for m >= 1
template<int LMAX>
struct SphNormTable {
    double norm_m0[LMAX + 1];
    double norm_m[LMAX + 1][LMAX + 1];
    SphNormTable() {
        for (int l = 0; l <= LMAX; ++l) {
            norm_m0[l] = sph_harm_normalization(l, 0);
            for (int m = 1; m <= l; ++m) {
                norm_m[l][m] = M_SQRT2 * sph_harm_normalization(l, m);
            }
        }
    }
};

template<int LMAX>
inline void compute_cos_sin_mphi(
    double cos_phi,
    double sin_phi,
    double (&cos_m)[LMAX + 1],
    double (&sin_m)[LMAX + 1]
) {
    cos_m[0] = 1.0;
    sin_m[0] = 0.0;

    for (int m = 1; m <= LMAX; ++m) {
        const double c_prev = cos_m[m - 1];
        const double s_prev = sin_m[m - 1];

        cos_m[m] = cos_phi * c_prev - sin_phi * s_prev;
        sin_m[m] = sin_phi * c_prev + cos_phi * s_prev;
    }
}

template<int LMAX>
inline void compute_associated_legendre_all(
    double cos_theta,
    double sin_theta,
    double P[LMAX + 1][LMAX + 1]
) {
    for (int l = 0; l <= LMAX; ++l) {
        for (int m = 0; m <= LMAX; ++m) {
            P[l][m] = 0.0;
        }
    }

    P[0][0] = 1.0;

    for (int m = 1; m <= LMAX; ++m) {
        P[m][m] = -(2 * m - 1) * sin_theta * P[m - 1][m - 1];
    }

    for (int m = 0; m < LMAX; ++m) {
        P[m + 1][m] = (2 * m + 1) * cos_theta * P[m][m];
    }

    for (int m = 0; m <= LMAX; ++m) {
        for (int l = m + 2; l <= LMAX; ++l) {
            P[l][m] =
                (
                    (2 * l - 1) * cos_theta * P[l - 1][m]
                    - (l + m - 1) * P[l - 2][m]
                ) / static_cast<double>(l - m);
        }
    }
}

template<int LMAX>
inline void compute_sph_harm_all(
    double cos_theta,
    double sin_theta,
    double cos_phi,
    double sin_phi,
    double (&ylm_re)[(LMAX + 1) * (LMAX + 1)],
    double (&ylm_im)[(LMAX + 1) * (LMAX + 1)]
) {
    constexpr int NLM = (LMAX + 1) * (LMAX + 1);

    for (int i = 0; i < NLM; ++i) {
        ylm_re[i] = 0.0;
        ylm_im[i] = 0.0;
    }

    double P[LMAX + 1][LMAX + 1];
    compute_associated_legendre_all<LMAX>(cos_theta, sin_theta, P);

    double cos_m[LMAX + 1];
    double sin_m[LMAX + 1];
    compute_cos_sin_mphi<LMAX>(cos_phi, sin_phi, cos_m, sin_m);

    static const SphNormTable<LMAX> NT;

    for (int l = 0; l <= LMAX; ++l) {
        const int idx0 = sph_harm_flat_index(l, 0);
        ylm_re[idx0] = NT.norm_m0[l] * P[l][0];
        ylm_im[idx0] = 0.0;

        for (int m = 1; m <= l; ++m) {
            const double sign_pos = (m % 2 == 0) ? 1.0 : -1.0;
            const double sign_neg = -1.0;
            const double amp = NT.norm_m[l][m] * P[l][m];
            const int idx_pos = sph_harm_flat_index(l, m);
            const int idx_neg = sph_harm_flat_index(l, -m);
            ylm_re[idx_pos] = sign_pos * amp * cos_m[m];
            ylm_re[idx_neg] = sign_neg * amp * sin_m[m];
            ylm_im[idx_pos] = 0.0;
            ylm_im[idx_neg] = 0.0;
        }
    }
}

inline void compute_spherical_info_from_dr(
    const Vec3& dr,
    double& dist,
    double& cos_theta,
    double& sin_theta,
    double& cos_phi,
    double& sin_phi
) {
    const double x = dr.x;
    const double y = dr.y;
    const double z = dr.z;

    dist = std::sqrt(x * x + y * y + z * z);

    if (dist < 1e-15) {
        cos_theta = 1.0;
        sin_theta = 0.0;
        cos_phi = 1.0;
        sin_phi = 0.0;
        return;
    }

    cos_theta = std::clamp(z / dist, -1.0, 1.0);
    sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

    const double rho2 = x * x + y * y;
    if (rho2 < 1e-30) {
        cos_phi = 1.0;
        sin_phi = 0.0;
    } else {
        const double rho = std::sqrt(rho2);
        cos_phi = x / rho;
        sin_phi = y / rho;
    }
}

inline double dY_dphi_from_Ylm(
    int l,
    int m,
    const double* ylm_block
) {
    if (m == 0) return 0.0;

    const double ratio = ((std::abs(m) % 2) == 0) ? 1.0 : -1.0;
    return ratio * static_cast<double>(m) * ylm_block[sph_harm_flat_index(l, -m)];
}


inline double dY_dtheta_from_Ylm(
    int l,
    int m,
    double cos_theta,
    double sin_theta,
    double Ylm,
    const double* ylm_block
) {
    const double eps = 1e-14;
    if (l == 0 || std::abs(sin_theta) <= eps) return 0.0;

    double coeff = 0.0;
    double prev = 0.0;
    if (std::abs(m) <= l - 1) {
        coeff = std::sqrt(
            ((2.0 * l + 1.0) / (2.0 * l - 1.0)) *
            static_cast<double>(l * l - m * m)
        );
        prev = ylm_block[sph_harm_flat_index(l - 1, m)];
    }

    return (static_cast<double>(l) * cos_theta * Ylm - coeff * prev) / sin_theta;
}
// ============================================================
// cell list / pair list
// ============================================================

inline CellList build_cell_list_for_pairs(
    const std::vector<Vec3>& atom_pos,
    const Mat3& lattice,
    const Mat3& lattice_inv,
    double r_c
) {
    CellList cl;
    const int n_atoms = static_cast<int>(atom_pos.size());

    Vec3 a(lattice.a[0][0], lattice.a[0][1], lattice.a[0][2]);
    Vec3 b(lattice.a[1][0], lattice.a[1][1], lattice.a[1][2]);
    Vec3 c(lattice.a[2][0], lattice.a[2][1], lattice.a[2][2]);

    auto cross = [](const Vec3& u, const Vec3& v) {
        return Vec3(
            u.y * v.z - u.z * v.y,
            u.z * v.x - u.x * v.z,
            u.x * v.y - u.y * v.x
        );
    };

    auto dot = [](const Vec3& u, const Vec3& v) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    };

    Vec3 bc = cross(b, c);
    Vec3 ca = cross(c, a);
    Vec3 ab = cross(a, b);

    double vol = std::abs(dot(a, bc));
    double h_a = vol / norm3(bc);
    double h_b = vol / norm3(ca);
    double h_c = vol / norm3(ab);

    cl.Mx = std::max(1, static_cast<int>(std::floor(h_a / r_c)));
    cl.My = std::max(1, static_cast<int>(std::floor(h_b / r_c)));
    cl.Mz = std::max(1, static_cast<int>(std::floor(h_c / r_c)));

    cl.frac_pos.resize(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        Vec3 s = vecmat(atom_pos[i], lattice_inv);
        cl.frac_pos[i] = wrap_fractional(s);
    }

    cl.buckets.assign(cl.Mx * cl.My * cl.Mz, {});
    for (int i = 0; i < n_atoms; ++i) {
        const Vec3& s = cl.frac_pos[i];

        int ix = std::min(cl.Mx - 1, static_cast<int>(std::floor(s.x * cl.Mx)));
        int iy = std::min(cl.My - 1, static_cast<int>(std::floor(s.y * cl.My)));
        int iz = std::min(cl.Mz - 1, static_cast<int>(std::floor(s.z * cl.Mz)));

        cl.buckets[cell_index_3d(ix, iy, iz, cl.My, cl.Mz)].push_back(i);
    }

    double cell_h_a = h_a / static_cast<double>(cl.Mx);
    double cell_h_b = h_b / static_cast<double>(cl.My);
    double cell_h_c = h_c / static_cast<double>(cl.Mz);

    cl.rx = std::max(1, static_cast<int>(std::ceil(r_c / cell_h_a)));
    cl.ry = std::max(1, static_cast<int>(std::ceil(r_c / cell_h_b)));
    cl.rz = std::max(1, static_cast<int>(std::ceil(r_c / cell_h_c)));

    return cl;
}

inline PairList build_half_pair_list_minimum_image(
    const std::vector<Vec3>& atom_pos,
    const Mat3& lattice,
    const CellList& cl,
    double r_c
) {
    PairList out;
    const int n_atoms = static_cast<int>(atom_pos.size());
    const double rc2 = r_c * r_c;

    const int nbuckets = cl.Mx * cl.My * cl.Mz;
    std::vector<int> bucket_stamp(nbuckets, -1);

    for (int i = 0; i < n_atoms; ++i) {
        const Vec3& si = cl.frac_pos[i];

        const int iix = std::min(cl.Mx - 1, static_cast<int>(std::floor(si.x * cl.Mx)));
        const int iiy = std::min(cl.My - 1, static_cast<int>(std::floor(si.y * cl.My)));
        const int iiz = std::min(cl.Mz - 1, static_cast<int>(std::floor(si.z * cl.Mz)));

        const int stamp = i;

        for (int dx_cell = -cl.rx; dx_cell <= cl.rx; ++dx_cell) {
            for (int dy_cell = -cl.ry; dy_cell <= cl.ry; ++dy_cell) {
                for (int dz_cell = -cl.rz; dz_cell <= cl.rz; ++dz_cell) {
                    const int wrap_x = mod_int(iix + dx_cell, cl.Mx);
                    const int wrap_y = mod_int(iiy + dy_cell, cl.My);
                    const int wrap_z = mod_int(iiz + dz_cell, cl.Mz);

                    const int bidx = cell_index_3d(wrap_x, wrap_y, wrap_z, cl.My, cl.Mz);

                    if (bucket_stamp[bidx] == stamp) continue;
                    bucket_stamp[bidx] = stamp;

                    const auto& bucket = cl.buckets[bidx];

                    for (int j : bucket) {
                        if (j <= i) continue;

                        const Vec3& sj = cl.frac_pos[j];
                        const Vec3 dr = minimum_image_disp(si, sj, lattice);

                        const double d2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
                        if (d2 > rc2 || d2 <= 1e-20) continue;

                        double dist, cos_theta, sin_theta, cos_phi, sin_phi;
                        compute_spherical_info_from_dr(
                            dr,
                            dist,
                            cos_theta,
                            sin_theta,
                            cos_phi,
                            sin_phi
                        );

                        out.pair_i.push_back(i);
                        out.pair_j.push_back(j);

                        out.dx.push_back(dr.x);
                        out.dy.push_back(dr.y);
                        out.dz.push_back(dr.z);

                        out.dist.push_back(dist);
                        out.cos_theta.push_back(cos_theta);
                        out.sin_theta.push_back(sin_theta);
                        out.cos_phi.push_back(cos_phi);
                        out.sin_phi.push_back(sin_phi);
                    }
                }
            }
        }
    }

    return out;
}

// Reduce lattice axes using Gauss-Lagrange style iteration (NeighborCell::refine_axis).
// Reduces the number of periodic translation vectors for non-orthogonal cells.
inline Mat3 reduce_lattice(const Mat3& lattice) {
    Mat3 lat = lattice;
    auto dot_col = [&](int c1, int c2) {
        return lat.a[0][c1]*lat.a[0][c2]
             + lat.a[1][c1]*lat.a[1][c2]
             + lat.a[2][c1]*lat.a[2][c2];
    };
    for (int iter = 0; iter < 100; ++iter) {
        const double d00 = dot_col(0, 0), d11 = dot_col(1, 1), d22 = dot_col(2, 2);
        const double d01 = dot_col(0, 1), d02 = dot_col(0, 2), d12 = dot_col(1, 2);
        const bool r01 = (std::abs(d01) > 0.5*d00 || std::abs(d01) > 0.5*d11);
        const bool r02 = (std::abs(d02) > 0.5*d00 || std::abs(d02) > 0.5*d22);
        const bool r12 = (std::abs(d12) > 0.5*d11 || std::abs(d12) > 0.5*d22);
        if (!r01 && !r02 && !r12) break;
        if (r01) {
            const int col_src = (d00 > d11) ? 1 : 0;
            const int col_tgt = (d00 > d11) ? 0 : 1;
            const int n = static_cast<int>(std::round(-d01 / dot_col(col_src, col_src)));
            for (int row = 0; row < 3; ++row)
                lat.a[row][col_tgt] += n * lat.a[row][col_src];
        }
        if (r02) {
            const int col_src = (d00 > d22) ? 2 : 0;
            const int col_tgt = (d00 > d22) ? 0 : 2;
            const int n = static_cast<int>(std::round(-d02 / dot_col(col_src, col_src)));
            for (int row = 0; row < 3; ++row)
                lat.a[row][col_tgt] += n * lat.a[row][col_src];
        }
        if (r12) {
            const int col_src = (d11 > d22) ? 2 : 1;
            const int col_tgt = (d11 > d22) ? 1 : 2;
            const int n = static_cast<int>(std::round(-d12 / dot_col(col_src, col_src)));
            for (int row = 0; row < 3; ++row)
                lat.a[row][col_tgt] += n * lat.a[row][col_src];
        }
    }
    return lat;
}

inline std::vector<Vec3> build_periodic_translations_uncached(
    const Mat3& lattice,
    double r_c
) {
    const Mat3 lat = reduce_lattice(lattice);

    const Vec3 a = lattice_vector(lat, 0);
    const Vec3 b = lattice_vector(lat, 1);
    const Vec3 c = lattice_vector(lat, 2);

    const double da = std::max(norm3(a), 1e-12);
    const double db = std::max(norm3(b), 1e-12);
    const double dc = std::max(norm3(c), 1e-12);

    const int ia_max = static_cast<int>(std::ceil(r_c / da)) + 1;
    const int ib_max = static_cast<int>(std::ceil(r_c / db)) + 1;
    const int ic_max = static_cast<int>(std::ceil(r_c / dc)) + 1;

    std::vector<Vec3> vertices = {
        frac_to_cart(lat, 0, 0, 1),  frac_to_cart(lat, 0, 1, 0),
        frac_to_cart(lat, 0, 1, 1),  frac_to_cart(lat, 1, 0, 0),
        frac_to_cart(lat, 1, 0, 1),  frac_to_cart(lat, 1, 1, 0),
        frac_to_cart(lat, 1, 1, 1),  frac_to_cart(lat, 0, 0, -1),
        frac_to_cart(lat, 0, -1, 0), frac_to_cart(lat, 0, -1, -1),
        frac_to_cart(lat, -1, 0, 0), frac_to_cart(lat, -1, 0, -1),
        frac_to_cart(lat, -1, -1, 0),frac_to_cart(lat, -1, -1, -1)
    };

    double max_diag = 0.0;
    for (const auto& v : vertices) {
        max_diag = std::max(max_diag, norm3(v));
    }

    std::vector<Vec3> trans;
    trans.reserve((2 * ia_max + 1) * (2 * ib_max + 1) * (2 * ic_max + 1));
    for (int i = -ia_max; i <= ia_max; ++i) {
        for (int j = -ib_max; j <= ib_max; ++j) {
            for (int k = -ic_max; k <= ic_max; ++k) {
                Vec3 t = frac_to_cart(lat, i, j, k);
                if (norm3(t) < max_diag + r_c) {
                    trans.push_back(t);
                }
            }
        }
    }
    return trans;
}

inline std::vector<Vec3> build_periodic_translations(
    const Mat3& lattice,
    double r_c
) {
    // Cache keyed on the 9 lattice doubles + r_c to avoid recomputing
    // reduce_lattice and translation enumeration for repeated identical lattices.
    struct CacheKey {
        double v[10];  // lattice[0..8] + r_c
        bool operator==(const CacheKey& o) const {
            for (int i = 0; i < 10; ++i) if (v[i] != o.v[i]) return false;
            return true;
        }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const {
            size_t h = 0;
            for (int i = 0; i < 10; ++i) {
                uint64_t bits;
                static_assert(sizeof(double) == sizeof(uint64_t), "");
                std::memcpy(&bits, &k.v[i], sizeof(bits));
                h ^= std::hash<uint64_t>{}(bits) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    static std::unordered_map<CacheKey, std::vector<Vec3>, CacheKeyHash> cache;
    static std::mutex cache_mutex;

    CacheKey key;
    int idx = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            key.v[idx++] = lattice.a[i][j];
    key.v[9] = r_c;

    // Fast path: check without lock first (cache is append-only, no deletion).
    // Reads of stable pointers after initial population are safe in practice,
    // but we guard the first miss with a lock.
    {
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
    }

    std::vector<Vec3> trans = build_periodic_translations_uncached(lattice, r_c);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto [it, inserted] = cache.emplace(key, std::move(trans));
        return it->second;
    }
}

inline bool use_half_translation_lexicographic(
    double dx,
    double dy,
    double dz,
    double tol = 1e-12
) {
    if (dz >= tol) return true;
    if (std::abs(dz) < tol && dy >= tol) return true;
    if (std::abs(dz) < tol && std::abs(dy) < tol && dx >= tol) return true;
    return false;
}

inline PairList build_half_pair_list_periodic_images(
    const std::vector<Vec3>& atom_pos,
    const Mat3& lattice,
    double r_c
) {
    PairList out;
    const int n_atoms = static_cast<int>(atom_pos.size());
    if (n_atoms == 0) return out;

    const double rc2 = r_c * r_c;
    const double tol = 1e-12;
    const auto& trans = build_periodic_translations(lattice, r_c);
    if (trans.empty()) return out;

    // ------------------------------------------------------------------
    // Spatial cell list: bin atom_pos into cubic cells of side r_c.
    // For each atom i we only scan the 27 neighboring cells for j candidates,
    // then apply each translation — same polymlp NeighborCell philosophy.
    // ------------------------------------------------------------------
    double min_x = atom_pos[0].x, min_y = atom_pos[0].y, min_z = atom_pos[0].z;
    for (const auto& p : atom_pos) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        min_z = std::min(min_z, p.z);
    }
    const double inv_cell = 1.0 / std::max(r_c, 1e-12);

    auto to_cell = [&](double x, double y, double z) -> CellKey3D {
        return CellKey3D{
            static_cast<int>(std::floor((x - min_x) * inv_cell)),
            static_cast<int>(std::floor((y - min_y) * inv_cell)),
            static_cast<int>(std::floor((z - min_z) * inv_cell))
        };
    };

    std::unordered_map<CellKey3D, std::vector<int>, CellKey3DHash> cell_list;
    cell_list.reserve(static_cast<size_t>(n_atoms) * 2);
    for (int j = 0; j < n_atoms; ++j)
        cell_list[to_cell(atom_pos[j].x, atom_pos[j].y, atom_pos[j].z)].push_back(j);

    out.pair_i.reserve(static_cast<size_t>(n_atoms) * 64);
    out.pair_j.reserve(static_cast<size_t>(n_atoms) * 64);
    out.dx.reserve(static_cast<size_t>(n_atoms) * 64);
    out.dy.reserve(static_cast<size_t>(n_atoms) * 64);
    out.dz.reserve(static_cast<size_t>(n_atoms) * 64);
    out.dist.reserve(static_cast<size_t>(n_atoms) * 64);
    out.cos_theta.reserve(static_cast<size_t>(n_atoms) * 64);
    out.sin_theta.reserve(static_cast<size_t>(n_atoms) * 64);
    out.cos_phi.reserve(static_cast<size_t>(n_atoms) * 64);
    out.sin_phi.reserve(static_cast<size_t>(n_atoms) * 64);

    for (int i = 0; i < n_atoms; ++i) {
        const Vec3& pi = atom_pos[i];
        const CellKey3D ci = to_cell(pi.x, pi.y, pi.z);

        // i-j pairs (j < i): scan 27 neighboring cells, then all translations
        for (int ox = -1; ox <= 1; ++ox) {
        for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
            auto it = cell_list.find(CellKey3D{ci.ix+ox, ci.iy+oy, ci.iz+oz});
            if (it == cell_list.end()) continue;

            for (int j : it->second) {
                if (j >= i) continue;  // half condition: j < i only
                const double dx_ij = atom_pos[j].x - pi.x;
                const double dy_ij = atom_pos[j].y - pi.y;
                const double dz_ij = atom_pos[j].z - pi.z;

                for (const auto& tr : trans) {
                    const double dx = dx_ij + tr.x;
                    const double dy = dy_ij + tr.y;
                    const double dz = dz_ij + tr.z;
                    const double d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 >= rc2 || d2 <= 1e-20) continue;

                    double dist, cos_theta, sin_theta, cos_phi, sin_phi;
                    compute_spherical_info_from_dr(
                        Vec3(dx, dy, dz),
                        dist, cos_theta, sin_theta, cos_phi, sin_phi
                    );
                    out.pair_i.push_back(i);
                    out.pair_j.push_back(j);
                    out.dx.push_back(dx);
                    out.dy.push_back(dy);
                    out.dz.push_back(dz);
                    out.dist.push_back(dist);
                    out.cos_theta.push_back(cos_theta);
                    out.sin_theta.push_back(sin_theta);
                    out.cos_phi.push_back(cos_phi);
                    out.sin_phi.push_back(sin_phi);
                }
            }
        }}}

        // self-image pairs (j == i): keep only lexicographically positive half
        for (const auto& tr : trans) {
            const double dx = tr.x, dy = tr.y, dz = tr.z;
            const double d2 = dx*dx + dy*dy + dz*dz;
            if (d2 >= rc2 || d2 <= 1e-20) continue;
            if (!use_half_translation_lexicographic(dx, dy, dz, tol)) continue;

            double dist, cos_theta, sin_theta, cos_phi, sin_phi;
            compute_spherical_info_from_dr(
                Vec3(dx, dy, dz),
                dist, cos_theta, sin_theta, cos_phi, sin_phi
            );
            out.pair_i.push_back(i);
            out.pair_j.push_back(i);
            out.dx.push_back(dx);
            out.dy.push_back(dy);
            out.dz.push_back(dz);
            out.dist.push_back(dist);
            out.cos_theta.push_back(cos_theta);
            out.sin_theta.push_back(sin_theta);
            out.cos_phi.push_back(cos_phi);
            out.sin_phi.push_back(sin_phi);
        }
    }
    return out;
}

inline PairList build_half_pair_list_periodic_images_naive(
    const std::vector<Vec3>& atom_pos,
    const Mat3& lattice,
    double r_c
) {
    PairList out;
    const int n_atoms = static_cast<int>(atom_pos.size());
    const double tol = 1e-12;
    const double rc2 = r_c * r_c;
    const auto trans = build_periodic_translations(lattice, r_c);

    out.pair_i.reserve(static_cast<size_t>(n_atoms) * 64);
    out.pair_j.reserve(static_cast<size_t>(n_atoms) * 64);

    for (int i = 0; i < n_atoms; ++i) {
        for (int j = 0; j < i; ++j) {
            const double dx_ij = atom_pos[j].x - atom_pos[i].x;
            const double dy_ij = atom_pos[j].y - atom_pos[i].y;
            const double dz_ij = atom_pos[j].z - atom_pos[i].z;

            for (const auto& tr : trans) {
                const double dx = dx_ij + tr.x;
                const double dy = dy_ij + tr.y;
                const double dz = dz_ij + tr.z;
                const double d2 = dx * dx + dy * dy + dz * dz;
                if (d2 >= rc2 || d2 <= 1e-20) continue;

                double dist, cos_theta, sin_theta, cos_phi, sin_phi;
                compute_spherical_info_from_dr(
                    Vec3(dx, dy, dz),
                    dist,
                    cos_theta,
                    sin_theta,
                    cos_phi,
                    sin_phi
                );

                out.pair_i.push_back(i);
                out.pair_j.push_back(j);
                out.dx.push_back(dx);
                out.dy.push_back(dy);
                out.dz.push_back(dz);
                out.dist.push_back(dist);
                out.cos_theta.push_back(cos_theta);
                out.sin_theta.push_back(sin_theta);
                out.cos_phi.push_back(cos_phi);
                out.sin_phi.push_back(sin_phi);
            }
        }

        for (const auto& tr : trans) {
            const double dx = tr.x;
            const double dy = tr.y;
            const double dz = tr.z;
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 >= rc2 || d2 <= 1e-20) continue;

            bool use_half = false;
            if (dz >= tol) use_half = true;
            else if (std::abs(dz) < tol && dy >= tol) use_half = true;
            else if (std::abs(dz) < tol && std::abs(dy) < tol && dx >= tol) {
                use_half = true;
            }
            if (!use_half) continue;

            double dist, cos_theta, sin_theta, cos_phi, sin_phi;
            compute_spherical_info_from_dr(
                Vec3(dx, dy, dz),
                dist,
                cos_theta,
                sin_theta,
                cos_phi,
                sin_phi
            );

            out.pair_i.push_back(i);
            out.pair_j.push_back(i);
            out.dx.push_back(dx);
            out.dy.push_back(dy);
            out.dz.push_back(dz);
            out.dist.push_back(dist);
            out.cos_theta.push_back(cos_theta);
            out.sin_theta.push_back(sin_theta);
            out.cos_phi.push_back(cos_phi);
            out.sin_phi.push_back(sin_phi);
        }
    }
    return out;
}


// ============================================================
// pair angular cache / pair derivative
// ============================================================

template<int LMAX>
inline void build_pair_angular_cache_one(
    const PairList& pairs,
    size_t p,
    PairAngularCache<LMAX>& out,
    bool need_deriv = true
) {
    constexpr int NLM = (LMAX + 1) * (LMAX + 1);

    out.i = pairs.pair_i[p];
    out.j = pairs.pair_j[p];

    out.x = pairs.dx[p];
    out.y = pairs.dy[p];
    out.z = pairs.dz[p];
    out.r = pairs.dist[p];

    out.cos_theta = pairs.cos_theta[p];
    out.sin_theta = pairs.sin_theta[p];
    out.cos_phi = pairs.cos_phi[p];
    out.sin_phi = pairs.sin_phi[p];

    // +dr
    double ylm_re[NLM], ylm_im[NLM];
    compute_sph_harm_all<LMAX>(
        out.cos_theta, out.sin_theta, out.cos_phi, out.sin_phi,
        ylm_re, ylm_im
    );

    for (int idx = 0; idx < NLM; ++idx) {
        out.ylm[idx] = ylm_re[idx];
    }

    if (need_deriv) {
        for (int l = 0; l <= LMAX; ++l) {
            for (int m = -l; m <= l; ++m) {
                const int idx = sph_harm_flat_index(l, m);
                out.dYdtheta[idx] = dY_dtheta_from_Ylm(
                    l, m,
                    out.cos_theta, out.sin_theta,
                    out.ylm[idx], out.ylm.data()
                );
            }
        }
    }
    // -dr is derived on-the-fly from parity: ylm_minus = (-1)^l * ylm
}

template<int LMAX>
inline void compute_pair_anlm_derivative_from_cache(
    const PairAngularCache<LMAX>& pc,
    double r_n,
    double beta_n,
    double r_c,
    double (&dax)[(LMAX + 1) * (LMAX + 1)],
    double (&day)[(LMAX + 1) * (LMAX + 1)],
    double (&daz)[(LMAX + 1) * (LMAX + 1)],
    bool use_minus
) {
    constexpr int NLM = (LMAX + 1) * (LMAX + 1);
    constexpr double eps = 1e-14;

    const double r = pc.r;
    if (r < eps || r >= r_c) {
        for (int idx = 0; idx < NLM; ++idx) {
            dax[idx] = 0.0;
            day[idx] = 0.0;
            daz[idx] = 0.0;
        }
        return;
    }

    double f, fp;
    radial_gaussian_cutoff_and_deriv(r, r_n, beta_n, r_c, f, fp);

    // For -dr: geometric sign changes and parity relation on Y/dYdt.
    // All derived from +dr data stored in pc (no separate ylm_minus array needed).
    const double inv_r = 1.0 / r;
    const double inv_r_sin =
        (std::abs(pc.sin_theta) > eps) ? (1.0 / (r * pc.sin_theta)) : 0.0;

    // +dr geometry
    const double rx_p =  pc.x * inv_r;
    const double ry_p =  pc.y * inv_r;
    const double rz_p =  pc.z * inv_r;
    const double tx_p =  pc.cos_theta * pc.cos_phi * inv_r;
    const double ty_p =  pc.cos_theta * pc.sin_phi * inv_r;
    const double tz_p = -pc.sin_theta * inv_r;
    const double px_p = -pc.sin_phi * inv_r_sin;
    const double py_p =  pc.cos_phi * inv_r_sin;

    // -dr geometry (derived from parity analysis; see compute_pair_anlm_derivative_all_radials)
    const double rx = use_minus ? -rx_p : rx_p;
    const double ry = use_minus ? -ry_p : ry_p;
    const double rz = use_minus ? -rz_p : rz_p;
    const double tx = tx_p;   // cos_theta_m*cos_phi_m = (-ct)*(-cp) = ct*cp
    const double ty = ty_p;
    const double tz = tz_p;
    const double px = use_minus ? -px_p : px_p;
    const double py = use_minus ? -py_p : py_p;

    const double* Y    = pc.ylm.data();
    const double* dYdt = pc.dYdtheta.data();

    for (int l = 0; l <= LMAX; ++l) {
        const double parity = (l % 2 == 0) ? 1.0 : -1.0;
        // Y_m = parity*Y_p, dYdt_m = -parity*dYdt_p, phi_part_m = parity*phi_part_p
        const double Yscale    = use_minus ?  parity : 1.0;
        const double dYdtscale = use_minus ? -parity : 1.0;
        for (int m = -l; m <= l; ++m) {
            const int idx = sph_harm_flat_index(l, m);

            const double Yval      = Yscale    * Y[idx];
            const double dYdt_val  = dYdtscale * dYdt[idx];
            const double phi_part  = Yscale    * dY_dphi_from_Ylm(l, m, Y);

            dax[idx] = rx * fp * Yval + f * (tx * dYdt_val + px * phi_part);
            day[idx] = ry * fp * Yval + f * (ty * dYdt_val + py * phi_part);
            daz[idx] = rz * fp * Yval + f * (tz * dYdt_val);
        }
    }
}


// 案7: compute derivatives for all radial params in one pass per pair,
// hoisting angle-dependent terms (rx,ry,rz,tx,ty,tz,px,py,Y,dYdt) out of the ir loop.
// Output: dpx[ir][idx], dpy[ir][idx], dpz[ir][idx] for +dr and -dr, interleaved as:
//   out_p[ir * NLM + idx] = dpx/dpy/dpz, similarly out_m for -dr
template<int LMAX>
inline void compute_pair_anlm_derivative_all_radials(
    const PairAngularCache<LMAX>& pc,
    const std::vector<RadialParam>& radial_params,
    double r_c,
    double* dpx_all,  // [nrad * NLM]
    double* dpy_all,
    double* dpz_all,
    double* dmx_all,
    double* dmy_all,
    double* dmz_all
) {
    constexpr int NLM = (LMAX + 1) * (LMAX + 1);
    constexpr double eps = 1e-14;
    const int nrad = static_cast<int>(radial_params.size());

    const double r = pc.r;
    if (r < eps || r >= r_c) {
        const int total = nrad * NLM;
        for (int k = 0; k < total; ++k) {
            dpx_all[k] = dpy_all[k] = dpz_all[k] = 0.0;
            dmx_all[k] = dmy_all[k] = dmz_all[k] = 0.0;
        }
        return;
    }

    // --- angle-dependent terms for +dr (computed once) ---
    const double inv_r = 1.0 / r;
    const double inv_r_sin_p =
        (std::abs(pc.sin_theta) > eps) ? (1.0 / (r * pc.sin_theta)) : 0.0;

    const double rx_p = pc.x * inv_r;
    const double ry_p = pc.y * inv_r;
    const double rz_p = pc.z * inv_r;
    const double tx_p = pc.cos_theta * pc.cos_phi * inv_r;
    const double ty_p = pc.cos_theta * pc.sin_phi * inv_r;
    const double tz_p = -pc.sin_theta * inv_r;
    const double px_p = -pc.sin_phi * inv_r_sin_p;
    const double py_p =  pc.cos_phi * inv_r_sin_p;

    // --- angle-dependent terms for -dr (computed once) ---
    // For -dr: x_m=-x, cos_theta_m=-cos_theta, cos_phi_m=-cos_phi, sin_phi_m=-sin_phi
    // sin_theta_m = sin_theta (unchanged, sin_theta = sqrt(1 - cos_theta^2) >= 0)
    // rx_m = x_m/r = -rx_p
    // tx_m = cos_theta_m * cos_phi_m / r = (-cos_theta)*(-cos_phi)/r = +cos_theta*cos_phi/r = +tx_p
    // ty_m = cos_theta_m * sin_phi_m / r = (-cos_theta)*(-sin_phi)/r = +ty_p
    // tz_m = -sin_theta_m / r = -sin_theta/r = tz_p (same)
    // px_m = -sin_phi_m / (r*sin_theta) = -(-sin_phi)/(r*sin_theta) = +px_p ... wait
    // px_m = -sin_phi_m * inv_r_sin = sin_phi * inv_r_sin = -px_p
    // py_m =  cos_phi_m * inv_r_sin = -cos_phi * inv_r_sin = -py_p
    const double inv_r_sin_m = inv_r_sin_p;  // sin_theta is same for ±dr
    const double rx_m = -rx_p;
    const double ry_m = -ry_p;
    const double rz_m = -rz_p;
    const double tx_m = tx_p;   // cos_theta_m*cos_phi_m = (-ct)*(-cp) = ct*cp
    const double ty_m = ty_p;   // cos_theta_m*sin_phi_m = (-ct)*(-sp) = ct*sp
    const double tz_m = tz_p;   // -sin_theta_m = -sin_theta (same)
    const double px_m = -px_p;  // -sin_phi_m = -(-sp) = sp; px = sp*inv_r_sin = -px_p
    const double py_m = -py_p;  // cos_phi_m = -cp; py = -cp*inv_r_sin = -py_p

    const double* Y_p    = pc.ylm.data();
    const double* dYdt_p = pc.dYdtheta.data();

    // Pre-compute per-(l,m) angle factors (shared across all ir).
    // For -dr: parity = (-1)^l
    //   Y_m[idx]    = parity * Y_p[idx]
    //   dYdt_m[idx] = -parity * dYdt_p[idx]
    //   dY/dphi is linear in Y, so phi_part_m = parity * phi_part_p
    // Combined geometric signs: rx_m=-rx_p, tx_m=tx_p, px_m=-px_p
    // Result: dmx = rx_m*fp*Y_m + f*(tx_m*dYdt_m + px_m*Aph_m)
    //       = -rx_p*fp*par*Y_p + f*(tx_p*(-par)*dYdt_p + (-px_p)*par*Aph_p)
    //       = -par*(rx_p*fp*Y_p + f*(tx_p*dYdt_p + px_p*Aph_p)) = -par*dpx
    // Similarly: dmy = -par*dpy, dmz = -par*dpz
    // So we only need to compute +dr, then scale by -parity for -dr.

    // Precompute angle-only vectors (hoisted out of radial loop):
    //   A[idx]  = Y_p[idx]
    //   Cx[idx] = tx_p * dYdt + px_p * dY/dphi
    //   Cy[idx] = ty_p * dYdt + py_p * dY/dphi
    //   Cz[idx] = tz_p * dYdt
    //   neg_par[idx] = -(-1)^l   (for -dr scaling)
    double A[NLM], Cx[NLM], Cy[NLM], Cz[NLM], neg_par[NLM];

    for (int l = 0; l <= LMAX; ++l) {
        const double par = (l % 2 == 0) ? 1.0 : -1.0;
        const double npar = -par;
        // m == 0
        {
            const int idx = l * l + l;  // sph_harm_flat_index(l, 0)
            A[idx]       = Y_p[idx];
            Cx[idx]      = tx_p * dYdt_p[idx];   // phi part = 0 for m=0
            Cy[idx]      = ty_p * dYdt_p[idx];
            Cz[idx]      = tz_p * dYdt_p[idx];
            neg_par[idx] = npar;
        }
        for (int m = 1; m <= l; ++m) {
            const double ratio = (m % 2 == 0) ? 1.0 : -1.0;
            const int idx_pos = l * l + l + m;   // sph_harm_flat_index(l,  m)
            const int idx_neg = l * l + l - m;   // sph_harm_flat_index(l, -m)
            const double aph_pos = ratio * static_cast<double>( m) * Y_p[idx_neg];
            const double aph_neg = ratio * static_cast<double>(-m) * Y_p[idx_pos];

            A[idx_pos]       = Y_p[idx_pos];
            Cx[idx_pos]      = tx_p * dYdt_p[idx_pos] + px_p * aph_pos;
            Cy[idx_pos]      = ty_p * dYdt_p[idx_pos] + py_p * aph_pos;
            Cz[idx_pos]      = tz_p * dYdt_p[idx_pos];
            neg_par[idx_pos] = npar;

            A[idx_neg]       = Y_p[idx_neg];
            Cx[idx_neg]      = tx_p * dYdt_p[idx_neg] + px_p * aph_neg;
            Cy[idx_neg]      = ty_p * dYdt_p[idx_neg] + py_p * aph_neg;
            Cz[idx_neg]      = tz_p * dYdt_p[idx_neg];
            neg_par[idx_neg] = npar;
        }
    }

    // Radial loop: only f and fp change.
    // +dr: dpx = rx_p*fp*A + f*Cx,  -dr: dmx = neg_par * dpx
    for (int ir = 0; ir < nrad; ++ir) {
        double f, fp;
        radial_gaussian_cutoff_and_deriv(r, radial_params[ir].r_n, radial_params[ir].beta, r_c, f, fp);
        double* dpx = dpx_all + ir * NLM;
        double* dpy = dpy_all + ir * NLM;
        double* dpz = dpz_all + ir * NLM;
        double* dmx = dmx_all + ir * NLM;
        double* dmy = dmy_all + ir * NLM;
        double* dmz = dmz_all + ir * NLM;

        for (int idx = 0; idx < NLM; ++idx) {
            const double fp_A = fp * A[idx];
            const double px = rx_p * fp_A + f * Cx[idx];
            const double py = ry_p * fp_A + f * Cy[idx];
            const double pz = rz_p * fp_A + f * Cz[idx];
            dpx[idx] = px;
            dpy[idx] = py;
            dpz[idx] = pz;
            const double np = neg_par[idx];
            dmx[idx] = np * px;
            dmy[idx] = np * py;
            dmz[idx] = np * pz;
        }
    }
}


// ============================================================
// anlm construction
// ============================================================

template<int LMAX>
inline std::vector<double> compute_anlm_from_pairlist_omp_impl(
    const PairList& pairs,
    int n_atoms,
    double r_n,
    double beta_n,
    double r_c
) {
    constexpr int NLM = (LMAX + 1) * (LMAX + 1);
    const size_t anlm_size =
        static_cast<size_t>(n_atoms) * static_cast<size_t>(NLM);

    const int nthreads = omp_get_max_threads();

    std::vector<std::vector<double>> anlm_private(
        nthreads, std::vector<double>(anlm_size, 0.0)
    );

    const size_t npairs = pairs.pair_i.size();

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local = anlm_private[tid];

        double ylm_re[NLM];
        double ylm_im[NLM];

        #pragma omp for schedule(static)
        for (size_t p = 0; p < npairs; ++p) {
            const int i = pairs.pair_i[p];
            const int j = pairs.pair_j[p];

            const double r = pairs.dist[p];
            const double radial = radial_gaussian_cutoff(r, r_n, beta_n, r_c);
            if (radial == 0.0) continue;

            compute_sph_harm_all<LMAX>(
                pairs.cos_theta[p],
                pairs.sin_theta[p],
                pairs.cos_phi[p],
                pairs.sin_phi[p],
                ylm_re,
                ylm_im
            );

            const size_t base_i = static_cast<size_t>(i) * static_cast<size_t>(NLM);
            const size_t base_j = static_cast<size_t>(j) * static_cast<size_t>(NLM);

            for (int l = 0; l <= LMAX; ++l) {
                const double parity = (l % 2 == 0) ? 1.0 : -1.0;
                const int block_begin = l * l;
                const int block_size  = 2 * l + 1;

                for (int q = 0; q < block_size; ++q) {
                    const int idx = block_begin + q;
                    const double val = radial * ylm_re[idx];

                    local[base_i + idx] += val;
                    local[base_j + idx] += parity * val;
                }
            }
        }
    }

    std::vector<double> anlm(anlm_size, 0.0);

    for (int t = 0; t < nthreads; ++t) {
        const auto& src = anlm_private[t];
        for (size_t k = 0; k < anlm_size; ++k) {
            anlm[k] += src[k];
        }
    }

    return anlm;
}

inline std::vector<double> compute_anlm_from_pairlist_omp(
    const PairList& pairs,
    int n_atoms,
    int l_max,
    double r_n,
    double beta_n,
    double r_c
) {
    switch (l_max) {
        case 0:
            return compute_anlm_from_pairlist_omp_impl<0>(pairs, n_atoms, r_n, beta_n, r_c);
        case 1:
            return compute_anlm_from_pairlist_omp_impl<1>(pairs, n_atoms, r_n, beta_n, r_c);
        case 2:
            return compute_anlm_from_pairlist_omp_impl<2>(pairs, n_atoms, r_n, beta_n, r_c);
        case 3:
            return compute_anlm_from_pairlist_omp_impl<3>(pairs, n_atoms, r_n, beta_n, r_c);
        case 4:
            return compute_anlm_from_pairlist_omp_impl<4>(pairs, n_atoms, r_n, beta_n, r_c);
        case 5:
            return compute_anlm_from_pairlist_omp_impl<5>(pairs, n_atoms, r_n, beta_n, r_c);
        case 6:
            return compute_anlm_from_pairlist_omp_impl<6>(pairs, n_atoms, r_n, beta_n, r_c);
        case 7:
            return compute_anlm_from_pairlist_omp_impl<7>(pairs, n_atoms, r_n, beta_n, r_c);
        case 8:
            return compute_anlm_from_pairlist_omp_impl<8>(pairs, n_atoms, r_n, beta_n, r_c);
        case 9:
            return compute_anlm_from_pairlist_omp_impl<9>(pairs, n_atoms, r_n, beta_n, r_c);
        case 10:
            return compute_anlm_from_pairlist_omp_impl<10>(pairs, n_atoms, r_n, beta_n, r_c);
        case 11:
            return compute_anlm_from_pairlist_omp_impl<11>(pairs, n_atoms, r_n, beta_n, r_c);
        case 12:
            return compute_anlm_from_pairlist_omp_impl<12>(pairs, n_atoms, r_n, beta_n, r_c);
        default:
            throw std::runtime_error("compute_anlm_from_pairlist_omp: unsupported l_max");
    }
}

template<int LMAX>
inline std::vector<std::vector<double>> compute_anlm_from_pairlist_omp_batch_impl(
    const std::vector<PairAngularCache<LMAX>>& pcache,
    int n_atoms,
    const std::vector<RadialParam>& radial_params,
    double r_c
) {
    constexpr int NLM = (LMAX + 1) * (LMAX + 1);
    const size_t anlm_size = static_cast<size_t>(n_atoms) * static_cast<size_t>(NLM);
    const int nrad = static_cast<int>(radial_params.size());
    const size_t n_pairs = pcache.size();

    // Precompute radial values for all (pair, ir) once — pure function, same r per pair.
    // Layout: [p * nrad + ir]. ~n_pairs*nrad*8 bytes (e.g. 2000*4*8 = 64 KB).
    std::vector<double> radial_cache(n_pairs * static_cast<size_t>(nrad));
    for (size_t p = 0; p < n_pairs; ++p) {
        const double r = pcache[p].r;
        for (int ir = 0; ir < nrad; ++ir) {
            radial_cache[p * static_cast<size_t>(nrad) + ir] =
                radial_gaussian_cutoff(r, radial_params[ir].r_n, radial_params[ir].beta, r_c);
        }
    }

    // Build atom -> [pair_index, is_j_side] reverse lookup.
    // is_j_side=false: atom is pc.i (add +ylm), true: atom is pc.j (add parity*ylm).
    struct AtomPairRef { int p; bool is_j; };
    std::vector<std::vector<AtomPairRef>> atom_pairs(static_cast<size_t>(n_atoms));
    for (size_t p = 0; p < n_pairs; ++p) {
        const int ai = pcache[p].i;
        const int aj = pcache[p].j;
        atom_pairs[static_cast<size_t>(ai)].push_back({static_cast<int>(p), false});
        if (aj != ai) {
            atom_pairs[static_cast<size_t>(aj)].push_back({static_cast<int>(p), true});
        }
    }

    // Output buffer: no private copies, no reduction needed.
    // Each atom i owns slots [i*NLM .. (i+1)*NLM) in each anlm_all[ir].
    // Atom-axis OMP ensures no two threads write to the same slot.
    std::vector<std::vector<double>> out(
        nrad, std::vector<double>(anlm_size, 0.0)
    );

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_atoms; ++i) {
        const size_t base_i = static_cast<size_t>(i) * NLM;
        for (const auto& ref : atom_pairs[static_cast<size_t>(i)]) {
            const double* rc = radial_cache.data() + static_cast<size_t>(ref.p) * nrad;
            const double* ylm = pcache[static_cast<size_t>(ref.p)].ylm.data();
            for (int ir = 0; ir < nrad; ++ir) {
                const double radial = rc[ir];
                if (radial == 0.0) continue;
                double* dst = out[ir].data() + base_i;
                if (!ref.is_j) {
                    for (int l = 0; l <= LMAX; ++l) {
                        const int block_begin = l * l;
                        const int block_size  = 2 * l + 1;
                        const double* ylm_l = ylm + block_begin;
                        double* di = dst + block_begin;
                        for (int q = 0; q < block_size; ++q) {
                            di[q] += radial * ylm_l[q];
                        }
                    }
                } else {
                    for (int l = 0; l <= LMAX; ++l) {
                        const double parity = (l % 2 == 0) ? 1.0 : -1.0;
                        const int block_begin = l * l;
                        const int block_size  = 2 * l + 1;
                        const double* ylm_l = ylm + block_begin;
                        double* di = dst + block_begin;
                        for (int q = 0; q < block_size; ++q) {
                            di[q] += parity * radial * ylm_l[q];
                        }
                    }
                }
            }
        }
    }

    return out;
}

// Variant that builds PairAngularCache on the stack per-pair instead of reading
// from a pre-allocated heap vector. Eliminates the large pcache heap allocation.
template<int LMAX>
inline std::vector<std::vector<double>> compute_anlm_from_pairs_stackcache(
    const PairList& pairs,
    int n_atoms,
    const std::vector<RadialParam>& radial_params,
    double r_c
) {
    constexpr int NLM = (LMAX + 1) * (LMAX + 1);
    const size_t anlm_size = static_cast<size_t>(n_atoms) * static_cast<size_t>(NLM);
    const int nrad = static_cast<int>(radial_params.size());
    const int nthreads = omp_get_max_threads();
    const size_t n_pairs = pairs.pair_i.size();

    std::vector<std::vector<std::vector<double>>> private_anlm(
        nthreads, std::vector<std::vector<double>>(nrad, std::vector<double>(anlm_size, 0.0))
    );

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (size_t p = 0; p < n_pairs; ++p) {
            // Build angular cache on the stack — no heap allocation.
            PairAngularCache<LMAX> pc;
            build_pair_angular_cache_one<LMAX>(pairs, p, pc);

            const size_t base_i = static_cast<size_t>(pc.i) * static_cast<size_t>(NLM);
            const size_t base_j = static_cast<size_t>(pc.j) * static_cast<size_t>(NLM);

            for (int ir = 0; ir < nrad; ++ir) {
                const double radial = radial_gaussian_cutoff(
                    pc.r, radial_params[ir].r_n, radial_params[ir].beta, r_c
                );
                if (radial == 0.0) continue;

                auto& local = private_anlm[tid][ir];

                for (int l = 0; l <= LMAX; ++l) {
                    const double parity = (l % 2 == 0) ? 1.0 : -1.0;
                    const int block_begin = l * l;
                    const int block_size  = 2 * l + 1;

                    for (int q = 0; q < block_size; ++q) {
                        const int idx = block_begin + q;
                        const double val = radial * pc.ylm[idx];
                        local[base_i + idx] += val;
                        local[base_j + idx] += parity * val;
                    }
                }
            }
        }
    }

    std::vector<std::vector<double>> out(
        nrad, std::vector<double>(anlm_size, 0.0)
    );
    for (int ir = 0; ir < nrad; ++ir) {
        for (int t = 0; t < nthreads; ++t) {
            for (size_t k = 0; k < anlm_size; ++k) {
                out[ir][k] += private_anlm[t][ir][k];
            }
        }
    }
    return out;
}


// ============================================================
// feature metadata / utilities
// ============================================================

// Max order supported inline (order > MAX_TERM_ORDER uses fallback heap vectors)
static constexpr int MAX_TERM_ORDER = 4;

struct SparseTermFast {
    double coeff;
    // local_idx and flat_idx stored inline for small orders to avoid heap indirection.
    // For order <= MAX_TERM_ORDER, use inline arrays.
    int n_idx = 0;
    std::array<int, MAX_TERM_ORDER> local_idx_arr{};
    std::array<int, MAX_TERM_ORDER> flat_idx_arr{};
    // Fallback for order > MAX_TERM_ORDER (rare / unused in practice)
    std::vector<int> local_idx;
    std::vector<int> flat_idx;

    // Accessors — always use these
    int get_flat_idx(int t) const {
        return (t < MAX_TERM_ORDER) ? flat_idx_arr[t] : flat_idx[t];
    }
    int get_local_idx(int t) const {
        return (t < MAX_TERM_ORDER) ? local_idx_arr[t] : local_idx[t];
    }
};

struct FeatureMetaFast {
    std::vector<int> radial_ids;   // n1, n2, ...
    std::vector<int> ls;           // l1, l2, ...
    std::vector<int> dims;
    std::vector<SparseTermFast> terms;
    bool valid = false;
    std::string key;
    int column = 0;
};

struct FeatureLambdaLayout {
    std::vector<int> offsets;
    int total_dim = 0;
};

inline std::vector<SparseTermFast> build_sparse_terms_fast(
    const double* eig_ptr,
    size_t eig_size,
    const std::vector<int>& ls,
    double eps = 1e-15
) {
    std::vector<SparseTermFast> terms;
    if (eig_ptr == nullptr || eig_size == 0) return terms;

    const int order = static_cast<int>(ls.size());
    std::vector<int> dims(order, 0);
    for (int i = 0; i < order; ++i) {
        dims[i] = 2 * ls[i] + 1;
    }

    std::vector<int> strides(order, 1);
    int total_size = 1;
    for (int i = order - 1; i >= 0; --i) {
        strides[i] = total_size;
        total_size *= dims[i];
    }

    if (static_cast<size_t>(total_size) != eig_size) {
        throw std::runtime_error("build_sparse_terms_fast: eig_size mismatch");
    }

    for (int linear = 0; linear < total_size; ++linear) {
        const double c = eig_ptr[linear];
        if (std::abs(c) < eps) continue;

        SparseTermFast term;
        term.coeff = c;
        term.n_idx = order;
        term.local_idx.resize(order);
        term.flat_idx.resize(order);

        int rem = linear;
        for (int d = 0; d < order; ++d) {
            const int idx = rem / strides[d];
            rem %= strides[d];

            const int l = ls[d];
            const int m = idx - l;
            const int fidx = sph_harm_flat_index(l, m);
            term.local_idx[d] = idx;
            term.flat_idx[d]  = fidx;
            if (d < MAX_TERM_ORDER) {
                term.local_idx_arr[d] = idx;
                term.flat_idx_arr[d]  = fidx;
            }
        }
        terms.push_back(std::move(term));
    }

    return terms;
}

inline int compute_total_feature_dim(const std::vector<int>& ls) {
    int total_size = 1;
    for (int l : ls) {
        total_size *= (2 * l + 1);
    }
    return total_size;
}

inline std::vector<std::vector<SparseTermFast>> build_sparse_terms_from_eig_array(
    const py::array_t<double>& eig_py,
    const std::vector<int>& ls
) {
    py::buffer_info eig_buf = eig_py.request();
    const int total_size = compute_total_feature_dim(ls);

    std::vector<std::vector<SparseTermFast>> columns;

    if (eig_buf.ndim == 1) {
        columns.push_back(
            build_sparse_terms_fast(
                static_cast<const double*>(eig_buf.ptr),
                static_cast<size_t>(eig_buf.size),
                ls
            )
        );
        return columns;
    }

    if (eig_buf.ndim != 2) {
        throw std::runtime_error(
            "eigenvector array must be 1D or 2D"
        );
    }

    const auto* eig_ptr = static_cast<const double*>(eig_buf.ptr);
    const ssize_t rows = eig_buf.shape[0];
    const ssize_t cols = eig_buf.shape[1];

    if (rows == total_size) {
        columns.reserve(static_cast<size_t>(cols));
        for (ssize_t col = 0; col < cols; ++col) {
            std::vector<double> column_data(static_cast<size_t>(total_size));
            for (int row = 0; row < total_size; ++row) {
                column_data[static_cast<size_t>(row)] =
                    eig_ptr[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                            static_cast<size_t>(col)];
            }
            columns.push_back(
                build_sparse_terms_fast(column_data.data(), column_data.size(), ls)
            );
        }
        return columns;
    }

    if (cols == total_size) {
        columns.reserve(static_cast<size_t>(rows));
        for (ssize_t row = 0; row < rows; ++row) {
            const double* row_ptr =
                eig_ptr + static_cast<size_t>(row) * static_cast<size_t>(cols);
            columns.push_back(
                build_sparse_terms_fast(row_ptr, static_cast<size_t>(cols), ls)
            );
        }
        return columns;
    }

    throw std::runtime_error(
        "2D eigenvector array shape is incompatible with l_list dimensions"
    );
}

// Lookup eigenvector data from the statically embedded header by key string.
// Returns nullptr if not found.
inline const eigvec_data::EigvecEntry* find_eigvec_entry(const std::string& key) {
    for (int i = 0; i < eigvec_data::REGISTRY_SIZE; ++i) {
        if (key == eigvec_data::REGISTRY[i].key) {
            return &eigvec_data::REGISTRY[i];
        }
    }
    return nullptr;
}

// Build SparseTermFast columns from a statically embedded raw array (rows x cols, row-major).
inline std::vector<std::vector<SparseTermFast>> build_sparse_terms_from_raw(
    const double* data,
    int rows,
    int cols,
    const std::vector<int>& ls
) {
    const int total_size = compute_total_feature_dim(ls);
    std::vector<std::vector<SparseTermFast>> columns;

    if (rows == total_size) {
        // Each column is one eigenvector
        columns.reserve(static_cast<size_t>(cols));
        for (int col = 0; col < cols; ++col) {
            std::vector<double> column_data(static_cast<size_t>(total_size));
            for (int row = 0; row < total_size; ++row) {
                column_data[static_cast<size_t>(row)] =
                    data[static_cast<size_t>(row) * static_cast<size_t>(cols)
                         + static_cast<size_t>(col)];
            }
            columns.push_back(build_sparse_terms_fast(column_data.data(), column_data.size(), ls));
        }
    } else if (cols == total_size) {
        columns.reserve(static_cast<size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            const double* row_ptr = data + static_cast<size_t>(row) * static_cast<size_t>(cols);
            columns.push_back(build_sparse_terms_fast(row_ptr, static_cast<size_t>(cols), ls));
        }
    } else {
        throw std::runtime_error("build_sparse_terms_from_raw: shape incompatible with ls");
    }
    return columns;
}

inline double eval_feature_sparse_fast(
    const double* anlm_center_base,
    const FeatureMetaFast& fm
) {
    double out = 0.0;
    for (const auto& term : fm.terms) {
        double prod = 1.0;
        for (size_t t = 0; t < term.flat_idx.size(); ++t) {
            prod *= anlm_center_base[term.flat_idx[t]];
        }
        out += term.coeff * prod;
    }
    return out;
}
inline double eval_feature_sparse_fast_multi(
    const std::vector<std::vector<double>>& anlm_all,
    size_t base,
    const FeatureMetaFast& fm
) {
    double out = 0.0;
    const int order = static_cast<int>(fm.ls.size());

    // Fast paths for common orders — avoid inner loop and heap vector access
    if (order == 1) {
        const double* anlm = anlm_all[fm.radial_ids[0]].data() + base;
        for (const auto& term : fm.terms) {
            out += term.coeff * anlm[term.flat_idx_arr[0]];
        }
        return out;
    }
    if (order == 2) {
        const double* anlm = anlm_all[fm.radial_ids[0]].data() + base;
        for (const auto& term : fm.terms) {
            out += term.coeff * anlm[term.flat_idx_arr[0]] * anlm[term.flat_idx_arr[1]];
        }
        return out;
    }
    if (order == 3) {
        const double* anlm = anlm_all[fm.radial_ids[0]].data() + base;
        for (const auto& term : fm.terms) {
            out += term.coeff
                * anlm[term.flat_idx_arr[0]]
                * anlm[term.flat_idx_arr[1]]
                * anlm[term.flat_idx_arr[2]];
        }
        return out;
    }

    // Generic fallback
    for (const auto& term : fm.terms) {
        double prod = 1.0;
        for (size_t t = 0; t < term.flat_idx.size(); ++t) {
            const int ir = fm.radial_ids[t];
            prod *= anlm_all[ir][base + term.flat_idx[t]];
        }
        out += term.coeff * prod;
    }

    return out;
}
inline std::vector<std::vector<double>> compute_feature_partials_fast(
    const double* anlm_center_base,
    const FeatureMetaFast& fm
) {
    const int order = static_cast<int>(fm.ls.size());
    std::vector<std::vector<double>> lambda(order);

    for (int t = 0; t < order; ++t) {
        lambda[t].assign(fm.dims[t], 0.0);
    }

    for (const auto& term : fm.terms) {
        const int p = static_cast<int>(term.flat_idx.size());

        std::vector<double> prefix(p + 1, 1.0);
        std::vector<double> suffix(p + 1, 1.0);

        for (int t = 0; t < p; ++t) {
            prefix[t + 1] = prefix[t] * anlm_center_base[term.flat_idx[t]];
        }
        for (int t = p - 1; t >= 0; --t) {
            suffix[t] = suffix[t + 1] * anlm_center_base[term.flat_idx[t]];
        }

        for (int t = 0; t < p; ++t) {
            lambda[t][term.local_idx[t]] += term.coeff * prefix[t] * suffix[t + 1];
        }
    }

    return lambda;
}

inline std::vector<double> compute_feature_partials_fast_multi(
    const std::vector<std::vector<double>>& anlm_all,
    int center,
    int NLM,
    int nrad,
    const FeatureMetaFast& fm
) {
    // flattened layout: lambda[ir * NLM + flat_idx]
    std::vector<double> lambda(static_cast<size_t>(nrad) * NLM, 0.0);

    for (const auto& term : fm.terms) {
        const int p = static_cast<int>(term.flat_idx.size());

        std::vector<double> vals(p);
        for (int t = 0; t < p; ++t) {
            const int ir = fm.radial_ids[t];
            vals[t] = anlm_all[ir][static_cast<size_t>(center) * NLM + term.flat_idx[t]];
        }

        std::vector<double> prefix(p + 1, 1.0);
        std::vector<double> suffix(p + 1, 1.0);

        for (int t = 0; t < p; ++t) {
            prefix[t + 1] = prefix[t] * vals[t];
        }
        for (int t = p - 1; t >= 0; --t) {
            suffix[t] = suffix[t + 1] * vals[t];
        }

        for (int t = 0; t < p; ++t) {
            const int ir = fm.radial_ids[t];
            const int idx = term.flat_idx[t];
            lambda[static_cast<size_t>(ir) * NLM + idx] += term.coeff * prefix[t] * suffix[t + 1];
        }
    }

    return lambda;
}

inline std::vector<double> compute_feature_partials_single_radial(
    const std::vector<std::vector<double>>& anlm_all,
    int center,
    int NLM,
    const FeatureMetaFast& fm
) {
    std::vector<double> lambda(NLM, 0.0);

    for (const auto& term : fm.terms) {
        const int p = static_cast<int>(term.flat_idx.size());

        std::vector<double> vals(p);
        for (int t = 0; t < p; ++t) {
            const int ir = fm.radial_ids[t];
            vals[t] = anlm_all[ir][static_cast<size_t>(center) * NLM + term.flat_idx[t]];
        }

        std::vector<double> prefix(p + 1, 1.0);
        std::vector<double> suffix(p + 1, 1.0);

        for (int t = 0; t < p; ++t) {
            prefix[t + 1] = prefix[t] * vals[t];
        }
        for (int t = p - 1; t >= 0; --t) {
            suffix[t] = suffix[t + 1] * vals[t];
        }

        for (int t = 0; t < p; ++t) {
            const int idx = term.flat_idx[t];
            lambda[idx] += term.coeff * prefix[t] * suffix[t + 1];
        }
    }

    return lambda;
}

inline void compute_feature_partials_single_radial_reuse(
    const std::vector<std::vector<double>>& anlm_all,
    size_t base,
    const FeatureMetaFast& fm,
    std::vector<double>& lambda,
    std::vector<double>& vals,
    std::vector<double>& prefix,
    std::vector<double>& suffix
) {
    std::fill(lambda.begin(), lambda.end(), 0.0);

    const int order = static_cast<int>(fm.ls.size());

    // Fast paths for common orders (all radial_ids are the same for single-radial features)
    if (order == 1) {
        // dX/da_i = coeff * prefix[0] * suffix[1] = coeff * 1 * 1 = coeff
        // (no anlm factor — the product has only one term so the "leave one out" product is 1)
        for (const auto& term : fm.terms) {
            lambda[static_cast<size_t>(term.flat_idx_arr[0])] += term.coeff;
        }
        return;
    }

    if (order == 2) {
        const double* anlm = anlm_all[fm.radial_ids[0]].data() + base;
        for (const auto& term : fm.terms) {
            const double v0 = anlm[term.flat_idx_arr[0]];
            const double v1 = anlm[term.flat_idx_arr[1]];
            const double c  = term.coeff;
            lambda[static_cast<size_t>(term.flat_idx_arr[0])] += c * v1;
            lambda[static_cast<size_t>(term.flat_idx_arr[1])] += c * v0;
        }
        return;
    }

    if (order == 3) {
        const double* anlm = anlm_all[fm.radial_ids[0]].data() + base;
        for (const auto& term : fm.terms) {
            const double v0 = anlm[term.flat_idx_arr[0]];
            const double v1 = anlm[term.flat_idx_arr[1]];
            const double v2 = anlm[term.flat_idx_arr[2]];
            const double c  = term.coeff;
            lambda[static_cast<size_t>(term.flat_idx_arr[0])] += c * v1 * v2;
            lambda[static_cast<size_t>(term.flat_idx_arr[1])] += c * v0 * v2;
            lambda[static_cast<size_t>(term.flat_idx_arr[2])] += c * v0 * v1;
        }
        return;
    }

    // Generic fallback for order >= 4
    for (const auto& term : fm.terms) {
        const int p = static_cast<int>(term.flat_idx.size());
        if (p == 0) continue;

        for (int t = 0; t < p; ++t) {
            const int ir = fm.radial_ids[t];
            vals[static_cast<size_t>(t)] =
                anlm_all[ir][base + term.flat_idx[t]];
        }

        prefix[0] = 1.0;
        for (int t = 0; t < p; ++t) {
            prefix[static_cast<size_t>(t + 1)] =
                prefix[static_cast<size_t>(t)] * vals[static_cast<size_t>(t)];
        }

        suffix[static_cast<size_t>(p)] = 1.0;
        for (int t = p - 1; t >= 0; --t) {
            suffix[static_cast<size_t>(t)] =
                suffix[static_cast<size_t>(t + 1)] * vals[static_cast<size_t>(t)];
        }

        for (int t = 0; t < p; ++t) {
            const int idx = term.flat_idx[t];
            lambda[static_cast<size_t>(idx)] +=
                term.coeff * prefix[static_cast<size_t>(t)] * suffix[static_cast<size_t>(t + 1)];
        }
    }
}

inline std::vector<RadialParam> parse_radial_params(py::list radial_param_py) {
    std::vector<RadialParam> radial_params;
    radial_params.reserve(py::len(radial_param_py));

    for (py::handle item : radial_param_py) {
        py::tuple tup = py::cast<py::tuple>(item);
        if (py::len(tup) != 2) {
            throw std::runtime_error(
                "Each element of radial_param must be a tuple/list of length 2: (beta, r_n)"
            );
        }

        // Keep polymlp.in convention: each tuple is (beta, r_n).
        RadialParam rp;
        rp.beta = py::cast<double>(tup[0]);
        rp.r_n  = py::cast<double>(tup[1]);
        radial_params.push_back(rp);
    }

    if (radial_params.empty()) {
        throw std::runtime_error("radial_param must not be empty");
    }

    return radial_params;
}

inline void enumerate_non_decreasing_radial_ids_rec(
    int depth,
    int order,
    int nrad,
    int start,
    std::vector<int>& current,
    std::vector<std::vector<int>>& out
) {
    if (depth == order) {
        out.push_back(current);
        return;
    }

    for (int ir = start; ir < nrad; ++ir) {
        current[depth] = ir;
        enumerate_non_decreasing_radial_ids_rec(
            depth + 1, order, nrad, ir, current, out
        );
    }
}

inline std::vector<std::vector<int>> enumerate_non_decreasing_radial_ids(
    int order,
    int nrad
) {
    std::vector<std::vector<int>> out;
    std::vector<int> current(order, 0);
    enumerate_non_decreasing_radial_ids_rec(
        0, order, nrad, 0, current, out
    );
    return out;
}

inline std::string make_ls_key_bracket(const std::vector<int>& ls) {
    std::string key = "l_list[";
    for (size_t i = 0; i < ls.size(); ++i) {
        key += std::to_string(ls[i]);
        if (i + 1 < ls.size()) key += ", ";
    }
    key += "]";
    return key;
}

inline std::string make_ls_key_tuple(const std::vector<int>& ls) {
    std::string key = "l_list(";
    for (size_t i = 0; i < ls.size(); ++i) {
        key += std::to_string(ls[i]);
        if (i + 1 < ls.size()) key += ", ";
    }
    if (ls.size() == 1) {
        key += ", ";
    }
    key += ")";
    return key;
}

inline std::vector<FeatureMetaFast> build_feature_meta_from_all_ls_and_radials(
    py::list all_ls_py,
    int nrad
) {
    std::vector<FeatureMetaFast> feat_meta;
    const int n_base_feat = static_cast<int>(py::len(all_ls_py));

    // polymlp(model_type=1) compatible order:
    // [radial0_feat0..featN-1, radial1_feat0..featN-1, ...]
    for (int ir = 0; ir < nrad; ++ir) {
        for (int kfeat = 0; kfeat < n_base_feat; ++kfeat) {
            std::vector<int> ls = all_ls_py[kfeat].cast<std::vector<int>>();
            const int order = static_cast<int>(ls.size());

            const std::string key_bracket = make_ls_key_bracket(ls);
            const std::string key_tuple = make_ls_key_tuple(ls);

            std::vector<std::vector<SparseTermFast>> all_terms;
            bool valid = false;
            std::string used_key = key_bracket;

            const eigvec_data::EigvecEntry* entry = find_eigvec_entry(key_bracket);
            if (entry == nullptr) {
                entry = find_eigvec_entry(key_tuple);
                if (entry != nullptr) used_key = key_tuple;
            }

            if (entry != nullptr) {
                all_terms = build_sparse_terms_from_raw(
                    entry->data, entry->rows, entry->cols, ls);
                valid = true;
            }

            if (!valid) {
                FeatureMetaFast fm;
                fm.radial_ids.assign(order, ir);
                fm.ls = ls;
                fm.key = used_key;
                fm.valid = false;
                fm.dims.resize(ls.size());
                for (size_t i = 0; i < ls.size(); ++i) {
                    fm.dims[i] = 2 * ls[i] + 1;
                }
                feat_meta.push_back(std::move(fm));
                continue;
            }

            for (size_t icol = 0; icol < all_terms.size(); ++icol) {
                FeatureMetaFast fm;
                fm.radial_ids.assign(order, ir);
                fm.ls = ls;
                fm.key = used_key;
                fm.column = static_cast<int>(icol);
                fm.terms = std::move(all_terms[icol]);
                fm.valid = true;

                fm.dims.resize(ls.size());
                for (size_t i = 0; i < ls.size(); ++i) {
                    fm.dims[i] = 2 * ls[i] + 1;
                }

                feat_meta.push_back(std::move(fm));
            }
        }
    }
    return feat_meta;
}

inline std::string make_feature_meta_cache_key(
    py::list all_ls_py,
    int nrad
) {
    std::ostringstream oss;
    oss << "nrad=" << nrad << "|ls=";
    for (py::handle item : all_ls_py) {
        auto ls = py::cast<std::vector<int>>(item);
        oss << "[";
        for (size_t i = 0; i < ls.size(); ++i) {
            oss << ls[i];
            if (i + 1 < ls.size()) oss << ",";
        }
        oss << "]";
    }
    return oss.str();
}

inline const std::vector<FeatureMetaFast>& get_cached_feature_meta(
    py::list all_ls_py,
    int nrad
) {
    static std::unordered_map<std::string, std::shared_ptr<std::vector<FeatureMetaFast>>> cache;
    static std::mutex cache_mutex;

    const std::string key = make_feature_meta_cache_key(all_ls_py, nrad);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return *(it->second);
        }
    }

    auto built = std::make_shared<std::vector<FeatureMetaFast>>(
        build_feature_meta_from_all_ls_and_radials(all_ls_py, nrad)
    );

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto [it, inserted] = cache.emplace(key, built);
        if (!inserted) {
            return *(it->second);
        }
        return *built;
    }
}
// ============================================================
// main descriptor implementation
// ============================================================

template<int LMAX>
py::tuple compute_all_descriptors_batch_impl(
    py::array_t<double> pos_py,
    py::array_t<double> lat_py,
    double n_param,
    py::list all_ls_py,
    const std::vector<RadialParam>& radial_params,
    double r_c,
    bool compute_force
) {
    (void)n_param;

    constexpr int NLM = (LMAX + 1) * (LMAX + 1);

    auto pos = pos_py.unchecked<2>();
    auto lat = lat_py.unchecked<2>();

    const int n_atoms = static_cast<int>(pos.shape(0));
    const int nrad    = static_cast<int>(radial_params.size());

    const auto& feat_meta = get_cached_feature_meta(all_ls_py, nrad);
    
    const int n_feat = static_cast<int>(feat_meta.size());
    // --------------------------------------------------
    // geometry
    // --------------------------------------------------
    std::vector<Vec3> atom_pos(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        atom_pos[i] = Vec3(pos(i, 0), pos(i, 1), pos(i, 2));
    }

    Mat3 lattice{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            lattice.a[i][j] = lat(i, j);
        }
    }
    // --------------------------------------------------
    // feature metadata
    // --------------------------------------------------
    int invalid_count = 0;
for (const auto& fm : feat_meta) {
    if (!fm.valid) invalid_count++;
}
if (invalid_count > 0) {
    std::cout << "[WARN] " << invalid_count
              << " features skipped: not in data_dict" << std::endl;
}
    // --------------------------------------------------
    // outputs
    // feature axis = [radial0_feat0 ... radial0_feat(n_feat-1),
    //                 radial1_feat0 ...]
    // --------------------------------------------------
    py::array_t<double> np_e({n_atoms, n_feat});
    py::array_t<double> np_f({n_atoms * 3, n_feat});
    py::array_t<double> np_s({n_atoms * 9, n_feat});

    auto ptr_e = np_e.mutable_unchecked<2>();
    auto ptr_f = np_f.mutable_unchecked<2>();
    auto ptr_s = np_s.mutable_unchecked<2>();

    for (int i = 0; i < n_atoms; ++i) {
        for (int k = 0; k < n_feat; ++k) {
            ptr_e(i, k) = 0.0;
        }
    }

    for (int i = 0; i < n_atoms * 3; ++i) {
        for (int k = 0; k < n_feat; ++k) {
            ptr_f(i, k) = 0.0;
        }
    }

    for (int i = 0; i < n_atoms * 9; ++i) {
        for (int k = 0; k < n_feat; ++k) {
            ptr_s(i, k) = 0.0;
        }
    }

    // --------------------------------------------------
    // pair list + angular cache
    // --------------------------------------------------
    PairList pairs =
#ifdef BENCH_BEFORE
        build_half_pair_list_periodic_images_naive(atom_pos, lattice, r_c);
#else
        build_half_pair_list_periodic_images(atom_pos, lattice, r_c);
#endif

    // --------------------------------------------------
    // pair angular cache: build once, reuse for both anlm and force loops.
    // static vector avoids repeated heap alloc/free across calls.
    // --------------------------------------------------
    static std::vector<PairAngularCache<LMAX>> pcache;
    const size_t n_pairs_now = pairs.pair_i.size();
    if (pcache.capacity() < n_pairs_now) {
        pcache.reserve(n_pairs_now);
    }
    pcache.resize(n_pairs_now);

    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < n_pairs_now; ++p) {
        build_pair_angular_cache_one<LMAX>(pairs, p, pcache[p], compute_force);
    }

    // --------------------------------------------------
    // anlm for all radial params
    // --------------------------------------------------
    std::vector<std::vector<double>> anlm_all =
        compute_anlm_from_pairlist_omp_batch_impl<LMAX>(pcache, n_atoms, radial_params, r_c);

// --------------------------------------------------
// Xe : compute energy features first
// --------------------------------------------------
#pragma omp parallel for schedule(static)
for (int center = 0; center < n_atoms; ++center) {
    const size_t base = static_cast<size_t>(center) * NLM;
    for (int kfeat = 0; kfeat < n_feat; ++kfeat) {
        const auto& fm = feat_meta[static_cast<size_t>(kfeat)];
        if (fm.valid) {
            ptr_e(center, kfeat) =
                eval_feature_sparse_fast_multi(anlm_all, base, fm);
        } else {
            ptr_e(center, kfeat) = 0.0;
        }
    }
}

if (!compute_force) {
    return py::make_tuple(np_e, np_f, np_s);
}

// --------------------------------------------------
// Precompute d(feature_k)/d(a_nlm) for each atom-center and feature.
// --------------------------------------------------
std::vector<int> feature_ir(n_feat, 0);
std::vector<std::vector<int>> features_by_ir(static_cast<size_t>(nrad));
for (int kfeat = 0; kfeat < n_feat; ++kfeat) {
    if (!feat_meta[kfeat].valid || feat_meta[kfeat].radial_ids.empty()) {
        feature_ir[kfeat] = -1;
        continue;
    }
    const int ir = feat_meta[kfeat].radial_ids[0];
    if (ir >= 0 && ir < nrad) {
        feature_ir[kfeat] = ir;
        features_by_ir[static_cast<size_t>(ir)].push_back(kfeat);
    } else {
        feature_ir[kfeat] = -1;
    }
}

int max_feature_order = 0;
for (const auto& fm : feat_meta) {
    max_feature_order = std::max(max_feature_order, static_cast<int>(fm.ls.size()));
}

// Plan 1: pre-compute lambda for all (atom, feat) into a dense buffer [n_atoms, n_feat, NLM].
// CSR conversion (old step 6) is eliminated — the force loop uses dense dot products.
const size_t n_center_feat = static_cast<size_t>(n_atoms) * static_cast<size_t>(n_feat);
const size_t lambda_all_size = n_center_feat * static_cast<size_t>(NLM);
std::vector<double> lambda_all(lambda_all_size, 0.0);

#pragma omp parallel
{
    std::vector<double> lambda_buf(static_cast<size_t>(NLM), 0.0);
    std::vector<double> vals_buf(static_cast<size_t>(max_feature_order), 0.0);
    std::vector<double> prefix_buf(static_cast<size_t>(max_feature_order + 1), 1.0);
    std::vector<double> suffix_buf(static_cast<size_t>(max_feature_order + 1), 1.0);

    #pragma omp for schedule(static)
    for (int center = 0; center < n_atoms; ++center) {
        const size_t anlm_base = static_cast<size_t>(center) * NLM;
        for (int kfeat = 0; kfeat < n_feat; ++kfeat) {
            const auto& fm = feat_meta[static_cast<size_t>(kfeat)];
            if (!fm.valid) continue;

            compute_feature_partials_single_radial_reuse(
                anlm_all, anlm_base, fm,
                lambda_buf, vals_buf, prefix_buf, suffix_buf
            );

            const size_t dst_base =
                (static_cast<size_t>(center) * static_cast<size_t>(n_feat)
                 + static_cast<size_t>(kfeat)) * static_cast<size_t>(NLM);
            std::copy(lambda_buf.begin(), lambda_buf.end(),
                      lambda_all.begin() + static_cast<std::ptrdiff_t>(dst_base));
        }
    }
}

#ifdef BENCH_BEFORE
// (BENCH_BEFORE path uses lambda_all built above; the force loop below handles both paths)
#endif

// Optional pair-derivative cache:
// cache d(a_nlm)/dr for each (pair, radial) to avoid repeated derivative evaluations
const size_t n_pairs = pairs.pair_i.size();
const size_t deriv_stride_pair = static_cast<size_t>(nrad) * static_cast<size_t>(NLM);
const size_t deriv_total = n_pairs * deriv_stride_pair;
const size_t deriv_cache_bytes = deriv_total * sizeof(double) * 6ULL;
const size_t deriv_work_units =
    n_pairs * static_cast<size_t>(nrad) * static_cast<size_t>(NLM);
constexpr size_t DERIV_CACHE_MAX_BYTES = 1536ULL * 1024ULL * 1024ULL; // 1.5 GiB
constexpr size_t DERIV_CACHE_MIN_WORK_UNITS = 8000000ULL;
const bool cache_memory_ok = (deriv_cache_bytes <= DERIV_CACHE_MAX_BYTES);
const bool cache_problem_large = (deriv_work_units >= DERIV_CACHE_MIN_WORK_UNITS);
const bool use_pair_deriv_cache_auto = cache_memory_ok && cache_problem_large;
const DerivCacheMode deriv_cache_mode = get_deriv_cache_mode();
bool use_pair_deriv_cache = use_pair_deriv_cache_auto;
if (deriv_cache_mode == DerivCacheMode::On) {
    // Keep memory guard even in ON mode to avoid excessive allocations.
    use_pair_deriv_cache = cache_memory_ok;
} else if (deriv_cache_mode == DerivCacheMode::Off) {
    use_pair_deriv_cache = false;
}

std::vector<double> dpx_cache, dpy_cache, dpz_cache;
std::vector<double> dmx_cache, dmy_cache, dmz_cache;
if (use_pair_deriv_cache) {
    dpx_cache.resize(deriv_total);
    dpy_cache.resize(deriv_total);
    dpz_cache.resize(deriv_total);
    dmx_cache.resize(deriv_total);
    dmy_cache.resize(deriv_total);
    dmz_cache.resize(deriv_total);

    constexpr int MAX_NRAD_CACHE = 16;
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < n_pairs; ++p) {
        const auto& pc = pcache[p];
        const size_t base = p * static_cast<size_t>(nrad) * static_cast<size_t>(NLM);
        compute_pair_anlm_derivative_all_radials<LMAX>(
            pc, radial_params, r_c,
            dpx_cache.data() + base,
            dpy_cache.data() + base,
            dpz_cache.data() + base,
            dmx_cache.data() + base,
            dmy_cache.data() + base,
            dmz_cache.data() + base
        );
    }
}

const int nthreads = omp_get_max_threads();
// Fix 1: flat 1D layout [tid * n_atoms*3*n_feat + atom_alpha*n_feat + kfeat]
// Eliminates vector<vector> pointer indirection; one contiguous allocation
const size_t f_stride = static_cast<size_t>(n_atoms) * 3 * static_cast<size_t>(n_feat);
std::vector<double> f_private_all(static_cast<size_t>(nthreads) * f_stride, 0.0);

#pragma omp parallel
{
    const int tid = omp_get_thread_num();
    double* f_private = f_private_all.data() + static_cast<size_t>(tid) * f_stride;

#ifdef BENCH_BEFORE
    #pragma omp for schedule(static)
    for (size_t p = 0; p < pcache.size(); ++p) {
        const auto& pc = pcache[p];
        const int i = pc.i;
        const int j = pc.j;

        std::vector<char> deriv_ready(static_cast<size_t>(nrad), 0);
        std::vector<double> dpx_all(static_cast<size_t>(nrad) * NLM, 0.0);
        std::vector<double> dpy_all(static_cast<size_t>(nrad) * NLM, 0.0);
        std::vector<double> dpz_all(static_cast<size_t>(nrad) * NLM, 0.0);
        std::vector<double> dmx_all(static_cast<size_t>(nrad) * NLM, 0.0);
        std::vector<double> dmy_all(static_cast<size_t>(nrad) * NLM, 0.0);
        std::vector<double> dmz_all(static_cast<size_t>(nrad) * NLM, 0.0);

        for (int kfeat = 0; kfeat < n_feat; ++kfeat) {
            const auto& fm = feat_meta[kfeat];
            if (!fm.valid) continue;

            const int ir = feature_ir[kfeat];
            if (ir < 0) continue;
            const size_t doff = static_cast<size_t>(ir) * NLM;
            if (!deriv_ready[static_cast<size_t>(ir)]) {
                double dpx[NLM], dpy[NLM], dpz[NLM];
                double dmx[NLM], dmy[NLM], dmz[NLM];

                compute_pair_anlm_derivative_from_cache<LMAX>(
                    pc, radial_params[ir].r_n, radial_params[ir].beta, r_c,
                    dpx, dpy, dpz, false
                );
                compute_pair_anlm_derivative_from_cache<LMAX>(
                    pc, radial_params[ir].r_n, radial_params[ir].beta, r_c,
                    dmx, dmy, dmz, true
                );

                for (int idx = 0; idx < NLM; ++idx) {
                    const size_t didx = doff + static_cast<size_t>(idx);
                    dpx_all[didx] = dpx[idx]; dpy_all[didx] = dpy[idx]; dpz_all[didx] = dpz[idx];
                    dmx_all[didx] = dmx[idx]; dmy_all[didx] = dmy[idx]; dmz_all[didx] = dmz[idx];
                }
                deriv_ready[static_cast<size_t>(ir)] = 1;
            }

            const size_t lbase_i = (static_cast<size_t>(i) * static_cast<size_t>(n_feat) + static_cast<size_t>(kfeat)) * lambda_stride;
            const size_t lbase_j = (static_cast<size_t>(j) * static_cast<size_t>(n_feat) + static_cast<size_t>(kfeat)) * lambda_stride;
            const double* lambda_i = lambda_all.data() + lbase_i;
            const double* lambda_j = lambda_all.data() + lbase_j;

            double fij_x = 0.0, fij_y = 0.0, fij_z = 0.0;
            double fji_x = 0.0, fji_y = 0.0, fji_z = 0.0;
            for (int idx = 0; idx < NLM; ++idx) {
                const size_t didx = doff + static_cast<size_t>(idx);
                const double lam_i = lambda_i[idx];
                const double lam_j = lambda_j[idx];
                if (lam_i * lam_i > LAMBDA_EPS2) { fij_x += -lam_i * dpx_all[didx]; fij_y += -lam_i * dpy_all[didx]; fij_z += -lam_i * dpz_all[didx]; }
                if (lam_j * lam_j > LAMBDA_EPS2) { fji_x += -lam_j * dmx_all[didx]; fji_y += -lam_j * dmy_all[didx]; fji_z += -lam_j * dmz_all[didx]; }
            }

            f_private[(static_cast<size_t>(3*j+0)*n_feat)+kfeat] += fij_x;
            f_private[(static_cast<size_t>(3*j+1)*n_feat)+kfeat] += fij_y;
            f_private[(static_cast<size_t>(3*j+2)*n_feat)+kfeat] += fij_z;
            f_private[(static_cast<size_t>(3*i+0)*n_feat)+kfeat] -= fij_x;
            f_private[(static_cast<size_t>(3*i+1)*n_feat)+kfeat] -= fij_y;
            f_private[(static_cast<size_t>(3*i+2)*n_feat)+kfeat] -= fij_z;
            f_private[(static_cast<size_t>(3*i+0)*n_feat)+kfeat] += fji_x;
            f_private[(static_cast<size_t>(3*i+1)*n_feat)+kfeat] += fji_y;
            f_private[(static_cast<size_t>(3*i+2)*n_feat)+kfeat] += fji_z;
            f_private[(static_cast<size_t>(3*j+0)*n_feat)+kfeat] -= fji_x;
            f_private[(static_cast<size_t>(3*j+1)*n_feat)+kfeat] -= fji_y;
            f_private[(static_cast<size_t>(3*j+2)*n_feat)+kfeat] -= fji_z;
        }
    }
#else
    // Plan 1: lambda_all is dense [n_atoms, n_feat, NLM].
    // Force pair loop uses dense dot products — no CSR scatter/gather.
    constexpr int MAX_NRAD = 16;

    #pragma omp for schedule(guided)
    for (size_t p = 0; p < n_pairs; ++p) {
        const auto& pc = pcache[p];
        const int i = pc.i;
        const int j = pc.j;

        double dpx_all_local[MAX_NRAD * NLM], dpy_all_local[MAX_NRAD * NLM], dpz_all_local[MAX_NRAD * NLM];
        double dmx_all_local[MAX_NRAD * NLM], dmy_all_local[MAX_NRAD * NLM], dmz_all_local[MAX_NRAD * NLM];
        if (!use_pair_deriv_cache) {
            compute_pair_anlm_derivative_all_radials<LMAX>(
                pc, radial_params, r_c,
                dpx_all_local, dpy_all_local, dpz_all_local,
                dmx_all_local, dmy_all_local, dmz_all_local
            );
        }

        for (int ir = 0; ir < nrad; ++ir) {
            const auto& kfeat_list = features_by_ir[static_cast<size_t>(ir)];
            if (kfeat_list.empty()) continue;

            const double* dpx_ptr;
            const double* dpy_ptr;
            const double* dpz_ptr;
            const double* dmx_ptr;
            const double* dmy_ptr;
            const double* dmz_ptr;
            if (use_pair_deriv_cache) {
                const size_t base =
                    (p * static_cast<size_t>(nrad) + static_cast<size_t>(ir))
                    * static_cast<size_t>(NLM);
                dpx_ptr = dpx_cache.data() + base;
                dpy_ptr = dpy_cache.data() + base;
                dpz_ptr = dpz_cache.data() + base;
                dmx_ptr = dmx_cache.data() + base;
                dmy_ptr = dmy_cache.data() + base;
                dmz_ptr = dmz_cache.data() + base;
            } else {
                const size_t base = static_cast<size_t>(ir) * NLM;
                dpx_ptr = dpx_all_local + base;
                dpy_ptr = dpy_all_local + base;
                dpz_ptr = dpz_all_local + base;
                dmx_ptr = dmx_all_local + base;
                dmy_ptr = dmy_all_local + base;
                dmz_ptr = dmz_all_local + base;
            }

            for (int kfeat : kfeat_list) {
                // Dense lambda vectors from pre-computed buffer.
                const size_t lbase_i =
                    (static_cast<size_t>(i) * static_cast<size_t>(n_feat)
                     + static_cast<size_t>(kfeat)) * static_cast<size_t>(NLM);
                const size_t lbase_j =
                    (static_cast<size_t>(j) * static_cast<size_t>(n_feat)
                     + static_cast<size_t>(kfeat)) * static_cast<size_t>(NLM);
                const double* lambda_i = lambda_all.data() + lbase_i;
                const double* lambda_j = lambda_all.data() + lbase_j;

                // Dense dot products: fij = -lambda_i · dp,  fji = -lambda_j · dm
                double fij_x = 0.0, fij_y = 0.0, fij_z = 0.0;
                double fji_x = 0.0, fji_y = 0.0, fji_z = 0.0;
                for (int idx = 0; idx < NLM; ++idx) {
                    const double li = lambda_i[idx];
                    const double lj = lambda_j[idx];
                    fij_x -= li * dpx_ptr[idx];
                    fij_y -= li * dpy_ptr[idx];
                    fij_z -= li * dpz_ptr[idx];
                    fji_x -= lj * dmx_ptr[idx];
                    fji_y -= lj * dmy_ptr[idx];
                    fji_z -= lj * dmz_ptr[idx];
                }

                f_private[(static_cast<size_t>(3 * j + 0) * n_feat) + kfeat] += fij_x;
                f_private[(static_cast<size_t>(3 * j + 1) * n_feat) + kfeat] += fij_y;
                f_private[(static_cast<size_t>(3 * j + 2) * n_feat) + kfeat] += fij_z;
                f_private[(static_cast<size_t>(3 * i + 0) * n_feat) + kfeat] -= fij_x;
                f_private[(static_cast<size_t>(3 * i + 1) * n_feat) + kfeat] -= fij_y;
                f_private[(static_cast<size_t>(3 * i + 2) * n_feat) + kfeat] -= fij_z;
                f_private[(static_cast<size_t>(3 * i + 0) * n_feat) + kfeat] += fji_x;
                f_private[(static_cast<size_t>(3 * i + 1) * n_feat) + kfeat] += fji_y;
                f_private[(static_cast<size_t>(3 * i + 2) * n_feat) + kfeat] += fji_z;
                f_private[(static_cast<size_t>(3 * j + 0) * n_feat) + kfeat] -= fji_x;
                f_private[(static_cast<size_t>(3 * j + 1) * n_feat) + kfeat] -= fji_y;
                f_private[(static_cast<size_t>(3 * j + 2) * n_feat) + kfeat] -= fji_z;
            }
        }
    }
#endif
}

// Fix 3: transpose reduction loop — iterate t in serial outer, atom_alpha in parallel.
for (int t = 0; t < nthreads; ++t) {
    const double* src_t = f_private_all.data() + static_cast<size_t>(t) * f_stride;
    #pragma omp parallel for schedule(static)
    for (int atom_alpha = 0; atom_alpha < n_atoms * 3; ++atom_alpha) {
        const double* src = src_t + static_cast<size_t>(atom_alpha) * static_cast<size_t>(n_feat);
        for (int kfeat = 0; kfeat < n_feat; ++kfeat) {
            ptr_f(atom_alpha, kfeat) += src[kfeat];
        }
    }
}

// --------------------------------------------------
// S : zero for now
// --------------------------------------------------
for (int i = 0; i < n_atoms * 9; ++i) {
    for (int kfeat = 0; kfeat < n_feat; ++kfeat) {
        ptr_s(i, kfeat) = 0.0;
    }
}

    return py::make_tuple(np_e, np_f, np_s);
}


// ============================================================
// python wrapper
// ============================================================

py::tuple compute_all_descriptors_batch(
    py::array_t<double> pos_py,
    py::array_t<double> lat_py,
    py::dict data_dict,   // accepted for API compatibility; eigenvectors are compiled-in
    double n_param,
    py::list all_ls_py,
    py::list radial_param_py,
    double r_c,
    bool compute_force
) {
    (void)data_dict;  // unused — data is embedded in eigenvectors_R_data.h
    std::vector<RadialParam> radial_params = parse_radial_params(radial_param_py);

    int l_max_global = 0;
    for (py::handle item : all_ls_py) {
        std::vector<int> ls = py::cast<std::vector<int>>(item);
        for (int l : ls) {
            l_max_global = std::max(l_max_global, l);
        }
    }

    switch (l_max_global) {
        case 0:
            return compute_all_descriptors_batch_impl<0>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 1:
            return compute_all_descriptors_batch_impl<1>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 2:
            return compute_all_descriptors_batch_impl<2>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 3:
            return compute_all_descriptors_batch_impl<3>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 4:
            return compute_all_descriptors_batch_impl<4>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 5:
            return compute_all_descriptors_batch_impl<5>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 6:
            return compute_all_descriptors_batch_impl<6>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 7:
            return compute_all_descriptors_batch_impl<7>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 8:
            return compute_all_descriptors_batch_impl<8>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 9:
            return compute_all_descriptors_batch_impl<9>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 10:
            return compute_all_descriptors_batch_impl<10>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 11:
            return compute_all_descriptors_batch_impl<11>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        case 12:
            return compute_all_descriptors_batch_impl<12>(
                pos_py, lat_py, n_param, all_ls_py, radial_params, r_c, compute_force
            );
        default:
            throw std::runtime_error("compute_all_descriptors_batch: unsupported l_max_global");
    }
}


// ============================================================
// pybind
// ============================================================

PYBIND11_MODULE(DESCRIPTOR_MODULE_NAME, m) {
    m.def(
        "compute_all_descriptors",
        &compute_all_descriptors_batch,
        py::arg("pos"),
        py::arg("lat"),
        py::arg("data_dict"),
        py::arg("n_param"),
        py::arg("all_ls"),
        py::arg("radial_param"),
        py::arg("r_c"),
        py::arg("compute_force") = true
    );
}
