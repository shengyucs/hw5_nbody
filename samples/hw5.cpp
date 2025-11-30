#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <hip/hip_runtime.h>

#define HIP_CHECK(call)                                                          \
    do {                                                                         \
        hipError_t err = (call);                                                 \
        if (err != hipSuccess) {                                                 \
            std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;   \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace param {
const int    n_steps = 200000;
const double dt      = 60.0;
const double eps     = 1e-3;
const double G       = 6.674e-11;

__host__ __device__
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000.0));
}

const double planet_radius = 1e7;
const double missile_speed = 1e6;

inline double get_missile_cost(double t) { return 1e5 + 1e3 * t; }

}  // namespace param

struct SystemState {
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz;
    std::vector<double> vx, vy, vz;
    std::vector<double> m;
    std::vector<std::string> type;
};

void read_input(const char* filename, SystemState& s) {
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("cannot open input file");

    fin >> s.n >> s.planet >> s.asteroid;
    int n = s.n;

    s.qx.resize(n); s.qy.resize(n); s.qz.resize(n);
    s.vx.resize(n); s.vy.resize(n); s.vz.resize(n);
    s.m.resize(n);  s.type.resize(n);

    for (int i = 0; i < n; i++)
        fin >> s.qx[i] >> s.qy[i] >> s.qz[i]
            >> s.vx[i] >> s.vy[i] >> s.vz[i]
            >> s.m[i] >> s.type[i];
}

void write_output(const char* filename, double min_dist, int hit_time_step,
                  int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1)
         << min_dist << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

struct DeviceBuffersP1P2 {
    double *d_qx=nullptr,*d_qy=nullptr,*d_qz=nullptr;
    double *d_vx=nullptr,*d_vy=nullptr,*d_vz=nullptr;
    double *d_m0=nullptr,*d_m_eff=nullptr;
    int    *d_is_device=nullptr;
    double *d_min_dist=nullptr;
    int    *d_hit_step=nullptr;
    int    *d_destroy_step=nullptr;
    int n_alloc=0;
};

static void allocate_device_buffers_p1p2(DeviceBuffersP1P2& buf, int n) {
    if (buf.n_alloc == n && buf.d_qx) return;
    if (buf.n_alloc > 0) {
        (void)hipFree(buf.d_qx); (void)hipFree(buf.d_qy); (void)hipFree(buf.d_qz);
        (void)hipFree(buf.d_vx); (void)hipFree(buf.d_vy); (void)hipFree(buf.d_vz);
        (void)hipFree(buf.d_m0); (void)hipFree(buf.d_m_eff);
        (void)hipFree(buf.d_is_device);
        (void)hipFree(buf.d_min_dist);
        (void)hipFree(buf.d_hit_step);
        (void)hipFree(buf.d_destroy_step);
    }
    HIP_CHECK(hipMalloc(&buf.d_qx,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_qy,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_qz,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_vx,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_vy,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_vz,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_m0,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_m_eff,n*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_is_device,n*sizeof(int)));
    HIP_CHECK(hipMalloc(&buf.d_min_dist,sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_hit_step,sizeof(int)));
    HIP_CHECK(hipMalloc(&buf.d_destroy_step,n*sizeof(int)));
    buf.n_alloc = n;
}

static void free_device_buffers_p1p2(DeviceBuffersP1P2& buf) {
    if (!buf.n_alloc) return;
    (void)hipFree(buf.d_qx); (void)hipFree(buf.d_qy); (void)hipFree(buf.d_qz);
    (void)hipFree(buf.d_vx); (void)hipFree(buf.d_vy); (void)hipFree(buf.d_vz);
    (void)hipFree(buf.d_m0); (void)hipFree(buf.d_m_eff);
    (void)hipFree(buf.d_is_device);
    (void)hipFree(buf.d_min_dist);
    (void)hipFree(buf.d_hit_step);
    (void)hipFree(buf.d_destroy_step);
    buf = DeviceBuffersP1P2{};
}

struct DeviceBuffersP3 {
    double *d_qx=nullptr,*d_qy=nullptr,*d_qz=nullptr;
    double *d_vx=nullptr,*d_vy=nullptr,*d_vz=nullptr;
    double *d_m0=nullptr,*d_m_eff=nullptr;
    int *d_is_device=nullptr;
    int *d_scenario_dev_id=nullptr;
    int *d_scenario_destroy_step=nullptr;
    int *d_hit_step=nullptr;
    int n_alloc=0,max_scen=0;
};

static void allocate_device_buffers_p3(DeviceBuffersP3& buf, int n, int n_scen) {
    if (buf.n_alloc == n && buf.max_scen == n_scen && buf.d_qx) return;

    if (buf.n_alloc > 0) {
        (void)hipFree(buf.d_qx); (void)hipFree(buf.d_qy); (void)hipFree(buf.d_qz);
        (void)hipFree(buf.d_vx); (void)hipFree(buf.d_vy); (void)hipFree(buf.d_vz);
        (void)hipFree(buf.d_m0); (void)hipFree(buf.d_m_eff);
        (void)hipFree(buf.d_is_device);
        (void)hipFree(buf.d_scenario_dev_id);
        (void)hipFree(buf.d_scenario_destroy_step);
        (void)hipFree(buf.d_hit_step);
    }

    std::size_t total = (size_t)n * n_scen;
    HIP_CHECK(hipMalloc(&buf.d_qx,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_qy,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_qz,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_vx,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_vy,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_vz,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_m0,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_m_eff,total*sizeof(double)));
    HIP_CHECK(hipMalloc(&buf.d_is_device,total*sizeof(int)));

    HIP_CHECK(hipMalloc(&buf.d_scenario_dev_id,n_scen*sizeof(int)));
    HIP_CHECK(hipMalloc(&buf.d_scenario_destroy_step,n_scen*sizeof(int)));
    HIP_CHECK(hipMalloc(&buf.d_hit_step,n_scen*sizeof(int)));

    buf.n_alloc  = n;
    buf.max_scen = n_scen;
}

static void free_device_buffers_p3(DeviceBuffersP3& buf) {
    if (!buf.n_alloc) return;
    (void)hipFree(buf.d_qx); (void)hipFree(buf.d_qy); (void)hipFree(buf.d_qz);
    (void)hipFree(buf.d_vx); (void)hipFree(buf.d_vy); (void)hipFree(buf.d_vz);
    (void)hipFree(buf.d_m0); (void)hipFree(buf.d_m_eff);
    (void)hipFree(buf.d_is_device);
    (void)hipFree(buf.d_scenario_dev_id);
    (void)hipFree(buf.d_scenario_destroy_step);
    (void)hipFree(buf.d_hit_step);
    buf = DeviceBuffersP3{};
}

constexpr int BLOCK_SIZE = 256;

__device__ inline void atomicMinDouble(double* addr, double val) {
    auto ptr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *ptr;
    while (true) {
        unsigned long long assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val <= val) break;
        unsigned long long new_val = __double_as_longlong(val);
        old = atomicCAS(ptr, assumed, new_val);
        if (old == assumed) break;
    }
}

__global__ void compute_step_kernel(
    int n, int step,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* m_eff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double xi=qx[i], yi=qy[i], zi=qz[i];
    double vxi=vx[i], vyi=vy[i], vzi=vz[i];

    double axi=0, ayi=0, azi=0;

    __shared__ double sh_qx[BLOCK_SIZE], sh_qy[BLOCK_SIZE], sh_qz[BLOCK_SIZE], sh_m[BLOCK_SIZE];
    int tid = threadIdx.x;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        int j = tile + tid;
        if (j < n) {
            sh_qx[tid]=qx[j];
            sh_qy[tid]=qy[j];
            sh_qz[tid]=qz[j];
            sh_m [tid]=m_eff[j];
        }
        __syncthreads();

        int tileSize = min(BLOCK_SIZE, n - tile);
        #pragma unroll 16
        for (int tj = 0; tj < tileSize; ++tj) {
            int jj = tile + tj;
            if (jj == i) continue;
            double dx = sh_qx[tj] - xi;
            double dy = sh_qy[tj] - yi;
            double dz = sh_qz[tj] - zi;
            double dist2 = dx*dx + dy*dy + dz*dz + param::eps*param::eps;
            double inv = rsqrt(dist2);
            double inv3 = inv*inv*inv;
            double f = param::G * sh_m[tj] * inv3;
            axi += f*dx; ayi += f*dy; azi += f*dz;
        }
        __syncthreads();
    }

    vxi+=axi*param::dt; vyi+=ayi*param::dt; vzi+=azi*param::dt;
    xi+=vxi*param::dt; yi+=vyi*param::dt; zi+=vzi*param::dt;

    vx[i]=vxi; vy[i]=vyi; vz[i]=vzi;
    qx[i]=xi;  qy[i]=yi;  qz[i]=zi;
}

__global__ void update_min_dist_kernel(
    const double* qx, const double* qy, const double* qz,
    int planet, int asteroid, double* min_dist)
{
    double dx=qx[planet]-qx[asteroid];
    double dy=qy[planet]-qy[asteroid];
    double dz=qz[planet]-qz[asteroid];
    double d = sqrt(dx*dx + dy*dy + dz*dz);
    atomicMinDouble(min_dist, d);
}

static double solve_problem1_gpu(const SystemState& initial) {
    int n = initial.n;
    std::vector<int> is_dev(n);
    std::vector<double> m_eff(n);

    for (int i = 0; i < n; i++) {
        is_dev[i] = (initial.type[i]=="device");
        m_eff[i]  = is_dev[i]? 0.0 : initial.m[i];
    }

    DeviceBuffersP1P2 buf;
    HIP_CHECK(hipSetDevice(0));
    allocate_device_buffers_p1p2(buf,n);

    HIP_CHECK(hipMemcpy(buf.d_qx, initial.qx.data(),n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_qy, initial.qy.data(),n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_qz, initial.qz.data(),n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_vx, initial.vx.data(),n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_vy, initial.vy.data(),n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_vz, initial.vz.data(),n*sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_m_eff,m_eff.data(),n*sizeof(double), hipMemcpyHostToDevice));

    double init_min = std::numeric_limits<double>::max();
    HIP_CHECK(hipMemcpy(buf.d_min_dist,&init_min,sizeof(double),hipMemcpyHostToDevice));

    int blocks = (n + BLOCK_SIZE - 1)/BLOCK_SIZE;

    for (int step=0; step<=param::n_steps; step++) {
        if (step>0) {
            hipLaunchKernelGGL(compute_step_kernel,
                dim3(blocks),dim3(BLOCK_SIZE),0,0,
                n,step,buf.d_qx,buf.d_qy,buf.d_qz,
                buf.d_vx,buf.d_vy,buf.d_vz,
                buf.d_m_eff);
            HIP_CHECK(hipDeviceSynchronize());
        }
        hipLaunchKernelGGL(update_min_dist_kernel,
            dim3(1),dim3(1),0,0,
            buf.d_qx,buf.d_qy,buf.d_qz,
            initial.planet, initial.asteroid, buf.d_min_dist);
        HIP_CHECK(hipDeviceSynchronize());
    }

    double out;
    HIP_CHECK(hipMemcpy(&out, buf.d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
    free_device_buffers_p1p2(buf);
    return out;
}

__global__ void persistent_kernel_p2_base(
    int n,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    double* m0, int* is_device, double* m_eff,
    int planet, int asteroid,
    int* hit_step,
    int* destroy_step)
{
    int tid = threadIdx.x;

    __shared__ double sh_qx[BLOCK_SIZE], sh_qy[BLOCK_SIZE], sh_qz[BLOCK_SIZE], sh_m[BLOCK_SIZE];

    for (int step_internal=0; step_internal<param::n_steps; ++step_internal) {
        int step = step_internal+1;
        double t = step * param::dt;

        for (int j=tid;j<n;j+=blockDim.x) {
            double mj = m0[j];
            int    fd = is_device[j];
            if (fd) mj = param::gravity_device_mass(mj,t);
            m_eff[j] = mj;
        }
        __syncthreads();

        for (int i=tid;i<n;i+=blockDim.x) {
            double xi=qx[i], yi=qy[i], zi=qz[i];
            double vxi=vx[i], vyi=vy[i], vzi=vz[i];
            double axi=0, ayi=0, azi=0;

            for (int tile=0; tile<n; tile+=BLOCK_SIZE) {
                int j=tile+tid;
                if (j<n) {
                    sh_qx[tid]=qx[j];
                    sh_qy[tid]=qy[j];
                    sh_qz[tid]=qz[j];
                    sh_m [tid]=m_eff[j];
                }
                __syncthreads();

                int ts = min(BLOCK_SIZE, n-tile);
                #pragma unroll 8
                for (int tj=0;tj<ts;++tj) {
                    int jj = tile+tj;
                    if (jj==i) continue;
                    double dx=sh_qx[tj]-xi, dy=sh_qy[tj]-yi, dz=sh_qz[tj]-zi;
                    double d2 = dx*dx+dy*dy+dz*dz+param::eps*param::eps;
                    double inv = rsqrt(d2);
                    double inv3=inv*inv*inv;
                    double f=param::G*sh_m[tj]*inv3;
                    axi+=f*dx; ayi+=f*dy; azi+=f*dz;
                }
                __syncthreads();
            }

            vxi+=axi*param::dt; vyi+=ayi*param::dt; vzi+=azi*param::dt;
            xi+=vxi*param::dt; yi+=vyi*param::dt; zi+=vzi*param::dt;

            vx[i]=vxi; vy[i]=vyi; vz[i]=vzi;
            qx[i]=xi;  qy[i]=yi;  qz[i]=zi;
        }
        __syncthreads();

        if (tid==0 && *hit_step==-2) {
            double dx=qx[planet]-qx[asteroid];
            double dy=qy[planet]-qy[asteroid];
            double dz=qz[planet]-qz[asteroid];
            double d2 = dx*dx+dy*dy+dz*dz;
            if (d2 < param::planet_radius*param::planet_radius)
                *hit_step = step;
        }
        __syncthreads();

        if (*hit_step != -2) break;

        if (tid==0) {
            double px=qx[planet], py=qy[planet], pz=qz[planet];
            double R = step * param::dt * param::missile_speed;
            for (int j=0;j<n;j++) {
                if (!is_device[j]) continue;
                if (destroy_step[j]!=-1) continue;
                double dx=qx[j]-px, dy=qy[j]-py, dz=qz[j]-pz;
                double dist = sqrt(dx*dx+dy*dy+dz*dz);
                if (R > dist) destroy_step[j] = step;
            }
        }
        __syncthreads();
    }
}

static int solve_problem2_and_destroy_steps(
    const SystemState& initial,
    std::vector<int>& destroy_steps_host)
{
    int n = initial.n;
    destroy_steps_host.assign(n, -1);

    std::vector<double> m0(initial.m);
    std::vector<int> is_dev(n);
    for (int i=0;i<n;i++) is_dev[i] = (initial.type[i]=="device");

    DeviceBuffersP1P2 buf;
    HIP_CHECK(hipSetDevice(0));
    allocate_device_buffers_p1p2(buf,n);

    HIP_CHECK(hipMemcpy(buf.d_qx,initial.qx.data(),n*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_qy,initial.qy.data(),n*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_qz,initial.qz.data(),n*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_vx,initial.vx.data(),n*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_vy,initial.vy.data(),n*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_vz,initial.vz.data(),n*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_m0,m0.data(),n*sizeof(double),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_is_device,is_dev.data(),n*sizeof(int),hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(buf.d_destroy_step,0xff,n*sizeof(int)));
    int init_hit=-2;
    HIP_CHECK(hipMemcpy(buf.d_hit_step,&init_hit,sizeof(int),hipMemcpyHostToDevice));

    hipLaunchKernelGGL(persistent_kernel_p2_base,
        dim3(1),dim3(BLOCK_SIZE),0,0,
        n,
        buf.d_qx,buf.d_qy,buf.d_qz,
        buf.d_vx,buf.d_vy,buf.d_vz,
        buf.d_m0,buf.d_is_device,buf.d_m_eff,
        initial.planet,initial.asteroid,
        buf.d_hit_step,buf.d_destroy_step);
    HIP_CHECK(hipDeviceSynchronize());

    int hit;
    HIP_CHECK(hipMemcpy(&hit, buf.d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(destroy_steps_host.data(), buf.d_destroy_step,
                        n*sizeof(int), hipMemcpyDeviceToHost));

    free_device_buffers_p1p2(buf);
    return hit;
}

__global__ void p3_multi_scenario_kernel(
    int n, int n_scen,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    double* m0, int* is_device, double* m_eff,
    int planet, int asteroid,
    const int* scenario_dev_id,
    const int* scenario_destroy_step,
    int* hit_step_out)
{
    int scen = blockIdx.x;
    if (scen>=n_scen) return;

    int tid = threadIdx.x;
    int offset = scen * n;

    double* qx_s = qx + offset;
    double* qy_s = qy + offset;
    double* qz_s = qz + offset;
    double* vx_s = vx + offset;
    double* vy_s = vy + offset;
    double* vz_s = vz + offset;
    double* m0_s = m0 + offset;
    double* m_eff_s = m_eff + offset;
    int* is_dev_s = is_device + offset;

    int target = scenario_dev_id[scen];
    int destroy_s = scenario_destroy_step[scen];

    __shared__ double sh_qx[BLOCK_SIZE], sh_qy[BLOCK_SIZE], sh_qz[BLOCK_SIZE], sh_m[BLOCK_SIZE];
    int local_hit = -2;

    for (int step_internal=0; step_internal<param::n_steps; ++step_internal) {
        int step = step_internal+1;
        double t = step * param::dt;

        for (int j=tid;j<n;j+=blockDim.x) {
            double mj = m0_s[j];
            int fd = is_dev_s[j];
            if (j==target && step>=destroy_s) {
                fd=0; mj=0.0;
            }
            if (fd) mj = param::gravity_device_mass(mj,t);
            m_eff_s[j] = mj;
        }
        __syncthreads();

        for (int i=tid;i<n;i+=blockDim.x) {
            double xi=qx_s[i], yi=qy_s[i], zi=qz_s[i];
            double vxi=vx_s[i], vyi=vy_s[i], vzi=vz_s[i];
            double axi=0, ayi=0, azi=0;

            for (int tile=0; tile<n; tile+=BLOCK_SIZE) {
                int j=tile+tid;
                if (j<n) {
                    sh_qx[tid]=qx_s[j];
                    sh_qy[tid]=qy_s[j];
                    sh_qz[tid]=qz_s[j];
                    sh_m [tid]=m_eff_s[j];
                }
                __syncthreads();

                int ts=min(BLOCK_SIZE,n-tile);
                #pragma unroll 8
                for (int tj=0;tj<ts;++tj) {
                    int jj=tile+tj;
                    if (jj==i) continue;
                    double dx=sh_qx[tj]-xi, dy=sh_qy[tj]-yi, dz=sh_qz[tj]-zi;
                    double d2=dx*dx+dy*dy+dz*dz+param::eps*param::eps;
                    double inv=rsqrt(d2);
                    double inv3=inv*inv*inv;
                    double f=param::G*sh_m[tj]*inv3;
                    axi+=f*dx; ayi+=f*dy; azi+=f*dz;
                }
                __syncthreads();
            }

            vxi+=axi*param::dt; vyi+=ayi*param::dt; vzi+=azi*param::dt;
            xi+=vxi*param::dt; yi+=vyi*param::dt; zi+=vzi*param::dt;

            vx_s[i]=vxi; vy_s[i]=vyi; vz_s[i]=vzi;
            qx_s[i]=xi;  qy_s[i]=yi;  qz_s[i]=zi;
        }
        __syncthreads();

        if (tid==0 && local_hit==-2) {
            double dx=qx_s[planet]-qx_s[asteroid];
            double dy=qy_s[planet]-qy_s[asteroid];
            double dz=qz_s[planet]-qz_s[asteroid];
            double d2=dx*dx+dy*dy+dz*dz;
            if (d2 < param::planet_radius*param::planet_radius)
                local_hit=step;
        }
        __syncthreads();

        if (local_hit!=-2) break;
    }

    if (tid==0) hit_step_out[scen]=local_hit;
}

static void solve_problem3_single_gpu(
    const SystemState& initial,
    int hit_time_step_p2,
    const std::vector<int>& destroy_steps_base,
    int& best_id,
    double& best_cost)
{
    int n = initial.n;

    if (hit_time_step_p2 == -2) { best_id=-1; best_cost=0; return; }

    std::vector<int> cand;
    for (int i=0;i<n;i++)
        if (initial.type[i]=="device" &&
            destroy_steps_base[i]!=-1 &&
            destroy_steps_base[i] <= hit_time_step_p2)
            cand.push_back(i);

    if (cand.empty()) { best_id=-1; best_cost=0; return; }

    int n_scen = cand.size();

    DeviceBuffersP3 buf;
    HIP_CHECK(hipSetDevice(0));
    allocate_device_buffers_p3(buf,n,n_scen);

    std::vector<double> m0(initial.m);
    std::vector<int> is_dev(n);
    for (int i=0;i<n;i++) is_dev[i]=(initial.type[i]=="device");

    for (int s=0;s<n_scen;s++) {
        int off = s*n;
        HIP_CHECK(hipMemcpy(buf.d_qx+off, initial.qx.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buf.d_qy+off, initial.qy.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buf.d_qz+off, initial.qz.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buf.d_vx+off, initial.vx.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buf.d_vy+off, initial.vy.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buf.d_vz+off, initial.vz.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buf.d_m0+off, m0.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(buf.d_is_device+off, is_dev.data(), n*sizeof(int), hipMemcpyHostToDevice));
    }

    std::vector<int> scen_dev(n_scen), scen_ds(n_scen);
    for (int s=0;s<n_scen;s++) {
        scen_dev[s]=cand[s];
        scen_ds[s]=destroy_steps_base[cand[s]];
    }

    HIP_CHECK(hipMemcpy(buf.d_scenario_dev_id, scen_dev.data(),
                        n_scen*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(buf.d_scenario_destroy_step, scen_ds.data(),
                        n_scen*sizeof(int), hipMemcpyHostToDevice));

    std::vector<int> init_hit(n_scen,-2);
    HIP_CHECK(hipMemcpy(buf.d_hit_step, init_hit.data(),
                        n_scen*sizeof(int), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(p3_multi_scenario_kernel,
        dim3(n_scen), dim3(BLOCK_SIZE),0,0,
        n,n_scen,
        buf.d_qx,buf.d_qy,buf.d_qz,
        buf.d_vx,buf.d_vy,buf.d_vz,
        buf.d_m0,buf.d_is_device,buf.d_m_eff,
        initial.planet,initial.asteroid,
        buf.d_scenario_dev_id,buf.d_scenario_destroy_step,
        buf.d_hit_step);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> hit_res(n_scen);
    HIP_CHECK(hipMemcpy(hit_res.data(), buf.d_hit_step,
                        n_scen*sizeof(int), hipMemcpyDeviceToHost));
    free_device_buffers_p3(buf);

    best_id=-1; best_cost=0;

    for (int s=0;s<n_scen;s++) {
        if (hit_res[s]==-2) {
            int d=cand[s];
            double t= scen_ds[s]*param::dt;
            double cost = param::get_missile_cost(t);
            if (best_id==-1 || cost<best_cost ||
                (fabs(cost-best_cost)<1e-6 && d<best_id)) {
                best_id=d; best_cost=cost;
            }
        }
    }

    if (best_id==-1) best_cost=0;
}

static void solve_problem3_gpu(
    const SystemState& initial,
    int hit_time_step_p2,
    const std::vector<int>& destroy_steps_base,
    int& best_id,
    double& best_cost)
{
    int dev_count=0;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    int use = std::min(2,dev_count);

    if (use<2) {
        solve_problem3_single_gpu(initial,hit_time_step_p2,destroy_steps_base,best_id,best_cost);
        return;
    }

    int n = initial.n;
    if (hit_time_step_p2==-2) { best_id=-1; best_cost=0; return; }

    std::vector<int> cand;
    for (int i=0;i<n;i++)
        if (initial.type[i]=="device" &&
            destroy_steps_base[i]!=-1 &&
            destroy_steps_base[i] <= hit_time_step_p2)
            cand.push_back(i);

    if (cand.empty()) { best_id=-1; best_cost=0; return; }

    int total_scen = cand.size();
    int mid = total_scen/2;

    std::vector<int> cand0(cand.begin(),cand.begin()+mid);
    std::vector<int> cand1(cand.begin()+mid,cand.end());

    struct Ctx {
        int dev_id=0,n_scen=0;
        bool used=false;
        DeviceBuffersP3 buf;
        std::vector<int> scen_dev, scen_ds, hit;
    };

    Ctx ctx[2];

    std::vector<double> m0(initial.m);
    std::vector<int> is_dev(n);
    for (int i=0;i<n;i++) is_dev[i]=(initial.type[i]=="device");

    auto setup = [&](Ctx& c, int dev, const std::vector<int>& v){
        if (v.empty()) return;
        c.used=true; c.dev_id=dev; c.n_scen=v.size();
        c.scen_dev=v; 
        c.scen_ds.resize(c.n_scen); 
        c.hit.assign(c.n_scen,-2);

        HIP_CHECK(hipSetDevice(dev));
        allocate_device_buffers_p3(c.buf,n,c.n_scen);

        for (int s=0;s<c.n_scen;s++) {
            int off=s*n;
            HIP_CHECK(hipMemcpy(c.buf.d_qx+off,initial.qx.data(),n*sizeof(double),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(c.buf.d_qy+off,initial.qy.data(),n*sizeof(double),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(c.buf.d_qz+off,initial.qz.data(),n*sizeof(double),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(c.buf.d_vx+off,initial.vx.data(),n*sizeof(double),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(c.buf.d_vy+off,initial.vy.data(),n*sizeof(double),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(c.buf.d_vz+off,initial.vz.data(),n*sizeof(double),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(c.buf.d_m0+off,m0.data(),n*sizeof(double),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(c.buf.d_is_device+off,is_dev.data(),n*sizeof(int),hipMemcpyHostToDevice));
            c.scen_ds[s] = destroy_steps_base[c.scen_dev[s]];
        }

        HIP_CHECK(hipMemcpy(c.buf.d_scenario_dev_id,c.scen_dev.data(),c.n_scen*sizeof(int),hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(c.buf.d_scenario_destroy_step,c.scen_ds.data(),c.n_scen*sizeof(int),hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(c.buf.d_hit_step,c.hit.data(),c.n_scen*sizeof(int),hipMemcpyHostToDevice));
    };

    setup(ctx[0],0,cand0);
    setup(ctx[1],1,cand1);

    for (auto& c:ctx) {
        if (!c.used) continue;
        HIP_CHECK(hipSetDevice(c.dev_id));
        hipLaunchKernelGGL(p3_multi_scenario_kernel,
            dim3(c.n_scen), dim3(BLOCK_SIZE),0,0,
            n,c.n_scen,
            c.buf.d_qx,c.buf.d_qy,c.buf.d_qz,
            c.buf.d_vx,c.buf.d_vy,c.buf.d_vz,
            c.buf.d_m0,c.buf.d_is_device,c.buf.d_m_eff,
            initial.planet,initial.asteroid,
            c.buf.d_scenario_dev_id,c.buf.d_scenario_destroy_step,
            c.buf.d_hit_step);
    }

    for (auto& c:ctx) {
        if (!c.used) continue;
        HIP_CHECK(hipSetDevice(c.dev_id));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(c.hit.data(),c.buf.d_hit_step,
                            c.n_scen*sizeof(int), hipMemcpyDeviceToHost));
        free_device_buffers_p3(c.buf);
    }

    best_id=-1; best_cost=0;

    auto upd=[&](Ctx& c){
        for (int s=0;s<c.n_scen;s++) {
            if (c.hit[s]==-2) {
                int d=c.scen_dev[s];
                double t=c.scen_ds[s]*param::dt;
                double cost=param::get_missile_cost(t);
                if (best_id==-1 || cost<best_cost ||
                    (fabs(cost-best_cost)<1e-6 && d<best_id)) {
                    best_id=d; best_cost=cost;
                }
            }
        }
    };

    upd(ctx[0]);
    upd(ctx[1]);

    if (best_id==-1) best_cost=0;
}

int main(int argc, char** argv) {
    if (argc!=3) throw std::runtime_error("must supply 2 arguments");

    SystemState state;
    read_input(argv[1],state);

    HIP_CHECK(hipSetDevice(0));

    double min_dist = solve_problem1_gpu(state);

    std::vector<int> destroy_steps_base;
    int hit_time_step = solve_problem2_and_destroy_steps(state, destroy_steps_base);

    int best_id;
    double best_cost;
    solve_problem3_gpu(state,hit_time_step,destroy_steps_base,best_id,best_cost);

    write_output(argv[2],min_dist,hit_time_step,best_id,best_cost);
    return 0;
}
