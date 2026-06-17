/* ----------------------------------------------------------------------
   FastSOG GPU — Optimized pipeline:
     GPU spread → CPU FFT/Green → GPU interp
   Uses CubeS2 CUDA kernels for O(N_atoms) operations.
   FFT stays on CPU (grid is small: 24³～14400 points).
------------------------------------------------------------------------- */

#include "fastsog_gpu.h"
#include "fastsog_spline.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "gpu_extra.h"
#include "math_const.h"
#include "modify.h"

using namespace LAMMPS_NS;
using namespace MathConst;

extern "C" {
  int  fastsog_gpu_spread(const int nlocal, const float *x, const float *y,
        const float *z, const float *q, const int nlocal_pad, float *rho,
        const float rho_scale, const float bx, const float by, const float bz,
        const float dxi, const float dyi, const float dzi, const int nx,
        const int ny, const int nz, const float xi);
  int  fastsog_gpu_interp(const int nlocal, const float *x, const float *y,
        const float *z, const float *q, const int nlocal_pad,
        const float *gx, const float *gy, const float *gz, const float qscale,
        const float bx, const float by, const float bz, const float dxi,
        const float dyi, const float dzi, const int nx, const int ny,
        const int nz, const float xi, float *fx, float *fy, float *fz);
  void fastsog_gpu_clear();
}

FastSOGGPU::FastSOGGPU(LAMMPS *lmp) : FastSOG(lmp) {
  gpu_ready=false; GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}
FastSOGGPU::~FastSOGGPU() { fastsog_gpu_clear(); }
void FastSOGGPU::init() { FastSOG::init(); gpu_ready=true; }
void FastSOGGPU::setup() { FastSOG::setup(); }

void FastSOGGPU::compute(int eflag, int vflag) {
  ev_init(eflag, vflag, 0);
  if (atom->natoms != natoms_original) { qsum_qsq(); natoms_original=atom->natoms; }
  if (qsqsum==0.0) { energy=0.0; for(int j=0;j<6;j++) virial[j]=0.0; return; }
  ensure_fft_plan();

  const bool we=(eflag&ENERGY_GLOBAL), wv=(vflag&(VIRIAL_PAIR|VIRIAL_FDOTR));
  energy=0.0; for(int j=0;j<6;j++) virial[j]=0.0;
  int nlocal=atom->nlocal; if (nlocal<=0) return;

  double **x=atom->x; double *q=atom->q;

  // Atom data → float
  std::vector<float> hx(nlocal),hy(nlocal),hz(nlocal),hq(nlocal);
  for(int i=0;i<nlocal;i++){hx[i]=(float)x[i][0];hy[i]=(float)x[i][1];hz[i]=(float)x[i][2];hq[i]=(float)q[i];}

  double vol=mesh_lx*mesh_ly*mesh_lz;
  float rs=(float)((double)(mesh_nx*mesh_ny*mesh_nz)/vol);
  float bx=(float)domain->boxlo[0],by=(float)domain->boxlo[1],bz=(float)domain->boxlo[2];
  float dxi=(float)((double)mesh_nx/mesh_lx),dyi=(float)((double)mesh_ny/mesh_ly),dzi=(float)((double)mesh_nz/mesh_lz);
  float xi=(float)((spline_type==4)?kCubes2Xi4:kCubes2Xi4);
  double qsc=force->qqrd2e*scale; float qsf=(float)qsc;
  size_t ng=mesh_rho.size();

  // ── Step 1: GPU charge spread ──
  std::vector<float> h_rho(ng,0.0f);
  int s=fastsog_gpu_spread(nlocal,hx.data(),hy.data(),hz.data(),hq.data(),
                            nlocal,h_rho.data(),rs,bx,by,bz,dxi,dyi,dzi,
                            mesh_nx,mesh_ny,mesh_nz,xi);
  if(s!=0) { FastSOG::compute(eflag,vflag); return; }

  // Copy to mesh_rho
  std::fill(mesh_rho.begin(),mesh_rho.end(),0.0);
  for(size_t i=0;i<ng;i++) mesh_rho[i]=(FFT_SCALAR)h_rho[i];

  // ── Step 2: CPU FFT + Green multiply ──
  double el=0.0, dl=0.0;
  std::array<double,6> fv_local = {0.0};
  {
  std::fill(mesh_fft_work.begin(),mesh_fft_work.end(),0.0);
  for(size_t i=0;i<ng;i++) mesh_fft_work[2*i]=mesh_rho[i];
  mesh_fft->compute(mesh_fft_work.data(),mesh_fft_work.data(),FFT3d::FORWARD);

  double si=1.0/(double)ng, s2=si*si;
  double tpx=MY_2PI/mesh_lx,tpy=MY_2PI/mesh_ly,tpz=MY_2PI/mesh_lz;

  for(int iz=0;iz<mesh_nz;iz++){
    int kzm=iz-mesh_nz*(2*iz/mesh_nz); double kz=tpz*(double)kzm;
    for(int iy=0;iy<mesh_ny;iy++){
      int kym=iy-mesh_ny*(2*iy/mesh_ny); double ky=tpy*(double)kym;
      for(int ix=0;ix<mesh_nx;ix++){
        int kxm=ix-mesh_nx*(2*ix/mesh_nx); double kx=tpx*(double)kxm;
        size_t idx=mesh_index(ix,iy,iz);
        double ge=mesh_green_energy[idx],gf=mesh_green_force[idx];
        if(ge==0.0&&gf==0.0){mesh_gradx[2*idx]=mesh_gradx[2*idx+1]=0.0;mesh_grady[2*idx]=mesh_grady[2*idx+1]=0.0;mesh_gradz[2*idx]=mesh_gradz[2*idx+1]=0.0;continue;}
        dl+=mesh_green_self[idx];
        double rr=(double)mesh_fft_work[2*idx],ri=(double)mesh_fft_work[2*idx+1];
        if(we) el+=s2*ge*(rr*rr+ri*ri);
        // Fourier virial (SOG-correct): W_{αβ} = s2*|ρ|²*(ge*δ_{αβ} - gv*k_α*k_β)
        if(wv) {
          double rho2=rr*rr+ri*ri;
          double gv=mesh_green_virial[idx];
          fv_local[0]+=s2*rho2*(ge - gv*kx*kx);
          fv_local[1]+=s2*rho2*(ge - gv*ky*ky);
          fv_local[2]+=s2*rho2*(ge - gv*kz*kz);
          fv_local[3]+=s2*rho2*(-gv*kx*ky);
          fv_local[4]+=s2*rho2*(-gv*kx*kz);
          fv_local[5]+=s2*rho2*(-gv*ky*kz);
        }
        double vr=si*gf*rr,vi=si*gf*ri;
        mesh_gradx[2*idx]=(FFT_SCALAR)(-kx*vi);mesh_gradx[2*idx+1]=(FFT_SCALAR)(kx*vr);
        mesh_grady[2*idx]=(FFT_SCALAR)(-ky*vi);mesh_grady[2*idx+1]=(FFT_SCALAR)(ky*vr);
        mesh_gradz[2*idx]=(FFT_SCALAR)(-kz*vi);mesh_gradz[2*idx+1]=(FFT_SCALAR)(kz*vr);
      }
    }
  }
  mesh_fft->compute(mesh_gradx.data(),mesh_gradx.data(),FFT3d::BACKWARD);
  mesh_fft->compute(mesh_grady.data(),mesh_grady.data(),FFT3d::BACKWARD);
  mesh_fft->compute(mesh_gradz.data(),mesh_gradz.data(),FFT3d::BACKWARD);
  } // end CPU k-space block

  // ── Step 3: GPU force interpolation ──
  std::vector<float> h_gx(ng),h_gy(ng),h_gz(ng);
  for(size_t i=0;i<ng;i++){h_gx[i]=(float)mesh_gradx[2*i];h_gy[i]=(float)mesh_grady[2*i];h_gz[i]=(float)mesh_gradz[2*i];}
  std::vector<float> h_fx(nlocal),h_fy(nlocal),h_fz(nlocal);
  s=fastsog_gpu_interp(nlocal,hx.data(),hy.data(),hz.data(),hq.data(),nlocal,
                        h_gx.data(),h_gy.data(),h_gz.data(),qsf,
                        bx,by,bz,dxi,dyi,dzi,mesh_nx,mesh_ny,mesh_nz,xi,
                        h_fx.data(),h_fy.data(),h_fz.data());
  if(s!=0) { FastSOG::compute(eflag,vflag); return; }

  // Accumulate forces
  std::array<double,6> vl={0.0};
  for(int i=0;i<nlocal;i++){
    double fsx=h_fx[i],fsy=h_fy[i],fsz=h_fz[i];
    atom->f[i][0]+=fsx;atom->f[i][1]+=fsy;atom->f[i][2]+=fsz;
    if(wv){vl[0]+=x[i][0]*fsx;vl[1]+=x[i][1]*fsy;vl[2]+=x[i][2]*fsz;vl[3]+=x[i][0]*fsy;vl[4]+=x[i][0]*fsz;vl[5]+=x[i][1]*fsz;}
  }

  if(we){double ea,da;MPI_Allreduce(&el,&ea,1,MPI_DOUBLE,MPI_SUM,world);MPI_Allreduce(&dl,&da,1,MPI_DOUBLE,MPI_SUM,world);
    double qsl=0;for(int i=0;i<nlocal;i++)qsl+=q[i]*q[i];double qsa;MPI_Allreduce(&qsl,&qsa,1,MPI_DOUBLE,MPI_SUM,world);
    energy=0.5*vol*ea;if(remove_self_interaction)energy-=qsa*da/(2.0*vol);energy-=self_coeff*qsa;energy*=qsc;}
  if(wv){
    double va[6];MPI_Allreduce(vl.data(),va,6,MPI_DOUBLE,MPI_SUM,world);
    // force·r retained for diagnostic (va), but primary virial is Fourier
    double vf_all[6];MPI_Allreduce(fv_local.data(),vf_all,6,MPI_DOUBLE,MPI_SUM,world);
    double vs=0.5*vol*qsc;
    for(int j=0;j<6;j++)virial[j]=vs*vf_all[j];
  }
  return;
}
