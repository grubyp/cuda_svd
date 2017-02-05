#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

struct ringPair{
	int ro;
	int co;
};

//获得列范数
__global__ static void norm(int row,int col,double *a,double *anorm){
	int tid=threadIdx.x;
	int i;
	anorm[tid]=0;
	for(i=0;i<row;i++){
		anorm[tid]+=a[i*col+tid]*a[i*col+tid];
	}
}

//归并排序
__global__ static void sort(int col,double *anorm,double *tnorm,int *p,int *tp){
	int tid=threadIdx.x;
	int i,j,k;
	int n=1;  //归并排序两组元素个数
	while(n<col){
		if(tid%(n*2)==0){
			for(i=tid,j=tid+n,k=tid;;k++){
				if(anorm[i]>anorm[j]){
					tnorm[k]=anorm[i];
					tp[k]=p[i];
					i++;
					if(i==tid+n){
						for(;j<tid+n+n;j++){
							k++;
							tnorm[k]=anorm[j];
							tp[k]=p[j];
						}
						break;
					}
				}else{
					tnorm[k]=anorm[j];
					tp[k]=p[j];
					j++;
					if(j==tid+n+n){
						for(;i<tid+n;i++){
							k++;
							tnorm[k]=anorm[i];
							tp[k]=p[i];
						}
						break;
					}
				}
			}
		}
		n*=2;
		__syncthreads();
		anorm[tid]=tnorm[tid];
		p[tid]=tp[tid];
		__syncthreads();
	}
}

__global__ static void one_side_jacobi(int i,int row,int col,double *a,double *v,double *anorm,struct ringPair *rp,int *n_clear){
	int bid=blockIdx.x;
	int tid=threadIdx.x;
	int e=1e-8;
	__shared__ int ro;
	__shared__ int co;
	__shared__ double d;
	d=0;
	if(tid==0){
		int n=i*(col/2)+bid;
		ro=rp[n].ro;
		co=rp[n].co;
		int j;
		for(j=0;j<row;j++){
			d+=a[j*col+ro]*a[j*col+co];
		}
	}
	__syncthreads();
	if(fabs(d)>col*e*sqrt(anorm[ro]*anorm[co])){
		double ct=(anorm[ro]-anorm[co])/(2*d);
		int sign;
		if(ct>0){
			sign=1;
		}else if(ct==0){
			sign=0;
		}else{
			sign=-1;
		}
		double t=sign/(fabs(ct)+sqrt(1+ct*ct));
		double c=1/(sqrt(1+t*t));
		double s=t*c;
		double vr=c*a[tid*col+ro]+s*a[tid*col+co];
		double vc=-s*a[tid*col+ro]+c*a[tid*col+co];
		a[tid*col+ro]=vr;
		a[tid*col+co]=vc;
		vr=c*v[tid*col+ro]+s*v[tid*col+co];
		vc=-s*v[tid*col+ro]+c*v[tid*col+co];
		v[tid*col+ro]=vr;
		v[tid*col+co]=vc;
	}else{
		(*n_clear)++;
	}
}

//矩阵排序
__global__ static void svdSort(int col,double *anorm,int *p,double *a,double *at,double *v,double *vt){
	int bid=blockIdx.x;
	int tid=threadIdx.x;
	int rows=tid*col;
	at[rows+bid]=a[rows+p[bid]];
	vt[rows+bid]=v[rows+p[bid]];
	//列范数开方（W）
	__syncthreads();
	anorm[bid]=sqrt(anorm[bid]);
	a[rows+bid]=at[rows+bid]/anorm[bid];
}