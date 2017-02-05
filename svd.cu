#include "kernel.cu"

#include <stdio.h>
#include <stdlib.h>

int isNumber(char c){
	if(c<'0'||c>'9'){
		return 1;
	}else{
		return 0;
	}
}
  
//初始化一个二维矩阵
double* getMatrix(int rows,int columns){
    double *rect=(double*)calloc(rows*columns,sizeof(double));
    return rect;
}
  
//返回一个单位矩阵
double* getIndentityMatrix(int columns){
    double* IM=getMatrix(columns,columns);
    int i;
    for(i=0;i<columns;i++)
        IM[i*columns+i]=1.0;
    return IM;
}

//从文件读取矩阵
double* getMatrixFromFile(int *r,int *c){
	FILE *fp;
	fp=fopen("D:\\visual studio 2010\\Projects\\svd_two\\svd_two\\matrix.txt","r");
	int i,j,k;
	int sign;
	float d;
	int row=0,col=0;
	char buf[8];
	fscanf(fp,"%s",buf);
	for(i=0;i<8;i++){
		if(isNumber(buf[i]))
			break;
		row*=10;
		row+=buf[i]-'0';
	}
	fscanf(fp,"%s",buf);
	for(i=0;i<8;i++){
		if(isNumber(buf[i]))
			break;
		col*=10;
		col+=buf[i]-'0';
	}
	*r=row;
	*c=col;
	double *a=getMatrix(row,col);
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			sign=1;
			d=0;
			a[i*col+j]=0;
			fscanf(fp,"%s",buf);
			for(k=0;k<8;k++){
				if(buf[k]=='-'){
					sign=-1;
					continue;
				}else if(buf[k]=='.'){
					d=0.1;
					continue;
				}else if(isNumber(buf[k])){
					break;					
				}
				if(d==0.0){
					a[i*col+j]*=10;	
					a[i*col+j]+=(buf[k]-'0')*sign;
				}else{
					a[i*col+j]+=(buf[k]-'0')*sign*d;
					d/=10;
				}
			}
		}
	}
	fclose(fp);
	return a;
}

//根据列范数大小排序信息（P）生成ring序列(待完善)
struct ringPair* getRingPair(int * p,int col){
	int i,j,temp;
	int n=col/2;
	int k=n-1;
	struct ringPair *rp=(struct ringPair *)malloc(col*(col-1)/2*sizeof(struct ringPair));
	int *list1=(int *)malloc(n*sizeof(int));
	int *list2=(int *)malloc(n*sizeof(int));
	for(i=0;i<n;i++){
		list1[i]=p[i*2];
		list2[i]=p[i*2+1];
	}
	for(i=0;i<(col-1);i++){
		for(j=0;j<n;j++){
			rp[i*n+j].ro=list1[j];
			rp[i*n+j].co=list2[j];
		}
		if(k>=0){
			temp=list1[k];
			list1[k]=list2[k];
			list2[k]=temp;
			if(i%2==1){
				k--;
			}
			temp=list2[0];
			for(j=0;j<n-1;j++){
				list2[j]=list2[j+1];
			}
			list2[j]=temp;
		}
	}
	free(list1);
	free(list2);
	return rp;
}

int main(){
	int row=0,col=0;
	int i,j,nBlock,nThread;
	double *a=getMatrixFromFile(&row,&col);
	printf("/--  A  --/\n");
	for(i=0;i<row;i++){
		for(j=0;j<col;j++){
			printf("%.4f ",a[i*col+j]);
		}
		printf("\n");
	}
	printf("\n");

	//获得列范数
	int aSize=row*col;
	int vSize=col*col;
	double *anorm=(double *)malloc(sizeof(double)*col);
	
	double *dev_a;
	double *dev_anorm;
	cudaMalloc((void **)&dev_a,sizeof(double)*aSize);
	cudaMalloc((void **)&dev_anorm,sizeof(double)*col);
	cudaMemcpy(dev_a,a,sizeof(double)*aSize,cudaMemcpyHostToDevice);
	norm<<<1,col>>>(row,col,dev_a,dev_anorm);
	cudaMemcpy(anorm,dev_anorm,sizeof(double)*col,cudaMemcpyDeviceToHost);
	
	//jacobi旋转
	int n_element=col-1;
	int convergence=0;
	int *p=(int *)malloc(sizeof(int)*col);  //排序信息
	double *v=getIndentityMatrix(col);
	int n_clear;
	struct ringPair *rp;

	double *dev_tnorm;
	int *dev_p;
	int *dev_tp;
	double *dev_v;
	struct ringPair *dev_rp;
	int *dev_n_clear;
	cudaMalloc((void **)&dev_tnorm,sizeof(double)*col);
	cudaMalloc((void **)&dev_p,sizeof(int)*col);
	cudaMalloc((void **)&dev_tp,sizeof(int)*col);
	cudaMalloc((void **)&dev_v,sizeof(double)*vSize);
	cudaMalloc((void **)&dev_rp,sizeof(struct ringPair)*col*(col-1)/2);
	cudaMalloc((void **)&dev_n_clear,sizeof(int));
	cudaMemcpy(dev_v,v,sizeof(double)*vSize,cudaMemcpyHostToDevice);

	nBlock=col/2;
	nThread=col;
	int iter;
	for(iter=0;iter<30;iter++){
		//生成ring序列
		for(i=0;i<col;i++){
			p[i]=i;
		}
		cudaMemcpy(dev_p,p,sizeof(int)*col,cudaMemcpyHostToDevice);
		sort<<<1,col>>>(col,dev_anorm,dev_tnorm,dev_p,dev_tp);
		cudaMemcpy(p,dev_p,sizeof(int)*col,cudaMemcpyDeviceToHost);
		rp=getRingPair(p,col);

		n_clear=0;
		cudaMemcpy(dev_anorm,anorm,sizeof(double)*col,cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rp,rp,sizeof(struct ringPair)*col*(col-1)/2,cudaMemcpyHostToDevice);
		cudaMemcpy(dev_n_clear,&n_clear,sizeof(int),cudaMemcpyHostToDevice);
		
		//迭代
		for(i=0;i<(col-1);i++){
			one_side_jacobi<<<nBlock,nThread>>>(i,row,col,dev_a,dev_v,dev_anorm,dev_rp,dev_n_clear);
			norm<<<1,col>>>(row,col,dev_a,dev_anorm);
		}
		cudaMemcpy(anorm,dev_anorm,sizeof(double)*col,cudaMemcpyDeviceToHost);
		cudaMemcpy(&n_clear,dev_n_clear,sizeof(int),cudaMemcpyDeviceToHost);
		if(n_clear==n_element){
			convergence=1;
			break;
		}
	}
	if(convergence==0){
		printf("Did not converge in 30 sweeps！！\n");
	}

	//矩阵列按奇异值从大到小排序
	double *dev_at;
	double *dev_vt;
	cudaMalloc((void **)&dev_at,sizeof(double)*aSize);
	cudaMalloc((void **)&dev_vt,sizeof(double)*vSize);
	for(i=0;i<col;i++){
		p[i]=i;
	}
	cudaMemcpy(dev_p,p,sizeof(int)*col,cudaMemcpyHostToDevice);
	sort<<<1,col>>>(col,dev_anorm,dev_tnorm,dev_p,dev_tp);
	nBlock=col;
	nThread=row;
	svdSort<<<nBlock,nThread>>>(col,dev_anorm,dev_p,dev_a,dev_at,dev_v,dev_vt);
	
	//打印结果
	cudaMemcpy(a,dev_a,sizeof(double)*aSize,cudaMemcpyDeviceToHost);
	cudaMemcpy(anorm,dev_anorm,sizeof(double)*col,cudaMemcpyDeviceToHost);
	cudaMemcpy(v,dev_vt,sizeof(double)*vSize,cudaMemcpyDeviceToHost);
	printf("/--  U  --/\n");
	for(i=0;i<row;i++){
		for(j=0;j<row;j++){
			printf("%.4f ",a[i*col+j]);
		}
		printf("\n");
	}
	printf("\n");
	printf("/--  W  --/\n");
	for(i=0;i<row;i++){
		printf("%.4f ",anorm[i]);
	}
	printf("\n\n");
	printf("/--  V  --/\n");
	for(i=0;i<col;i++){
		for(j=0;j<col;j++){
			printf("%.4f ",v[i*col+j]);
		}
		printf("\n");
	}

	cudaFree(dev_a);
	cudaFree(dev_at);
	cudaFree(dev_anorm);
	cudaFree(dev_tnorm);
	cudaFree(dev_p);
	cudaFree(dev_tp);
	cudaFree(dev_v);
	cudaFree(dev_vt);
	cudaFree(dev_rp);
	cudaFree(dev_n_clear);
	free(a);
	free(anorm);
	free(p);
	free(v);
	free(rp);
	getchar();
	return 0;
}
