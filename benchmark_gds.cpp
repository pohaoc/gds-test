#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include "cufile.h"

#define CHUNK_SIZE 100
#define NUM_THREADS 10

#define cudaCheckError() {\
	cudaError_t e = cudaGetLastError();\
	if(e!=cudaSuccess){\
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
		exit(EXIT_FAILURE);\
	}	\
}

using namespace std;
using namespace chrono;


typedef struct thread_data
{
	void *devPtr;
	loff_t offset;
	loff_t devPtr_offset;
	CUfileHandle_t cfr_handle;
}thread_data_t;

static void *thread_fn(void *data)
{
	int ret;
	thread_data_t *t = (thread_data_t *)data;

	cudaSetDevice(0);
	cudaCheckError();

	for(int i = 0 ; i < 100 ; i++){
		ret = cuFileRead(t->cfr_handle, t->devPtr, CHUNK_SIZE, t->offset,
				t->devPtr_offset);
		if(ret < 0){
			fprintf(stderr, "cuFileRead failed with ret=%d\n", ret);
		}
	}
	fprintf(stdout, "Read Success file-offset %ld readSize %ld to GPU 0 buffer offset %ld size %ld\n",
			(unsigned long) t->offset, CHUNK_SIZE, (unsigned long) t->offset, CHUNK_SIZE);
	return NULL;
}

int main(int argc, char **argv){
	void *devPtr;
	size_t offset = 0;
	int fd;
	CUfileError_t status;
	CUfileDescr_t cfr_descr;
	CUfileHandle_t cfr_handle;

	thread_data t[NUM_THREADS];
	pthread_t thread[NUM_THREADS];

	if(argc < 2){
		fprintf(stderr, "Invalid Input.\n");
		exit(1);
	}

	fd = open(argv[1], O_RDWR | O_DIRECT);

	if(fd < 0){
		fprintf(stderr, "Unable to open file %s fd %d errno %d\n",
				argv[1], fd, errno);
		exit(1);
	}

	memset((void *)&cfr_descr, 0, sizeof(CUfileDescr_t));

	memset((void *)&cfr_descr, 0, sizeof(CUfileDescr_t));
	cfr_descr.handle.fd = fd;
	cfr_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cfr_handle, &cfr_descr);
	if (status.err != CU_FILE_SUCCESS) {
		printf("file register error: %s\n", CUFILE_ERRSTR(status.err));
		close(fd);
		exit(1);
	}

	cudaSetDevice(0);
	cudaCheckError();

	cudaMalloc(&devPtr, CHUNK_SIZE * NUM_THREADS);
	cudaCheckError();

	status = cuFileBufRegister(devPtr, CHUNK_SIZE * NUM_THREADS, 0);
	if (status.err != CU_FILE_SUCCESS){
		printf("Buffer register failed :%s\n", CUFILE_ERRSTR(status.err));
		cuFileHandleDeregister(cfr_handle);
		close(fd);
		exit(1);
	}

	for(int i = 0 ; i < NUM_THREADS ; i++){
	/* Every thread has same devPtr address; every thread 
	 * will share the same cuFileHandle
	 */
		t[i].devPtr = devPtr;
		t[i] .cfr_handle = cfr_handle;

	/*
	 * Every thread will work on different offset
	 * */
		t[i].offset = offset;
		t[i].devPtr_offset = offset;
		offset += CHUNK_SIZE;
	}

	auto start = high_resolution_clock::now();
	for(int i = 0; i < 10 ; i++){
		pthread_create(&thread[i], NULL, &thread_fn, &t[i]);
	}

	for(int i =0 ; i < 10 ; i++){
		pthread_join(thread[i], NULL);
	}
	auto end = high_resolution_clock::now();
	cout << "Time : " << (float)duration_cast<milliseconds>(end - start).count() << " msec" << endl;

	status = cuFileBufDeregister(devPtr);
	if(status.err != CU_FILE_SUCCESS){
		fprintf(stderr, "cuFileBufDeregister failed :%s\n", CUFILE_ERRSTR(status.err));
	}

	cuFileHandleDeregister(cfr_handle);
	close(fd);
	cudaFree(devPtr);
	return 0;
}

