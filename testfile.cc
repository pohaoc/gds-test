#include <cstdlib>
#include <cstring>
#include <iostream>

#include <fcntl.h>
#include <assert.h>
#include <unistd.h>


#include <cuda_runtime.h>
#include "cufile.h"
#include "cufile_sample_utils.h"

using namespace std;

#define CHUNK_SIZE (64 * 1024UL)
#define MIN_FILE_SIZE ( 16 * 1024 * 1024 * 1024UL)

int main(int argc, char *argv[]){
	int fd, fdw;
	ssize_t ret = -1, count = 0;
	void *devPtr = NULL;
	unsigned long size;
       	size_t nbytes, bufOff = 0, fileOff = 0;
	CUfileError_t status;
	const char *TESTFILE;

	CUfileDescr_t cf_descr;
	CUfileHandle_t cf_handle;

	if(argc < 3){
		std::cerr << argv[0] << " readfile writefile gpuid " << std::endl;
		exit(1);
	}

	TESTFILE = argv[1];
	check_cudaruntimecall(cudaSetDevice(atoi(argv[2])));

	ret = open(TESTFILE, O_RDONLY | O_DIRECT);
	if(ret < 0){
		std::cerr << "file open error : " << TESTFILE << " "
			<< cuFileGetErrorString(errno) << std::endl;
		return -1;
	}
	fd = ret;
	memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
	cf_descr.handle.fd = fd;
	cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	status = cuFileHandleRegister(&cf_handle, &cf_descr);
	if ( status.err != CU_FILE_SUCCESS){
		std::cerr << "file register error: "
			<< cuFileGetErrorString(status) << std::endl;
		close(fd);
		return -1;
	}
	size = GetFileSize(fd);
	if (!size) {
		ret = -1;
		std::cerr << "file size empty: " << TESTFILE << std::endl;
		goto error;
	}
	size = std::min(size, MIN_FILE_SIZE);

	check_cudaruntimecall(cudaMalloc(&devPtr, size));
	check_cudaruntimecall(cudaMemset(devPtr, 0x00, size));
	check_cudaruntimecall(cudaStreamSynchronize(0));

	std::cout << size << '\n';
	std::cout << "reading file sequentially: " << TESTFILE 
		<< " chunk size : "  << CHUNK_SIZE  << std::endl;

	do {
		nbytes = std::min((size - fileOff), CHUNK_SIZE);
		ret = cuFileRead(cf_handle, (char *) devPtr + bufOff, nbytes, fileOff, 0);
		if(ret <0 ){
			if(IS_CUFILE_ERR(ret))
				std::cerr << "read failed :"
					<< cuFileGetErrorString(ret) << std::endl;
			else
				std::cerr << "read failed : "
					<< cuFileGetErrorString(errno) << std::endl;
			goto error1;
		}
		bufOff += nbytes;
		fileOff += nbytes;
		count++;
	}while (fileOff < size);

	std::cout << "Total chunks read : " << count  << std::endl;


error1:
	check_cudaruntimecall(cudaFree(devPtr));
error:
	cuFileHandleDeregister(cf_handle);
	close(fd);
	return ret;
}





