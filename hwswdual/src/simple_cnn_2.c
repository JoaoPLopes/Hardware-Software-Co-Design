/*
 * simple_cnn_2.c
 *
 *  Created on: 24/05/2019
 *      Author: User
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "image.h"
#include "simple_cnn.h"
#include "xaxidma.h"
#include "xparameters.h"
#include <stdio.h>
#include "xtime_l.h"
#include "xil_mmu.h"
#include "xil_cache.h"
//#include "xil_cache_l.h"

/* Device hardware build related constants. */
#define DMA_DEV_ID	 XPAR_AXIDMA_0_DEVICE_ID
#define XPAR_CPU_ID 1U
#define HALF_A (12 * 24 * 25)
#define HALF_BIAS (12 * 24)

volatile unsigned char *ch_images;  // Images data region
volatile float *fp_weights; // Network weights data region

/************************** Function Prototypes ******************************/

int XAxiDma_Simple_MatProd(u16 DeviceId);
int init_XAxiDma_SimplePollMode(u16 DeviceId);


/************************** Variable Definitions *****************************/
/*
 * Device instance definitions
 */
XAxiDma AxiDma;

volatile float *fp_image; // Scaled floating-point image to be processed


volatile int * sync_f = (int *) 0xFFFF0000;

#define PROC0_STARTED 11
#define PROC1_STARTED 22
#define PROC1_COMPLETED 33

volatile float *matA;  // Auxiliary matrix A
volatile float *matAT;  // Transpose of matA
volatile float *matB;   // Auxiliary matrix B of 22 feature maps with 5*5 weights each
volatile float *matBT;  // Transpose of matB
volatile float *matC;   // Auxiliary matrix C with intermediate output (before adding bias) in convolutional layer
volatile float *matCT;   // Transpose of matC
volatile float *matCbias; // Output of convolutional layer 22 images of size 24*24
volatile float *matCpool; // Output of pooling layer 22 images of size 12*12
volatile float *matConn;  // Intermediate output (before adding bias) of fully connected layer (10 elements)
volatile float *matConnB; // Output of fully connected layer (10 elements)
volatile float *matSoftM; // Output of softmax layer (10 elements)


void define_memory_regions()
{
  static float *paddress = (float *)MEM_DATA_BASE_ADDRESS;

  // Region Size NIMAGES*IMAGE_HEIGTH*IMAGE_WIDTH+16 = 78416 Bytes (100 images)
  ch_images = (unsigned char *)MEM_IMAGES_BASE_ADDRESS;
  // Region Size TOTAL_WEIGTHS*sizeof(float) = 29330*4 = 117320 Bytes
  fp_weights = (volatile float *)MEM_WEIGTHS_BASE_ADDRESS;

  // Region Size IMAGE_HEIGTH*IMAGE_WIDTH*sizeof(float) = 28*28*4 = 3136 Bytes
  fp_image = paddress;
  paddress += 28*28;

  // Aux matrix of (24*24)*(25) elements. Region Size = 14400 * 4 = 57600 Bytes
  matA = paddress;
  paddress += (24*24)*(25);
  // Transpose of matA. Region Size = 14400 * 4
  matAT = paddress;
  paddress += (24*24)*(25);
  // Aux matrix of (22)*(25) elements. Region Size = 550 * 4 Bytes;
  matB = paddress;
  paddress += (22)*(25);
  // Transpose of matB. Region Size = 550 * 4 Bytes
  matBT = paddress;
  paddress += (22)*(25);
  // Aux matrix of (24*24)*(22) elements. Region Size = 12672 * 4 Bytes;
  matC = paddress;
  paddress += (24*24)*(22);
  // Transpose of matC. Region Size = 11520 * 4 Bytes
  matCT = paddress;
  paddress += (24*24)*(22);
  // Aux matrix of (22)*(24*24) elements. Region Size = 11520 * 4 Bytes
  matCbias = paddress;
  paddress += (22)*(24*24);
  // Aux matrix of (22)*(12*12) elements. Region Size = 3168 * 4 Bytes
  matCpool = paddress;
  paddress += (22)*(12*12);
  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes;
  matConn = paddress;
  paddress += 10;
  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes
  matConnB = paddress;
  paddress += 10;
  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes
  matSoftM = paddress;

  // printf("%p, %d\n", (void *)paddress+10, (paddress+10)-(float *)MEM_DATA_BASE_ADDRESS);
  // Total data region size is 71898 * 4 = 287,592 Bytes

}

void prepare_matrixA()
{
  while (*sync_f != PROC0_STARTED); // Wait for P0
  *sync_f = PROC1_STARTED;

  int i, j, k, row, col;

  for (i=12; i<24; i++) {
    for (j=0; j<24; j++) {
      for (k=0; k<25; k++) {
        row = i + k/5;
        col = j + (k%5);

        // Matrix A has (24*24) rows and 25 columns
        matA[(i*24+j)*25+k] = fp_image[row*IMAGE_WIDTH+col];
      }
    }
  }

  *sync_f = PROC1_COMPLETED;
  Xil_DCacheFlushRange((INTPTR)((float *)matA+ 24*24*25*(1/2)+1)  , (unsigned)(4 * 24*24*25*(1/2)));
}


void gemm(float *A, float *B, float *C, float *bias, int rowsA, int colsA, int colsB)
{
  while (*sync_f != PROC0_STARTED); // Wait for P0
  *sync_f = PROC1_STARTED;
  int i, j, k;

  for (i=rowsA/2; i<rowsA; i++) {
    for (j=0; j<colsB; j++) {
      C[i*colsB+j] = bias[j];
      for (k=0; k<colsA; k++) {
	C[i*colsB+j] += A[i*colsA+k] * B[k*colsB+j];
      }
    }
  }
  *sync_f = PROC1_COMPLETED;
   Xil_DCacheFlushRange((INTPTR)((float *)matA+ 5)  , (unsigned)(4*5));
}

void forward_connected_layer()
{
    float *matW, *matIN, *mbias, *matOUT, *matOutB;

    // The 10 bias values of this layer are stored after the 22+550 convolutional bias+weigths
    mbias = (float *)fp_weights + 22 + 550;
    // The 10*2880 weights are stored after the 10 bias values
    matW = (float *)fp_weights + 22 + 550 + 10;

    matIN = (float *)matCpool;
    matOUT = (float *)matConn;
    matOutB = (float *)matConnB;

    // A(10*3168) * B(3168*1) -> C(10*1)
    gemm(matW, matIN, (float *)matOutB, mbias, 10, 3168, 1);
    // print_fp((float *)matConn, 10, "Connected");
    // print_fp(mbias, 10, "Bias");

    //add_bias(matOUT, 10, 1, mbias, (float *)matOutB, 0);
    // print_fp((float *)matConnB, 10, "Connected+Bias");
    // Output vector ConnB has 10 values, one for each digit
}

void forward_maxpool_layer()
{
	while (*sync_f != PROC0_STARTED); // Wait for P0
	*sync_f = PROC1_STARTED;


  int i, j, k, n, m, row, col, index;
  int size=2, stride=2;
  int oh=12, ow=12;
  int ih=24, iw=24, chan=22;
  float max = -FLT_MAX, val;
  float *pout, *pin;

  pin = (float *)matCbias;
  pout = (float *)matCpool;

  for(k = chan/2; k < chan; ++k){
    for(i = 0; i < oh; ++i) {
      for(j = 0; j < ow; ++j) {
	max = -FLT_MAX;
	for(n = 0; n < size; ++n){
	  for(m = 0; m < size; ++m){
	    row = i*stride + n;
	    col = j*stride + m;
	    index = col + iw * (row + ih * k);
	    val = pin[index] ;
	    max = (val > max) ? val : max;
	  }
	}
	pout[j + ow * (i + oh * k)] = max;
      }
    }
  }
  *sync_f = PROC1_COMPLETED;
   Xil_DCacheFlushRange((INTPTR)(pout +11*12*12)  , (unsigned)(4*11*12*12));
  // print_fp((float *)matCpool, 120, "Pool");
  // Output matrix Cpool is 22*144, that is this layer outputs 22 12*12 images.
}



int main(int argc, char **argv){

  //Xil_DCacheDisable();
  Xil_SetTlbAttributes(0xFFFF0000,0x14de2);


  define_memory_regions();
 // prepare_matrixA();
  forward_maxpool_layer();
  forward_connected_layer();

}
