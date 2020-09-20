#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<32,16> datai_t;
typedef ap_fixed<64,32> datao_t;
typedef ap_fixed<32,16> op_t;
typedef ap_fixed<64,32> mul_t;
typedef ap_fixed<64,32> acc_t;

struct ap_i_axis{
  datai_t data;
  ap_uint<1> last;
};
struct ap_o_axis{
  datao_t  data;
  ap_uint<1> last;
};

// The top-level function
void axis_fixed_macc(
      hls::stream<ap_o_axis> &strm_out,
      hls::stream<ap_i_axis> &strm_in
      )
{
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS interface axis port=strm_in
#pragma HLS INTERFACE axis port=strm_out

   struct ap_i_axis tmp;
   struct ap_o_axis tmpa;
   static op_t op1, op2;
   static mul_t mult;
   static acc_t acc;
   static ap_uint<9> vect_size;
   static datai_t localmem[512];
   int i;
   static datai_t bias;

   for (i=0; i<512; i++) {
	   tmp = strm_in.read();
	   localmem[i] = tmp.data;
	   if (tmp.last == 1) break ;
   }
   vect_size = i;
   tmp = strm_in.read();
   bias = tmp.data;
   for (; ; ) {
#pragma HLS loop_flatten off
	   acc = bias;
	   for (i=0; i<512; i++) {
#pragma HLS pipeline
		   tmp = strm_in.read();
		   op1 = (op_t)(localmem[i]);
		   op2 = (op_t)(tmp.data);
		   mult = op1 * op2;
		   acc += mult;
		   if (i==vect_size) break;
	   }
	   tmpa.last = tmp.last;
	   tmpa.data = (datao_t)acc;
	   strm_out.write(tmpa);
	   if (tmp.last == 1) break ;
   }
}
