#include <THC/THC.h>

#include "gpc.h"
#include "pia_wrapper_gpu.h"

extern THCState *state;

// preds(b, n, bpp, 7) boxes(b, nb, 7)
// boxes format: [w, d, h, theta, lx, ly, tz]
// preds format: [w, d, h, theta, lx, ly, tz]
// output: iou_out (b, nb, n, bpp)
int gpc_wrapper(int b, int n, int bpp, int nb, THCudaTensor *preds_tensor,
		THCudaTensor *boxes_tensor, /*THCudaTensor *i_obj_tensor, */
		/* THCudaLongTensor *target_box_tensor, */
		THCudaTensor *iou_out_tensor) {

	const float *preds_gpu = THCudaTensor_data(state, preds_tensor);
	const float *boxes_gpu = THCudaTensor_data(state, boxes_tensor);
	/* float *i_obj = THCudaTensor_data(state, i_obj_tensor); */
	/* long *target_box = THCudaLongTensor_data(state, target_box_tensor);
	 */
	float *iou_out_gpu = THCudaTensor_data(state, iou_out_tensor);

	const size_t preds_size = b * n * bpp * 7 * sizeof(float);
	float *preds_mem = (float *)malloc(preds_size);
	float *__restrict__ preds = preds_mem;

	const size_t boxes_size = b * nb * 7 * sizeof(float);
	float *boxes_mem = (float *)malloc(boxes_size);
	float *__restrict__ boxes = boxes_mem;

	const size_t iou_size = b * nb * n * bpp * sizeof(float);
	float *iou_out_mem = (float *)malloc(iou_size);
	float *__restrict__ iout_out = iou_out_mem;

	cudaMemcpy(preds, preds_gpu, preds_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(boxes, boxes_gpu, preds_size, cudaMemcpyDeviceToHost);

	for (int batch_index = 0; batch_index < b; ++batch_index) {
		for (int box_index = 0; box_index < nb; ++box_index) {
			float b_lx = boxes[j * 7 + 4];
			float b_ly = boxes[j * 7 + 5];
			float b_tz = boxes[j * 7 + 6];

			float b_theta = boxes[j * 7 + 3];

			float b_w = boxes[j * 7 + 0];
			float b_d = boxes[j * 7 + 1];
			float b_h = boxes[j * 7 + 2];

			float true_rect_vol = b_w * b_d * b_h;
		}

		boxes += nb * 7;
		preds += n * bpp * 7;
		iou_out +=
	}

	free(preds_mem);
	free(boxes_mem);
	free(iou_out_mem);
	return 1;
}
