import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1' #Necessary
import pyopencl as cl
import numpy as np
from scipy.misc import imread, imsave

#Read in image
img = imread('mid5.jpg', flatten=True).astype(np.float32)

# Get platforms, both CPU and GPU
plat = cl.get_platforms()
CPU = plat[0].get_devices()
try:
    GPU = plat[1].get_devices()
except IndexError:
    GPU = "none"

#Create context for GPU/CPU
if GPU!= "none":
    ctx = cl.Context(GPU)
else:
    ctx = cl.Context(CPU)

# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

# Kernel function
src = '''
void sort(int *a, int *b, int *c) {
   int swap;
   if(*a > *b) {
      swap = *a;
      *a = *b;
      *b = swap;
   }
   if(*a > *c) {
      swap = *a;
      *a = *c;
      *c = swap;
   }
   if(*b > *c) {
      swap = *b;
      *b = *c;
      *c = swap;
   }
}
__kernel void medianFilter(__global float *img, __global float *result, __global int *width, __global int *height)
{
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    // Keeping the edge pixels the same
    if( posx == 0 || posy == 0 || posx == w-1 || posy == h-1 )
    {
        result[i] = img[i];
    }
    else
    {
       // int pixel00, pixel01, pixel02, pixel10, pixel11, pixel12, pixel20, pixel21, pixel22;
        
        
        int pixel02 = img[i-2*w-1];
        
        int pixel12 = img[i-w-1];
        
        int pixel20 = img[i-3];
        int pixel21 = img[i-2];
        int pixel22 = img[i-1];
        int pixel23 = img[i];
        int pixel24 = img[i+1];
        
        int pixel32 = img[i+w-1];
        
        int pixel42 = img[i+2*w-1];
        
        
        //step-1
        sort( &(pixel02), &(pixel12), &(pixel20) );
        sort( &(pixel21), &(pixel22), &(pixel23) );
        sort( &(pixel24), &(pixel32), &(pixel42) );
        
        //step-2    
        sort( &(pixel02), &(pixel21), &(pixel24) );
        
        if(pixel02>=70){
            
            //70 is based on threshold
            
            if(img[i]>=255-pixel02){
            
            result[i] = 0; //pixel02;
         }
         else{
         
         result[i]=img[i];
         }
        
        }
        else{
        result[i] = img[i];
        
        }
        
       
    }
}
'''

#Kernel function instantiation
prg = cl.Program(ctx, src).build()
#Allocate memory for variables on the device
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))
# Call Kernel. Automatically takes care of block/grid distribution
prg.medianFilter(queue, img.shape, None , img_g, result_g, width_g, height_g)
result = np.empty_like(img)
cl.enqueue_copy(queue, result, result_g)

# Show the blurred image
imsave('mid4.jpg',result)
