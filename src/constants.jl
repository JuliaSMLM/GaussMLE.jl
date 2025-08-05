#List of Global Constants

global const BSZ = 128      # max number of threads per block 
global const MEM = 3872     #3872		11616//!< shared 
global const IMSZBIG = 21	#maximum fitting window size
global const NK = 128		#number of blocks to run in each kernel
global const MAXPARAMS = 6  #maximum number of fitting parameters
global const NV_P = 4		#number of fitting parameters for MLEfit (x,y,bg,I)
global const NV_PS = 5		#number of fitting parameters for MLEFit_sigma (x,y,bg,I,Sigma)
global const NV_PS2 = 6		#number of fitting parameters for MLEFit_sigmaxy (x,y,bg,I,Sx,Sy)
global const NV_PZ = 5		#number of fitting parameters for astigmatic z-fit (x,y,z,bg,I)