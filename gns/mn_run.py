#import standard modules 
import numpy as np
import sys

#import custom modules
try:
	import pymultinest
except:
	pass
from . import ns_loop_funcs
from . import prob_funcs

def MNRun(priorParams, LLhoodParams, paramNames):
	"""
	Run pyMultiNest for toy models specified by priorParams, LLhoodParams.
	This is pretty much deprecated.
	"""
	output = '../text_output/mn_test'
	nDims = len(paramNames)
	ns_loop_funcs.checkInputParamsShape(priorParams, LLhoodParams, nDims)
	priorObjs = prob_funcs.fitPriors(priorParams)
	priorFuncsPpf = prob_funcs.getPriorPpfs(priorObjs)

	def invPriorWrap(cube, nDims, nParams):
		"""
		wrapper around prob_funcs.invPrior(), as MN only takes prior function with cube, ndim, nparams arguments.
		Need to be higher order function (defined inside MNRUN()) so that priorFuncsPpf is in scope. Can't pass it as argument
		to function as MN calls function with only arguments described above
		"""
		cubeArr = cube2numpyArr(cube, nDims)
		physArr = prob_funcs.invPrior(cubeArr, priorFuncsPpf)
		numpyArr2Cube(physArr, cube, nDims)
	LhoodObj = prob_funcs.fitLhood(LLhoodParams)
	LLhoodFunc = prob_funcs.LLhood(LhoodObj)
	
	def LLhoodWrap(cube, nDims, nParams):
		"""
		wrapper around evaluating, as MN only takes LLhood function with cube, ndim, nparams arguments.
		Need to be higher order function (defined inside MNRUN()) so that LLhoodFunc is in scope. Can't pass it as argument
		to function as MN calls function with only arguments described above
		"""
		cubeArr = cube2numpyArr(cube, nDims)
		return LLhoodFunc(cubeArr)

	pymultinest.run(LLhoodWrap, invPriorWrap, nDims, verbose = True, resume = False, sampling_efficiency = 0.8, n_live_points = 500, outputfiles_basename = output)

def MNRun2(invPriorFunc, LLhoodFunc, paramNames, outputFile, wrapped = False):
	"""
	Run pyMultiNest for given invPriorFunc and LLhoodFunc.
	Should be more useful than MNRun() 
	"""
	nDims = len(paramNames)

	def invPriorWrap(cube, nDims, nParams):
		"""
		As above but don't need to fit inv prior func.
		Not sure the cube->array->cube conversion still necessary
		"""
		cubeArr = cube2numpyArr(cube, nDims)
		physArr = invPriorFunc(cubeArr)
		numpyArr2Cube(physArr, cube, nDims)

	def LLhoodWrap(cube, nDims, nParams):
		"""
		As above but don't need to fit LLhood func
		Not sure the cube->array conversion still necessary
		"""
		cubeArr = cube2numpyArr(cube, nDims)
		return LLhoodFunc(cubeArr)

	if wrapped:
		wrapped_params = [1] * nDims
	else:
		wrapped_params = None
	pymultinest.run(LLhoodWrap, invPriorWrap, nDims, verbose = True, resume = False, sampling_efficiency = 0.8, n_live_points = 500, outputfiles_basename = outputFile, wrapped_params = wrapped_params) #nlive low value is 50, high value is 500, massive value is 2000. Since May 19, running with wrapped params.

def cube2numpyArr(cube, nDims):
	"""
	Convert from 'cube' object given by MN to np array
	used by prior/ Lhood functions
	"""
	arr = np.zeros(nDims)
	for i in range(nDims):
		arr[i] = cube[i]
	return arr.reshape(1, -1)

def numpyArr2Cube(physArr, cube, nDims):
	"""
	Convert back from np array to MN cube object ready for MN 
	to manipulate
	"""
	for i in range(nDims):
		cube[i] = physArr[0,i]
