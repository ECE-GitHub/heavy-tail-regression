import numpy as np
import statsmodels.api as sma
import statsmodels.robust.scale as smrs
import statsmodels.nonparametric.kernel_regression as KernelReg



#Ordinary Least Square
def ols_reg(data):
	"""
	:param data: csv data
	:retruns : regression object 
	"""
	y = data.y
	x = data.x
	x = sma.add_constant(x)
	reg=sma.OLS(y, x).fit()
	theta= reg.params
	return reg


#Weighted Least Square
def wls_reg(data):
	"""
	:param data: csv data
	:returns: regression coefficients
	"""
	x, y = data.x, data.y
	x = sma.add_constant(x)
	#First estimate variance
	reg = sma.WLS(y, x).fit()
	log_resid =  np.log(reg.resid**2)
	varEst = np.exp(KernelReg.KernelReg(endog=y,
						exog=log_resid, 
						var_type="c",
						reg_type="ll").fit()[0])
	#Use the inverted variance as the weight
	reg=sma.WLS(endog=y,
				exog=sma.add_constant(x),
				weights=varEst**(-1)).fit()
	theta = reg.params
	return theta


#Robust Huber Regression
def huber_reg(data, budget = 10, eps_tolerance = 1e-6):
	"""
	:param data: csv data
	:param budget: computational budget
	:param eps_tolerance: epsilon tolerance
	returns: regression coefficients
	"""

	# Huber loss
	def huber_W(e,k):
		"""
		param e: residual
		param k: tuning constant
		returns: huber weight
		"""
		return 1.0 if abs(e)<=k else  k/float(abs(e))

	#initialize Weight and regression coefficients
	x, y = data.x, data.y
	x= sma.add_constant(x)
	reg = sma.WLS(y, x).fit()
	last_theta = reg.params
	W = np.diag(1.0/(reg.resid**2))
	huber_W = np.vectorize(huber_W)
	step =0
	while step <= budget:
		step +=1
		#Update Weight and regression coefficients
		theta = np.linalg.solve(np.dot(np.dot(x.T,W),x),np.dot(np.dot(x.T,W),y))
		norm_resid = (y - np.dot(theta,x.T))/smrs.mad(y - np.dot(theta,x.T))
		W = np.diag(huber_W(e=norm_resid,k=1.345))
		if np.max(theta - last_theta) < eps_tolerance: break
		last_theta = theta
	return theta
