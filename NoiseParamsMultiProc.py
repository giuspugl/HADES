import numpy as np
import os
import subprocess

if __name__=='__main__':
	indices=np.arange(400,500)
	def runner(index):
		print 'starting index %s' %index
		job="python -u hades/NoiseParams.py %s" %index
		p=subprocess.Popen(job)#Popen(string,stdout=subprocess.PIPE,shell=True)
		(output,err)=p.communicate()
		p_status=p.wait()
		print 'index %s complete' %index
		#print "Command output: " + output
		
	import os,subprocess
	#os.system("python -u hades/NoiseParams.py 345")
	#runner(345)
	
	import multiprocessing as mp
	count=mp.cpu_count()
	p=mp.Pool(processes=40)
	#p.map(runner,indices)
	import tqdm
	res=tqdm.tqdm(p.imap(runner,indices),total=len(indices))
	#subprocess.check_output(['ls','-ls'])
	#p.join()
	print 'All tasks complete'

