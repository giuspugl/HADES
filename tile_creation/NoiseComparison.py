import fileinput
import os 
import sys

tempFile=open('hades/params.py','r+')

for line in fileinput.input('hades/params.py'):
	lineIn='noise_power = *.'
	lineOut='noise_power = 3.'
	tempFile.write(line.replace(lineIn,lineOut))
	
tempFile.close()
	
