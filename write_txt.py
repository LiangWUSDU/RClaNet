import os


def ListFilesToTxt(dir, file, wildcard, recursion):
	exts = wildcard.split(" ")
	files = os.listdir(dir)
	#files.sort(key=lambda x:int(x[:-15]))
	for name in files:
		fullname = os.path.join(dir, name)
		if (os.path.isdir(fullname) & recursion):
			ListFilesToTxt(fullname, file, wildcard, recursion)
		else:
			for ext in exts:
				if (name.endswith(ext)):
					(filename, extension) = os.path.splitext(name)
					file.write(name + "\n")
					break


def Test():
	dir = "new_weights/MA/MHCNN_kf10_10/"
	outfile = "new_weights/MA/MHCNN_kf10_10/weights.txt"
	wildcard = ".hdf5"

	file = open(outfile, "w")
	ListFilesToTxt(dir, file, wildcard, 1)

	file.close()


Test()