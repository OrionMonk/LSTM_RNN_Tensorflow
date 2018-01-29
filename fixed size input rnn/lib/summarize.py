def summarize_epoch(epoch, n_epochs, error):
	# init
	n_loader_units = 20
	epoch += 1

	print("\r", epoch,"/", n_epochs,end = "\t[")
	print(int(((epoch-1)/n_epochs)*n_loader_units)*"=", end = "")
	if epoch >= n_epochs:
		print("=",end = "")
	else:
		print(">", end = "")
	print(int(((n_epochs - epoch)/n_epochs)*n_loader_units)*".", end = "]\t")

	print("Error:",error, end = "\t\t")
	if epoch >= n_epochs:
		print()