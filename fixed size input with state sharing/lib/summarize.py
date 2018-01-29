def summarize_epoch(epoch, n_epochs, error):
	# init
	n_loader_units = 20
	n_epochs -= 1

	print("\r",end = "[")
	print(int(((epoch)/n_epochs)*n_loader_units)*"=", end = "")
	if epoch >= n_epochs:
		print("=",end = "")
	else:
		print(">", end = "")
	print(int(((n_epochs - epoch-1)/n_epochs)*n_loader_units)*".", end = "]\t")

	print("Accuracy:",error, end = "\t\t")
	if epoch >= n_epochs:
		print()