from os.path import join as pjoin
import os 
import pickle
class ResultSaver(object):
	def __init__(self, path):
		self.path = pjoin(path, "data.p")
		self.data = { 
			"f1_train" : [],
			"EM_train" : [],
			"f1_val" : [],
			"EM_val" : [],
			"batch_indicies" : [],
			"losses" : []
		}
		if os.path.exists(path):
			self.load_data()

	def load_data(self):
		prev_data = pickle.load(open(self.path, "rb"))
		self.data["f1_train"] = prev_data["f1_train"]
		self.data["EM_train"] = prev_data["EM_train"]
		self.data["f1_val"] = prev_data["f1_val"]
		self.data["EM_val"] = prev_data["EM_val"]
		self.data["batch_indicies"] = prev_data["batch_indicies"]
		self.data["losses"] = prev_data["losses"]

	def save(self, key, value):
		self.data[key].append(value)
		pickle.dump(self.data, open(self.path, "wb"))

	def get(self, key):
		return(self.data[key])

	def is_empty(self, key):
		if len(self.data[key]) == 0:
			return(True)
		else:
			return(False)

