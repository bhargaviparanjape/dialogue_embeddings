import json
import h5py
import pickle
import numpy as np

input_file = "swda_mrda_ami_avg_elmo.json"
output_hdf5_file = "swda_mrda_ami_avg_elmo.hdf5"
output_pkl_file = "swda_mrda_ami_avg_elmo.pkl"

embeddings = {}
with open(input_file) as fin:
	for line in fin:
		dict = json.loads(line)
		embeddings[dict["id"]] = dict["embeddings"]

h_file = h5py.File(output_hdf5_file,  "w")
dt = h5py.special_dtype(vlen=np.dtype('float64'))
feature_length = 1024
pkl_file = output_pkl_file

conversation_id_map = {}
num_conversations = len(embeddings)

average_elmo_features = h_file.create_dataset(
			'average_elmo', (num_conversations, feature_length), dtype=dt)

conversation_processed = 0
for key,values in embeddings.items():
	conversation_id_map[key] = conversation_processed
	average_elmo = np.array(values)
	average_elmo_features[conversation_processed] = average_elmo.transpose()

h_file.close()
pickle.dump(conversation_id_map, open(pkl_file, 'wb'))