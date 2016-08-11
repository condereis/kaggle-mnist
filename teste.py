import requests
import sys

# The direct link to the Kaggle data set
data_url = 'https://www.kaggle.com/c/digit-recognizer/download/test.csv'

# The local path where the data set is saved.
local_filename = "train.csv"
print('before request')
# Kaggle Username and Password
kaggle_info = {'UserName': "rafael.lcreis@gmail.com", 'Password': "bhunji030615"}

# Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.get(data_url)
print('after request')

# Login to Kaggle and retrieve the data.
r = requests.post(r.url, data = kaggle_info, stream = True)
print('after login')
# Writes the data to a local file one chunk at a time.
with open(local_filename, "wb") as f:
	print "Downloading %s" % local_filename
	total_length = r.headers.get('content-length')
	dl = 0
	for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
	    if chunk: # filter out keep-alive new chunks
	    	f.write(chunk)
	    	if total_length != None: # no content length header
	    		dl += len(chunk)
	        	total_length = int(total_length)
		        done = int(50 * dl / total_length)
		        sys.stdout.write("\r[%s%s] %s%%" % ('=' * done, ' ' * (50-done), 2*done) )    
		        sys.stdout.flush()
