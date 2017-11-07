import requests
from pymongo import MongoClient
client = MongoClient('mongodb://dataminer:miner1@ds151955.mlab.com:51955/firedata')
db = client.firedata

header = {
	'token':'gNiZQiWhajbhniafBVHeloreGplUGuZR'
}
api_url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
def makeRequest(endpoint,payload):
	r = requests.get(api_url + endpoint,headers= header,params =payload)
	try:
		return r.json()
	except:
		return r.text
makeRequest('datasets',{})
req = makeRequest('data',{'datasetid':'GSOM','startdate':'2000-01-01','enddate':'2000-12-31'})

db.weather.insert_one(req)