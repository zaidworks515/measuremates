from flask import Flask, request, render_template, g
# import cv2
# import base64
# from io import BytesIO
# from PIL import Image
# from ultralytics import YOLO
# from datetime import datetime, timedelta
# from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from flask_cors import cross_origin
# ###

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SUPPRESS_EXCEPTIONS'] = True

# @app.before_request
# def load_model():
	
# 	# MODEL LOADING FROM AZURE BLOB STORAGE
# 	# enter credentials
# 	account_name = 'measuremates1'
# 	account_key = 'mxZFnhTH5KfuKufbdNXxZVV3gzeJPL6rjE9eLAxKPu26nZ2GEim6LfpDkKS5l3LnYAoLT4Aumzlw+AStnpLzCw=='
# 	container_name = 'yolo-model'
	
	
# 	#create a client to interact with blob storage
# 	connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
# 	blob_service_client = BlobServiceClient.from_connection_string(connect_str)
	
	
# 	# use the client to connect to the container
# 	container_client = blob_service_client.get_container_client(container_name)
	
	
# 	# get a list of all blob files in the container
# 	blob_list = []
# 	for blob_i in container_client.list_blobs():
# 		blob_list.append(blob_i.name)
	
	
# 	#generate a shared access signature for each blob file
# 	url_list=[]
	
# 	for blob_i in blob_list:
# 		sas_i = generate_blob_sas(account_name = account_name,
# 									container_name = container_name,
# 									blob_name = blob_i,
# 									account_key=account_key,
# 									permission=BlobSasPermissions(read=True),
# 									expiry=datetime.utcnow() + timedelta(hours=1))
		
# 		sas_url = 'https://' + account_name+'.blob.core.windows.net/' + container_name + '/' + blob_i + '?' + sas_i

# 		url_list.append(sas_url)
	
# 	model = YOLO(url_list[0], task='pose')
# 	g.model = model

@app.route('/')
@cross_origin()
def home():
	return render_template('index.html')



# @app.route('/evaluate', methods=['GET', 'POST'])
# @cross_origin()
# def detect_objects():
#     model = getattr(g, 'model', None)
    
#     if model is None:
#         return {'error': 'Model not loaded'}
    
#     if request.method =='POST':
#         try:
#             loaded_image = request.files['image']
#             image = cv2.imread(loaded_image)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             results = model(source=image, save=False, conf=0.6, task='pose')
            
#             pred_image = results[0].plot()
#             pred_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
            
#             # Convert images to base64 format
#             input_image_base64 = encode_image(image_rgb)
#             output_image_base64 = encode_image(pred_rgb)
            
#             return render_template('result.html', input_image_base64=input_image_base64, output_image_base64=output_image_base64)
        
#         except Exception as e:
#             return {'error': str(e)}
        
#     else:
    	# return render_template('index.html')


def encode_image(image):
	image_pil = Image.fromarray(image)
	buffer = BytesIO()
	image_pil.save(buffer, format="PNG")
	image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
	return f"data:image/png;base64,{image_base64}"

if __name__ == "__main__":
	app.run(host='0.0.0.0', threaded=True, port=8000)
	
