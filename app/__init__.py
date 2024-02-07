from flask import Flask
from ultralytics import YOLO
# from datetime import datetime, timedelta
# from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SUPPRESS_EXCEPTIONS'] = True

# Model Loading Logic
def load_model():
    model_path = 'model/last.pt'
    model = YOLO(model_path, task='pose')
    return model

app.config['yolo_model'] = load_model()

# Azure Blob Storage Configuration (commented out for now)

# app.config['azure_credentials'] = {
#     'account_name': 'measuremates1',
#     'account_key': 'mxZFnhTH5KfuKufbdNXxZVV3gzeJPL6rjE9eLAxKPu26nZ2GEim6LfpDkKS5l3LnYAoLT4Aumzlw+AStnpLzCw==',
#     'container_name': 'yolo-model',
# }

# Define other configurations as needed

# Import your routes
from app import routes
